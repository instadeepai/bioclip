# -*- coding: utf-8 -*-
#
import collections
import json
import logging
import time
from functools import partial

import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np
import optax
import torch as pt
from sklearn import metrics
from tqdm import tqdm

from bio_clip.data.downstream.pesto_src.scoring import (
    bc_score_names,
    bc_scoring,
    nanmean,
)
from bio_clip.data.downstream.residue_ppi import prepare_resppi_dataloaders, tree_stack
from bio_clip.model.downstream.res_ppi import DownstreamModel
from bio_clip.model.esm.esm_haiku import get_pretrained_esm_model
from bio_clip.train.downstream.partition_params import parameters_partition_fn
from bio_clip.train.downstream.trainer import Trainer, prepare_downstream_model
from bio_clip.train.pretrain.trainer import TrainingState
from bio_clip.utils.checkpointing import device
from bio_clip.utils.neptune import prepare_downstream_neptune
from bio_clip.utils.utils import (
    ThreadedIterable,
    convert_to_ml_dict,
    get_logger,
    pad,
    show_git,
)

logger = get_logger(__name__)

jax.config.update("jax_platform_name", "cpu")


def prepare(cfg):
    config = convert_to_ml_dict(cfg)
    n_classes = config.training.metadata.nclasses

    # # Find devices
    # devices = jax.devices("tpu")
    config, devices = device(config)
    num_devices = len(devices)

    dataloaders, baselines_dict, clip_cfg = prepare_resppi_dataloaders(
        config, num_devices
    )
    assert all(v in dataloaders for v in config.training.validation_sets)

    batch_cfg = config.training.batch
    assert num_devices * batch_cfg.num_per_device_update == batch_cfg.batch_size

    # get a dummy batch with a target...
    for i in range(num_devices):
        print(f"INDEX: {i}")
        print(jax.tree_map(lambda x: x.shape, dataloaders["train"].dataset[i]))

    loss_vars = {
        "pos_ratios": jnp.ones(n_classes),
        "global_step": jnp.ones(1),
        "pos_weight_factor": jnp.ones(1),
    }
    dummy_batch_data = tree_stack(
        [dataloaders["train"].dataset[i] + (loss_vars,) for i in range(num_devices)]
    )

    def reshape(x, num_per_device):
        return jnp.reshape(x, (num_devices, num_per_device) + x.shape[1:])

    inf_reshape = partial(
        reshape, num_per_device=config.training.batch.num_per_device_inference
    )
    update_reshape = partial(
        reshape, num_per_device=config.training.batch.num_per_device_update
    )

    dummy_batch_data = jax.tree_map(
        partial(reshape, num_per_device=1), dummy_batch_data
    )
    print("BATCH DATA:")
    print(jax.tree_map(lambda x: x.shape, dummy_batch_data))

    number_of_transformer_blocks = int(
        (cfg.model.plm.esm_model_name).split("_")[1].replace("t", "")
    )
    esm_parameters, forward_esm_fn, tokenizer = get_pretrained_esm_model(
        cfg.model.plm.esm_model_name,
        embeddings_layers_to_save=[number_of_transformer_blocks],
    )

    number_of_transformer_blocks = int(
        (clip_cfg.model.plm.esm_model_name).split("_")[1].replace("t", "")
    )
    random_key = jax.random.PRNGKey(config.random_seed)
    fine_tune_config = config
    fine_tune_config.training.pad_token_id = tokenizer.pad_token_id
    fine_tune_config.training.embeddings_layer_to_save = number_of_transformer_blocks
    model_name = "fnpr_downstream_model"
    loss_fn, inference_fn, params, *prev_training_state = prepare_downstream_model(
        lambda *args: DownstreamModel(
            *args,
            n_classes=n_classes,
            esm_forward=forward_esm_fn,
            architecture=clip_cfg.model.architecture,
        ),
        model_name,
        clip_cfg,
        dummy_batch_data,
        random_key,
        fine_tune_config,
        return_training_state=config.training.checkpoints.write.resume,
    )
    if fine_tune_config.training.use_esm_embedding:
        # load esm params into params:
        for k, v in esm_parameters.items():
            params[f"{model_name}/{k}"] = v
        print("PREPARED MODEL")

    # Optimizer, loss function
    def lr_schedule(step):
        return config.training.optimiser.learning_rate

    if config.training.optimiser.optimiser_type == "adamw":
        _optimizer = optax.adamw(
            learning_rate=config.training.optimiser.learning_rate,
            weight_decay=config.training.optimiser.weight_decay,
        )
        if config.training.optimiser.learning_rate_from > 0:
            copt = config.training.optimiser

            def lr_schedule(step):  # noqa: F811
                return jnp.where(
                    step < copt.learning_rate_switch_at,
                    copt.learning_rate_from,
                    copt.learning_rate_to,
                )

            optimizer = optax.chain(
                optax.scale_by_schedule(lr_schedule),
                _optimizer,
            )
        else:
            optimizer = _optimizer
    elif config.training.optimiser.optimiser_type == "adam":
        optimizer = optax.adam(learning_rate=config.training.optimiser.learning_rate)
    else:
        raise NotImplementedError

    train_esm_from = cfg.training.train_esm_from
    if train_esm_from < 0:
        train_esm_from = number_of_transformer_blocks + train_esm_from - 1

    if config.training.optimise_everything_overide:

        def _partition_fn(module_name, _name, _value):
            return True

    else:
        _partition_fn = partial(
            parameters_partition_fn,
            first_trainable_gnn_layer=config.training.first_trainable_gnn_layer,
            gnn_layer_name="gat_layer",  # "mpnn_layer",
            model_name=model_name,
            train_esm_from=train_esm_from,
        )

    trainer = Trainer(
        inference_fn,
        loss_fn,
        optimizer,
        config.training.optimiser.optimiser_type,
        _partition_fn,
    )
    # Pmap the update and prediction function
    pmap_update = jax.pmap(
        trainer.update,
        axis_name="p",
        devices=devices,
        donate_argnums=(0,),
    )
    pmap_inference = jax.pmap(
        trainer.inference_and_loss, axis_name="p", devices=devices
    )

    param_sizes = jax.tree_map(lambda x: x.size, params)
    param_sizes = {k: sum(jax.tree_util.tree_leaves(v)) for k, v in param_sizes.items()}
    esm_size = sum(v for k, v in param_sizes.items() if "esm" in k)
    non_esm = {k: v for k, v in param_sizes.items() if "esm" not in k}
    print(non_esm)
    print(f"ESM NUM PARAMS: {esm_size}; NON-ESM PARAMS: {sum(non_esm.values())}")

    if len(prev_training_state):
        training_state = prev_training_state
    else:
        trainable_params, _fixed_params = hk.data_structures.partition(
            _partition_fn, params
        )
        print(
            "optimising these parameters: "
            + "".join(["\n\t" + k for k in trainable_params])
        )
        print(
            "fixing these parameters: " + "".join(["\n\t" + k for k in _fixed_params])
        )
        optimizer_state = optimizer.init(trainable_params)

        training_state = TrainingState(
            step=jnp.array(0),
            best_validation_cluster_loss=jnp.array(-1),
            best_validation_unif_loss=jnp.array(-1),
            params=params,
            optimizer_state=optimizer_state,
            random_key=random_key,
        )
    training_state = jax.device_put_replicated(training_state, devices)

    neptune_run = prepare_downstream_neptune(
        config, name_expe="Interface-Pred", other_tags=[]
    )

    show_git()
    return (
        config,
        num_devices,
        dataloaders,
        baselines_dict,
        inf_reshape,
        pmap_inference,
        n_classes,
        neptune_run,
        update_reshape,
        pmap_update,
        training_state,
        lr_schedule,
        _partition_fn,
    )


@hydra.main(
    config_path="../../bio_clip/config",
    version_base=None,
    config_name="residue_ppi",
)
def main(cfg):
    (
        config,
        num_devices,
        dataloaders,
        baselines_dict,
        inf_reshape,
        pmap_inference,
        n_classes,
        neptune_run,
        update_reshape,
        pmap_update,
        training_state,
        lr_schedule,
        _partition_fn,
    ) = prepare(cfg)

    # Set-up checkpointing:
    # Find the bioclip pre-trained run
    ckpt_cfg = cfg.training.checkpoints
    bioc_run = ckpt_cfg.checkpoint_dir.split("BIOC-")[-1].split("_")[0]
    ckpt_cfg.write.checkpoint_base = (
        f"{ckpt_cfg.write.checkpoint_base}using_BIOC_{bioc_run}"
    )

    class_names = ["protein", "DNA/RNA", "ion", "ligand", "lipid"]
    metrics_dict = collections.defaultdict(list)

    # LR: TODO; make global step and pos_ratios resumable
    pos_ratios = 0.5 * jnp.ones(n_classes)
    global_step = jnp.ones(1)
    pos_weight_factor = jnp.array(config.training.pos_weight_factor)
    inf_batch_size = num_devices * config.training.batch.num_per_device_inference

    # repeat the loss vars data along devices
    (pos_ratios, global_step, pos_weight_factor) = jax.tree_map(
        lambda x: x[None, ...].repeat(num_devices, axis=0),
        (pos_ratios, global_step, pos_weight_factor),
    )

    def nan(data):
        return any(jax.tree_leaves(jax.tree_map(lambda x: np.isnan(x).any(), data)))

    def catfl(x):
        return np.concatenate(x, axis=0).reshape(-1)

    def nanmedian(x):
        return np.median(x[~np.isnan(x)])

    def run_inference(name, dataloader, training_state, loss_vars, metrics_dict):
        val_losses = []
        scores = []
        # repeat along the per-device dim
        loss_vars = jax.tree_map(
            lambda x: x.repeat(config.training.batch.num_per_device_inference, axis=0),
            loss_vars,
        )
        if name == "masif":
            aucs = []

        pbar = tqdm(ThreadedIterable(dataloader))
        _start = time.time()
        for batch_fine_tuning in pbar:
            batch_time = time.time() - _start
            _start = time.time()
            # padding only occurs on the final items in the test set
            batch_fine_tuning, batch_mask = pad(batch_fine_tuning, inf_batch_size)

            batch_fine_tuning = jax.tree_map(
                inf_reshape, batch_fine_tuning + (loss_vars,)
            )
            pad_shape_time = time.time() - _start

            _start = time.time()
            loss, aux = pmap_inference(training_state, batch_fine_tuning)
            inference_time = time.time() - _start

            _start = time.time()

            def remove_batch_mask(x):
                return np.array(x).reshape(inf_batch_size, *x.shape[2:])[
                    np.array(batch_mask) == 1.0
                ]

            loss, aux = jax.tree_map(remove_batch_mask, (loss, aux))
            if nan(loss):
                print("NAN in eval loss")
            if nan(aux):
                print("NAN in eval aux")
            graph = batch_fine_tuning[0].graph
            # LR: It would be great to standardise the data tree,
            if config.model.architecture == "graph_transformer":
                *_, atom14_mask = graph
                residue_masks = atom14_mask.any(-1)
            else:
                residue_masks = jnp.squeeze(graph.nodes_mask, axis=-1)
            residue_masks = remove_batch_mask(residue_masks)
            unpad_time = time.time() - _start

            _start = time.time()
            # this is quite slow... could re-write in jax
            scores += [
                bc_scoring(pt.Tensor(targ[rmask]), pt.Tensor(pred[rmask]))
                for pred, targ, rmask in zip(
                    aux["prediction"],
                    remove_batch_mask(batch_fine_tuning[1]),
                    residue_masks,
                )
            ]
            scores_time = time.time() - _start
            pbar.set_description(
                f"inf: {inference_time:.3f}; metr: {scores_time:.3f}s; "
                f"loss: {np.mean(loss):.3f}; "
                f"bt: {batch_time:.3f}; ps: {pad_shape_time:.3f}; up: {unpad_time:.3f}"
            )
            val_losses.append(loss)

            if name == "masif":
                # calculate the auc method as in the benchmark notebook, only for
                # protein interactions.
                aucs += [
                    metrics.auc(
                        *metrics.precision_recall_curve(
                            targ[rmask, :1], pred[rmask, :1]
                        )[:2][::-1]
                    )
                    for pred, targ, rmask in zip(
                        aux["prediction"],
                        remove_batch_mask(batch_fine_tuning[1]),
                        residue_masks,
                    )
                ]
            _start = time.time()

        global_val_losses = catfl(val_losses)
        global_val_losses = np.mean(global_val_losses)

        neptune_run[f"{name}/loss"].log(
            value=float(global_val_losses), step=current_training_time
        )
        metrics_dict[f"{name}/loss"].append(float(global_val_losses))
        m_scores = nanmean(pt.stack(scores, dim=0)).numpy()
        scores_arr = np.stack(list(map(np.array, scores)), axis=-1)
        med_scores = np.array(
            [nanmedian(mtr) for mtr in scores_arr.reshape(-1, scores_arr.shape[-1])]
        ).reshape(m_scores.shape)
        print("COMPUTING METRICS")
        print(
            f"bc_score_names.shape: {len(bc_score_names)}; "
            f"m_scores.shape: {m_scores.shape}"
        )
        for k, mean_v, med_v in zip(bc_score_names, m_scores, med_scores):
            for interaction_type, mn_sc, md_sc in zip(class_names, mean_v, med_v):
                if not np.isnan(md_sc):
                    neptune_run[f"{name}/{interaction_type}/median/{k}"].log(
                        value=float(md_sc), step=current_training_time
                    )
                metrics_dict[f"{name}/{interaction_type}/median/{k}"].append(
                    float(md_sc)
                )
                if not np.isnan(mn_sc):
                    neptune_run[f"{name}/{interaction_type}/mean/{k}"].log(
                        value=float(mn_sc), step=current_training_time
                    )
                metrics_dict[f"{name}/{interaction_type}/mean/{k}"].append(float(mn_sc))

        if name == "masif":
            for stat, auc in {
                "mean": float(np.mean(aucs)),
                "median": float(np.median(aucs)),
            }.items():
                neptune_run[f"{name}/ppi_auc/{stat}_bioclip"].log(
                    value=auc, step=current_training_time
                )
                metrics_dict[f"{name}/ppi_auc/{stat}_bioclip"].append(auc)
        return metrics_dict

    def validate(metrics_dict, training_state, loss_vars):
        for val_name in config.training.validation_sets:
            metrics_dict = run_inference(
                val_name, dataloaders[val_name], training_state, loss_vars, metrics_dict
            )
        # upload the baseline...
        for stat, bd in baselines_dict.items():
            for k, v in bd["pr_aucs_bench"].items():
                if not np.isnan(v):
                    neptune_run[f"masif/ppi_auc/{stat}_{k}"].log(
                        value=v, step=current_training_time
                    )
                    metrics_dict[f"masif/ppi_auc/{stat}_{k}"].append(v)
            for baseline, vmet in bd["all_metrics_masif"].items():
                for metric_name, sc in vmet.items():
                    if not np.isnan(sc):
                        neptune_run[
                            f"masif/protein/{baseline}/{stat}_{metric_name}"
                        ].log(value=sc, step=current_training_time)
                    metrics_dict[
                        f"masif/protein/{baseline}/{stat}_{metric_name}"
                    ].append(sc)
        return metrics_dict

    # Training loop
    validation_frequency = config.training.validation_frequency
    within_train_epoch_validation = validation_frequency < 1
    if within_train_epoch_validation:
        step_freq = int(len(dataloaders["train"]) * validation_frequency)

    if ckpt_cfg.write.resume:
        df = neptune_run["masif/protein/median/auc"].fetch_values()
        if within_train_epoch_validation:
            pass
        else:
            starting_epoch = len(df.step.to_numpy())
        current_training_time = df.step.to_numpy()[-1]
        global_step = ...
        pos_ratios = ...
        pos_weight_factor = ...
        dont_validate_initially = True
    else:
        starting_epoch = 0
        current_training_time = 0.0
        dont_validate_initially = False
    for epoch_num in range(starting_epoch, config.training.num_epochs):
        logging.warning(f"Epoch number {epoch_num}")

        if epoch_num % validation_frequency == 0 and not within_train_epoch_validation:
            if dont_validate_initially:
                dont_validate_initially = False
            else:
                loss_vars = {
                    "pos_ratios": pos_ratios,
                    "global_step": global_step,
                    "pos_weight_factor": pos_weight_factor,
                }
                metrics_dict = validate(metrics_dict, training_state, loss_vars)

        # Training epoch
        epoch_metrics = []
        _start = time.time()
        pbar = tqdm(ThreadedIterable(dataloaders["train"]))
        times = {
            "prepare_batch": [],
            "update_model": [],
            "val_if": [],
            "pad": [],
            "loss_var": [],
            "ureshape": [],
        }
        for batch_fine_tuning in pbar:
            times["prepare_batch"].append(time.time() - _start)
            step_id = int(np.array(global_step[0]))

            _start = time.time()
            if within_train_epoch_validation and step_id % step_freq == 1:
                if dont_validate_initially:
                    dont_validate_initially = False
                else:
                    loss_vars = {
                        "pos_ratios": pos_ratios,
                        "global_step": global_step,
                        "pos_weight_factor": pos_weight_factor,
                    }
                    metrics_dict = validate(metrics_dict, training_state, loss_vars)
                    loss = metrics_dict["test/protein/median/auc"][
                        -1
                    ]  # auc == aucroc, using test as to not eval on masif
                    # checkpointer.step = step_id
                    # checkpointer.periodic_checkpoint(config, training_state, step_id)
            times["val_if"].append(time.time() - _start)

            global_step += 1

            _start = time.time()
            batch_fine_tuning, mask = pad(
                batch_fine_tuning, config.training.batch.batch_size
            )
            times["pad"].append(time.time() - _start)

            # LR temp hack:
            if np.any(np.array(mask) == 0):
                print(f"MASK: {np.array(mask)}")
                continue

            _start = time.time()
            loss_vars = {
                "pos_ratios": pos_ratios,
                "global_step": global_step,
                "pos_weight_factor": pos_weight_factor,
            }
            # repeat along the per-device dim, note that we usually have [device, 1, ...
            # ] the 1 is squeezed in the trainer.update
            loss_vars = jax.tree_map(
                lambda x: x.repeat(config.training.batch.num_per_device_update, axis=0),
                loss_vars,
            )
            times["loss_var"].append(time.time() - _start)

            _start = time.time()
            batch_fine_tuning = jax.tree_map(
                update_reshape, batch_fine_tuning + (loss_vars,)
            )
            mask = jax.tree_map(update_reshape, mask)
            times["ureshape"].append(time.time() - _start)

            _start = time.time()
            training_state, training_metrics, other = pmap_update(
                training_state, batch_fine_tuning, mask
            )
            update_time = time.time() - _start
            times["update_model"].append(update_time)
            current_training_time += update_time
            pos_ratios = other["pos_ratios"]
            if nan(training_metrics):
                print("NAN in training_metrics")
                continue
            epoch_metrics.append(training_metrics)
            loss = np.array(training_metrics.loss).reshape(-1)
            pbar.set_description(f"upd: {update_time:.3f}; loss: {np.mean(loss):.3f}")
            lr = np.squeeze(lr_schedule(global_step[0]))
            neptune_run["learning_rate"].log(
                value=float(lr), step=current_training_time
            )
            neptune_run["training/loss"].log(
                value=float(np.mean(loss)), step=current_training_time
            )
            metrics_dict[f"epoch_{epoch_num}/training/loss"].append(
                float(np.mean(loss))
            )

            logger.info(
                "Timings: " + str([f"{k}: {v[-1]:.4f}" for k, v in times.items()])
            )
            _start = time.time()
        neptune_run["timings/update_model"].log(
            value=float(np.mean(times["update_model"])), step=current_training_time
        )
        neptune_run["timings/update_model_total"].log(
            value=float(np.sum(times["update_model"])), step=current_training_time
        )
        neptune_run["timings/prepare_batch"].log(
            value=float(np.mean(times["prepare_batch"])), step=current_training_time
        )
        neptune_run["timings/prepare_batch_total"].log(
            value=float(np.sum(times["prepare_batch"])), step=current_training_time
        )

        project = config.training.neptune.project_name
        with open(f"/app/bio-clip/neptune_runs/{project}/metrics.json", "w") as f:
            f.write(json.dumps(metrics_dict))


if __name__ == "__main__":
    main()
