# -*- coding: utf-8 -*-
#
import logging
import os
from functools import partial

import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np
import optax
from tqdm import tqdm

from bio_clip.data.downstream.go_term import prepare_dataloaders
from bio_clip.model.downstream.go_ec_term import DownstreamModel
from bio_clip.model.esm.esm_haiku import get_pretrained_esm_model
from bio_clip.train.downstream.partition_params import parameters_partition_fn
from bio_clip.train.downstream.trainer import Trainer, prepare_downstream_model
from bio_clip.train.pretrain.trainer import TrainingState
from bio_clip.utils.checkpointing import prepare_ft_pretraining_cfgs
from bio_clip.utils.go_ec_metrics import get_all_metrics
from bio_clip.utils.neptune import prepare_downstream_neptune
from bio_clip.utils.shared_metrics import compute_metrics
from bio_clip.utils.utils import ThreadedIterable, convert_to_ml_dict, show_git

jax.config.update("jax_platform_name", "cpu")


def prepare(cfg):
    config = convert_to_ml_dict(cfg)
    clip_cfg, config, devices = prepare_ft_pretraining_cfgs(config)

    batch_cfg = config.training.batch

    num_devices = len(devices)
    assert num_devices * batch_cfg.num_per_device_update == batch_cfg.batch_size

    get_train_dataloader, get_test_dataloader, baseline = prepare_dataloaders(
        cfg, num_devices
    )

    with get_train_dataloader() as dataloader:
        for i, (ids, dummy_batch_data, mask) in enumerate(ThreadedIterable(dataloader)):
            break

    # Get results for DeepFRI baseline for a given set of IDs
    k2i = {k: i for i, k in enumerate(baseline["proteins"])}
    baseline_pred = baseline["Y_pred"]
    baseline_target = baseline["Y_true"]

    def get_baseline_for_ids(ids):
        print(
            f"Using {len(ids)} ids of the {baseline_target.shape[0]} available in the "
            "baseline"
        )
        ixs = [k2i[k] for k in ids]
        return baseline_pred[ixs], baseline_target[ixs]

    n_classes = dummy_batch_data.target.shape[-1]

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
    loss_fn, inference_fn, params = prepare_downstream_model(
        lambda *args: DownstreamModel(
            *args, n_classes=n_classes, esm_forward=forward_esm_fn
        ),
        model_name,
        clip_cfg,
        dummy_batch_data,
        random_key,
        fine_tune_config,
    )
    if fine_tune_config.training.use_esm_embedding:
        # load esm params into params:
        for k, v in esm_parameters.items():
            params[f"{model_name}/{k}"] = v
        print("PREPARED MODEL")

    # Optimizer, loss function
    if config.training.optimiser.optimiser_type == "adamw":
        optimizer = optax.adamw(
            learning_rate=config.training.optimiser.learning_rate,
            weight_decay=config.training.optimiser.weight_decay,
        )
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

    trainable_params, _fixed_params = hk.data_structures.partition(
        _partition_fn, params
    )
    print(
        "optimising these parameters: "
        + "".join(["\n\t" + k for k in trainable_params])
    )
    print("fixing these parameters: " + "".join(["\n\t" + k for k in _fixed_params]))
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
        config,
        name_expe=f"FN-Pred-{config.training.data.ontology}",
        other_tags=[
            f"ontology_{config.training.data.ontology}",
        ],
    )

    show_git()
    return (
        config,
        n_classes,
        get_train_dataloader,
        get_test_dataloader,
        pmap_inference,
        pmap_update,
        neptune_run,
        training_state,
        get_baseline_for_ids,
    )


@hydra.main(
    config_path="../../bio_clip/config",
    version_base=None,
    config_name="go_term",
)
def main(cfg):
    required_vars = prepare(cfg)
    (
        config,
        n_classes,
        get_train_dataloader,
        get_test_dataloader,
        pmap_inference,
        pmap_update,
        neptune_run,
        training_state,
        get_baseline_for_ids,
    ) = required_vars

    # Training loop
    validation_frequency = config.training.validation_frequency
    for epoch_num in range(config.training.num_epochs):
        logging.warning(f"Epoch number {epoch_num}")

        if epoch_num % validation_frequency == 0:
            # Evaluation on test set (compute metrics like mcc, f1, etc.)
            all_predictions = []
            all_targets = []
            val_losses = []
            all_ids = []

            with get_test_dataloader() as dataloader:
                for ids, batch_fine_tuning, mask in tqdm(ThreadedIterable(dataloader)):
                    mask = mask.reshape(-1)  # this mask is only over batch dims.
                    loss, aux = pmap_inference(training_state, batch_fine_tuning)
                    all_predictions.append(
                        np.array(aux["prediction"]).reshape(-1, n_classes)[mask]
                    )
                    all_targets.append(
                        np.array(batch_fine_tuning.target).reshape(-1, n_classes)[mask]
                    )
                    val_losses.append(np.array(loss).reshape(-1)[mask])
                    all_ids += ids
            print("\n\nVALIDATION COMPLETE\n\n")
            all_targets = np.concatenate(all_targets, axis=0)
            all_predictions = np.concatenate(all_predictions, axis=0)
            val_losses = np.concatenate(val_losses, axis=0)
            assert len(all_ids) == val_losses.shape[0]

            deepfri_pred, deepfri_targets = get_baseline_for_ids(all_ids)

            metrics = {
                "bioclip": get_all_metrics(
                    target=all_targets.astype(np.int32), predicted=all_predictions
                ),
                "deepfri": get_all_metrics(
                    target=deepfri_targets.astype(np.int32), predicted=deepfri_pred
                ),
            }
            for algo, results in metrics.items():
                for k, v in results.items():
                    neptune_run[f"test_metrics/{algo}_{k}"].log(value=float(v))

            print("\nComputed metrics 1\n")

            all_predictions = all_predictions.reshape(-1).astype(np.float32)
            all_targets = all_targets.reshape(-1).astype(np.float32)
            print("uploading loss to neptune.")
            neptune_run[f"validation/loss"].log(value=float(np.mean(val_losses)))

            metrics = compute_metrics(all_targets, all_predictions)

            print("\nComputed metrics 2\n")

            logging.warning(f"Metrics  {metrics}")
            for k, v in metrics.items():
                neptune_run[f"test_metrics/{k}"].log(value=float(v))

        losses = []
        total_examples = 0
        print("\n\nRunning training...\n\n")

        with get_train_dataloader() as dataloader:
            thread = ThreadedIterable(dataloader)
            for ids, batch_fine_tuning, mask in thread:
                if (mask == 0).sum() > 0:
                    print("LR drop last batch hack for update")
                    break
                training_state, training_metrics = pmap_update(
                    training_state, batch_fine_tuning, mask
                )
                total_examples += np.sum(mask)
                loss = np.array(np.mean(training_metrics.loss) * np.sum(mask))
                losses.append(loss)
        avg_train_loss = np.sum(losses) / total_examples
        neptune_run[f"training/loss"].log(value=float(avg_train_loss))
    neptune_run.stop()
    os._exit(0)


if __name__ == "__main__":
    main()
