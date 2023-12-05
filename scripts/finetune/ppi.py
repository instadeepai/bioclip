# -*- coding: utf-8 -*-
#
import logging
from functools import partial

import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np
import optax

from bio_clip.data.downstream.ppi import cropping_dependent_processing
from bio_clip.model.downstream.ppi import DownstreamModel
from bio_clip.model.esm.esm_haiku import get_pretrained_esm_model
from bio_clip.train.downstream.partition_params import parameters_partition_fn
from bio_clip.train.downstream.trainer import Trainer, prepare_downstream_model
from bio_clip.train.pretrain.trainer import TrainingState
from bio_clip.utils.checkpointing import prepare_ft_pretraining_cfgs
from bio_clip.utils.neptune import prepare_downstream_neptune
from bio_clip.utils.shared_metrics import compute_metrics
from bio_clip.utils.utils import convert_to_ml_dict, show_git

logger = logging.getLogger()
logger.setLevel("INFO")

jax.config.update("jax_platform_name", "cpu")


@hydra.main(
    config_path="../../bio_clip/config",
    version_base=None,
    config_name="ppi",
)
def main(cfg):
    config = convert_to_ml_dict(cfg)
    clip_cfg, config, devices = prepare_ft_pretraining_cfgs(config)

    print("CONFIG:")
    print(config)
    batch_cfg = config.training.batch

    # Find devices
    DEBUG = False
    if DEBUG:
        devices = jax.devices()
        num_devices = len(devices)
        batch_cfg.num_per_device_update = 2
        batch_cfg.batch_size = 2
    else:
        num_devices = len(devices)

    assert num_devices * batch_cfg.num_per_device_update == batch_cfg.batch_size

    number_of_transformer_blocks = int(
        (config.model.plm.esm_model_name).split("_")[1].replace("t", "")
    )
    esm_parameters, forward_esm_fn, tokenizer = get_pretrained_esm_model(
        config.model.plm.esm_model_name,
        embeddings_layers_to_save=[number_of_transformer_blocks],
    )
    get_dataloader, n_data, processing_fn, rows = cropping_dependent_processing(
        config,
        tokenizer,
        num_devices,
        processed_h5_path="/app/bio-clip/h5_ppi_processed.h5",
    )
    neptune_run = prepare_downstream_neptune(
        config,
        name_expe="PPI-ppi",
        other_tags=[
            config.training.data.benchmark,
        ],
    )
    show_git()

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

    model_name = "ppi_downstream_model"
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
    trainer = None
    validation_frequency = config.training.validation_frequency

    prop = 0.8
    n_train = int(n_data * prop)
    all_ixs = list(range(n_data))
    np.random.shuffle(all_ixs)
    train_ixs = all_ixs[:n_train]
    test_ixs = all_ixs[n_train:]

    random_key = jax.random.PRNGKey(config.random_seed)

    number_of_transformer_blocks = int(
        (clip_cfg.model.plm.esm_model_name).split("_")[1].replace("t", "")
    )
    fine_tune_config = config
    fine_tune_config.training.pad_token_id = tokenizer.pad_token_id
    fine_tune_config.training.embeddings_layer_to_save = number_of_transformer_blocks

    ixs, dummy_batch, mask = next(iter(get_dataloader(test_ixs)))

    print(
        "fine_tune_config.use_gnn_embedding: "
        f"{fine_tune_config.training.use_gnn_embedding}"
    )
    loss_fn, inference_fn, params = prepare_downstream_model(
        lambda *args: DownstreamModel(*args, esm_forward=forward_esm_fn),
        model_name,
        clip_cfg,
        dummy_batch,
        random_key,
        fine_tune_config,
    )

    if trainer is None:
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

    # Training loop
    for epoch_num in range(config.training.num_epochs):
        logger.warning(f"Epoch number {epoch_num}")

        if epoch_num % validation_frequency == 0:
            # Evaluation on test set (compute metrics like mcc, f1, etc.)
            global_predictions = []
            global_target = []
            global_masks = []
            val_losses = []
            for ixs, batch_fine_tuning, mask in get_dataloader(test_ixs):
                valloss, auxiliary_data = pmap_inference(
                    training_state, batch_fine_tuning
                )
                print(f"Loss: {float(np.mean(valloss, where=np.array(mask))):.5f}")
                if np.array(jnp.isnan(valloss).any()):
                    continue

                val_losses.append(valloss)
                global_predictions.append(auxiliary_data["prediction"])
                global_target.append(batch_fine_tuning.target)
                global_masks.append(mask)

            def catsh(x):
                return np.concatenate(x, axis=0).reshape(-1)

            global_predictions = catsh(global_predictions)
            global_target = catsh(global_target)
            global_val_losses = catsh(val_losses)
            global_mask = catsh(global_masks).astype(bool)

            global_predictions = global_predictions[global_mask]
            global_target = global_target[global_mask]
            global_val_losses = global_val_losses[global_mask]
            neptune_run[f"validation/loss"].log(value=float(np.mean(global_val_losses)))

            # Remove NANs
            global_predictions = np.array(global_predictions.astype(jnp.float32))
            not_nan_indices = np.where(np.isnan(global_predictions) == 0)[0]
            global_predictions_ = global_predictions[not_nan_indices]
            global_target_ = global_target[not_nan_indices]

            print(f"global_target_: {global_target_.shape}: {global_target_.dtype}")
            print(
                f"global_predictions_: {global_predictions_.shape}: "
                f"{global_predictions_.dtype}"
            )
            print(f"global_target_: {global_target_}")
            print(f"global_predictions_: {global_predictions_}")
            metrics_test_set = compute_metrics(global_target_, global_predictions_)

            logger.warning(f"Metrics  {metrics_test_set}")
            for k, v in metrics_test_set.items():
                neptune_run[f"test_metrics/{k}"].log(value=float(v))

        # Training epoch
        epoch_metrics = []
        masks = []
        for ixs, batch_fine_tuning, mask in get_dataloader(train_ixs):
            training_state, training_metrics = pmap_update(
                training_state, batch_fine_tuning, mask
            )
            done_count = len(masks) * mask.reshape(-1).shape[0]
            prog = (
                f"Done: {done_count}/{n_train} "
                f"({epoch_num}/{config.training.num_epochs})"
            )
            print(f"Loss: {float(np.mean(training_metrics.loss)):.5f}; {prog}")
            epoch_metrics.append(training_metrics)
            masks.append(mask)
        losses = jnp.stack([t.loss for t in epoch_metrics])
        masks = np.stack(masks)
        neptune_run[f"training/loss"].log(value=float(jnp.sum(losses) / mask.sum()))


if __name__ == "__main__":
    main()
