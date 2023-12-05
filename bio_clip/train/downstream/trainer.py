# -*- coding: utf-8 -*-
#
import functools
import os
import subprocess
from typing import Any, Callable, NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.profiler
import optax
from typing_extensions import TypeAlias

from bio_clip.train.pretrain.trainer import TrainingState

Metrics: TypeAlias = Tuple[Any, ...]


class TrainingMetrics(NamedTuple):
    """
    Contain a training metrics.
    """

    step: jnp.ndarray
    loss: jnp.ndarray
    metrics: Metrics


class Trainer:
    """A stateless abstraction around an init_fn/update_fn pair.
    This extracts some common boilerplate from the training loop.
    """

    def __init__(
        self,
        inference_fn: Callable,
        loss_fn: Callable,
        optimizer: optax.GradientTransformation,
        optimizer_type: str,
        parameters_partition_fn: Optional[Callable] = None,
    ):
        """
        Args:
            net_init (Callable): The forward function initialization function. Takes a
                an input and returns a state of initialized weights.
            loss_fn (Callable): Loss function.
            optimizer (optax.GradientTransformation): Optimizer.
            optimizer_type (str): which optimizer is used (adam or adamw)
        """
        self._inference_fn = inference_fn
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self.optimizer_type = optimizer_type
        self.parameters_partition_fn = parameters_partition_fn

    @functools.partial(jax.jit, static_argnums=0)
    def update(
        self, state: TrainingState, batch_data, mask
    ) -> Tuple[TrainingState, TrainingMetrics]:
        """Updates the training state.

        Args:
            state (TrainingState): Current training state.
            batch_data (BatchDataGeneric): batch data

        Returns:
            new_state (TrainingState): New training state.
            metrics (TrainingMetrics): Metrics.
        """
        rng = state.random_key
        params = state.params

        vloss = jax.vmap(self._loss_fn, in_axes=(None, None, 0))
        print(
            f"Trainer.update; batch_data: {jax.tree_map(lambda x: x.shape, batch_data)}"
        )
        print(f"Trainer.update; mask {mask}")

        def mask_mean(data, mask):
            broadcast_shape = tuple([slice(None)] + [None] * (data.ndim - 1))
            return jnp.mean(data, axis=0, where=mask[broadcast_shape])

        def average_vmapped_loss(params, rng, device_batch_data, mask):
            losses, auxs = vloss(params, rng, device_batch_data)
            return mask_mean(losses, mask), jax.tree_map(
                lambda x: mask_mean(x, mask), auxs
            )

        loss_grad_fn = jax.value_and_grad(average_vmapped_loss, argnums=0, has_aux=True)

        (loss, auxiliary_data), gradient = loss_grad_fn(params, rng, batch_data, mask)
        pesto_vars = "pos_ratios" in auxiliary_data
        if pesto_vars:
            other = {"pos_ratios": auxiliary_data["pos_ratios"]}
            del auxiliary_data["pos_ratios"]

        auxiliary_metrics = auxiliary_data["metrics"]

        loss = jax.lax.pmean(loss, axis_name="p")
        auxiliary_metrics_across_devices = tuple(
            jax.lax.pmean(a, axis_name="p") for a in auxiliary_metrics
        )
        # gradient = jax.tree_map(lambda g: jax.lax.pmean(g, axis_name="p"), gradient)
        gradient = jax.lax.pmean(gradient, axis_name="p")

        # update optimizer and weights
        optimizer_state = state.optimizer_state

        if self.parameters_partition_fn is not None:
            trainable_params, fixed_params = hk.data_structures.partition(
                self.parameters_partition_fn, params
            )
            trainable_grads, _fixed_grads = hk.data_structures.partition(
                self.parameters_partition_fn, gradient
            )
        else:
            trainable_params = params
            fixed_params = {}
            trainable_grads = gradient
        print(
            "optimising these parameters: "
            + "".join(["\n\t" + k for k in trainable_params])
        )
        print("fixing these parameters: " + "".join(["\n\t" + k for k in fixed_params]))

        if self.optimizer_type == "adamw":
            updates, optimizer_state = self._optimizer.update(
                trainable_grads, optimizer_state, params=trainable_params
            )
        elif self.optimizer_type == "adam":
            updates, optimizer_state = self._optimizer.update(
                trainable_grads, optimizer_state
            )

        trainable_params = optax.apply_updates(trainable_params, updates)
        rng, new_rng = jax.random.split(rng, num=2)

        params = hk.data_structures.merge(trainable_params, fixed_params)

        new_state = TrainingState(
            step=state.step + 1,
            best_validation_cluster_loss=state.best_validation_cluster_loss,
            best_validation_unif_loss=state.best_validation_unif_loss,
            params=params,
            optimizer_state=optimizer_state,
            random_key=new_rng,
        )

        metrics = TrainingMetrics(
            step=state.step, loss=loss, metrics=auxiliary_metrics_across_devices
        )
        if pesto_vars:
            return new_state, metrics, other
        else:
            return new_state, metrics

    @functools.partial(jax.jit, static_argnums=0)
    def inference_and_loss(self, state: TrainingState, batch_data) -> Any:
        """Computes inference over a batch of data and returns embeddings.

        Args:
            state (TrainingState): Training state.
            batch_data (BatchDataGeneric): batch data

        Returns:
            TrainingMetrics (TrainingMetrics): Training Metrics.
        """
        rng = state.random_key
        rng, _ = jax.random.split(rng)
        vinf = jax.vmap(self._loss_fn, in_axes=(None, None, 0))
        loss, auxiliary_data = vinf(state.params, rng, batch_data)
        return loss, auxiliary_data

    @functools.partial(jax.jit, static_argnums=0)
    def inference(self, state: TrainingState, batch_data) -> Any:
        """Computes inference over a batch of data and returns embeddings.

        Args:
            state (TrainingState): Training state.
            batch_data (BatchDataGeneric): batch data

        Returns:
            TrainingMetrics (TrainingMetrics): Training Metrics.
        """
        rng = state.random_key
        rng, _ = jax.random.split(rng)
        vinf = jax.vmap(self._inference_fn, in_axes=(None, None, 0))
        auxiliary_data = vinf(state.params, rng, batch_data)
        return auxiliary_data


def prepare_downstream_model(
    downstream_model_cls,
    downstream_model,
    clip_cfg,
    dummy_batch,
    random_key,
    fine_tune_config,
    return_training_state=False,
):
    feat = jax.tree_map(lambda x: x[0, 0], dummy_batch)

    transform = hk.transform(
        lambda x: downstream_model_cls(clip_cfg, fine_tune_config.training)(
            x, compute_loss=True
        )
    )
    inf_transform = hk.transform(
        lambda x: downstream_model_cls(clip_cfg, fine_tune_config.training)(
            x, compute_loss=False
        )
    )
    params = transform.init(random_key, feat)

    bio_clip_fn_name = "forward_bio_clip"
    if fine_tune_config.training.load_bioclip_params:
        if fine_tune_config.training.checkpoints.checkpoint_dir.startswith("s3://"):
            local_path = "checkpoint"
            os.makedirs(local_path, exist_ok=True)
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    fine_tune_config.training.checkpoints.checkpoint_dir,
                    local_path,
                    f"--endpoint={fine_tune_config.training.checkpoints.aws_endpoint}",
                    "--recursive",
                ],
                capture_output=True,
                check=True,
            )
        else:
            local_path = fine_tune_config.training.checkpoints.checkpoint_dir
        training_state = TrainingState.load(local_path)

        assert all(
            k.startswith(bio_clip_fn_name) or k.startswith("esm2")
            for k in training_state.params
        )
        not_loaded = []
        loaded = []
        for k, v in training_state.params.items():
            if "node_proj" in k:
                continue  # with the addition of the MLP to the GNN this is now a
            # different shape. in the ESM+GNN experiments we dont use the output of the
            # MLP after the GNN representation shapes: [(1024, 512), (1024, 512), (1024,
            # 512), (1024, 512), (1024, 2048)] these go into the
            # "fnpr_downstream_model/forward_bio_clip/node_weight" parameter
            if "node_weight" in k:
                continue

            if k.startswith("esm2"):
                newk = f"{downstream_model}/{k}"
            else:
                newk = k.replace(
                    bio_clip_fn_name, f"{downstream_model}/{bio_clip_fn_name}"
                )
            if "seq_" not in k:
                if newk in params:
                    # check the shapes match
                    old = jax.tree_util.tree_structure(params[newk])
                    new = jax.tree_util.tree_structure(v)
                    assert old == new, f"old: {old}; new: {new}"
                    shapes_match = jax.tree_map(
                        lambda a, b: a.shape == b.shape, params[newk], v
                    )
                    valid = all(jax.tree_util.tree_leaves(shapes_match))
                    assert valid, (
                        f"[{newk}] shape mismatch: (params[newk] == v) = "
                        f"{jax.tree_map(lambda x: x.shape, (params[newk], v))}"
                    )
                    params[newk] = v
                    loaded.append(newk)
            else:
                not_loaded.append(newk)
        print("-" * 80)
        print("SUCCESSFULLY LOADED:\n\t" + "\n\t".join(loaded))
        print("FAILED TO LOAD:\n\t" + "\n\t".join(not_loaded))
        print("-" * 80)
    else:
        print("NOT LOADING BIOCLIP PARAMS")
    ret = (transform.apply, inf_transform.apply, params)
    if return_training_state:
        return ret + (training_state,)
    else:
        return ret
