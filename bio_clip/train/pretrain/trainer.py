import functools
import os
from typing import Any, Callable, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bio_clip.types import Dict, Metrics, RNGKey


def save_params(params: hk.Params, filename: str) -> jax.tree_util.PyTreeDef:
    """Save a neural network parameters to a .npz file.


    Args:
        params (hk.Params): Pytree of the parameters to be saved.
        filename (str): File name where it is to be saved

    Returns:
        tree_def: Parameters pytree definition.
    """
    flatten_params, tree_def = jax.tree_util.tree_flatten(params)
    jnp.savez(filename, *flatten_params)
    return tree_def


def load_params(filename: str, tree_def: jax.tree_util.PyTreeDef) -> hk.Params:
    """Load params from a .npz file.

    Args:
        filename (str): File name where it is to be saved
        tree_def: Parameters pytree definition.

    Returns:
        hk.Params: Parameters pytree.
    """
    with open(filename, "rb") as f:
        uploaded = jnp.load(f)
        arrays = [jnp.asarray(uploaded[file]) for file in uploaded.files]
        reconstructed_params = jax.tree_util.tree_unflatten(tree_def, arrays)
        return reconstructed_params


class TrainingState(NamedTuple):
    """
    Contain a full training state.
    """

    step: jnp.ndarray
    best_validation_cluster_loss: jnp.ndarray
    best_validation_unif_loss: jnp.ndarray
    params: hk.Params
    optimizer_state: optax.OptState
    random_key: jnp.ndarray

    @property
    def tree_def(self) -> jax.tree_util.PyTreeDef:
        return jax.tree_util.tree_structure(self)

    def save(self, save_dir: str, partition_fn: Callable = None):
        """
        Save state into .npz and .npy arrays. Filename should end with .npz.
        It returns the tree structure of the state, which is used to
        load the state.

        Args:
            save_dir (str): Directory where the training state is to be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        if partition_fn is not None:
            trainable_params, _ = hk.data_structures.partition(
                partition_fn, self.params
            )
        else:
            trainable_params = self.params
        params_treedef = save_params(
            params=trainable_params, filename=os.path.join(save_dir, "params.npz")
        )
        optstate_treedef = save_params(
            params=self.optimizer_state,
            filename=os.path.join(save_dir, "opt_state.npz"),
        )
        remaining_information = {
            "step": int(self.step),
            "best_validation_cluster_loss": float(self.best_validation_cluster_loss),
            "best_validation_unif_loss": float(self.best_validation_unif_loss),
            "key": np.array(self.random_key),
            "params_treedef": params_treedef,
            "optstate_treedef": optstate_treedef,
        }
        np.save(os.path.join(save_dir, "state_variables.npy"), remaining_information)

    @classmethod
    def load(
        cls,
        save_dir: str,
        fixed_params=None,
    ) -> "TrainingState":
        """
        Load the training state from a given directory.

        Args:
            save_dir (str): Directory where the training state is saved.

        Returns:
            TrainingState (TrainingState): Loaded training state.
        """
        information = np.load(
            os.path.join(save_dir, "state_variables.npy"), allow_pickle=True
        ).item()

        params = load_params(
            filename=os.path.join(save_dir, "params.npz"),
            tree_def=information["params_treedef"],
        )
        if fixed_params is not None:
            params = hk.data_structures.merge(params, fixed_params)
        optimizer_state = load_params(
            filename=os.path.join(save_dir, "opt_state.npz"),
            tree_def=information["optstate_treedef"],
        )
        return TrainingState(
            step=jnp.array(information["step"]),
            best_validation_cluster_loss=jnp.array(
                information["best_validation_cluster_loss"]
            ),
            best_validation_unif_loss=jnp.array(
                information["best_validation_unif_loss"]
            ),
            params=params,
            optimizer_state=optimizer_state,
            random_key=jnp.array(information["key"]),
        )


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
        esm_init: Callable,
        net_init: Callable,
        loss_fn: Callable,
        optimizer: optax.GradientTransformation,
        optimizer_type: str,
        parameters_partition_fn: Callable = None,
        turn_off_pmap: bool = False,
    ):
        """
        Args:
            net_init (Callable): The forward function initialization function. Takes a
                an input and returns a state of initialized weights.
            loss_fn (Callable): Loss function.
            optimizer (optax.GradientTransformation): Optimizer.
            optimizer_type (str): which optimizer is used (adam or adamw)
        """
        self._net_init = net_init
        self._esm_init = esm_init
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self.optimizer_type = optimizer_type
        self.parameters_partition_fn = parameters_partition_fn
        self.turn_off_pmap = turn_off_pmap

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, random_key: RNGKey, esm_params, gnn_batch_data) -> TrainingState:
        """Initializes state of the updater.

        Args:
            random_key (RandomKey): Random JAX key.
            batch_data (BatchDataGeneric): batch data

        Returns:
            TrainingState (TrainingState): Training state.
        """
        out_rng, init_rng = jax.random.split(random_key)
        # esm_params = self._esm_init(init_rng, esm_batch_data.tokens)
        gnn_params = self._net_init(init_rng, gnn_batch_data)
        params = hk.data_structures.merge(esm_params, gnn_params)
        print("loading sizes:")
        print(jax.tree_map(lambda x: x.size, params))
        if self.parameters_partition_fn is not None:
            trainable_params, fixed_params = hk.data_structures.partition(
                self.parameters_partition_fn, params
            )
        else:
            trainable_params = params
            fixed_params = {}
        print(
            "optimising these parameters: "
            + "".join(["\n\t" + k for k in trainable_params])
        )
        print("fixing these parameters: " + "".join(["\n\t" + k for k in fixed_params]))
        optimizer_state = self._optimizer.init(trainable_params)
        return TrainingState(
            step=jnp.array(0),
            best_validation_cluster_loss=jnp.array(1e10),
            best_validation_unif_loss=jnp.array(1e10),
            params=params,
            optimizer_state=optimizer_state,
            random_key=out_rng,
        )

    @functools.partial(jax.jit, static_argnums=0)
    def update(
        self, state: TrainingState, batch_data  #: BatchDataGeneric
    ) -> Tuple[TrainingState, TrainingMetrics, Dict[str, Any]]:
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

        (loss, aux_metrics), gradient = jax.value_and_grad(self._loss_fn, has_aux=True)(
            params, rng, batch_data
        )
        if self.parameters_partition_fn is not None:
            trainable_gradient, _ = hk.data_structures.partition(
                self.parameters_partition_fn, gradient
            )
            trainable_params, fixed_params = hk.data_structures.partition(
                self.parameters_partition_fn, params
            )
        else:
            trainable_gradient = gradient
            trainable_params = params
            fixed_params = {}

        mean = (
            jnp.mean
            if self.turn_off_pmap
            else (lambda x: jax.lax.pmean(x, axis_name="p"))
        )
        jaxmax = (
            jnp.max
            if self.turn_off_pmap
            else (lambda x: jax.lax.pmax(x, axis_name="p"))
        )

        # gather losses and gradients across devices
        loss = mean(loss)
        aux_metrics = jax.tree_map(lambda x: mean(x), aux_metrics)
        trainable_gradient = jax.tree_map(lambda g: mean(g), trainable_gradient)

        # update optimizer and weights
        optimizer_state = state.optimizer_state

        if self.optimizer_type == "adamw":
            updates, optimizer_state = self._optimizer.update(
                trainable_gradient, optimizer_state, params=trainable_params
            )
        elif self.optimizer_type == "adam":
            updates, optimizer_state = self._optimizer.update(
                trainable_gradient, optimizer_state
            )

        def _diagnostics(par):
            a = jnp.abs(par)
            notnan = ~jnp.isnan(a)
            return (
                notnan.sum(),
                jnp.mean(a, where=notnan),
                jnp.max(a, where=notnan, initial=0),
                jnp.any(~notnan),
            )

        def _summarise(dat):
            diags = [_diagnostics(p) for p in jax.tree_util.tree_leaves(dat)]
            notnans, avg, mx, nans = [
                jnp.stack([d[i] for d in diags]) for i in range(4)
            ]
            avg = avg.dot(notnans / notnans.sum())
            return {"avg_mag": avg, "max_mag": jnp.max(mx), "any_nans": nans.any()}

        summaries = {
            "params": _summarise(trainable_params),
            "grads": _summarise(trainable_gradient),
            "updates": _summarise(updates),
        }

        def agg(dat):
            return {
                "avg_mag": mean(dat["avg_mag"]),
                "max_mag": jaxmax(dat["max_mag"]),
                "any_nans": mean(dat["any_nans"]),
            }

        summaries = {k: agg(v) for k, v in summaries.items()}

        trainable_params = optax.apply_updates(trainable_params, updates)
        params = hk.data_structures.merge(trainable_params, fixed_params)
        rng, new_rng = jax.random.split(rng, num=2)

        new_state = TrainingState(
            step=state.step + 1,
            best_validation_cluster_loss=state.best_validation_cluster_loss,
            best_validation_unif_loss=state.best_validation_unif_loss,
            random_key=new_rng,
            optimizer_state=optimizer_state,
            params=params,
        )

        metrics = TrainingMetrics(step=state.step, loss=loss, metrics=aux_metrics)
        return new_state, metrics, summaries

    @functools.partial(jax.jit, static_argnums=0)
    def predict(
        self,
        key: jax.random.PRNGKey,
        params: hk.Params,
        batch_data,  #: BatchDataGeneric
    ) -> Tuple[float, dict]:
        """Computes inference over a batch of data and returns metrics.
        Might be used for validation purpose.

        Args:
            key (jax.random.PRNGKey): Random key.
            params (jax.random.PRNGKey): Model parameters.
            batch_data (BatchDataGeneric): Data.

        Returns:
            loss (float):  Training Loss.
            aux_metrics (dict): Auxillary metrics.
        """
        mean = (
            jnp.mean
            if self.turn_off_pmap
            else (lambda x: jax.lax.pmean(x, axis_name="p"))
        )

        loss, aux_metrics = self._loss_fn(params, key, batch_data)

        # gather losses and metrics across devices.
        loss = mean(loss)
        aux_metrics = jax.tree_map(lambda x: mean(x), aux_metrics)

        return loss, aux_metrics

    @functools.partial(jax.jit, static_argnums=0)
    def inference(self, state: TrainingState, batch_data) -> Any:
        """Computes inference over a batch of data and returns embeddings.

        Args:
            state (TrainingState): Training state.
            batch_data (BatchDataGeneric): batch data

        Returns:
            TrainingMetrics (TrainingMetrics): Training Metrics.
        """
        raise NotImplementedError()
