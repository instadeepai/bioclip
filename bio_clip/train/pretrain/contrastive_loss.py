from typing import Any, Callable, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from bio_clip.types import BatchDataWithTokensBioClip, Loss, MultimodalEmbeddings


def l2_normalize(x: jnp.ndarray, axis: int = 0, eps: float = 1e-12) -> jnp.ndarray:
    """Normalizes along dimension `axis` using an L2 norm.

    Args:
        x (jnp.ndarray): An input ndarray.
        axis (int): Dimension along which to normalize
        eps (float): Epsilon to avoid dividing by zero.

    Returns:
        An jnp.ndarray of the same shape as 'x' L2-normalized along 'axis'.
    """
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


def loss_bioclip_distributed_fn(
    params: hk.Params,
    random_key: jnp.ndarray,
    batch_data: BatchDataWithTokensBioClip,
    forward_fn: Callable,
    train_cfg,
    esm_forward: Callable,
    gnn_type: str,
    turn_off_pmap: bool = False,
) -> Tuple[Loss, Dict[str, jnp.ndarray]]:
    """Loss function of the model

    Args:
        forward_fn (Callable): forward function yielding multimodal embeddings
            of sequence and structure
        hyperparameters (BioClipConfig): hyperparameters
        params (hk.Params): Parameters associated to the forward function.
        random_key (jnp.ndarray): Random JAX key.
        batch_data (BatchDataWithTokensBioClip): batch data for BioCLIP
    Returns:
        Loss function value, metrics and bioclip embeddings
    """
    esm_params, gnn_params = hk.data_structures.partition(
        lambda k, _m, _n: k.startswith("esm2"), params
    )
    batch_esm_embeddings = esm_forward(
        esm_params,
        jax.random.split(random_key, num=batch_data.tokens.shape[0]).reshape(
            (*batch_data.tokens.shape[:-1], 2)
        ),
        batch_data.tokens,
    )
    if gnn_type == "graph_transformer":
        nodes_mask = batch_data.graph[-1].any(-1)[..., None]
    else:
        nodes_mask = batch_data.graph.nodes_mask

    batch_data = BatchDataWithTokensBioClip(
        graph=batch_data.graph, sequence=batch_esm_embeddings
    )

    num_samples = train_cfg.batch_size_gnn_per_device

    assert num_samples % train_cfg.chunk_size == 0, f"error: {train_cfg}"

    num_chunks = num_samples // train_cfg.chunk_size

    if num_chunks > 1:
        if train_cfg.chunk_size > 1:
            print(f"{num_chunks} sequential batching; vmapping {train_cfg.chunk_size}")
            batch_data = jax.tree_map(
                lambda x: x.reshape((num_chunks, train_cfg.chunk_size, *x.shape[1:])),
                batch_data,
            )

            def forward(x: BatchDataWithTokensBioClip) -> Any:  # MultimodalEmbeddings:
                multimodal_embedding = jax.vmap(forward_fn, in_axes=(None, None, 0))(
                    gnn_params, random_key, x
                )
                return multimodal_embedding

            if train_cfg.use_remat:
                print("REMAT ON")
                forward = jax.remat(forward)

            def _scan_fn(
                carry: int, x: BatchDataWithTokensBioClip
            ) -> Tuple[int, MultimodalEmbeddings]:
                y = forward(x)
                return carry, y

            _, multimodal_embedding = jax.lax.scan(
                _scan_fn, init=0, xs=batch_data, length=num_chunks
            )

            multimodal_embedding = jax.tree_map(
                lambda x: x.reshape((num_samples, *x.shape[2:])), multimodal_embedding
            )
        else:
            print(
                f"{num_chunks} sequential batching; no vmapping; {train_cfg.chunk_size}"
            )

            def forward(x: BatchDataWithTokensBioClip) -> Any:  # MultimodalEmbeddings:
                return forward_fn(gnn_params, random_key, x)

            if train_cfg.use_remat and num_samples > 1:
                print("REMAT ON")
                forward = jax.remat(forward)

            def _scan_fn(
                carry: int, x: BatchDataWithTokensBioClip
            ) -> Tuple[int, MultimodalEmbeddings]:
                y = forward(x)
                return carry, y

            _, multimodal_embedding = jax.lax.scan(_scan_fn, init=0, xs=batch_data)
    else:
        # # the sequential operations are 1, the jax.jit compiler is not able to remove
        # the scan itself if train_cfg.chunk_size > 1:
        print(f"no sequential batching; vmapping batch {train_cfg.chunk_size}")
        batch_data = jax.tree_map(
            lambda x: x.reshape((train_cfg.chunk_size, *x.shape[1:])), batch_data
        )

        def forward(x: BatchDataWithTokensBioClip) -> Any:  # MultimodalEmbeddings:
            multimodal_embedding = jax.vmap(forward_fn, in_axes=(None, None, 0))(
                gnn_params, random_key, x
            )
            return multimodal_embedding

        if train_cfg.use_remat and num_samples > 1:
            print("REMAT ON")
            forward = jax.remat(forward)
        multimodal_embedding = forward(batch_data)
        multimodal_embedding = jax.tree_map(
            lambda x: x.reshape((num_samples, *x.shape[1:])), multimodal_embedding
        )

    # should make this a tuple if providing both loss functions.
    g_emb = multimodal_embedding.projected_structure_embedding
    if train_cfg.use_projected_sequence_embedding:
        print(
            "\n\nWARNING! Using a projected ESM embedding as target. This will mean you"
            " are approximating a linear projection of ESM, this will not provide much "
            "complementary information if used downstream in combination with ESM. If "
            "using an earlier structure this should work however.\n\n"
        )
        s_emb = multimodal_embedding.projected_sequence_embedding
    else:
        s_emb = multimodal_embedding.sequence_embedding

    shape_info = str(jax.tree_map(lambda x: x.shape, (g_emb, s_emb)))
    assert jax.tree_map(
        lambda g, s: s.shape == g.shape, g_emb, s_emb
    ), f"Shape mismatch: [{shape_info}]"
    print(f"structure-emb, sequence-emb shape: {shape_info}")

    exp_tau = jnp.exp(multimodal_embedding.temperature)

    # Dictionary for metrics to return
    metrics = {}

    if train_cfg.objective_type == "supervision":
        print("USING DIRECT SUPERVISION")
        mask = jnp.squeeze(nodes_mask, axis=-1)  # shape: [batch, residue]
        # num_res = train_cfg.data.fixed_sizes.maximum_padding
        # if mask.shape == (num_samples, num_res):
        if train_cfg.target_type == "per-res":
            print("Using per-residue supervision")
            error = (g_emb - s_emb) ** 2
            error = jnp.mean(
                error, where=mask[:, :, None].repeat(error.shape[-1], axis=-1), axis=1
            )  # mean over residues
            loss = jnp.mean(error)  # mean over batch, hidden
        elif train_cfg.target_type == "both":
            # Unpack the two types of embeddings: per-residue, per-chain.
            per_res_gnn_emb, agg_gnn_emb = g_emb
            per_res_seq_emb, agg_seq_emb = s_emb

            per_res_error = (per_res_gnn_emb - per_res_seq_emb) ** 2
            per_res_loss = jnp.mean(
                per_res_error,
                where=mask[:, :, None].repeat(per_res_error.shape[-1], axis=-1),
                axis=1,
            )  # mean over residues
            per_chain_error = (agg_gnn_emb - agg_seq_emb) ** 2
            per_chain_loss = jnp.mean(per_chain_error)  # mean over batch, hidden
            loss = per_res_loss + per_chain_loss
        elif train_cfg.target_type == "per-chain":
            # assert mask.shape == (num_samples,), "mask is the incorrect shape"
            error = (g_emb - s_emb) ** 2
            print("Using per-protein supervision")
            loss = jnp.mean(error)  # mean over batch, hidden
        else:
            raise ValueError(
                f"target_type invalid: {train_cfg.target_type}; "
                "should be in ['per-chain', 'per-res', 'both']"
            )

        # dummy
        loss_str = -jnp.ones(1)
        loss_seq = -jnp.ones(1)
    elif train_cfg.objective_type == "clip":
        print("USING CLIP")

        def per_res_clip(g_emb, s_emb, exp_tau, random_key):
            mask = jnp.squeeze(nodes_mask, axis=-1)  # shape: [batch, residue]
            num_res = train_cfg.data.fixed_sizes.maximum_padding
            latent = g_emb.shape[-1]
            assert mask.shape == (
                num_samples,
                num_res,
            ), f"Invalid; mask shape {mask.shape}, should be {(num_samples, num_res)}"

            def gather(data, _ixs):
                """The tf.gather usage is very confusing, to keep it simple I vmap this
                simple function which performs the numpy operation: data[_ixs] where
                data is 2d and _ixs is 1d."""
                index_2d_matrix_by_leading_dim = jax.lax.GatherDimensionNumbers(
                    offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
                )
                slice_data = jax.lax.gather(
                    data,
                    start_indices=_ixs[:, None],
                    dimension_numbers=index_2d_matrix_by_leading_dim,
                    slice_sizes=(1, latent),
                )
                print(f"slice_data: {slice_data.shape}")
                return slice_data

            num_sample = (
                train_cfg.num_samples_per_sequence
                if train_cfg.sample_each_sequence
                else train_cfg.num_samples_per_sequence * num_samples
            )

            def sample(k, mask_vec):
                p = mask_vec / mask_vec.sum()
                ixs = jnp.arange(mask_vec.shape[0])
                # replace samples, to avoid the case where
                # sum(mask)<num_samples_per_sequence
                rand = jax.random.choice(
                    k, ixs, shape=(num_sample,), p=p, replace=False
                )
                return rand

            if train_cfg.sample_each_sequence:
                print("sampling each sequence individually")
                keys = jax.random.split(random_key, num=num_samples).reshape(
                    num_samples, 2
                )

                ixs = jax.vmap(sample)(keys, mask)  # [B, num_samples_per_sequence]
                desired_shape = (num_samples, train_cfg.num_samples_per_sequence)
                assert ixs.shape == desired_shape, (
                    f"sampled residues are the wrong shape:"
                    f" [ixs: {ixs.shape}; should be: {desired_shape}]"
                )

                vgather = jax.vmap(gather, in_axes=(0, 0))
                g_emb = vgather(g_emb, ixs)
                s_emb = vgather(s_emb, ixs)
                desired_shape = (
                    num_samples,
                    train_cfg.num_samples_per_sequence,
                    latent,
                )
                assert (
                    g_emb.shape == desired_shape
                ), f"g_emb wrong: [g_emb: {g_emb.shape}; should be {desired_shape}]"
                g_emb = g_emb.reshape(-1, latent)
                s_emb = s_emb.reshape(-1, latent)
            else:
                print("sampling all residues together")
                flat_mask = mask.reshape(-1)
                random_key, sub_key = jax.random.split(random_key)
                ixs = sample(sub_key, flat_mask)
                g_emb = gather(g_emb.reshape(-1, latent), ixs)
                s_emb = gather(s_emb.reshape(-1, latent), ixs)

            desired_shape = (num_samples * train_cfg.num_samples_per_sequence, latent)
            assert (
                g_emb.shape == desired_shape
            ), f"g_emb wrong: [g_emb: {g_emb.shape}; should be {desired_shape}]"

            exp_tau = exp_tau.repeat(train_cfg.num_samples_per_sequence, axis=0)

            clip_size = train_cfg.batch_size * train_cfg.num_samples_per_sequence
            labels = jnp.arange(clip_size)
            return g_emb, s_emb, exp_tau, labels, random_key

        def clip_loss(g_emb, s_emb, exp_tau, labels):
            g_emb = g_emb / jnp.linalg.norm(g_emb, axis=-1, keepdims=True)
            s_emb = s_emb / jnp.linalg.norm(s_emb, axis=-1, keepdims=True)

            print(
                f"Generate embs with shapes g_emb: {g_emb.shape}, s_emb: {s_emb.shape}"
            )

            if turn_off_pmap:
                labels = labels[: s_emb.shape[0]]
            else:
                g_emb = jax.lax.all_gather(g_emb, "p", axis=0).reshape(
                    -1, g_emb.shape[-1]
                )
                s_emb = jax.lax.all_gather(s_emb, "p", axis=0).reshape(
                    -1, s_emb.shape[-1]
                )
                exp_tau = jax.lax.all_gather(exp_tau, "p", axis=0).reshape(-1)

            print(
                f"Gathered embs to g_emb_all: {g_emb.shape}, s_emb_all: {s_emb.shape}"
            )

            print(f"exp_tau: {exp_tau.shape}")
            logits = jnp.einsum("gi, si -> gs", g_emb, s_emb) * exp_tau.clip(-100, 100)

            print(f"Generate logits with shape: {logits.shape}")

            loss_str = optax.softmax_cross_entropy_with_integer_labels(
                logits, labels
            ).mean()
            loss_seq = optax.softmax_cross_entropy_with_integer_labels(
                logits.T, labels
            ).mean()
            loss = (loss_str + loss_seq) / 2
            return loss, loss_str, loss_seq

        if train_cfg.target_type == "per-res":
            g_emb, s_emb, exp_tau, labels, random_key = per_res_clip(
                g_emb, s_emb, exp_tau, random_key
            )
            loss, loss_str, loss_seq = clip_loss(g_emb, s_emb, exp_tau, labels)
        elif train_cfg.target_type == "both":
            # Unpack the two types of embeddings: per-residue, per-chain.
            per_res_gnn_emb, agg_gnn_emb = g_emb
            per_res_seq_emb, agg_seq_emb = s_emb

            # Compute the per-chain CLIP loss. Note that only aggregation params are
            # trained.
            labels = jnp.arange(train_cfg.batch_size)
            loss1 = clip_loss(agg_gnn_emb, agg_seq_emb, exp_tau, labels)

            # Compute the per-residue CLIP loss.
            (
                per_res_gnn_emb,
                per_res_seq_emb,
                exp_tau,
                labels,
                random_key,
            ) = per_res_clip(per_res_gnn_emb, per_res_seq_emb, exp_tau, random_key)
            loss2 = clip_loss(per_res_gnn_emb, per_res_seq_emb, exp_tau, labels)
            loss, loss_str, loss_seq = jax.tree_map(lambda a, b: a + b, loss1, loss2)

            # Add per-chain and per-res losses for logging
            per_chain_loss, per_chain_loss_str, per_chain_loss_seq = loss1
            per_res_loss, per_res_loss_str, per_res_loss_seq = loss2
            metrics["per_chain_loss"] = per_chain_loss
            metrics["per_chain_loss_str"] = per_chain_loss_str
            metrics["per_chain_loss_seq"] = per_chain_loss_seq
            metrics["per_res_loss"] = per_res_loss
            metrics["per_res_loss_str"] = per_res_loss_str
            metrics["per_res_loss_seq"] = per_res_loss_seq

        elif train_cfg.target_type == "per-chain":
            print("PER RESIDUE CLIP IS NOT ACTIVE")
            exp_tau = jnp.exp(multimodal_embedding.temperature)
            labels = jnp.arange(train_cfg.batch_size)
            loss, loss_str, loss_seq = clip_loss(g_emb, s_emb, exp_tau, labels)
        else:
            raise ValueError(
                f"target_type invalid: {train_cfg.target_type}; "
                f"should be in ['per-chain', 'per-res', 'both']"
            )
    else:
        raise ValueError(
            f"objective_type invalid: {train_cfg.objective_type}; "
            f"should be in ['supervision', 'clip']"
        )

    metrics["loss"] = loss
    metrics["loss_structure"] = loss_str
    metrics["loss_sequence"] = loss_seq
    metrics["logit_temperature"] = multimodal_embedding.temperature.mean()

    return loss, metrics
