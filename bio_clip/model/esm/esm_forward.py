from typing import Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp

from bio_clip.model.esm.esm_haiku import (
    AttentionMask,
    Tokens,
    TransformerOutput,
    build_padding_attention_mask,
)


def forward_esm_model(
    parameters: hk.Params,
    batch_keys: jax.random.KeyArray,
    batch_tokens: jnp.ndarray,
    max_batch_size: int,
    embeddings_layer_to_save: int,
    compute_mean_embedding: bool,
    forward_esm_fn: Callable[
        [hk.Params, jax.random.KeyArray, Tokens, Optional[AttentionMask]],
        TransformerOutput,
    ],
    pad_token_id: int,
) -> jnp.ndarray:
    assert len(batch_tokens.shape) == 2
    assert batch_tokens.shape[0] % max_batch_size == 0
    assert batch_keys.shape[0] == batch_tokens.shape[0]

    num_steps = batch_tokens.shape[0] // max_batch_size
    num_res = batch_tokens.shape[1]

    batch_tokens, batch_keys = jax.tree_map(
        lambda x: x.reshape((num_steps, max_batch_size, x.shape[-1])),
        (batch_tokens, batch_keys),
    )

    def _one_forward(carry, data_batch):  # type: ignore
        (tokens, keys) = data_batch
        # breakpoint()
        pad_attention_mask = build_padding_attention_mask(tokens, pad_token_id)
        pad_sequence_mask = tokens != pad_token_id
        outs = forward_esm_fn(parameters, keys[0], tokens, pad_attention_mask)
        # outs = forward_esm_fn(parameters, keys[0], tokens)
        embeddings = outs[f"embeddings_{embeddings_layer_to_save}"]
        return (
            carry,
            jnp.mean(
                embeddings,
                axis=1,
                where=jnp.repeat(
                    pad_sequence_mask[..., None], embeddings.shape[-1], axis=-1
                ),
            )
            if compute_mean_embedding
            else embeddings,  # [:, 0],
        )

    esm_embeddings = jax.lax.scan(
        _one_forward,
        None,
        (batch_tokens, batch_keys),
    )[1]
    new_batch = num_steps * max_batch_size
    if compute_mean_embedding:
        shape = (new_batch, esm_embeddings.shape[-1])
    else:
        shape = (new_batch, num_res, esm_embeddings.shape[-1])
    return esm_embeddings.reshape(shape)


def prep_esm_wrapper(
    esm_forward,
    num_res,
    max_batch_size,
    embeddings_layer_to_save,
    compute_mean_embedding,
    pad_token_id,
):
    print("performing ESM forward.")

    def forward_esm_model(batch_tokens) -> jnp.ndarray:
        assert len(batch_tokens.shape) == 2, f"batch_tokens.shape: {batch_tokens.shape}"
        assert batch_tokens.shape[0] % max_batch_size == 0
        num_steps = batch_tokens.shape[0] // max_batch_size
        batch_tokens = jax.tree_map(
            lambda x: x.reshape((num_steps, max_batch_size, x.shape[-1])), batch_tokens
        )

        def _one_forward(carry, _tokens):
            pad_attention_mask = build_padding_attention_mask(_tokens, pad_token_id)
            pad_sequence_mask = _tokens != pad_token_id
            outs = esm_forward(_tokens, pad_attention_mask)
            embeddings = outs[f"embeddings_{embeddings_layer_to_save}"]
            print(f"EMBEDDINGS: {embeddings.shape}; {pad_sequence_mask.shape}")
            if compute_mean_embedding:
                emb = jnp.mean(
                    embeddings,
                    axis=1,
                    where=jnp.repeat(
                        pad_sequence_mask[..., None], embeddings.shape[-1], axis=-1
                    ),
                )
            else:
                emb = embeddings  # [:, 0]
            return carry, emb

        esm_embeddings = hk.scan(_one_forward, None, batch_tokens)[1]
        print(f"esm_embeddings: {esm_embeddings.shape}")
        new_batch = num_steps * max_batch_size
        if compute_mean_embedding:
            shape = (new_batch, esm_embeddings.shape[-1])
        else:
            shape = (new_batch, num_res, esm_embeddings.shape[-1])
        return esm_embeddings.reshape(shape)

    return forward_esm_model
