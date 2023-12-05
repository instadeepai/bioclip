import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import esm
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import regex
import torch
from haiku import initializers
from typing_extensions import TypeAlias

RNGKey: TypeAlias = jnp.ndarray
Embedding: TypeAlias = jnp.ndarray
Tokens: TypeAlias = jnp.ndarray
Labels: TypeAlias = jnp.ndarray
AttentionMask: TypeAlias = jnp.ndarray
SequenceMask: TypeAlias = jnp.ndarray
AttentionWeights: TypeAlias = jnp.ndarray
TransformerOutput: TypeAlias = Dict[str, jnp.ndarray]  # type: ignore
Metrics: TypeAlias = Dict[str, jnp.ndarray]

# Constant used in Sinusoidal/Rotary Embeddings, reference to this value can be found
# on page 6 of https://arxiv.org/pdf/1706.03762.pdf and page 5 of
# https://arxiv.org/abs/2104.09864
UPPER_FREQ = 10000


class RotaryEmbedding(hk.Module):
    """
    Rotary Positional Embedding inspired by RoFormer:
    https://arxiv.org/abs/2104.09864
    https://github.com/ZhuiyiTechnology/roformer .
    """

    def __init__(
        self,
        key_size: int,
        name: Optional[str] = None,
    ):

        """
        Args:
            key_size: Dimension of one head.
            name: Name of the layer. Defaults to None.
        """
        super().__init__(name=name)
        self._inv_freq = 1.0 / (UPPER_FREQ ** (jnp.arange(0, key_size, 2) / key_size))

    def _compute_cos_sin_tables(
        self,
        heads: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the cosinus and sinus for rotation.

        Args:
            heads: Query or key heads of shape (batch_size, seq_len, num_heads,
            key_size).

        Returns:
            Cosinus positional embedding of shape (1, seq_len, 1,
                key_size).
            Sinus positional embedding of shape (1, seq_len, 1,
                key_size).
        """
        seq_len = heads.shape[1]

        self._seq_len_cached = seq_len
        t = jnp.arange(seq_len)
        freqs = jnp.einsum("i,j->ij", t, self._inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)

        # Compute cos and cast is as (1, seq_len, 1, key_size) to be applied to queries
        # of shape (batch_size, seq_len, num_heads, key_size)
        cos_cached = jnp.cos(emb)[None, :, None, :]
        sin_cached = jnp.sin(emb)[None, :, None, :]

        return cos_cached, sin_cached

    def _apply_rotary_pos_emb(
        self, heads: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Applies the rotary positional embedding to the heads.

        Args:
            heads: Query or key heads of shape (batch_size, seq_len, num_heads,
                key_size).
            cos: Cosinus values.
            sin: Sinus values.

        Returns:
            Embedded heads of shape (batch_size, seq_len, num_heads,
                key_size).
        """

        # Rotate x
        x1, x2 = heads[..., : heads.shape[-1] // 2], heads[..., heads.shape[-1] // 2 :]
        heads_rotated = jnp.concatenate((-x2, x1), axis=-1)

        embedded_heads = (heads * cos) + (heads_rotated * sin)
        return embedded_heads

    def __call__(
        self, query_heads: jnp.ndarray, key_heads: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Applies rotary embeddings to query_heads and key_heads.

        Args:
            query_heads: Query heads of shape
                (batch_size, seq_len, num_heads, key_size).
            key_heads: Key heads of shape (batch_size, seq_len, num_heads, key_size).

        Returns:
            Embedded query heads.
            Embedded key heads.
        """
        cos, sin = self._compute_cos_sin_tables(query_heads)

        return (
            self._apply_rotary_pos_emb(query_heads, cos, sin),
            self._apply_rotary_pos_emb(key_heads, cos, sin),
        )


class ESMRobertaLMHead(hk.Module):
    """
    ESM Roberta Language Model head. Transform final attention layer output into a
    distribution over tokens at each position.
    """

    def __init__(self, embed_dim: int, alphabet_size: int, name: Optional[str] = None):
        """
        Args:
            embed_dim: Embedding dimension.
            alphabet_size: Number of tokens in the alphabet.
            name: Name of the layer. Defaults to None.
        """
        super(ESMRobertaLMHead, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.alphabet_size = alphabet_size

        # Define layers
        self._first_layer_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="emb_layer_norm_after"
        )
        self._fc1 = hk.Linear(self.embed_dim, name="lm_head_fc_1")
        self._final_fc = hk.Linear(self.alphabet_size, name="lm_final_fc")
        self._second_layer_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="lm_head_layer_norm"
        )

    def __call__(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        x = self._first_layer_norm(x)
        # Embeddings are computed after the first layer norm to be consistent with ESM
        embeddings = x
        x = self._fc1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self._second_layer_norm(x)

        # Compute logits
        logits = self._final_fc(x)
        return {"embeddings": embeddings, "logits": logits}


class ESMMultiHeadAttention(hk.MultiHeadAttention):
    """
    Multi-head attention with masking applied. Modified from the core implementation to
    support biases in keys and values.
    """

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        add_bias_kv: bool,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
        super(ESMMultiHeadAttention, self).__init__(
            num_heads=num_heads,
            key_size=key_size,
            w_init=w_init,
            value_size=value_size,
            model_size=model_size,
            name=name,
        )

        if add_bias_kv:
            self._bias_k = hk.get_parameter(
                "bias_k", [1, 1, self.num_heads, self.key_size], init=jnp.zeros
            )
            self._bias_v = hk.get_parameter(
                "bias_v", [1, 1, self.num_heads, self.value_size], init=jnp.zeros
            )
        else:
            self._bias_k = None
            self._bias_v = None

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
    ) -> TransformerOutput:
        """
        Computes both the embeddings and the attention weights.

        Args:
            query: Embedding sequence to compute queries.
            key: Embedding sequence to compute keys.
            value: Embedding sequence to compute values.
            attention_mask: Mask to be applied during the attention layers.
                Triangular for autoregressive models. Defaults to None.

        Returns:
            Dictionary containing the output embeddings and the attention weights.
        """

        attention_weights = self.attention_weights(query, key, attention_mask)
        embeddings = self.compute_embeddings(value, attention_weights)
        return {"embeddings": embeddings, "attention_weights": attention_weights}

    @hk.transparent
    def attention_weights(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        attention_mask: Optional[AttentionMask] = None,
    ) -> jnp.ndarray:
        """
        Computes the attention weights.

        Args:
            query: Embedding sequence to compute queries.
            key: Embedding sequence to compute keys.
            attention_mask: Input attention_mask. Defaults to None.

        Returns:
            Attention weights.
        """

        query_heads = self._linear_projection_he_init(query, self.key_size, "query")
        key_heads = self._linear_projection_he_init(key, self.key_size, "key")

        # Add bias for key (see ESM architecture)
        if self._bias_k is not None:
            batch_size = key_heads.shape[0]
            attention_bias = jnp.tile(self._bias_k, (batch_size, 1, 1, 1))
            key_heads = jnp.concatenate((key_heads, attention_bias), axis=1)
            if attention_mask is not None:
                attention_mask = jnp.concatenate(
                    (
                        attention_mask,
                        jnp.ones(attention_mask.shape[:-1] + (1,), dtype=jnp.bool_),
                    ),
                    axis=-1,
                )

        query_heads, key_heads = RotaryEmbedding(self.key_size, name="rotary_embed")(
            query_heads, key_heads
        )

        attention_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        sqrt_key_size = jnp.sqrt(self.key_size).astype(query.dtype)
        attention_logits = attention_logits / sqrt_key_size

        if attention_mask is not None:
            assert len(attention_mask.shape) == len(attention_logits.shape)
            attention_logits = jnp.where(attention_mask, attention_logits, -1e30)

        attention_weights = jax.nn.softmax(attention_logits)

        return attention_weights

    @hk.transparent
    def compute_embeddings(
        self,
        value: jnp.ndarray,
        attention_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the output embeddings.

        Args:
            value: Embedding sequence to compute values.
            attention_weights: Attention weights.

        Returns:
            Output embeddings.
        """

        # He initialization
        w_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")
        b_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")

        value_heads = self._linear_projection_he_init(value, self.value_size, "value")
        if self._bias_v is not None:
            batch_size = value_heads.shape[0]
            attention_bias = jnp.tile(self._bias_v, (batch_size, 1, 1, 1))
            value_heads = jnp.concatenate((value_heads, attention_bias), axis=1)

        attention = jnp.einsum("...htT,...Thd->...thd", attention_weights, value_heads)

        # Concatenate attention matrix of all heads into a single vector.
        attention_vec = jnp.reshape(attention, (*value.shape[:-1], -1))
        return hk.Linear(
            self.model_size, w_init=w_init, b_init=b_init, name="mha_output"
        )(attention_vec)

    @hk.transparent
    def _linear_projection_he_init(
        self, x: jnp.ndarray, head_size: int, name: Optional[str] = None
    ) -> jnp.ndarray:
        """
        Linear layer for multi-head attention mechanism. Initialized with the He method.

        Args:
            x: Input embeddings.
            head_size: Embedding size of each attention head.
            name: Name of the linear layer.

        Returns:
            Multi-head embeddings.
        """

        # He initialization
        w_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")
        b_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")

        y = hk.Linear(
            self.num_heads * head_size, w_init=w_init, b_init=b_init, name=name
        )(x)
        return y.reshape((*x.shape[:-1], self.num_heads, head_size))


class ESMAttentionBlock(hk.Module):
    """
    Attention block made of self-attention.
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        add_bias_kv: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        # Add checks on dimensions
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"The embedding dimension should be divisible by the number of heads, "
                f"however provided embedding dimension is {embed_dim} and the number of"
                f" heads is {num_heads}"
            )

        # Hyperparameters internalization
        key_size = embed_dim // num_heads

        # Define layers
        self.fc1 = hk.Linear(ffn_embed_dim, name="fc1")
        self.fc2 = hk.Linear(embed_dim, name="fc2")

        self.layer_norm_self_attention = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name="mha_layer_norm",
        )
        self.layer_norm_mlp = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="final_layer_norm"
        )
        self.sa_layer = ESMMultiHeadAttention(
            num_heads=num_heads,
            key_size=key_size,
            add_bias_kv=add_bias_kv,
            name="mha",
        )

    @hk.transparent
    def self_attention(
        self,
        x: Embedding,
        attention_mask: Optional[AttentionMask] = None,
    ) -> TransformerOutput:
        """
        Applies the self attention mechanism.

        Args:
            x: Input token embeddings of shape (batch_size, seq_len, embed_dim).
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            Dictionary containing the output embeddings and the attention weights.
        """

        return self.sa_layer(x, x, x, attention_mask=attention_mask)

    @hk.transparent
    def mlp(self, x: Embedding) -> Embedding:
        """
        Applies one layer-norm, one linear layer, a Gelu activation,
        then a final linear layer.

        Args:
            x: Embeddings of shape (batch_size, seq_len, key_size * num_heads).

        Returns:
            The transformed sequence embedding.
        """
        x = self.layer_norm_mlp(x)
        x = jax.nn.gelu(
            self.fc1(x),
            approximate=False,
        )
        x = self.fc2(x)
        return x

    def __call__(
        self,
        x: Tokens,
        attention_mask: Optional[AttentionMask] = None,
    ) -> TransformerOutput:
        """
        Computes the output of the attention layer.

        Args:
            x: Input token embeddings of shape (batch_size,seq_len,embed_dim).
            attention_mask: Attention mask of shape (batch_size, 1,seq_len, seq_len).

        Returns:
            A dictionary containing the output embeddings and the attention weights.
        """

        # Self-Attention
        res = x
        x = self.layer_norm_self_attention(x)
        output = self.self_attention(
            x=x,
            attention_mask=attention_mask,
        )
        x = output["embeddings"]
        x = res + x

        # MLP
        x = x + self.mlp(x)

        output["embeddings"] = x
        return output  # type: ignore  # type: ignore


class TokensDropout(hk.Module):
    """
    Tokens dropout layer.
    """

    def __init__(
        self,
        embed_dim: int,
        pad_token_id: int,
        mask_token_id: int,
        masking_ratio: float,
        masking_prob: float,
        name: Optional[str] = None,
    ):
        """
        Args:
            embed_dim: Embedding dimension.
            pad_token_id: ID of the pad token.
            mask_token_id: ID of the pad token.
            masking_ratio: Masking ratio.
            masking_prob: Probability to mask.
            name: Name of the layer. Defaults to None.
        """
        super().__init__(name=name)
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.masking_ratio = masking_ratio
        self.masking_prob = masking_prob
        self.embed_dim = embed_dim

    def __call__(self, x: jnp.ndarray, tokens: Tokens) -> jnp.ndarray:

        padding_mask_tokens = tokens == self.pad_token_id
        tokens_repeated = jnp.repeat(
            tokens[:, :, None], repeats=self.embed_dim, axis=-1
        )
        x = jnp.where(tokens_repeated == self.mask_token_id, 0.0, x)
        mask_ratio_train = self.masking_ratio * self.masking_prob
        src_lengths = (~padding_mask_tokens).sum(-1)
        mask_ratio_observed = (tokens == self.mask_token_id).sum(-1) / src_lengths
        x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
        return x


def build_padding_attention_mask(tokens: Tokens, pad_token: int) -> AttentionMask:
    """
    Builds a padding mask from a sequence of tokens by masking <pad> in the attention.

    Args:
        tokens: Batch of sequences of shape (batch_size, seq_len).
        pad_token: Int corresponding to the <pad> token to mask.

    Returns:
        Batch of attention masks, masking out <pad> tokens.
    """
    padding_mask = tokens != pad_token
    padding_mask = padding_mask[:, None, :]
    padding_mask = jnp.einsum("bhT, bht->bhtT", padding_mask, padding_mask)
    return padding_mask


class FixedSizeStandardTokenizer:
    """
    StandardTokenizer that tokenizes all sequences to a pre-defined fixed length.
    An exception will be raised if trying to tokenize a sequence longer than
    the pre-defined length. All sequences shorter than the fixed length will be
    padded. Tokenizing to a fixed length can become handy in Jax to avoid the need
    to recompile computations graph for different token sequences lengths. However, note
    that in some cases if the fixed length is high compared to the average length, this
    behavior might computationally inefficient.
    """

    def __init__(
        self,
        standard_tokens: List[str],
        fixed_length: int,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        class_token: str = "<cls>",
        eos_token: str = "<eos>",
        bos_token: str = "<bos>",
        prepend_bos_token: bool = False,
        prepend_cls_token: bool = False,
        append_eos_token: bool = False,
        extra_special_tokens: Optional[List[str]] = None,
        tokens_to_ids: Optional[Dict[str, int]] = None,
    ):
        """
        Initializes a FixedSizeStandardTokenizer.

        Args:
            standard_tokens: Standard tokens, where special tokens are omitted.
            unk_token: Unknown token.
            pad_token: Pad token.
            mask_token: Mask token.
            class_token: Class token.
            eos_token: End of speech tokens.
            bos_token: Beginning of sentence token.
            prepend_bos_token: Prepend beginning of sentence token.
            prepend_cls_token: Prepend class token.
            append_eos_token: Append end of speech token.
            fixed_length: Fixed length to pad all sequences in batches
            extra_special_tokens: (Optional) Enable the user to define optionally
                additional special tokens.
            tokens_to_ids: (Optional) Enable the user to optionally choose ids for the
                tokens. If this value is not provided, then ids are attributed
                automatically by the tokenizer during initialization.
        """
        # Define special tokens essential to masked language modelling
        special_tokens_1 = [
            unk_token,
            pad_token,
            mask_token,
            class_token,
        ]

        # Add extra special tokens
        special_tokens_2 = [eos_token, bos_token]

        if extra_special_tokens is not None:
            special_tokens_2.extend(extra_special_tokens)

        self._all_tokens = special_tokens_1 + standard_tokens + special_tokens_2
        self._standard_tokens = standard_tokens
        self._special_tokens = special_tokens_1 + special_tokens_2

        self._unk_token = unk_token
        self._pad_token = pad_token
        self._mask_token = mask_token
        self._class_token = class_token
        self._eos_token = eos_token
        self._bos_token = bos_token
        self._prepend_bos_token = prepend_bos_token
        self._prepend_cls_token = prepend_cls_token
        self._append_eos_token = append_eos_token

        # Can only
        if self._prepend_bos_token and self._prepend_cls_token:
            raise ValueError(
                "Cannot prepend both BOS and CLS token, must choose only one"
            )

        # Matching between tokens and ids
        if tokens_to_ids is not None:
            if set(tokens_to_ids.keys()) != set(self._all_tokens):
                raise ValueError(
                    f"Specified matching between tokens and ids, "
                    f"but some tokens are missing or mismatch. "
                    f"Got specifications for tokens: {set(tokens_to_ids.keys())} "
                    f"and expected for {set(self._all_tokens)}"
                )
            sorted_tokens = np.sort(list(tokens_to_ids.values()))
            if np.any(sorted_tokens != np.arange(len(self._all_tokens))):
                raise ValueError(
                    f"Specified matching between tokens and ids, "
                    f"but some ids are missing or mismatch. "
                    f"Got specifications for ids: {sorted_tokens} "
                    f"and expected for {np.arange(len(self._all_tokens))}"
                )
            self._tokens_to_ids = tokens_to_ids
        else:
            self._tokens_to_ids = {tok: i for i, tok in enumerate(self._all_tokens)}

        self._ids_to_tokens = {i: tok for tok, i in self._tokens_to_ids.items()}
        self._compiled_regex = regex.compile(
            "|".join(self._all_tokens + ["\S"])
        )  # noqa
        self._fixed_length = fixed_length

    @property
    def vocabulary(self) -> List[str]:
        return self._all_tokens

    @property
    def standard_tokens(self) -> List[str]:
        return self._standard_tokens

    @property
    def special_tokens(self) -> List[str]:
        return self._special_tokens

    @property
    def unk_token(self) -> str:
        return self._unk_token

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def mask_token(self) -> str:
        return self._mask_token

    @property
    def class_token(self) -> str:
        return self._class_token

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def bos_token(self) -> str:
        return self._bos_token

    @property
    def vocabulary_size(self) -> int:
        """
        Property that returns the total number of tokens.

        Returns:
            Total number of tokens.
        """
        return len(self.vocabulary)

    @property
    def unk_token_id(self) -> int:
        """
        Property that returns id (int representation) of the unknown token.

        Returns:
            Id (int representation) of the unknown token.
        """
        return self.token_to_id(self.unk_token)

    @property
    def pad_token_id(self) -> int:
        """
        Property that returns id (int representation) of the pad token.

        Returns:
            Id (int representation) of the pad token.
        """
        return self.token_to_id(self.pad_token)

    @property
    def mask_token_id(self) -> int:
        """
        Property that returns id (int representation) of the mask token.

        Returns:
            Id (int representation) of the mask token.
        """
        return self.token_to_id(self.mask_token)

    @property
    def class_token_id(self) -> int:
        """
        Property that returns id (int representation) of the class token.

        Returns:
            Id (int representation) of the class token.
        """
        return self.token_to_id(self.class_token)

    @property
    def eos_token_id(self) -> int:
        """
        Property that returns id (int representation) of the eos token.

        Returns:
            Id (int representation) of the eos token.
        """
        return self.token_to_id(self.eos_token)

    @property
    def bos_token_id(self) -> int:
        """
        Property that returns id (int representation) of the bos token.

        Returns:
            Id (int representation) of the bos token.
        """
        return self.token_to_id(self.bos_token)

    def id_to_token(self, token_id: int) -> str:
        try:
            token = self._ids_to_tokens[token_id]
            return token
        except KeyError:
            raise KeyError("Token id not found in vocabulary")

    def token_to_id(self, token: str) -> int:
        return self._tokens_to_ids.get(token, -1)

    def tokenize(self, sequence: str) -> Tuple[List[str], List[int]]:
        """
        Tokenizes a sequence and returns the list of tokens as well
        as the list of their IDs. Any character found in the sequence that does not
        correspond to any token in the vocabulary is replaced by the unk token.

        Args:
            sequence: Sequence to be tokenized.

        Returns:
            List of tokens.
            List of token ids.
        """
        tokens: List[str] = self._compiled_regex.findall(sequence)
        tokens = [
            tok if tok in self._tokens_to_ids.keys() else self._unk_token
            for tok in tokens
        ]
        if self._prepend_cls_token:
            tokens = [self._class_token] + tokens

        if self._prepend_bos_token:
            tokens = [self._bos_token] + tokens

        if self._append_eos_token:
            tokens.append(self._eos_token)

        try:
            tokens_ids = [self._tokens_to_ids[tok] for tok in tokens]
        except KeyError:
            print(
                "Found tokens in the sequence that are not supported by the tokenizer"
            )

        return tokens, tokens_ids

    def batch_tokenize(self, sequences: List[str]) -> List[Tuple[List[str], List[int]]]:
        """
        Tokenizes a batch of sequences.
        Sequences are padded to the maximum length in the batch.

        Args:
            sequences: Batch of sequences to be tokenized.

        Returns:
            Batch of tokenized sequences as well as their token ids,
            where every sequence has been padded to the maximum length
            in the batch.
        """
        return self.pad_tokens_batch(  # type: ignore
            [self.tokenize(seq) for seq in sequences]
        )

    @property
    def fixed_length(self) -> int:
        """
        Property that returns the pre-defined fixed sequence length.

        Returns:
             The pre-defined fixed sequence length.
        """
        return self._fixed_length

    def pad_tokens_batch(
        self, batch: List[Tuple[List[str], List[int]]]
    ) -> List[Tuple[List[str], List[int]]]:
        """
        Takes tokens and tokens ids of a batch of sequences and returns a batch of
        padded sequences.

        Args:
            batch: List of tuples, each composed of a sequence's tokens and token ids.

        Returns:
            The padded list, where every sequence is padded to the pre-defined
            max length.
        """
        lengths = [len(t[0]) for t in batch]
        maximum_length = max(lengths)
        if maximum_length > self._fixed_length:
            raise ValueError(
                "Found a sequence with length that "
                "exceeds the fixed length to tokenize."
            )
        deltas = [self._fixed_length - length for length in lengths]
        padded_tokens = [
            t[0] + ([self.pad_token] * delta) for t, delta in zip(batch, deltas)
        ]
        padded_tokens_ids = [
            t[1] + ([self.pad_token_id] * delta) for t, delta in zip(batch, deltas)
        ]
        return [
            (toks, toks_ids) for toks, toks_ids in zip(padded_tokens, padded_tokens_ids)
        ]


@dataclass
class ESMTransformerConfig:
    """
    Parameters to initialize an ESM model. While the ESM architecture is an encoder-only
    model, different choices have been made for each version and this configuration aims
    to cover most of them.

    Args:
        alphabet_size: Token vocabulary.
        pad_token_id: ID of pad token.
        mask_token_id: ID of mask token.
        max_positions: Maximum sequence length.
        embed_scale: Correction ratio applied to the embeddings to make up for the
            norm difference between the input during training and inference.
        attention_heads: Number of attention heads.
        embed_dim: Embedding dimension.
        ffn_embed_dim: Feed forward embedding dimension.
        num_layers: Number of attention blocks.
        add_bias_kv: Add bias in attention layer.
        mask_before_attention: Use mask before attention layers (for EMS1b and ESM2).
        token_dropout: Token dropout.
        masking_ratio: Masking ratio (used if token dropout is enabled).
        masking_prob: Masking probability (used if token dropout is enabled).
        use_gradient_checkpointing: Whether to use gradient checkpointing (checkpoint
            gradients in the forward pass to reduce the computation in the backward).
    """

    alphabet_size: int
    pad_token_id: int
    mask_token_id: int

    max_positions: int = 1024
    embed_scale: float = 1.0

    # architecture
    emb_layer_norm_before: bool = False
    attention_heads: int = 20
    embed_dim: int = 1280
    ffn_embed_dim: int = 5120
    num_layers: int = 24
    add_bias_kv: bool = False
    mask_before_attention: bool = False

    # dropout
    token_dropout: bool = False
    masking_ratio: float = 0.15
    masking_prob: float = 0.8

    # logging
    use_gradient_checkpointing: bool = False

    # return
    embeddings_layers_to_save: Tuple[int, ...] = ()
    attention_maps_to_save: Tuple[Tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        """
        Checks that the given values are compatible.
        """
        if not self.embed_dim % self.attention_heads == 0:
            raise ValueError(
                "The embedding dimension should be divisible by the number of heads, "
                f"however provided embedding dimension is {self.embed_dim} and the "
                f"number of heads is {self.attention_heads}."
            )


class ESMTransformer(hk.Module):
    """
    Jax implementation of ESM models. Covers ESM1, ESM1-b and ESM2 models.
    """

    def __init__(
        self,
        config: ESMTransformerConfig,
        name: Optional[str] = None,
    ):
        """
        Initialize an ESM model.

        Args:
            config: Dataclass containing model hyperparameters.
            name: Name for module (custom will break weight loading).
        """

        self._config = config
        super().__init__(name=name)

        self._embed_layer = hk.Embed(self._config.alphabet_size, self._config.embed_dim)
        self._lm_head = ESMRobertaLMHead(
            embed_dim=self._config.embed_dim,
            alphabet_size=self._config.alphabet_size,
        )

        # Process attention maps to save requirement into more suitable format
        attention_maps_to_save = config.attention_maps_to_save
        self._attention_layers_to_save = list(
            set([t[0] for t in attention_maps_to_save])
        )
        self._attention_maps_per_layer_to_save = {
            layer: [t[1] for t in attention_maps_to_save if t[0] == layer]
            for layer in self._attention_layers_to_save
        }

        # Checking user request can be executed, raise error otherwise
        max_layer = max(self._attention_layers_to_save + [0])
        if max_layer > config.num_layers:
            raise ValueError(
                f"You are requiring attention maps for layer {max_layer}, "
                f"while the model has {config.num_layers} layers only."
            )

        for layer, maps in self._attention_maps_per_layer_to_save.items():
            max_map = max(maps)
            if max_map > config.attention_heads:
                raise ValueError(
                    f"You are requiring attention maps number {max_map} "
                    f"at layer {layer}, while the model has {config.attention_heads} "
                    f"only."
                )

    @hk.transparent
    def _attention_block(self, layer_idx: int) -> ESMAttentionBlock:

        return ESMAttentionBlock(  # type: ignore
            num_heads=self._config.attention_heads,
            embed_dim=self._config.embed_dim,
            ffn_embed_dim=self._config.ffn_embed_dim,
            add_bias_kv=self._config.add_bias_kv,
            name=f"attention_layer_{layer_idx}",
        )

    @hk.transparent
    def apply_attention_blocks(
        self,
        x: Embedding,
        outs: Dict[str, Embedding],
        attention_mask: Optional[AttentionMask] = None,
    ) -> Tuple[Embedding, Dict[str, Embedding]]:
        """
        Create the blocks of attention layers and applies them.

        Args:
            x: The sequence embedding.
            outs: A dictionary to carry through the attention layers which stores the
                intermediate sequence embedding and attention maps.
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            The output sequence embedding.
            The optional intermediate results (embeddings of the layer and attention
                weights).
        """

        layers: List[Callable] = [
            self._attention_block(layer_idx)
            for layer_idx in range(self._config.num_layers)
        ]

        if self._config.use_gradient_checkpointing:
            # the remat-ed function cannot take control flow arguments
            layers = [hk.remat(layer) for layer in layers]

        for layer_idx, layer in enumerate(layers):
            output = layer(
                x=x,
                attention_mask=attention_mask,
            )
            x = output["embeddings"]
            # Save intermediate embeddings if needed
            if (layer_idx + 1) in self._config.embeddings_layers_to_save:
                outs[f"embeddings_{(layer_idx+1)}"] = output["embeddings"]
            # Save intermediate attention maps if needed
            if (layer_idx + 1) in self._attention_layers_to_save:
                for map_number in self._attention_maps_per_layer_to_save[layer_idx + 1]:
                    dkey = f"attention_map_layer_{layer_idx + 1}_number_{map_number}"
                    outs[dkey] = output["attention_weights"][:, map_number + 1]

        return x, outs

    def __call__(
        self,
        tokens: Tokens,
        attention_mask: Optional[AttentionMask] = None,
    ) -> TransformerOutput:
        """
        Computes the embeddings based on the input tokens.

        Args:
            tokens: Input tokens out of the tokenizer of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len).
                If none is provided, defaults to a mask over pad tokens where the
                pad_token_id is defined in the configuration.

        Returns:
            Dictionary containing the final embeddings and logits.
        """
        batch_size, seq_len = tokens.shape

        # Prepare outputs dict
        outs: Dict[str, jnp.ndarray] = {}

        # Compute embeddings
        x = self._embed_layer(tokens)
        # Tokens dropout if needed
        if self._config.token_dropout:
            x = TokensDropout(
                embed_dim=self._config.embed_dim,
                mask_token_id=self._config.mask_token_id,
                pad_token_id=self._config.pad_token_id,
                masking_ratio=self._config.masking_ratio,
                masking_prob=self._config.masking_prob,
            )(x, tokens)

        # RoBERTa's mask scaling factor
        x = self._config.embed_scale * x

        if self._config.emb_layer_norm_before:
            x = self.emb_ln_before(x)

        # Attention mask
        if attention_mask is None:
            attention_mask = build_padding_attention_mask(
                tokens, self._config.pad_token_id
            )

        # Mask before attention for models ESM1b and ESM2
        if self._config.mask_before_attention:
            x = x - jnp.where(attention_mask == 0, 1, 0)[:, 0, 0][..., None] * x

        # construct a tower of attention layers
        x, outs = self.apply_attention_blocks(
            x=x,
            outs=outs,
            attention_mask=attention_mask,
        )

        # Language Model Head
        lm_head_outs = self._lm_head(x)
        outs["logits"] = lm_head_outs["logits"]

        embeddings = lm_head_outs["embeddings"]

        # Save final embeddings if needed
        if self._config.num_layers in self._config.embeddings_layers_to_save:
            outs[f"embeddings_{self._config.num_layers}"] = embeddings

        return outs  # type: ignore


def build_esm_fn(
    model_config: ESMTransformerConfig,
    mixed_precision: bool = False,
    model_name: Optional[str] = None,
) -> Callable:
    """
    Creates the model's forward pass.

    Args:
        model_config: Model hyperparameters.
        mixed_precision: Whether to use mixed precision computation.

    Returns:
        ESM model forward function.
    """
    if mixed_precision:
        # Use mixed precision (only support A100 GPU and TPU for now)
        half = jnp.bfloat16
        full = jnp.float32

        policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=full)
        hk.mixed_precision.set_policy(ESMTransformer, policy)

        # Remove it in batch norm to avoid instabilities
        policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=half)
        hk.mixed_precision.set_policy(hk.BatchNorm, policy)
        hk.mixed_precision.set_policy(hk.LayerNorm, policy)

    def esm_fn(
        tokens: Tokens, attention_mask: Optional[AttentionMask] = None
    ) -> TransformerOutput:
        """Forward pass."""
        # Run the encoder over the inputs.
        encoder = ESMTransformer(config=model_config, name=model_name)
        outs = encoder(
            tokens=tokens,
            attention_mask=attention_mask,
        )
        return outs

    return esm_fn


def convert_model_torch_to_jax(
    model: torch.nn.ParameterDict, model_name: str
) -> hk.Params:
    """
    Converts ESM PyTorch model to Haiku parameters.

    Args:
        model: Pretrained PyTorch ESM model.
        model_name: Name of the model (see ESM Github repository).

    Returns:
       Haiku parameters.
    """
    param_dict: Dict = defaultdict(lambda: {})

    # copy weights for the initial embedding layers
    param_dict[f"{model_name}/~/embed"][
        "embeddings"
    ] = model.embed_tokens.weight.detach().numpy()
    if hasattr(model, "embed_positions"):
        if hasattr(model.embed_positions, "weight"):  # TODO: remove
            param_dict[f"{model_name}/~/esm_learned_positional_embeddings/~/embed"][
                "embeddings"
            ] = model.embed_positions.weight.detach().numpy()
        if hasattr(model.emb_layer_norm_before, "weight"):  # TODO: remove
            param_dict[f"{model_name}/~/emb_layer_norm_before"][
                "scale"
            ] = model.emb_layer_norm_before.weight.detach().numpy()
            param_dict[f"{model_name}/~/emb_layer_norm_before"][
                "offset"
            ] = model.emb_layer_norm_before.bias.detach().numpy()

    # copy weights for each of the transformer blocks
    num_layers = re.search("t(.+?)_", model_name)
    num_layers = int(num_layers.group(1))  # type: ignore
    for layer_idx in range(num_layers):
        # multiheaded self-attention weights
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/mha/query"]["w"] = (
            model.layers[layer_idx].self_attn.q_proj.weight.detach().numpy().T
        )
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/mha/query"]["b"] = (
            model.layers[layer_idx].self_attn.q_proj.bias.detach().numpy()
        )
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/mha/key"]["w"] = (
            model.layers[layer_idx].self_attn.k_proj.weight.detach().numpy().T
        )
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/mha/key"]["b"] = (
            model.layers[layer_idx].self_attn.k_proj.bias.detach().numpy()
        )
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/mha/value"]["w"] = (
            model.layers[layer_idx].self_attn.v_proj.weight.detach().numpy().T
        )
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/mha/value"]["b"] = (
            model.layers[layer_idx].self_attn.v_proj.bias.detach().numpy()
        )
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/mha/mha_output"][
            "w"
        ] = (model.layers[layer_idx].self_attn.out_proj.weight.detach().numpy().T)
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/mha/mha_output"][
            "b"
        ] = (model.layers[layer_idx].self_attn.out_proj.bias.detach().numpy())
        if model.layers[layer_idx].self_attn.bias_k is not None:
            num_heads = model.layers[0].self_attn.num_heads
            key_size = model.layers[0].self_attn.embed_dim // num_heads
            param_dict[f"{model_name}/attention_layer_{layer_idx}/~/mha"]["bias_k"] = (
                model.layers[layer_idx]
                .self_attn.bias_k.detach()
                .numpy()
                .reshape(1, 1, num_heads, key_size)
            )
            param_dict[f"{model_name}/attention_layer_{layer_idx}/~/mha"]["bias_v"] = (
                model.layers[layer_idx]
                .self_attn.bias_v.detach()
                .numpy()
                .reshape(1, 1, num_heads, key_size)
            )

        # weights for the linear layers
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/fc1"]["w"] = (
            model.layers[layer_idx].fc1.weight.detach().numpy().T
        )
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/fc1"]["b"] = (
            model.layers[layer_idx].fc1.bias.detach().numpy()
        )
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/fc2"]["w"] = (
            model.layers[layer_idx].fc2.weight.detach().numpy().T
        )
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/fc2"]["b"] = (
            model.layers[layer_idx].fc2.bias.detach().numpy()
        )

        # weights for the layer norms
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/mha_layer_norm"][
            "scale"
        ] = (model.layers[layer_idx].self_attn_layer_norm.weight.detach().numpy())
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/mha_layer_norm"][
            "offset"
        ] = (model.layers[layer_idx].self_attn_layer_norm.bias.detach().numpy())
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/final_layer_norm"][
            "scale"
        ] = (model.layers[layer_idx].final_layer_norm.weight.detach().numpy())
        param_dict[f"{model_name}/attention_layer_{layer_idx}/~/final_layer_norm"][
            "offset"
        ] = (model.layers[layer_idx].final_layer_norm.bias.detach().numpy())

    # weights for the post-transformer blocks layer norm.
    if hasattr(model, "emb_layer_norm_after"):
        param_dict[f"{model_name}/~/esm_roberta_lm_head/~/emb_layer_norm_after"][
            "scale"
        ] = model.emb_layer_norm_after.weight.detach().numpy()
        param_dict[f"{model_name}/~/esm_roberta_lm_head/~/emb_layer_norm_after"][
            "offset"
        ] = model.emb_layer_norm_after.bias.detach().numpy()

    param_dict[f"{model_name}/~/esm_roberta_lm_head/~/lm_head_fc_1"]["w"] = (
        model.lm_head.dense.weight.detach().numpy().T
    )
    param_dict[f"{model_name}/~/esm_roberta_lm_head/~/lm_head_fc_1"][
        "b"
    ] = model.lm_head.dense.bias.detach().numpy()
    param_dict[f"{model_name}/~/esm_roberta_lm_head/~/lm_head_layer_norm"][
        "scale"
    ] = model.lm_head.layer_norm.weight.detach().numpy()
    param_dict[f"{model_name}/~/esm_roberta_lm_head/~/lm_head_layer_norm"][
        "offset"
    ] = model.lm_head.layer_norm.bias.detach().numpy()

    param_dict[f"{model_name}/~/esm_roberta_lm_head/~/lm_final_fc"]["w"] = (
        model.lm_head.weight.detach().numpy().T
    )
    param_dict[f"{model_name}/~/esm_roberta_lm_head/~/lm_final_fc"][
        "b"
    ] = model.lm_head.bias.detach().numpy()
    return param_dict


def get_model_hyperparameters(
    model: torch.nn.ParameterDict, model_name: str, alphabet: esm.Alphabet
) -> Dict:
    """
    Gets ESM model hyperparameters to create Haiku model.

    Args:
        model: Pretrained PyTorch ESM model.
        model_name: Name of the model (see ESM Github repository).
        alphabet: Alphabet of ESM.

    Returns:
        Dictionary of hyperparameters of the model.
    """
    hyperparam_dict: Dict = {}
    hyperparam_dict[model_name] = {}

    # copy weights for each of the 33 transformer blocks
    num_layers = re.search("t(.+?)_", model_name)
    num_layers = int(num_layers.group(1))  # type: ignore
    for layer_idx in range(num_layers):
        if model.layers[layer_idx].self_attn.bias_k is not None:
            hyperparam_dict[model_name]["add_bias_kv"] = True
        else:
            hyperparam_dict[model_name]["add_bias_kv"] = False

    hyperparam_dict[model_name]["attention_heads"] = model.layers[0].self_attn.num_heads
    hyperparam_dict[model_name]["embed_dim"] = model.layers[0].self_attn.embed_dim
    hyperparam_dict[model_name]["ffn_embed_dim"] = model.layers[
        layer_idx
    ].fc1.out_features
    hyperparam_dict[model_name]["num_layers"] = num_layers
    hyperparam_dict[model_name]["alphabet_size"] = alphabet.__len__()
    hyperparam_dict[model_name]["model_type"] = (
        "esm-1" if model_name != "esm1b_t33_650M_UR50S" else "esm-1b"
    )
    hyperparam_dict[model_name]["alphabet_size"] = alphabet.__len__()
    hyperparam_dict[model_name]["tokens"] = alphabet.all_toks
    hyperparam_dict[model_name]["embed_scale"] = model.embed_scale
    hyperparam_dict[model_name]["token_dropout"] = model.token_dropout
    hyperparam_dict[model_name]["mask_before_attention"] = True
    hyperparam_dict[model_name]["append_eos"] = True
    hyperparam_dict[model_name]["prepend_cls"] = True
    return hyperparam_dict


def get_config_and_tokenizer(
    hyperparam_dict: Dict,
    model_name: str,
    embeddings_layers_to_save: Tuple[int, ...],
    attention_maps_to_save: Tuple[Tuple[int, ...], ...] = (),
    max_positions: int = 1024,
) -> Tuple[ESMTransformerConfig, FixedSizeStandardTokenizer]:
    """
    Creates the Transformer configuration for ESM.

    Args:
        hyperparam_dict: Dictionary of hyperparameters of the model.
        model_name: Name of the model (see ESM Github repository).
        embeddings_layers_to_save: Indices of the layers to save.
        attention_maps_to_save: (Indices layer, Indices maps) to save.
        max_positions: Maximum number of tokens (for padding).

    Returns:
        Model configuration.
        Model tokenizer.
    """
    hyperparam_dict = hyperparam_dict[model_name]
    tokens = hyperparam_dict["tokens"]

    unk_token = "<unk>"
    pad_token = "<pad>"
    mask_token = "<mask>"
    cls_token = "<cls>"
    eos_token = "<eos>"
    bos_token = "<bos>"
    extra_special_tokens = []
    if "<null_1>" in tokens:
        extra_special_tokens.append("<null_1>")
    if "<null_0>" in tokens:
        extra_special_tokens.append("<null_0>")
    if "<sep>" in tokens:
        extra_special_tokens.append("<sep>")

    special_tokens = [unk_token, pad_token, mask_token, cls_token, eos_token]
    special_tokens = special_tokens + extra_special_tokens

    standard_tokens = list(set(tokens).difference(set(special_tokens)))

    tokenizer = FixedSizeStandardTokenizer(
        fixed_length=max_positions,
        standard_tokens=standard_tokens,
        unk_token=unk_token,
        pad_token=pad_token,
        mask_token=mask_token,
        class_token=cls_token,
        eos_token=eos_token,
        prepend_cls_token=hyperparam_dict["prepend_cls"],
        append_eos_token=hyperparam_dict["append_eos"],
        extra_special_tokens=extra_special_tokens,
        tokens_to_ids={tok: i for i, tok in enumerate(tokens + [bos_token])},
    )

    config = ESMTransformerConfig(
        alphabet_size=tokenizer.vocabulary_size - 1,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        max_positions=max_positions,
        add_bias_kv=hyperparam_dict["add_bias_kv"],
        attention_heads=hyperparam_dict["attention_heads"],
        embed_dim=hyperparam_dict["embed_dim"],
        ffn_embed_dim=hyperparam_dict["ffn_embed_dim"],
        num_layers=hyperparam_dict["num_layers"],
        token_dropout=hyperparam_dict["token_dropout"],
        masking_ratio=0.15,  # hard-coded but it doesn't affect inference.
        masking_prob=0.8,  # hard-coded but it doesn't affect inference.
        embed_scale=hyperparam_dict["embed_scale"],
        mask_before_attention=hyperparam_dict["mask_before_attention"],
        embeddings_layers_to_save=embeddings_layers_to_save,
        attention_maps_to_save=attention_maps_to_save,
    )

    return config, tokenizer


def get_pretrained_esm_model(
    model_name: str,
    model_path: Optional[str] = None,
    mixed_precision: bool = False,
    embeddings_layers_to_save: Tuple[int, ...] = (),
    attention_maps_to_save: Tuple[Tuple[int, ...], ...] = (),
    max_positions: int = 1024,
) -> Tuple[hk.Params, Callable, FixedSizeStandardTokenizer, ESMTransformerConfig]:
    """
    Create a Haiku ESM model by downloading and transforming a pretrained PyTorch model.

    Args:
        model_name: Name of the model (see ESM Github repository).
        model_path: Path to the model if you want to load it from there.
        mixed_precision: Whether to use mixed precision.
        embedding_layers_to_save: Intermediate embeddings to return in the output.
        attention_maps_to_save: Intermediate attention maps to return in the output.
        max_positions: Maximum length of a token (for padding).

    Returns:
        Model parameters.
        Haiku function to call the model.
        Tokenizer.
        Model config (hyperparameters).
    """
    supported_models = [
        "esm2_t6_8M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t48_15B_UR50D",
    ]
    if not (model_name in supported_models):
        raise NotImplementedError(
            f"Unknown ESM {model_name} model. "
            f"Supported models are {supported_models}"
        )
    if model_path is None:
        # load the model from hub
        load_function = getattr(esm.pretrained, model_name)
        model, alphabet = load_function()
    else:
        # load the model from path
        model_path = Path(model_path)  # type: ignore
        assert model_path.is_file()  # type: ignore
        assert model_path.stem == model_name  # type: ignore
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)

    hyperparam_dict = get_model_hyperparameters(model, model_name, alphabet)
    parameters = convert_model_torch_to_jax(model, model_name)
    config, tokenizer = get_config_and_tokenizer(
        hyperparam_dict,
        model_name,
        embeddings_layers_to_save,
        attention_maps_to_save,
        max_positions,
    )

    forward_fn = build_esm_fn(
        model_config=config, mixed_precision=mixed_precision, model_name=model_name
    )

    return parameters, forward_fn, tokenizer
