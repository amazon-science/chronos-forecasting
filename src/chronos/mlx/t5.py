# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from ml-explore/mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/b8a348c1b8df4433cfacb9adbeb89b8aa3979ab2/t5/t5.py
# Modifications:
# - Added support for attention mask.
# - Added support for top_k and top_p sampling.


from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import T5Config


def _relative_position_bucket(
    relative_position, bidirectional=True, num_buckets=32, max_distance=128
):
    # Adapted from HuggingFace transformers:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).astype(mx.int16) * num_buckets
        relative_position = mx.abs(relative_position)
    else:
        relative_position = -mx.minimum(
            relative_position, mx.zeros_like(relative_position)
        )
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins
    # in positions up to max_distance
    scale = (num_buckets - max_exact) / np.log(max_distance / max_exact)
    relative_position_if_large = max_exact + (
        mx.log(relative_position.astype(mx.float32) / max_exact) * scale
    ).astype(mx.int16)
    relative_position_if_large = mx.minimum(relative_position_if_large, num_buckets - 1)
    relative_buckets += mx.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets


class RelativePositionBias(nn.Module):
    def __init__(self, config: T5Config, bidirectional: bool):
        self.bidirectional = bidirectional
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.n_heads = config.num_heads
        self.embeddings = nn.Embedding(
            config.relative_attention_num_buckets, config.num_heads
        )

    def __call__(self, query_length: int, key_length: int, offset: int = 0):
        """Compute binned relative position bias"""
        context_position = mx.arange(offset, query_length)[:, None]
        memory_position = mx.arange(key_length)[None, :]

        # shape (query_length, key_length)
        relative_position = memory_position - context_position
        relative_position_bucket = _relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )

        # shape (query_length, key_length, num_heads)
        values = self.embeddings(relative_position_bucket)

        # shape (num_heads, query_length, key_length)
        return values.transpose(2, 0, 1)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        inner_dim = config.d_kv * config.num_heads
        self.num_heads = config.num_heads
        self.query_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.key_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.value_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, config.d_model, bias=False)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=3)
            values = mx.concatenate([value_cache, values], axis=2)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scores = queries @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)

        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(values_hat), (keys, values)


class DenseActivation(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        mlp_dims = config.d_ff or config.d_model * 4
        self.gated = config.feed_forward_proj.startswith("gated")
        if self.gated:
            self.wi_0 = nn.Linear(config.d_model, mlp_dims, bias=False)
            self.wi_1 = nn.Linear(config.d_model, mlp_dims, bias=False)
        else:
            self.wi = nn.Linear(config.d_model, mlp_dims, bias=False)
        self.wo = nn.Linear(mlp_dims, config.d_model, bias=False)
        activation = config.feed_forward_proj.removeprefix("gated-")
        if activation == "relu":
            self.act = nn.relu
        elif activation == "gelu":
            self.act = nn.gelu
        elif activation == "silu":
            self.act = nn.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def __call__(self, x):
        if self.gated:
            hidden_act = self.act(self.wi_0(x))
            hidden_linear = self.wi_1(x)
            x = hidden_act * hidden_linear
        else:
            x = self.act(self.wi(x))
        return self.wo(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ln1 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dense = DenseActivation(config)

    def __call__(self, x, mask):
        y = self.ln1(x)
        y, _ = self.attention(y, y, y, mask=mask)
        x = x + y

        y = self.ln2(x)
        y = self.dense(y)
        return x + y


class TransformerEncoder(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(config) for i in range(config.num_layers)
        ]
        self.ln = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=True)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None):
        pos_bias = self.relative_attention_bias(x.shape[1], x.shape[1])[None]
        if mask is not None:
            mask = mask[:, None, None, :]
            pos_bias += mask
        for layer in self.layers:
            x = layer(x, mask=pos_bias)
        return self.ln(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.ln1 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln3 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dense = DenseActivation(config)

    def __call__(
        self,
        x: mx.array,
        memory: mx.array,
        mask: mx.array,
        memory_mask: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ):
        y = self.ln1(x)
        y, cache = self.self_attention(y, y, y, mask, cache)
        x = x + y

        y = self.ln2(x)
        y, _ = self.cross_attention(y, memory, memory, memory_mask)
        x = x + y

        y = self.ln3(x)
        y = self.dense(y)
        x = x + y

        return x, cache


class TransformerDecoder(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        n_layers = getattr(config, "num_decoder_layers", config.num_layers)
        self.layers = [TransformerDecoderLayer(config) for i in range(n_layers)]
        self.ln = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=False)

    def __call__(self, x, memory, mask, memory_mask, cache=None):
        if cache is not None:
            offset = cache[0][0].shape[3]
        else:
            offset = 0
            cache = [None] * len(self.layers)

        T = offset + x.shape[1]
        pos_bias = self.relative_attention_bias(T, T, offset=offset)
        if mask is not None:
            mask += pos_bias
        else:
            mask = pos_bias

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, memory, mask, memory_mask, cache=cache[e])
        x = self.ln(x)

        return x, cache


class OutputHead(nn.Module):
    def __init__(self, config: T5Config):
        self.linear = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(self, inputs):
        return self.linear(inputs)


def sample(logits, top_k=1, top_p=1.0, temperature=1.0):
    if top_p != 1.0:
        raise NotImplementedError("top_p sampling is not supported yet")
    if temperature == 0 or top_k == 1:
        return mx.argmax(logits, axis=-1)
    else:
        vocab_size = logits.shape[-1]
        if top_k >= vocab_size:
            return mx.random.categorical(logits)

        top_k_indices = mx.argpartition(logits, top_k, axis=-1)[..., -top_k:]
        top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)

        if temperature != 1.0:
            top_k_logits /= temperature

        return top_k_indices[
            mx.arange(top_k_indices.shape[0]), mx.random.categorical(top_k_logits)
        ]


class T5(nn.Module):
    def __init__(self, config: T5Config):
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = OutputHead(config)
        self.model_dim = config.d_model

    def encode(self, inputs: mx.array, mask: mx.array):
        return self.encoder(self.wte(inputs), mask)

    def decode(
        self,
        inputs: mx.array,
        memory: mx.array,
        memory_mask: mx.array,
        cache=None,
    ):
        inputs = self.wte(inputs)
        T = inputs.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(inputs.dtype)
        else:
            mask = None

        memory_mask = memory_mask[:, None, None, :]
        y, cache = self.decoder(
            inputs, memory=memory, mask=mask, memory_mask=memory_mask, cache=cache
        )
        if not self.tie_word_embeddings:
            y = self.lm_head(y)
        else:
            y *= self.model_dim**-0.5
            y = y @ self.wte.weight.T
        return y, cache

    def __call__(
        self,
        inputs: mx.array,
        decoder_inputs: mx.array,
    ):
        return self.decode(decoder_inputs, self.encode(inputs))[0]

    def generate(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        min_new_tokens: Optional[int] = None,
        max_new_tokens: int = 64,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
        temperature: Optional[float] = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ):
        self.eval()

        def should_stop(current_token, num_sampled_tokens):
            if eos_token_id is not None and (current_token == eos_token_id).all():
                return True
            if num_sampled_tokens >= max_new_tokens:
                return True
            return False

        top_k = top_k if do_sample else 1
        attention_mask = (1.0 - attention_mask.astype(mx.float32)) * -1e9
        memory = self.encode(input_ids, mask=attention_mask)

        repeated_memory = mx.repeat(memory, num_return_sequences, axis=0)
        repeated_attention_mask = mx.repeat(
            attention_mask, num_return_sequences, axis=0
        )
        decoder_start_id = pad_token_id
        decoder_inputs = mx.array([decoder_start_id] * len(repeated_attention_mask))[
            :, None
        ]

        cache = None
        prediction = [decoder_inputs]
        num_sampled_tokens = 0
        while not should_stop(prediction[-1], num_sampled_tokens):
            logits, cache = self.decode(
                prediction[-1],
                repeated_memory,
                memory_mask=repeated_attention_mask,
                cache=cache,
            )
            if (
                min_new_tokens is not None
                and eos_token_id is not None
                and num_sampled_tokens < min_new_tokens
            ):
                logits[..., eos_token_id] = -float("inf")

            y = sample(
                logits[:, -1, :], top_k=top_k, top_p=top_p, temperature=temperature
            )
            num_sampled_tokens += 1
            prediction.append(y[:, None])

        return mx.concatenate(prediction, axis=-1)
