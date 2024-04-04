# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from ml-explore/mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/b8a348c1b8df4433cfacb9adbeb89b8aa3979ab2/t5/convert.py

import mlx.core as mx
import torch
from transformers import T5ForConditionalGeneration

SHARED_REPLACEMENT_PATTERNS = [
    (".block.", ".layers."),
    (".k.", ".key_proj."),
    (".o.", ".out_proj."),
    (".q.", ".query_proj."),
    (".v.", ".value_proj."),
    ("shared.", "wte."),
    ("lm_head.", "lm_head.linear."),
    (".layer.0.layer_norm.", ".ln1."),
    (".layer.1.layer_norm.", ".ln2."),
    (".layer.2.layer_norm.", ".ln3."),
    (".final_layer_norm.", ".ln."),
    (
        "layers.0.layer.0.SelfAttention.relative_attention_bias.",
        "relative_attention_bias.embeddings.",
    ),
]

ENCODER_REPLACEMENT_PATTERNS = [
    (".layer.0.SelfAttention.", ".attention."),
    (".layer.1.DenseReluDense.", ".dense."),
]

DECODER_REPLACEMENT_PATTERNS = [
    (".layer.0.SelfAttention.", ".self_attention."),
    (".layer.1.EncDecAttention.", ".cross_attention."),
    (".layer.2.DenseReluDense.", ".dense."),
]


def replace_key(key: str) -> str:
    for old, new in SHARED_REPLACEMENT_PATTERNS:
        key = key.replace(old, new)
    if key.startswith("encoder."):
        for old, new in ENCODER_REPLACEMENT_PATTERNS:
            key = key.replace(old, new)
    elif key.startswith("decoder."):
        for old, new in DECODER_REPLACEMENT_PATTERNS:
            key = key.replace(old, new)
    return key


def translate_weights(model_name: str, dtype: mx.Dtype):
    """Translate a HuggingFace transformers T5 model to MLX.

    Parameters
    ----------
    model_name
        HuggingFace model name or local path.
    dtype
        mlx dtype for the resulting mlx weights.

    Returns
    -------
        A state dictionary with weights as mlx arrays.
    """
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    weights = {
        replace_key(k): mx.array(v.numpy(), dtype=dtype)
        for k, v in model.state_dict().items()
    }
    return weights
