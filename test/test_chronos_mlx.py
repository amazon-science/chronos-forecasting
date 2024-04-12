# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Tuple

import mlx.core as mx
import numpy as np
import pytest

from chronos_mlx.t5 import apply_top_p
from chronos_mlx import ChronosPipeline


def validate_array(samples: np.ndarray, shape: Tuple[int, ...]) -> None:
    assert isinstance(samples, np.ndarray)
    assert samples.shape == shape


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_pipeline_predict(dtype: str):
    pipeline = ChronosPipeline.from_pretrained(
        Path(__file__).parent / "dummy-chronos-model",
        dtype=dtype,
    )
    context = 10 * np.random.rand(4, 16) + 10

    # input: tensor of shape (batch_size, context_length)

    samples = pipeline.predict(context, num_samples=12, prediction_length=3)
    validate_array(samples, (4, 12, 3))

    with pytest.raises(ValueError):
        samples = pipeline.predict(context, num_samples=7, prediction_length=65)

    samples = pipeline.predict(
        context, num_samples=7, prediction_length=65, limit_prediction_length=False
    )
    validate_array(samples, (4, 7, 65))

    # input: batch_size-long list of tensors of shape (context_length,)

    samples = pipeline.predict(list(context), num_samples=12, prediction_length=3)
    validate_array(samples, (4, 12, 3))

    with pytest.raises(ValueError):
        samples = pipeline.predict(list(context), num_samples=7, prediction_length=65)

    samples = pipeline.predict(
        list(context),
        num_samples=7,
        prediction_length=65,
        limit_prediction_length=False,
    )
    validate_array(samples, (4, 7, 65))

    # input: tensor of shape (context_length,)

    samples = pipeline.predict(context[0, ...], num_samples=12, prediction_length=3)
    validate_array(samples, (1, 12, 3))

    with pytest.raises(ValueError):
        samples = pipeline.predict(context[0, ...], num_samples=7, prediction_length=65)

    samples = pipeline.predict(
        context[0, ...],
        num_samples=7,
        prediction_length=65,
        limit_prediction_length=False,
    )
    validate_array(samples, (1, 7, 65))

    # test non-default inference params
    samples = pipeline.predict(
        context,
        num_samples=12,
        prediction_length=3,
        top_p=0.7,
        top_k=32,
        temperature=0.9,
    )
    validate_array(samples, (4, 12, 3))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_pipeline_embed(dtype: str):
    pipeline = ChronosPipeline.from_pretrained(
        Path(__file__).parent / "dummy-chronos-model",
        dtype=dtype,
    )
    d_model = pipeline.model.model.model_dim
    context = 10 * np.random.rand(4, 16) + 10
    expected_embed_length = 16 + (1 if pipeline.model.config.use_eos_token else 0)

    # input: tensor of shape (batch_size, context_length)

    embedding, scale = pipeline.embed(context)
    validate_array(embedding, (4, expected_embed_length, d_model))
    validate_array(scale, (4,))

    # input: batch_size-long list of tensors of shape (context_length,)

    embedding, scale = pipeline.embed(list(context))
    validate_array(embedding, (4, expected_embed_length, d_model))
    validate_array(scale, (4,))

    # input: tensor of shape (context_length,)
    embedding, scale = pipeline.embed(context[0, ...])
    validate_array(embedding, (1, expected_embed_length, d_model))
    validate_array(scale, (1,))


@pytest.mark.parametrize(
    "top_p,expected_non_zero_probs",
    [
        (
            0.1,
            mx.array(
                [
                    [False, True, False, False],
                    [False, True, False, False],
                    [True, False, False, False],
                    [True, False, False, False],
                    [False, False, False, True],
                ]
            ),
        ),
        (
            0.5,
            mx.array(
                [
                    [False, True, False, False],
                    [False, True, False, False],
                    [True, False, False, False],
                    [True, False, False, False],
                    [False, False, True, True],
                ]
            ),
        ),
        (
            0.95,
            mx.array(
                [
                    [False, True, True, True],
                    [False, True, False, True],
                    [True, False, False, False],
                    [True, True, False, False],
                    [False, True, True, True],
                ]
            ),
        ),
    ],
)
def test_apply_top_p(top_p: float, expected_non_zero_probs: mx.array):
    probs = mx.array(
        [
            [0.1, 0.4, 0.3, 0.2],
            [0.01, 0.39, 0.25, 0.35],
            [0.9, 0.01, 0.01, 0.08],
            [0.7, 0.2, 0.05, 0.05],
            [0.25, 0.25, 0.25, 0.25],
        ],
    )
    top_p_probs = mx.softmax(apply_top_p(probs.log(), top_p=top_p), axis=-1)
    assert mx.all(mx.not_equal(top_p_probs, 0.0) == expected_non_zero_probs)
