# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from chronos.mlx import ChronosPipeline


def validate_array(samples: np.ndarray, shape: Tuple[int, int, int]) -> None:
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
