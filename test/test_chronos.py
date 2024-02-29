# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Tuple

import torch
import pytest

from chronos import ChronosConfig, ChronosPipeline


@pytest.mark.xfail
@pytest.mark.parametrize("n_numerical_tokens", [5, 10, 27])
@pytest.mark.parametrize("n_special_tokens", [2, 5, 13])
@pytest.mark.parametrize("use_eos_token", [False, True])
def test_tokenizer_fixed_data(
    n_numerical_tokens: int, n_special_tokens: int, use_eos_token: bool
):
    n_tokens = n_numerical_tokens + n_special_tokens
    context_length = 3

    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs=dict(low_limit=-1.0, high_limit=1.0),
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=use_eos_token,
        model_type="seq2seq",
        context_length=512,
        prediction_length=64,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )

    tokenizer = config.create_tokenizer()

    context = torch.tensor(
        [
            [-3.7, 3.7],
            [-42.0, 42.0],
        ]
    )
    batch_size, _ = context.shape

    token_ids, attention_mask, scale = tokenizer.input_transform(context)

    assert token_ids.shape == (batch_size, context_length + 1 * use_eos_token)
    assert all(token_ids[:, 0] == torch.tensor([0]).repeat(batch_size))
    assert all(token_ids[:, 1] == torch.tensor([n_special_tokens]).repeat(batch_size))
    assert all(token_ids[:, 2] == torch.tensor([n_tokens - 1]).repeat(batch_size))

    if use_eos_token:
        assert all(token_ids[:, 3] == torch.tensor([1]).repeat(batch_size))

    samples = tokenizer.output_transform(
        torch.arange(n_special_tokens, n_tokens).unsqueeze(0).repeat(batch_size, 1, 1),
        decoding_context=scale,
    )

    assert (samples[:, 0, [0, -1]] == context).all()


@pytest.mark.xfail
@pytest.mark.parametrize("use_eos_token", [False, True])
def test_tokenizer_random_data(use_eos_token: bool):
    context_length = 8
    n_tokens = 256
    n_special_tokens = 2

    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs=dict(low_limit=-1.0, high_limit=1.0),
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=use_eos_token,
        model_type="seq2seq",
        context_length=context_length,
        prediction_length=64,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )

    tokenizer = config.create_tokenizer()

    context = torch.tensor(
        [
            [torch.nan, torch.nan, 1.0, 1.1, torch.nan, 2.0],
            [3.0, torch.nan, 3.9, 4.0, 4.1, 4.9],
        ]
    )

    token_ids, attention_mask, scale = tokenizer.input_transform(context)

    assert token_ids.shape == (
        *context.shape[:-1],
        context_length + 1 * use_eos_token,
    )
    assert attention_mask.shape == (
        *context.shape[:-1],
        context_length + 1 * use_eos_token,
    )
    assert scale.shape == context.shape[:1]

    sample_ids = torch.randint(low=n_special_tokens, high=n_tokens, size=(2, 10, 4))
    sample_ids[0, 0, 0] = n_special_tokens
    sample_ids[-1, -1, -1] = n_tokens - 1

    samples = tokenizer.output_transform(sample_ids, scale)

    assert samples.shape == (2, 10, 4)


def validate_samples(samples: torch.Tensor, shape: Tuple[int, int, int]) -> None:
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == shape


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.bfloat16])
def test_pipeline(torch_dtype: str):
    pipeline = ChronosPipeline.from_pretrained(
        Path(__file__).parent / "dummy-chronos-model",
        device_map="cpu",
        torch_dtype=torch_dtype,
    )
    context = 10 * torch.rand(size=(4, 16)) + 10

    # input: tensor of shape (batch_size, context_length)

    samples = pipeline.predict(context, num_samples=12, prediction_length=3)
    validate_samples(samples, (4, 12, 3))

    with pytest.raises(ValueError):
        samples = pipeline.predict(context, num_samples=7, prediction_length=65)

    samples = pipeline.predict(
        context, num_samples=7, prediction_length=65, limit_prediction_length=False
    )
    validate_samples(samples, (4, 7, 65))

    # input: batch_size-long list of tensors of shape (context_length,)

    samples = pipeline.predict(list(context), num_samples=12, prediction_length=3)
    validate_samples(samples, (4, 12, 3))

    with pytest.raises(ValueError):
        samples = pipeline.predict(list(context), num_samples=7, prediction_length=65)

    samples = pipeline.predict(
        list(context),
        num_samples=7,
        prediction_length=65,
        limit_prediction_length=False,
    )
    validate_samples(samples, (4, 7, 65))

    # input: tensor of shape (context_length,)

    samples = pipeline.predict(context[0, ...], num_samples=12, prediction_length=3)
    validate_samples(samples, (1, 12, 3))

    with pytest.raises(ValueError):
        samples = pipeline.predict(context[0, ...], num_samples=7, prediction_length=65)

    samples = pipeline.predict(
        context[0, ...],
        num_samples=7,
        prediction_length=65,
        limit_prediction_length=False,
    )
    validate_samples(samples, (1, 7, 65))
