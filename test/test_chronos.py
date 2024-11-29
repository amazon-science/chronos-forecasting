# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

from chronos import (
    BaseChronosPipeline,
    ChronosConfig,
    ChronosPipeline,
    MeanScaleUniformBins,
)
from test.util import validate_tensor


def test_base_chronos_pipeline_loads_from_huggingface():
    BaseChronosPipeline.from_pretrained("amazon/chronos-t5-tiny", device_map="cpu")


@pytest.mark.parametrize("n_numerical_tokens", [5, 10, 27])
@pytest.mark.parametrize("n_special_tokens", [2, 5, 13])
def test_tokenizer_consistency(n_numerical_tokens: int, n_special_tokens: int):
    n_tokens = n_numerical_tokens + n_special_tokens

    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs=dict(low_limit=-1.0, high_limit=1.0),
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=512,
        prediction_length=64,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )

    tokenizer = config.create_tokenizer()
    assert isinstance(tokenizer, MeanScaleUniformBins)

    context = tokenizer.centers.unsqueeze(0)  # add batch dimension
    scale = torch.ones((1,))  # fix the scale to one to turn off scaling

    token_ids, _, _ = tokenizer._input_transform(context, scale=scale)

    samples = tokenizer.output_transform(
        token_ids.unsqueeze(1),  # add sample dimension
        scale=scale,
    )

    assert (samples[0, 0, :] == context).all()


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

    token_ids, attention_mask, scale = tokenizer.context_input_transform(context)

    assert token_ids.shape == (batch_size, context_length + 1 * use_eos_token)
    assert all(token_ids[:, 0] == torch.tensor([0]).repeat(batch_size))
    assert all(token_ids[:, 1] == torch.tensor([n_special_tokens]).repeat(batch_size))
    assert all(token_ids[:, 2] == torch.tensor([n_tokens - 1]).repeat(batch_size))

    if use_eos_token:
        assert all(token_ids[:, 3] == torch.tensor([1]).repeat(batch_size))

    samples = tokenizer.output_transform(
        torch.arange(n_special_tokens, n_tokens).unsqueeze(0).repeat(batch_size, 1, 1),
        tokenizer_state=scale,
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

    token_ids, attention_mask, scale = tokenizer.context_input_transform(context)

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


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_pipeline_predict(model_dtype: torch.dtype, input_dtype: torch.dtype):
    pipeline = ChronosPipeline.from_pretrained(
        Path(__file__).parent / "dummy-chronos-model",
        device_map="cpu",
        torch_dtype=model_dtype,
    )
    context = 10 * torch.rand(size=(4, 16)) + 10
    context = context.to(dtype=input_dtype)

    # input: tensor of shape (batch_size, context_length)

    samples = pipeline.predict(context, num_samples=12, prediction_length=3)
    validate_tensor(samples, shape=(4, 12, 3), dtype=torch.float32)

    with pytest.raises(ValueError):
        samples = pipeline.predict(
            context, num_samples=7, prediction_length=65, limit_prediction_length=True
        )

    samples = pipeline.predict(
        context, num_samples=7, prediction_length=65, limit_prediction_length=False
    )
    validate_tensor(samples, shape=(4, 7, 65), dtype=torch.float32)

    # input: batch_size-long list of tensors of shape (context_length,)

    samples = pipeline.predict(list(context), num_samples=12, prediction_length=3)
    validate_tensor(samples, shape=(4, 12, 3), dtype=torch.float32)

    with pytest.raises(ValueError):
        samples = pipeline.predict(
            list(context),
            num_samples=7,
            prediction_length=65,
            limit_prediction_length=True,
        )

    samples = pipeline.predict(
        list(context),
        num_samples=7,
        prediction_length=65,
        limit_prediction_length=False,
    )
    validate_tensor(samples, shape=(4, 7, 65), dtype=torch.float32)

    # input: tensor of shape (context_length,)

    samples = pipeline.predict(context[0, ...], num_samples=12, prediction_length=3)
    validate_tensor(samples, shape=(1, 12, 3), dtype=torch.float32)

    with pytest.raises(ValueError):
        samples = pipeline.predict(
            context[0, ...],
            num_samples=7,
            prediction_length=65,
            limit_prediction_length=True,
        )

    samples = pipeline.predict(
        context[0, ...],
        num_samples=7,
        prediction_length=65,
    )
    validate_tensor(samples, shape=(1, 7, 65), dtype=torch.float32)


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("prediction_length", [3, 65])
@pytest.mark.parametrize(
    "quantile_levels", [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.1, 0.5, 0.9]]
)
def test_pipeline_predict_quantiles(
    model_dtype: torch.dtype,
    input_dtype: torch.dtype,
    prediction_length: int,
    quantile_levels: list[int],
):
    pipeline = ChronosPipeline.from_pretrained(
        Path(__file__).parent / "dummy-chronos-model",
        device_map="cpu",
        torch_dtype=model_dtype,
    )
    context = 10 * torch.rand(size=(4, 16)) + 10
    context = context.to(dtype=input_dtype)

    num_expected_quantiles = len(quantile_levels)
    # input: tensor of shape (batch_size, context_length)

    quantiles, mean = pipeline.predict_quantiles(
        context,
        num_samples=12,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
    validate_tensor(
        quantiles, (4, prediction_length, num_expected_quantiles), dtype=torch.float32
    )
    validate_tensor(mean, (4, prediction_length), dtype=torch.float32)

    # input: batch_size-long list of tensors of shape (context_length,)

    quantiles, mean = pipeline.predict_quantiles(
        list(context),
        num_samples=12,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
    validate_tensor(
        quantiles, (4, prediction_length, num_expected_quantiles), dtype=torch.float32
    )
    validate_tensor(mean, (4, prediction_length), dtype=torch.float32)

    # input: tensor of shape (context_length,)

    quantiles, mean = pipeline.predict_quantiles(
        context[0, ...],
        num_samples=12,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
    validate_tensor(
        quantiles, (1, prediction_length, num_expected_quantiles), dtype=torch.float32
    )
    validate_tensor(mean, (1, prediction_length), dtype=torch.float32)


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_pipeline_embed(model_dtype: torch.dtype, input_dtype: torch.dtype):
    pipeline = ChronosPipeline.from_pretrained(
        Path(__file__).parent / "dummy-chronos-model",
        device_map="cpu",
        torch_dtype=model_dtype,
    )
    d_model = pipeline.model.model.config.d_model
    context = 10 * torch.rand(size=(4, 16)) + 10
    context = context.to(dtype=input_dtype)
    expected_embed_length = 16 + (1 if pipeline.model.config.use_eos_token else 0)

    # input: tensor of shape (batch_size, context_length)

    embedding, scale = pipeline.embed(context)
    validate_tensor(
        embedding, shape=(4, expected_embed_length, d_model), dtype=model_dtype
    )
    validate_tensor(scale, shape=(4,), dtype=torch.float32)

    # input: batch_size-long list of tensors of shape (context_length,)

    embedding, scale = pipeline.embed(list(context))
    validate_tensor(
        embedding, shape=(4, expected_embed_length, d_model), dtype=model_dtype
    )
    validate_tensor(scale, shape=(4,), dtype=torch.float32)

    # input: tensor of shape (context_length,)
    embedding, scale = pipeline.embed(context[0, ...])
    validate_tensor(
        embedding, shape=(1, expected_embed_length, d_model), dtype=model_dtype
    )
    validate_tensor(scale, shape=(1,), dtype=torch.float32)


@pytest.mark.parametrize("n_tokens", [10, 1000, 10000])
def test_tokenizer_number_of_buckets(n_tokens):
    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs=dict(low_limit=-1.0, high_limit=1.0),
        n_tokens=n_tokens,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=512,
        prediction_length=64,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    tokenizer = config.create_tokenizer()

    n_numerical_tokens = config.n_tokens - config.n_special_tokens

    # The tokenizer has one bucket too many as a result of an early bug. In order to
    # keep consistent with the original trained models, this is kept as it is. However,
    # token ids are clipped to a maximum of `n_tokens - 1` to avoid out-of-bounds errors.
    assert len(tokenizer.centers) == (n_numerical_tokens - 1)
    assert len(tokenizer.boundaries) == n_numerical_tokens


@pytest.mark.parametrize("n_tokens", [10, 1000, 10000])
def test_token_clipping(n_tokens):
    config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15, "high_limit": 15},
        n_tokens=n_tokens,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=512,
        prediction_length=64,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    tokenizer = config.create_tokenizer()

    huge_value = 1e22  # this large value is assigned to the largest bucket
    token_ids, _, _ = tokenizer._input_transform(
        context=torch.tensor([[huge_value]]), scale=torch.tensor(([1]))
    )
    assert token_ids[0, 0] == config.n_tokens - 1  # and it's clipped to n_tokens - 1
