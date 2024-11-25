from pathlib import Path
from typing import Tuple

import pytest
import torch

from chronos import BaseChronosPipeline, ChronosBoltPipeline
from chronos.chronos_bolt import InstanceNorm, Patch


def validate_tensor(input: torch.Tensor, shape: Tuple[int, ...]) -> None:
    assert isinstance(input, torch.Tensor)
    assert input.shape == shape


def test_base_chronos_pipeline_loads_from_huggingface():
    BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu")


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.bfloat16])
def test_pipeline_predict(torch_dtype: str):
    pipeline = ChronosBoltPipeline.from_pretrained(
        Path(__file__).parent / "dummy-chronos-bolt-model",
        device_map="cpu",
        torch_dtype=torch_dtype,
    )
    context = 10 * torch.rand(size=(4, 16)) + 10
    expected_num_quantiles = len(pipeline.quantiles)

    # input: tensor of shape (batch_size, context_length)

    quantiles = pipeline.predict(context, prediction_length=3)
    validate_tensor(quantiles, (4, expected_num_quantiles, 3))

    with pytest.raises(ValueError):
        quantiles = pipeline.predict(
            context, prediction_length=65, limit_prediction_length=True
        )

    quantiles = pipeline.predict(context, prediction_length=65)
    validate_tensor(quantiles, (4, expected_num_quantiles, 65))

    # input: batch_size-long list of tensors of shape (context_length,)

    quantiles = pipeline.predict(list(context), prediction_length=3)
    validate_tensor(quantiles, (4, expected_num_quantiles, 3))

    with pytest.raises(ValueError):
        quantiles = pipeline.predict(
            list(context),
            prediction_length=65,
            limit_prediction_length=True,
        )

    quantiles = pipeline.predict(list(context), prediction_length=65)
    validate_tensor(quantiles, (4, expected_num_quantiles, 65))

    # input: tensor of shape (context_length,)

    quantiles = pipeline.predict(context[0, ...], prediction_length=3)
    validate_tensor(quantiles, (1, expected_num_quantiles, 3))

    with pytest.raises(ValueError):
        quantiles = pipeline.predict(
            context[0, ...],
            prediction_length=65,
            limit_prediction_length=True,
        )

    quantiles = pipeline.predict(
        context[0, ...],
        prediction_length=65,
    )
    validate_tensor(quantiles, (1, expected_num_quantiles, 65))


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("prediction_length", [3, 65])
@pytest.mark.parametrize(
    "quantile_levels", [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.1, 0.5, 0.9]]
)
def test_pipeline_predict_quantiles(
    torch_dtype: str, prediction_length: int, quantile_levels: list[int]
):
    pipeline = ChronosBoltPipeline.from_pretrained(
        Path(__file__).parent / "dummy-chronos-bolt-model",
        device_map="cpu",
        torch_dtype=torch_dtype,
    )
    context = 10 * torch.rand(size=(4, 16)) + 10

    num_expected_quantiles = len(quantile_levels)
    # input: tensor of shape (batch_size, context_length)

    quantiles, mean = pipeline.predict_quantiles(
        context,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
    validate_tensor(quantiles, (4, prediction_length, num_expected_quantiles))
    validate_tensor(mean, (4, prediction_length))

    # input: batch_size-long list of tensors of shape (context_length,)

    quantiles, mean = pipeline.predict_quantiles(
        list(context),
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
    validate_tensor(quantiles, (4, prediction_length, num_expected_quantiles))
    validate_tensor(mean, (4, prediction_length))

    # input: tensor of shape (context_length,)

    quantiles, mean = pipeline.predict_quantiles(
        context[0, ...],
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
    validate_tensor(quantiles, (1, prediction_length, num_expected_quantiles))
    validate_tensor(mean, (1, prediction_length))


# The following tests have been taken from
# https://github.com/autogluon/autogluon/blob/f57beb26cb769c6e0d484a6af2b89eab8aee73a8/timeseries/tests/unittests/models/chronos/pipeline/test_chronos_bolt.py
# Author: Caner Turkmen <atturkm@amazon.com>


def test_given_even_data_patch_operator_output_is_correct():
    batch_size = 17
    patch_len = 16

    patch = Patch(patch_len, patch_len)

    batch = (
        torch.stack([torch.arange(512)] * batch_size)
        + torch.arange(batch_size)[:, None]
    )
    output = patch(batch)

    assert output.shape == (batch_size, 512 // patch_len, patch_len)

    assert torch.allclose(
        output[:, 0],
        torch.stack([torch.arange(patch_len)] * batch_size)
        + torch.arange(batch_size)[:, None],
        atol=1e-5,
    )
    assert torch.allclose(
        output[:, 1],
        torch.stack([torch.arange(patch_len, 2 * patch_len)] * batch_size)
        + torch.arange(batch_size)[:, None],
        atol=1e-5,
    )
    assert not torch.isnan(output).any()


def test_given_even_data_and_strides_patch_operator_output_is_correct():
    batch_size = 17
    patch_len, patch_stride = 16, 8

    patch = Patch(patch_len, patch_stride)

    offset = torch.arange(batch_size)[:, None]
    batch = torch.stack([torch.arange(512)] * batch_size) + offset
    output = patch(batch)

    assert torch.allclose(
        output[:, 1],
        torch.stack([torch.arange(patch_stride, patch_stride + patch_len)] * batch_size)
        + offset,
        atol=1e-5,
    )
    assert not torch.isnan(output).any()


def test_given_uneven_data_patch_operator_pads_and_output_is_correct():
    batch_size = 17
    patch_len = 16

    patch = Patch(patch_len, patch_len)

    batch = (
        torch.stack([torch.arange(512 - patch_len + 1)] * batch_size)
        + torch.arange(batch_size)[:, None]
    ).float()
    output = patch(batch)

    assert output.shape == (batch_size, 512 // patch_len, patch_len)

    # check the first portion is padded
    assert torch.isnan(output[:, 0, :-1]).all()

    # check nowhere else is nan
    assert not torch.isnan(output[:, 1:]).any()


def test_when_instancenorm_applied_then_standardization_correct():
    inorm = InstanceNorm()

    input_ = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
        ]
    ).float()

    normalized, (loc, scale) = inorm(input_)

    assert normalized.shape == input_.shape
    assert torch.allclose(normalized[0], normalized[1])
    assert torch.allclose(loc.squeeze(), torch.tensor([3.0, 4.0]))
    assert torch.allclose(scale.squeeze(), torch.tensor(1.41421))


def test_when_instancenorm_applied_and_reversed_then_nans_preserved():
    inorm = InstanceNorm()

    input_ = torch.tensor(
        [
            [1, torch.nan, 3, 4, 5],
            [2, 3, 4, 5, torch.nan],
        ]
    ).float()

    normalized, (loc, scale) = inorm(input_)
    assert torch.allclose(normalized.isnan(), input_.isnan())

    output = inorm.inverse(normalized, (loc, scale))
    assert torch.allclose(output, input_, equal_nan=True)


def test_when_instancenorm_applied_and_reversed_then_output_correct():
    inorm = InstanceNorm()

    input_ = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 1000],
        ]
    ).float()

    normalized, loc_scale = inorm(input_)
    output = inorm.inverse(normalized, loc_scale)

    assert torch.allclose(output, input_)
