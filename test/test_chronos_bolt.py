# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import datasets
import fev
import numpy as np
import pandas as pd
import pytest
import torch

from chronos import BaseChronosPipeline, ChronosBoltPipeline
from chronos.chronos_bolt import InstanceNorm, Patch
from test.util import create_df, get_forecast_start_times, validate_tensor

DUMMY_MODEL_PATH = Path(__file__).parent / "dummy-chronos-bolt-model"


@pytest.fixture
def pipeline() -> ChronosBoltPipeline:
    return BaseChronosPipeline.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu")


def test_base_chronos_pipeline_loads_from_huggingface():
    BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu")


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_pipeline_predict(torch_dtype: torch.dtype, input_dtype: torch.dtype):
    pipeline = ChronosBoltPipeline.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu", torch_dtype=torch_dtype)
    context = 10 * torch.rand(size=(4, 16)) + 10
    context = context.to(dtype=input_dtype)
    expected_num_quantiles = len(pipeline.quantiles)

    # input: tensor of shape (batch_size, context_length)

    quantiles = pipeline.predict(context, prediction_length=3)
    validate_tensor(quantiles, (4, expected_num_quantiles, 3), dtype=torch.float32)

    with pytest.raises(ValueError):
        quantiles = pipeline.predict(context, prediction_length=65, limit_prediction_length=True)

    quantiles = pipeline.predict(context, prediction_length=65)
    validate_tensor(quantiles, (4, expected_num_quantiles, 65))

    # input: batch_size-long list of tensors of shape (context_length,)

    quantiles = pipeline.predict(list(context), prediction_length=3)
    validate_tensor(quantiles, (4, expected_num_quantiles, 3), dtype=torch.float32)

    with pytest.raises(ValueError):
        quantiles = pipeline.predict(
            list(context),
            prediction_length=65,
            limit_prediction_length=True,
        )

    quantiles = pipeline.predict(list(context), prediction_length=65)
    validate_tensor(quantiles, (4, expected_num_quantiles, 65), dtype=torch.float32)

    # input: tensor of shape (context_length,)

    quantiles = pipeline.predict(context[0, ...], prediction_length=3)
    validate_tensor(quantiles, (1, expected_num_quantiles, 3), dtype=torch.float32)

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
    validate_tensor(quantiles, (1, expected_num_quantiles, 65), dtype=torch.float32)


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("prediction_length", [3, 65])
@pytest.mark.parametrize("quantile_levels", [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.1, 0.5, 0.9]])
def test_pipeline_predict_quantiles(
    torch_dtype: torch.dtype,
    input_dtype: torch.dtype,
    prediction_length: int,
    quantile_levels: list[float],
):
    pipeline = ChronosBoltPipeline.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu", torch_dtype=torch_dtype)
    context = 10 * torch.rand(size=(4, 16)) + 10
    context = context.to(dtype=input_dtype)

    num_expected_quantiles = len(quantile_levels)
    # input: tensor of shape (batch_size, context_length)

    quantiles, mean = pipeline.predict_quantiles(
        context,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
    validate_tensor(quantiles, (4, prediction_length, num_expected_quantiles), dtype=torch.float32)
    validate_tensor(mean, (4, prediction_length), dtype=torch.float32)

    # input: batch_size-long list of tensors of shape (context_length,)

    quantiles, mean = pipeline.predict_quantiles(
        list(context),
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
    validate_tensor(quantiles, (4, prediction_length, num_expected_quantiles), dtype=torch.float32)
    validate_tensor(mean, (4, prediction_length), dtype=torch.float32)

    # input: tensor of shape (context_length,)

    quantiles, mean = pipeline.predict_quantiles(
        context[0, ...],
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )
    validate_tensor(quantiles, (1, prediction_length, num_expected_quantiles), dtype=torch.float32)
    validate_tensor(mean, (1, prediction_length), dtype=torch.float32)


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_pipeline_embed(model_dtype: torch.dtype, input_dtype: torch.dtype):
    pipeline = ChronosBoltPipeline.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu", torch_dtype=model_dtype)
    d_model = pipeline.model.config.d_model
    context = 10 * torch.rand(size=(4, 16)) + 10
    context = context.to(dtype=input_dtype)

    # the patch size of dummy model is 16, so only 1 patch is created
    expected_embed_length = 1 + (1 if pipeline.model.config.chronos_config["use_reg_token"] else 0)

    # input: tensor of shape (batch_size, context_length)

    embedding, loc_scale = pipeline.embed(context)
    validate_tensor(embedding, shape=(4, expected_embed_length, d_model), dtype=model_dtype)
    validate_tensor(loc_scale[0], shape=(4,), dtype=torch.float32)
    validate_tensor(loc_scale[1], shape=(4,), dtype=torch.float32)

    # input: batch_size-long list of tensors of shape (context_length,)

    embedding, loc_scale = pipeline.embed(list(context))
    validate_tensor(embedding, shape=(4, expected_embed_length, d_model), dtype=model_dtype)
    validate_tensor(loc_scale[0], shape=(4,), dtype=torch.float32)
    validate_tensor(loc_scale[1], shape=(4,), dtype=torch.float32)

    # input: tensor of shape (context_length,)
    embedding, loc_scale = pipeline.embed(context[0, ...])
    validate_tensor(embedding, shape=(1, expected_embed_length, d_model), dtype=model_dtype)
    validate_tensor(loc_scale[0], shape=(1,), dtype=torch.float32)
    validate_tensor(loc_scale[1], shape=(1,), dtype=torch.float32)


@pytest.mark.parametrize(
    "context_setup, expected_rows",
    [
        # Targets only
        ({}, 6),  # 2 series * 3 predictions
        # Different context lengths
        (
            {"series_ids": ["X", "Y", "Z"], "n_points": [10, 17, 56], "target_cols": ["custom_target"]},
            9,
        ),  # 3 series * 3 predictions
    ],
)
@pytest.mark.parametrize("freq", ["s", "min", "30min", "h", "D", "W", "ME", "QE", "YE"])
def test_predict_df_works_for_valid_inputs(pipeline, context_setup, expected_rows, freq):
    prediction_length = 3
    df = create_df(**context_setup, freq=freq)
    forecast_start_times = get_forecast_start_times(df, freq)

    series_ids = context_setup.get("series_ids", ["A", "B"])
    target_columns = context_setup.get("target_cols", ["target"])
    n_series = len(series_ids)
    n_targets = len(target_columns)
    result = pipeline.predict_df(df, target=target_columns[0], prediction_length=prediction_length)

    assert len(result) == expected_rows
    assert "item_id" in result.columns and np.all(
        result["item_id"].to_numpy() == np.array(series_ids).repeat(n_targets * prediction_length)
    )
    assert "target_name" in result.columns and np.all(
        result["target_name"].to_numpy() == np.tile(np.array(target_columns).repeat(prediction_length), n_series)
    )
    assert "timestamp" in result.columns and np.all(
        result.groupby("item_id")["timestamp"].min().to_numpy() == pd.to_datetime(forecast_start_times).to_numpy()
    )
    assert "predictions" in result.columns
    assert all(str(q) in result.columns for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def test_predict_df_with_non_uniform_timestamps_raises_error(pipeline):
    df = create_df()
    # Make timestamps non-uniform for series A
    df.loc[df["item_id"] == "A", "timestamp"] = [
        "2023-01-01",
        "2023-01-02",
        "2023-01-04",
        "2023-01-05",
        "2023-01-06",
        "2023-01-07",
        "2023-01-08",
        "2023-01-09",
        "2023-01-10",
        "2023-01-11",
    ]

    with pytest.raises(ValueError, match="not infer frequency"):
        pipeline.predict_df(df)


def test_predict_df_with_inconsistent_frequencies_raises_error(pipeline):
    df = pd.DataFrame(
        {
            "item_id": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "timestamp": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
                "2023-01-01",
                "2023-02-01",
                "2023-03-01",
                "2023-04-01",
                "2023-05-01",
            ],
            "target": [1.0] * 10,
        }
    )

    with pytest.raises(ValueError, match="same frequency"):
        pipeline.predict_df(df)


# The following tests have been taken from
# https://github.com/autogluon/autogluon/blob/f57beb26cb769c6e0d484a6af2b89eab8aee73a8/timeseries/tests/unittests/models/chronos/pipeline/test_chronos_bolt.py
# Author: Caner Turkmen <atturkm@amazon.com>


def test_given_even_data_patch_operator_output_is_correct():
    batch_size = 17
    patch_len = 16

    patch = Patch(patch_len, patch_len)

    batch = torch.stack([torch.arange(512)] * batch_size) + torch.arange(batch_size)[:, None]
    output = patch(batch)

    assert output.shape == (batch_size, 512 // patch_len, patch_len)

    assert torch.allclose(
        output[:, 0],
        torch.stack([torch.arange(patch_len)] * batch_size) + torch.arange(batch_size)[:, None],
        atol=1e-5,
    )
    assert torch.allclose(
        output[:, 1],
        torch.stack([torch.arange(patch_len, 2 * patch_len)] * batch_size) + torch.arange(batch_size)[:, None],
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
        torch.stack([torch.arange(patch_stride, patch_stride + patch_len)] * batch_size) + offset,
        atol=1e-5,
    )
    assert not torch.isnan(output).any()


def test_given_uneven_data_patch_operator_pads_and_output_is_correct():
    batch_size = 17
    patch_len = 16

    patch = Patch(patch_len, patch_len)

    batch = (torch.stack([torch.arange(512 - patch_len + 1)] * batch_size) + torch.arange(batch_size)[:, None]).float()
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


@pytest.mark.parametrize("task_kwargs", [{}, {"eval_metric": "WQL", "quantile_levels": [0.1, 0.2]}])
def test_pipeline_can_evaluate_on_dummy_fev_task(task_kwargs):
    pipeline = BaseChronosPipeline.from_pretrained(
        Path(__file__).parent / "dummy-chronos-bolt-model", device_map="cpu"
    )
    task = fev.Task(
        dataset_path="autogluon/chronos_datasets",
        dataset_config="monash_m1_yearly",
        horizon=8,
        **task_kwargs,
    )
    predictions_per_window, inference_time_s = pipeline.predict_fev(task)

    assert isinstance(inference_time_s, float)
    assert isinstance(predictions_per_window, list) and all(
        isinstance(pred, datasets.DatasetDict) for pred in predictions_per_window
    )

    eval_summary = task.evaluation_summary(predictions_per_window, model_name="chronos-bolt")
    assert isinstance(eval_summary["test_error"], float)
