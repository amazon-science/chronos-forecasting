# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path

import datasets
import fev
import numpy as np
import pandas as pd
import pytest
import torch

from chronos import BaseChronosPipeline, Chronos2Pipeline
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.layers import MHA
from chronos.df_utils import convert_df_input_to_list_of_dicts_input
from test.util import create_df, create_future_df, get_forecast_start_times, validate_tensor

DUMMY_MODEL_PATH = Path(__file__).parent / "dummy-chronos2-model"

with open(DUMMY_MODEL_PATH / "config.json") as fp:
    config = json.load(fp)
    DEFAULT_MODEL_NUM_QUANTILES = len(config["chronos_config"]["quantiles"])


@pytest.fixture
def pipeline() -> Chronos2Pipeline:
    return BaseChronosPipeline.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu")


def test_base_chronos2_pipeline_loads_from_s3():
    BaseChronosPipeline.from_pretrained("s3://autogluon/chronos-2", device_map="cpu")


def test_base_chronos2_pipeline_loads_from_hf():
    BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")


@pytest.mark.parametrize(
    "inputs, prediction_length, expected_output_shapes",
    [
        # Homogenous univariate task
        (torch.rand(4, 1, 16), 7, [(1, DEFAULT_MODEL_NUM_QUANTILES, 7)] * 4),
        # Homogenous multivariate task
        (torch.rand(4, 3, 37), 27, [(3, DEFAULT_MODEL_NUM_QUANTILES, 27)] * 4),
        # Heterogenous tasks with different history lengths
        (
            [torch.rand(100), torch.rand(2, 150), torch.rand(120)],
            68,
            [
                (1, DEFAULT_MODEL_NUM_QUANTILES, 68),
                (2, DEFAULT_MODEL_NUM_QUANTILES, 68),
                (1, DEFAULT_MODEL_NUM_QUANTILES, 68),
            ],
        ),
        # Homogenous univariate list of dicts with target only
        (
            [{"target": torch.rand(10)}, {"target": torch.rand(110)}, {"target": torch.rand(17)}],
            5,
            [(1, DEFAULT_MODEL_NUM_QUANTILES, 5)] * 3,
        ),
        # Homogenous multivariate list of dicts with target only
        (
            [{"target": torch.rand(2, 10)}, {"target": torch.rand(2, 110)}, {"target": torch.rand(2, 17)}],
            16,
            [(2, DEFAULT_MODEL_NUM_QUANTILES, 16)] * 3,
        ),
        # Homogenous list of dicts with target and past-only covariates
        (
            [
                {"target": torch.rand(10), "past_covariates": {"feat_1": torch.rand(10)}},
                {"target": torch.rand(110), "past_covariates": {"feat_1": torch.rand(110)}},
                {"target": torch.rand(17), "past_covariates": {"feat_1": torch.rand(17)}},
            ],
            10,
            [(1, DEFAULT_MODEL_NUM_QUANTILES, 10)] * 3,
        ),
        # Homogenous list of dicts with target, past-only and known future covariates
        (
            [
                {
                    "target": torch.rand(10),
                    "past_covariates": {"feat_1": torch.rand(10), "feat_2": torch.rand(10)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
                {
                    "target": torch.rand(99),
                    "past_covariates": {"feat_1": torch.rand(99), "feat_2": torch.rand(99)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
                {
                    "target": torch.rand(17),
                    "past_covariates": {"feat_1": torch.rand(17), "feat_2": torch.rand(17)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
            ],
            15,
            [(1, DEFAULT_MODEL_NUM_QUANTILES, 15)] * 3,
        ),
        # Heterogenous list of dicts with different mix of tasks
        (
            [
                {
                    "target": torch.rand(100),
                    "past_covariates": {"temperature": torch.rand(100), "precipitation": torch.rand(100)},
                    "future_covariates": {"temperature": torch.rand(200)},
                },
                {"target": torch.rand(2, 150), "past_covariates": {"wind_speed": torch.rand(150)}},
                {
                    "target": np.random.rand(150),
                    "past_covariates": {
                        "numeric_covariate_1": np.random.rand(150),
                        "numeric_covariate_2": np.random.rand(150),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=150),
                    },
                    "future_covariates": {
                        "numeric_covariate_1": np.random.rand(200),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=200),
                    },
                },
                {
                    "target": np.random.rand(3, 150),
                    "past_covariates": {
                        "numeric_covariate_1": np.random.rand(150),
                        "numeric_covariate_2": np.random.rand(150),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=150),
                    },
                    "future_covariates": {
                        "numeric_covariate_1": np.random.rand(200),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=200),
                    },
                },
                {"target": torch.rand(1, 150)},
            ],
            200,
            [
                (1, DEFAULT_MODEL_NUM_QUANTILES, 200),
                (2, DEFAULT_MODEL_NUM_QUANTILES, 200),
                (1, DEFAULT_MODEL_NUM_QUANTILES, 200),
                (3, DEFAULT_MODEL_NUM_QUANTILES, 200),
                (1, DEFAULT_MODEL_NUM_QUANTILES, 200),
            ],
        ),
    ],
)
def test_when_input_is_valid_then_pipeline_can_predict(pipeline, inputs, prediction_length, expected_output_shapes):
    outputs = pipeline.predict(inputs, prediction_length=prediction_length)

    assert isinstance(outputs, list) and len(outputs) == len(expected_output_shapes)
    for out, expected_shape in zip(outputs, expected_output_shapes):
        validate_tensor(out, expected_shape, dtype=torch.float32)


@pytest.mark.parametrize(
    "inputs, prediction_length, quantile_levels, expected_output_shapes",
    [
        # Homogenous univariate task
        (torch.rand(4, 1, 16), 7, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [(1, 7, 9)] * 4),
        # Homogenous multivariate task
        (torch.rand(4, 3, 37), 27, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [(3, 27, 8)] * 4),
        # Heterogenous tasks with different history lengths
        (
            [torch.rand(100), torch.rand(2, 150), torch.rand(120)],
            68,
            [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [(1, 68, 10), (2, 68, 10), (1, 68, 10)],
        ),
        # Homogenous univariate list of dicts with target only
        (
            [{"target": torch.rand(10)}, {"target": torch.rand(110)}, {"target": torch.rand(17)}],
            5,
            [0.1, 0.5, 0.9],
            [(1, 5, 3)] * 3,
        ),
        # Homogenous multivariate list of dicts with target only
        (
            [{"target": torch.rand(2, 10)}, {"target": torch.rand(2, 110)}, {"target": torch.rand(2, 17)}],
            16,
            [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
            [(2, 16, 11)] * 3,
        ),
        # Homogenous list of dicts with target and past-only covariates
        (
            [
                {"target": torch.rand(10), "past_covariates": {"feat_1": torch.rand(10)}},
                {"target": torch.rand(110), "past_covariates": {"feat_1": torch.rand(110)}},
                {"target": torch.rand(17), "past_covariates": {"feat_1": torch.rand(17)}},
            ],
            10,
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [(1, 10, 9)] * 3,
        ),
        # Homogenous list of dicts with target, past-only and known future covariates
        (
            [
                {
                    "target": torch.rand(10),
                    "past_covariates": {"feat_1": torch.rand(10), "feat_2": torch.rand(10)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
                {
                    "target": torch.rand(99),
                    "past_covariates": {"feat_1": torch.rand(99), "feat_2": torch.rand(99)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
                {
                    "target": torch.rand(17),
                    "past_covariates": {"feat_1": torch.rand(17), "feat_2": torch.rand(17)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
            ],
            15,
            [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            [(1, 15, 13)] * 3,
        ),
        # Heterogenous list of dicts with different mix of tasks
        (
            [
                {
                    "target": torch.rand(100),
                    "past_covariates": {"temperature": torch.rand(100), "precipitation": torch.rand(100)},
                    "future_covariates": {"temperature": torch.rand(200)},
                },
                {"target": torch.rand(2, 150), "past_covariates": {"wind_speed": torch.rand(150)}},
                {
                    "target": np.random.rand(150),
                    "past_covariates": {
                        "numeric_covariate_1": np.random.rand(150),
                        "numeric_covariate_2": np.random.rand(150),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=150),
                    },
                    "future_covariates": {
                        "numeric_covariate_1": np.random.rand(200),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=200),
                    },
                },
                {
                    "target": np.random.rand(3, 150),
                    "past_covariates": {
                        "numeric_covariate_1": np.random.rand(150),
                        "numeric_covariate_2": np.random.rand(150),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=150),
                    },
                    "future_covariates": {
                        "numeric_covariate_1": np.random.rand(200),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=200),
                    },
                },
                {"target": torch.rand(1, 150)},
            ],
            200,
            [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
            [(1, 200, 11), (2, 200, 11), (1, 200, 11), (3, 200, 11), (1, 200, 11)],
        ),
    ],
)
def test_when_input_is_valid_then_pipeline_can_predict_quantiles(
    pipeline, inputs, prediction_length, quantile_levels, expected_output_shapes
):
    quantiles, mean = pipeline.predict_quantiles(
        inputs, prediction_length=prediction_length, quantile_levels=quantile_levels
    )

    assert isinstance(quantiles, list) and len(quantiles) == len(expected_output_shapes)
    assert isinstance(mean, list) and len(mean) == len(expected_output_shapes)
    for out_q, out_m, expected_shape in zip(quantiles, mean, expected_output_shapes):
        validate_tensor(out_q, expected_shape, dtype=torch.float32)
        validate_tensor(out_m, expected_shape[:-1], dtype=torch.float32)


@pytest.mark.parametrize(
    "inputs, error_match_string",
    [
        (torch.rand(16), "should be 3-d with shape"),
        (torch.rand(4, 3), "should be 3-d with shape"),
        ([torch.rand(1, 2, 100), torch.rand(120)], "the elements should either be 1-d"),
        ([{"target": torch.rand(10)}, {"target": torch.rand(1, 2, 17), "extra_key": []}], "Found invalid keys"),
        ([{"target": torch.rand(10)}, {"target": torch.rand(1, 2, 17)}], "`target` should either be 1-d with shape"),
        ([{"target": torch.rand(10), "past_covariates": torch.rand(10)}], "Found invalid type for `past_covariates`"),
        (
            [
                {"target": torch.rand(10), "past_covariates": {"feat_1": torch.rand(10)}},
                {"target": torch.rand(17), "past_covariates": {"feat_1": torch.rand(10)}},
            ],
            "`past_covariates` must be 1-d with length",
        ),
        (
            [
                {
                    "target": torch.rand(10),
                    "past_covariates": {"feat": torch.rand(10)},
                    "future_covariates": torch.rand(10),
                }
            ],
            "Found invalid type for `future_covariates`",
        ),
        (
            [
                {
                    "target": torch.rand(10),
                    "past_covariates": {"feat": torch.rand(10)},
                    "future_covariates": {"feat": torch.rand(10), "extra": torch.rand(10)},
                }
            ],
            "Expected keys in `future_covariates`",
        ),
        (
            [
                {
                    "target": torch.rand(10),
                    "past_covariates": {"feat": torch.rand(10)},
                    "future_covariates": {"feat": torch.rand(17)},
                }
            ],
            "`future_covariates` must be 1-d with length",
        ),
    ],
)
def test_when_input_is_invalid_then_predict_raises_value_error(pipeline, inputs, error_match_string):
    with pytest.raises(ValueError, match=error_match_string):
        _ = pipeline.predict(inputs, prediction_length=10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_pipeline_predict_can_handle_different_model_and_input_dtypes(dtype: torch.dtype, input_dtype: torch.dtype):
    pipeline = BaseChronosPipeline.from_pretrained(
        Path(__file__).parent / "dummy-chronos2-model", device_map="cpu", torch_dtype=dtype
    )
    context = 10 * torch.rand(size=(4, 3, 16)) + 10
    context = context.to(dtype=input_dtype)
    expected_num_quantiles = len(pipeline.quantiles)

    # input: tensor of shape (batch_size, n_variates, context_length)

    quantiles = pipeline.predict(context, prediction_length=7)
    for quantiles_item in quantiles:
        validate_tensor(quantiles_item, (3, expected_num_quantiles, 7), dtype=torch.float32)


@pytest.mark.parametrize(
    "inputs, expected_output_shapes",
    [
        # NOTE: d_model for the dummy model is 6
        # Homogenous univariate task
        (torch.rand(4, 1, 16), [(1, 3, 6)] * 4),
        # Homogenous multivariate task
        (torch.rand(4, 3, 37), [(3, 5, 6)] * 4),
        # Heterogenous tasks with different history lengths
        (
            [torch.rand(100), torch.rand(2, 150), torch.rand(120)],
            [(1, 12, 6), (2, 12, 6), (1, 12, 6)],
        ),
    ],
)
def test_when_input_is_valid_then_pipeline_can_embed(pipeline, inputs, expected_output_shapes):
    embeds, loc_scales = pipeline.embed(inputs)

    assert (
        isinstance(embeds, list)
        and len(embeds) == len(expected_output_shapes)
        and len(loc_scales) == len(expected_output_shapes)
    )
    for embed, loc_scale, expected_shape in zip(embeds, loc_scales, expected_output_shapes):
        validate_tensor(embed, expected_shape, dtype=torch.float32)
        validate_tensor(loc_scale[0], (expected_shape[0], 1), dtype=torch.float32)
        validate_tensor(loc_scale[1], (expected_shape[0], 1), dtype=torch.float32)


@pytest.mark.parametrize(
    "task_kwargs",
    [
        {"dataset_path": "autogluon/chronos_datasets", "dataset_config": "monash_m1_yearly", "horizon": 8},
        {
            "dataset_path": "autogluon/chronos_datasets",
            "dataset_config": "monash_m1_yearly",
            "horizon": 8,
            "eval_metric": "WQL",
            "quantile_levels": [0.1, 0.2],
        },
        {
            "dataset_path": "autogluon/fev_datasets",
            "dataset_config": "ETT_1H",
            "horizon": 27,
            "target": ["HULL", "HUFL", "OT"],
        },
        {
            "dataset_path": "autogluon/fev_datasets",
            "dataset_config": "ETT_1H",
            "horizon": 34,
            "target": "OT",
            "past_dynamic_columns": ["HULL", "HUFL"],
        },
        {
            "dataset_path": "autogluon/fev_datasets",
            "dataset_config": "ETT_1H",
            "horizon": 34,
            "target": "OT",
            "past_dynamic_columns": ["HULL", "HUFL"],
            "known_dynamic_columns": ["LULL"],
        },
    ],
)
def test_pipeline_can_evaluate_on_dummy_fev_task(pipeline, task_kwargs):
    task = fev.Task(**task_kwargs)
    predictions_per_window, inference_time_s = pipeline.predict_fev(task)

    assert isinstance(inference_time_s, float)
    assert isinstance(predictions_per_window, list) and all(
        isinstance(pred, datasets.DatasetDict) for pred in predictions_per_window
    )

    eval_summary = task.evaluation_summary(predictions_per_window, model_name="chronos-2")
    assert isinstance(eval_summary["test_error"], float)


@pytest.mark.parametrize(
    "context_setup, future_setup, expected_rows",
    [
        # Targets only
        ({}, None, 6),  # 2 series * 3 predictions
        # Multiple targets with different context lengths
        (
            {"target_cols": ["sales", "revenue", "profit"], "n_points": [10, 17]},
            None,
            18,
        ),  # 2 series * 3 targets * 3 predictions
        # With past covariates
        ({"covariates": ["cov1"]}, None, 6),
        # With future covariates
        ({"covariates": ["cov1"]}, {"covariates": ["cov1"], "n_points": [3, 3]}, 6),
        # With past-only and future covariates
        ({"covariates": ["cov1", "cov2"]}, {"covariates": ["cov1"], "n_points": [3, 3]}, 6),
        # With past-only and future covariates and different series order
        (
            {"series_ids": ["B", "C", "A", "Z"], "n_points": [10, 20, 100, 256], "covariates": ["cov1", "cov2"]},
            {
                "series_ids": ["B", "C", "A", "Z"],
                "covariates": ["cov1"],
                "n_points": [3, 3, 3, 3],
            },
            12,
        ),
    ],
)
@pytest.mark.parametrize("freq", ["s", "min", "30min", "h", "D", "W", "ME", "QE", "YE"])
def test_predict_df_works_for_valid_inputs(pipeline, context_setup, future_setup, expected_rows, freq):
    prediction_length = 3
    df = create_df(**context_setup, freq=freq)
    forecast_start_times = get_forecast_start_times(df, freq)
    future_df = create_future_df(forecast_start_times, **future_setup, freq=freq) if future_setup else None

    series_ids = context_setup.get("series_ids", ["A", "B"])
    target_columns = context_setup.get("target_cols", ["target"])
    n_series = len(series_ids)
    n_targets = len(target_columns)
    result = pipeline.predict_df(df, future_df=future_df, target=target_columns, prediction_length=prediction_length)

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


@pytest.mark.parametrize(
    "context_data, error_match",
    [
        # Missing timestamp column
        ({"item_id": ["A"], "target": [1.0]}, "df does not contain all"),
        # Insufficient data points
        ({"item_id": ["A"], "timestamp": ["2023-01-01"], "target": [1.0]}, "must have at least 3 data"),
    ],
)
def test_predict_df_df_validation_errors(pipeline, context_data, error_match):
    df = pd.DataFrame(context_data)

    with pytest.raises(ValueError, match=error_match):
        pipeline.predict_df(df)


@pytest.mark.parametrize(
    "future_data, error_match",
    [
        # Missing timestamp column
        ({"item_id": ["A"], "cov1": [1.0]}, "future_df does not contain all"),
        # target in future_df
        (
            {"item_id": ["A"], "timestamp": ["2023-01-01"], "cov1": [1.0], "target": [1.0]},
            "future_df cannot contain target",
        ),
        # Extra columns in future_df
        (
            {"item_id": ["A"], "timestamp": ["2023-01-01"], "cov1": [1.0], "cov2": [1.0]},
            "future_df cannot contain columns not present",
        ),
    ],
)
def test_predict_df_future_df_validation_errors(pipeline, future_data, error_match):
    df = create_df(series_ids=["A", "B"], covariates=["cov1"], freq="h")
    future_df = pd.DataFrame(future_data)
    with pytest.raises(ValueError, match=error_match):
        pipeline.predict_df(df, future_df=future_df)


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


def test_predict_df_with_future_df_missing_series_raises_error(pipeline):
    df = create_df(series_ids=["A", "B"], covariates=["cov1"])
    future_df = create_future_df(
        get_forecast_start_times(df), series_ids=["A"], covariates=["cov1"]
    )  # Missing Bs=["cov1"])  # Missing B

    with pytest.raises(ValueError, match="same time series IDs"):
        pipeline.predict_df(df, future_df=future_df)


def test_predict_df_with_future_df_with_different_lengths_raises_error(pipeline):
    df = create_df(series_ids=["A", "B"], covariates=["cov1"])
    future_df = create_future_df(
        get_forecast_start_times(df), series_ids=["A", "B"], n_points=[3, 7], covariates=["cov1"]
    )

    with pytest.raises(ValueError, match="all time series must have length"):
        pipeline.predict_df(df, future_df=future_df, prediction_length=3)


def test_predict_df_with_future_df_with_different_freq_raises_error(pipeline):
    df = create_df(series_ids=["A", "B"], covariates=["cov1"], freq="h")
    future_df = create_future_df(
        get_forecast_start_times(df), series_ids=["A", "B"], n_points=[3, 3], covariates=["cov1"], freq="D"
    )

    with pytest.raises(ValueError, match="must have the same frequency as context"):
        pipeline.predict_df(df, future_df=future_df, prediction_length=3)


@pytest.mark.parametrize(
    "inputs, prediction_length, expected_output_shapes",
    [
        # Homogenous univariate task
        (torch.rand(4, 1, 16), 7, [(1, DEFAULT_MODEL_NUM_QUANTILES, 7)] * 4),
        # Homogenous multivariate task
        (torch.rand(4, 3, 37), 27, [(3, DEFAULT_MODEL_NUM_QUANTILES, 27)] * 4),
        # Heterogenous tasks with different history lengths
        (
            [torch.rand(100), torch.rand(2, 150), torch.rand(120)],
            68,
            [
                (1, DEFAULT_MODEL_NUM_QUANTILES, 68),
                (2, DEFAULT_MODEL_NUM_QUANTILES, 68),
                (1, DEFAULT_MODEL_NUM_QUANTILES, 68),
            ],
        ),
        # Homogenous list of dicts with target, past-only and known future covariates
        (
            [
                {
                    "target": torch.rand(10),
                    "past_covariates": {"feat_1": torch.rand(10), "feat_2": torch.rand(10)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
                {
                    "target": torch.rand(99),
                    "past_covariates": {"feat_1": torch.rand(99), "feat_2": torch.rand(99)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
                {
                    "target": torch.rand(17),
                    "past_covariates": {"feat_1": torch.rand(17), "feat_2": torch.rand(17)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
            ],
            15,
            [(1, DEFAULT_MODEL_NUM_QUANTILES, 15)] * 3,
        ),
        # Heterogenous list of dicts with different mix of tasks
        (
            [
                {
                    "target": torch.rand(1000),
                    "past_covariates": {"temperature": torch.rand(1000), "precipitation": torch.rand(1000)},
                    "future_covariates": {"temperature": torch.rand(200)},
                },
                {"target": torch.rand(2, 150), "past_covariates": {"wind_speed": torch.rand(150)}},
                {
                    "target": np.random.rand(150),
                    "past_covariates": {
                        "numeric_covariate_1": np.random.rand(150),
                        "numeric_covariate_2": np.random.rand(150),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=150),
                    },
                    "future_covariates": {
                        "numeric_covariate_1": np.random.rand(200),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=200),
                    },
                },
                {
                    "target": np.random.rand(3, 150),
                    "past_covariates": {
                        "numeric_covariate_1": np.random.rand(150),
                        "numeric_covariate_2": np.random.rand(150),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=150),
                    },
                    "future_covariates": {
                        "numeric_covariate_1": np.random.rand(200),
                        "cat_covariate": np.random.choice(["A", "B", "C", "D", "E"], size=200),
                    },
                },
                {"target": torch.rand(1, 150)},
            ],
            200,
            [
                (1, DEFAULT_MODEL_NUM_QUANTILES, 200),
                (2, DEFAULT_MODEL_NUM_QUANTILES, 200),
                (1, DEFAULT_MODEL_NUM_QUANTILES, 200),
                (3, DEFAULT_MODEL_NUM_QUANTILES, 200),
                (1, DEFAULT_MODEL_NUM_QUANTILES, 200),
            ],
        ),
    ],
)
def test_when_input_is_valid_then_pipeline_can_be_finetuned(
    pipeline, inputs, prediction_length, expected_output_shapes
):
    # Get outputs before fine-tuning
    orig_outputs_before = pipeline.predict(inputs, prediction_length=prediction_length)
    ft_pipeline = pipeline.fit(inputs, prediction_length=prediction_length, num_steps=5, min_past=1, batch_size=32)
    # Get outputs from fine-tuned pipeline
    ft_outputs = ft_pipeline.predict(inputs, prediction_length=prediction_length)
    # Get outputs from original pipeline after fine-tuning
    orig_outputs_after = pipeline.predict(inputs, prediction_length=prediction_length)

    # Check output shapes are correct and output is different from the pretrained model outputs
    assert isinstance(ft_outputs, list) and len(ft_outputs) == len(expected_output_shapes)
    for orig_out_before, finetuned_out, orig_out_after, expected_shape in zip(
        orig_outputs_before, ft_outputs, orig_outputs_after, expected_output_shapes
    ):
        validate_tensor(finetuned_out, expected_shape, dtype=torch.float32)
        assert torch.allclose(orig_out_before, orig_out_after)
        assert not torch.allclose(orig_out_before, finetuned_out)
        assert not torch.isnan(finetuned_out).any()


@pytest.mark.parametrize(
    "inputs, prediction_length, expected_output_shapes",
    [
        # Homogenous list of dicts with target, past-only and known future covariates
        (
            [
                {
                    "target": torch.rand(10),
                    "past_covariates": {"feat_1": torch.rand(10), "feat_2": torch.rand(10)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
                {
                    "target": torch.rand(99),
                    "past_covariates": {"feat_1": torch.rand(99), "feat_2": torch.rand(99)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
                {
                    "target": torch.rand(17),
                    "past_covariates": {"feat_1": torch.rand(17), "feat_2": torch.rand(17)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
            ],
            15,
            [(1, DEFAULT_MODEL_NUM_QUANTILES, 15)] * 3,
        )
    ],
)
def test_pipeline_can_be_finetuned_with_validation(pipeline, inputs, prediction_length, expected_output_shapes):
    # Get outputs before fine-tuning
    orig_outputs_before = pipeline.predict(inputs, prediction_length=prediction_length)
    ft_pipeline = pipeline.fit(
        inputs,
        prediction_length=prediction_length,
        validation_inputs=inputs,
        num_steps=20,
        min_past=1,
        eval_steps=10,
        logging_steps=10,
        batch_size=32,
    )
    # Get outputs from fine-tuned pipeline
    ft_outputs = ft_pipeline.predict(inputs, prediction_length=prediction_length)
    # Get outputs from original pipeline after fine-tuning
    orig_outputs_after = pipeline.predict(inputs, prediction_length=prediction_length)

    # Check output shapes are correct and output is different from the pretrained model outputs
    assert isinstance(ft_outputs, list) and len(ft_outputs) == len(expected_output_shapes)
    for orig_out_before, finetuned_out, orig_out_after, expected_shape in zip(
        orig_outputs_before, ft_outputs, orig_outputs_after, expected_output_shapes
    ):
        validate_tensor(finetuned_out, expected_shape, dtype=torch.float32)
        assert torch.allclose(orig_out_before, orig_out_after)
        assert not torch.allclose(orig_out_before, finetuned_out)
        assert not torch.isnan(finetuned_out).any()


@pytest.mark.parametrize(
    "inputs, prediction_length, expected_output_shapes",
    [
        # Homogenous list of dicts with target, past-only and known future covariates
        (
            [
                {
                    "target": torch.rand(10),
                    "past_covariates": {"feat_1": torch.rand(10), "feat_2": torch.rand(10)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
                {
                    "target": torch.rand(99),
                    "past_covariates": {"feat_1": torch.rand(99), "feat_2": torch.rand(99)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
                {
                    "target": torch.rand(17),
                    "past_covariates": {"feat_1": torch.rand(17), "feat_2": torch.rand(17)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
            ],
            15,
            [(1, DEFAULT_MODEL_NUM_QUANTILES, 15)] * 3,
        )
    ],
)
@pytest.mark.parametrize("ft_future_values", [None, np.array([]), torch.zeros(0)])
def test_pipeline_can_be_finetuned_with_empty_future_covariates(
    pipeline, inputs, prediction_length, expected_output_shapes, ft_future_values
):
    # Get outputs before fine-tuning
    orig_outputs_before = pipeline.predict(inputs, prediction_length=prediction_length)
    # Replace future covariates with ft_future_values
    ft_inputs = deepcopy(inputs)
    for idx, task in enumerate(inputs):
        for key in task["future_covariates"]:
            ft_inputs[idx]["future_covariates"][key] = ft_future_values

    ft_pipeline = pipeline.fit(
        ft_inputs,
        prediction_length=prediction_length,
        validation_inputs=ft_inputs,
        num_steps=20,
        min_past=1,
        eval_steps=10,
        logging_steps=10,
        batch_size=32,
    )
    # Get outputs from fine-tuned pipeline
    ft_outputs = ft_pipeline.predict(inputs, prediction_length=prediction_length)
    # Get outputs from original pipeline after fine-tuning
    orig_outputs_after = pipeline.predict(inputs, prediction_length=prediction_length)

    # Check output shapes are correct and output is different from the pretrained model outputs
    assert isinstance(ft_outputs, list) and len(ft_outputs) == len(expected_output_shapes)
    for orig_out_before, finetuned_out, orig_out_after, expected_shape in zip(
        orig_outputs_before, ft_outputs, orig_outputs_after, expected_output_shapes
    ):
        validate_tensor(finetuned_out, expected_shape, dtype=torch.float32)
        assert torch.allclose(orig_out_before, orig_out_after)
        assert not torch.allclose(orig_out_before, finetuned_out)
        assert not torch.isnan(finetuned_out).any()


@pytest.mark.parametrize(
    "inputs, prediction_length",
    [
        # Homogenous univariate task
        (torch.rand(4, 1, 16), 10),
        # Homogenous multivariate task
        (torch.rand(4, 3, 37), 27),
        # Homogenous list of dicts with target, past-only and known future covariates
        (
            [
                {
                    "target": torch.rand(10),
                    "past_covariates": {"feat_1": torch.rand(10), "feat_2": torch.rand(10)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
                {
                    "target": torch.rand(29),
                    "past_covariates": {"feat_1": torch.rand(29), "feat_2": torch.rand(29)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
                {
                    "target": torch.rand(17),
                    "past_covariates": {"feat_1": torch.rand(17), "feat_2": torch.rand(17)},
                    "future_covariates": {"feat_1": torch.rand(15)},
                },
            ],
            15,
        ),
    ],
)
def test_when_input_time_series_are_too_short_then_finetuning_raises_error(pipeline, inputs, prediction_length):
    with pytest.raises(ValueError, match="The dataset is empty after filtering"):
        pipeline.fit(
            inputs, prediction_length=prediction_length, num_steps=5, min_past=prediction_length, batch_size=32
        )


@pytest.mark.parametrize(
    "context_setup, future_setup, expected_rows",
    [
        # Targets only
        ({}, None, 6),  # 2 series * 3 predictions
        # Multiple targets with different context lengths
        (
            {"target_cols": ["sales", "revenue", "profit"], "n_points": [10, 17]},
            None,
            18,
        ),  # 2 series * 3 targets * 3 predictions
        # With past covariates
        ({"covariates": ["cov1"]}, None, 6),
        # With future covariates
        ({"covariates": ["cov1"]}, {"covariates": ["cov1"], "n_points": [3, 3]}, 6),
        # With past-only and future covariates
        ({"covariates": ["cov1", "cov2"]}, {"covariates": ["cov1"], "n_points": [3, 3]}, 6),
        # With past-only and future covariates and different series order
        (
            {"series_ids": ["B", "C", "A", "Z"], "n_points": [10, 20, 100, 256], "covariates": ["cov1", "cov2"]},
            {
                "series_ids": ["B", "C", "A", "Z"],
                "covariates": ["cov1"],
                "n_points": [3, 3, 3, 3],
            },
            12,
        ),
    ],
)
@pytest.mark.parametrize("freq", ["h", "D", "ME"])
def test_two_step_finetuning_with_df_input_works(pipeline, context_setup, future_setup, expected_rows, freq):
    prediction_length = 3
    df = create_df(**context_setup, freq=freq)
    forecast_start_times = get_forecast_start_times(df, freq)
    future_df = create_future_df(forecast_start_times, **future_setup, freq=freq) if future_setup else None

    series_ids = context_setup.get("series_ids", ["A", "B"])
    target_columns = context_setup.get("target_cols", ["target"])
    n_series = len(series_ids)
    n_targets = len(target_columns)

    # Get predictions from the pretrained model
    orig_result_before = pipeline.predict_df(
        df, future_df=future_df, target=target_columns, prediction_length=prediction_length
    )

    # Convert df inputs to list of dicts inputs expected by finetune
    inputs, _, _ = convert_df_input_to_list_of_dicts_input(
        df,
        future_df=future_df,
        id_column="item_id",
        timestamp_column="timestamp",
        target_columns=target_columns,
        prediction_length=prediction_length,
    )
    # Finetune the model
    ft_pipeline = pipeline.fit(inputs, prediction_length=prediction_length, num_steps=5, min_past=1, batch_size=32)
    # Predict with fine-tuned model
    result = ft_pipeline.predict_df(
        df, future_df=future_df, target=target_columns, prediction_length=prediction_length
    )
    # Get predictions from the original pipeline again
    orig_result_after = pipeline.predict_df(
        df, future_df=future_df, target=target_columns, prediction_length=prediction_length
    )

    # Check predictions from the fine-tuned model are valid
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

    # Check predictions from the original pipeline are the same before and after fine-tuning
    assert np.allclose(orig_result_before["predictions"].to_numpy(), orig_result_after["predictions"].to_numpy())

    # Check predictions from the fine-tuned model are different from the original predictions
    assert not np.allclose(orig_result_before["predictions"].to_numpy(), result["predictions"].to_numpy())


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_pipeline_works_with_different_attention_implementations(attn_implementation):
    """Test that the pipeline works with different attention implementations."""
    # Load the dummy model
    model_path = Path(__file__).parent / "dummy-chronos2-model"

    # Load with specified attention implementation
    pipeline = BaseChronosPipeline.from_pretrained(
        model_path, device_map="cpu", attn_implementation=attn_implementation
    )

    # Verify the config has the correct attention implementation
    assert pipeline.model.config._attn_implementation == attn_implementation

    # Test prediction with simple input
    inputs = torch.rand(2, 1, 16)
    prediction_length = 7

    outputs = pipeline.predict(inputs, prediction_length=prediction_length)

    # Check outputs are valid
    assert isinstance(outputs, list) and len(outputs) == 2
    for out in outputs:
        validate_tensor(out, (1, DEFAULT_MODEL_NUM_QUANTILES, 7), dtype=torch.float32)


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("output_attentions", [False, True])
def test_attention_implementations_with_output_attentions(attn_implementation, output_attentions):
    """Test that attention implementations handle output_attentions correctly."""
    # Create config with specified attention implementation
    config = Chronos2CoreConfig(
        d_model=128,
        d_kv=32,
        num_heads=4,
        dropout_rate=0.1,
        attn_implementation=attn_implementation,
    )

    # Create MHA layer
    mha = MHA(config, use_rope=True)
    mha.eval()

    # Create dummy inputs
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.d_model)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    mask = torch.zeros(batch_size, config.num_heads, seq_len, seq_len)

    # Test forward pass
    output = mha(
        hidden_states=hidden_states,
        mask=mask,
        position_ids=position_ids,
        output_attentions=output_attentions,
    )

    # Check output shape
    assert output.hidden_states.shape == (batch_size, seq_len, config.d_model)

    # Check attention weights - should only be returned when output_attentions=True
    if output_attentions:
        assert output.attn_weights is not None
        assert output.attn_weights.shape == (batch_size, config.num_heads, seq_len, seq_len)
    else:
        # SDPA doesn't return weights
        if attn_implementation == "sdpa":
            assert output.attn_weights is None


def test_eager_and_sdpa_produce_identical_outputs(pipeline):
    """Test that eager and SDPA implementations produce identical outputs on full pipeline."""
    # Reload pipeline with SDPA
    model_path = Path(__file__).parent / "dummy-chronos2-model"
    pipeline_sdpa = BaseChronosPipeline.from_pretrained(
        model_path, device_map="cpu", attn_implementation="sdpa", torch_dtype=torch.float32
    )

    # Note: the original pipeline fixture uses default attn_implementation which should be sdpa
    # Force eager for comparison
    pipeline_eager = BaseChronosPipeline.from_pretrained(
        model_path, device_map="cpu", attn_implementation="eager", torch_dtype=torch.float32
    )

    # Test 1: Simple univariate input
    inputs_simple = torch.rand(2, 1, 16)
    prediction_length = 7

    with torch.no_grad():
        outputs_eager = pipeline_eager.predict(inputs_simple, prediction_length=prediction_length)
        outputs_sdpa = pipeline_sdpa.predict(inputs_simple, prediction_length=prediction_length)

    # Verify outputs match exactly
    assert len(outputs_eager) == len(outputs_sdpa)
    for out_eager, out_sdpa in zip(outputs_eager, outputs_sdpa):
        # Should match exactly or very close (numerical precision)
        assert torch.allclose(out_eager, out_sdpa, atol=1e-5, rtol=1e-4)

    # Test 2: Multivariate inputs with covariates to test group attention
    inputs_grouped = [
        {
            "target": np.random.randn(2, 36),
            "past_covariates": {
                "temperature": np.random.randn(36),
                "weather_type": np.random.choice(["sunny", "cloudy", "rainy"], size=36),
            },
            "future_covariates": {
                "temperature": np.random.randn(prediction_length),
                "weather_type": np.random.choice(["sunny", "cloudy", "rainy"], size=prediction_length),
            },
        }
        for _ in range(5)
    ]

    with torch.no_grad():
        outputs_eager_grouped = pipeline_eager.predict(inputs_grouped, prediction_length=prediction_length)
        outputs_sdpa_grouped = pipeline_sdpa.predict(inputs_grouped, prediction_length=prediction_length)

    # Verify outputs match for grouped inputs
    assert len(outputs_eager_grouped) == len(outputs_sdpa_grouped)
    for out_eager, out_sdpa in zip(outputs_eager_grouped, outputs_sdpa_grouped):
        # Should match exactly or very close (numerical precision)
        assert torch.allclose(out_eager, out_sdpa, atol=1e-5, rtol=1e-4)
