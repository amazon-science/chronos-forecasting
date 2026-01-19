# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from chronos.df_utils import (
    convert_df_input_to_list_of_dicts_input,
    validate_df_inputs,
)
from test.util import create_df, create_future_df, get_forecast_start_times

# Tests for validate_df_inputs function


@pytest.mark.parametrize("freq", ["s", "min", "30min", "h", "D", "W", "ME", "QE", "YE"])
def test_validate_df_inputs_returns_correct_metadata_for_valid_inputs(freq):
    """Test that function returns validated dataframes, frequency, series lengths, and original order."""
    # Create test data with 2 series
    df = create_df(series_ids=["A", "B"], n_points=[10, 15], target_cols=["target"], freq=freq)

    # Call validate_df_inputs
    validated_df, validated_future_df, inferred_freq, series_lengths, original_order = validate_df_inputs(
        df=df,
        future_df=None,
        target_columns=["target"],
        prediction_length=5,
        id_column="item_id",
        timestamp_column="timestamp",
    )

    # Verify key return values
    assert validated_future_df is None
    assert inferred_freq is not None
    assert series_lengths == [10, 15]
    assert list(original_order) == ["A", "B"]
    # Verify dataframe is sorted
    assert validated_df["item_id"].iloc[0] == "A"
    assert validated_df["item_id"].iloc[10] == "B"


def test_validate_df_inputs_casts_mixed_dtypes_correctly():
    """Test that numeric columns are cast to float32 and categorical/string/object columns are cast to category."""
    # Create dataframe with mixed column types
    df = pd.DataFrame(
        {
            "item_id": ["A"] * 10,
            "timestamp": pd.date_range(end="2001-10-01", periods=10, freq="h"),
            "target": np.random.randn(10),  # numeric
            "numeric_cov": np.random.randint(0, 10, 10),  # integer numeric
            "string_cov": ["cat1"] * 5 + ["cat2"] * 5,  # string
            "bool_cov": [True, False] * 5,  # boolean
        }
    )

    # Call validate_df_inputs
    validated_df, _, _, _, _ = validate_df_inputs(
        df=df,
        future_df=None,
        target_columns=["target"],
        prediction_length=5,
    )

    # Verify dtypes after validation
    assert validated_df["target"].dtype == np.float32
    assert validated_df["numeric_cov"].dtype == np.float32
    assert validated_df["string_cov"].dtype.name == "category"
    assert validated_df["bool_cov"].dtype == np.float32  # booleans are cast to float32


def test_validate_df_inputs_raises_error_when_series_has_insufficient_data():
    """Test that ValueError is raised for series with < 3 data points."""
    # Create dataframe with one series having only 2 points
    df = create_df(series_ids=["A", "B"], n_points=[10, 2], target_cols=["target"], freq="h")

    # Verify error is raised with series ID in message
    with pytest.raises(ValueError, match=r"Every time series must have at least 3 data points.*series B"):
        validate_df_inputs(
            df=df,
            future_df=None,
            target_columns=["target"],
            prediction_length=5,
        )


def test_validate_df_inputs_raises_error_when_future_df_has_mismatched_series_ids():
    """Test that ValueError is raised when future_df has different series IDs than df."""
    # Create df with series A and B
    df = create_df(series_ids=["A", "B"], n_points=[10, 15], target_cols=["target"], freq="h")

    # Create future_df with only series A
    forecast_start_times = get_forecast_start_times(df, freq="h")
    future_df = create_future_df(
        forecast_start_times=[forecast_start_times[0]], series_ids=["A"], n_points=[5], covariates=None, freq="h"
    )

    # Verify appropriate error is raised
    with pytest.raises(ValueError, match=r"future_df must contain the same time series IDs as df"):
        validate_df_inputs(
            df=df,
            future_df=future_df,
            target_columns=["target"],
            prediction_length=5,
        )


def test_validate_df_inputs_raises_error_when_future_df_has_incorrect_lengths():
    """Test that ValueError is raised when future_df lengths don't match prediction_length."""
    # Create df with series A and B with a covariate
    df = create_df(series_ids=["A", "B"], n_points=[10, 13], target_cols=["target"], covariates=["cov1"], freq="h")

    # Create future_df with varying lengths per series (3 and 7 instead of 5)
    forecast_start_times = get_forecast_start_times(df, freq="h")
    future_df = create_future_df(
        forecast_start_times=forecast_start_times,
        series_ids=["A", "B"],
        n_points=[3, 7],  # incorrect lengths
        covariates=["cov1"],
        freq="h",
    )

    # Verify error message indicates which series have incorrect lengths
    with pytest.raises(
        ValueError, match=r"future_df must contain prediction_length=5 values for each series.*different lengths"
    ):
        validate_df_inputs(
            df=df,
            future_df=future_df,
            target_columns=["target"],
            prediction_length=5,
        )


# Tests for convert_df_input_to_list_of_dicts_input function


def test_convert_df_with_single_target_preserves_values():
    """Test conversion with single target column."""
    df = create_df(series_ids=["A", "B"], n_points=[10, 12], target_cols=["target"], freq="h")

    inputs, original_order, prediction_timestamps = convert_df_input_to_list_of_dicts_input(
        df=df,
        future_df=None,
        target_columns=["target"],
        prediction_length=5,
    )

    # Verify output list has correct length (one per series)
    assert len(inputs) == 2

    # Verify target arrays have correct shape and values match input
    assert inputs[0]["target"].shape == (1, 10)  # (n_targets=1, n_timesteps=10)
    assert inputs[1]["target"].shape == (1, 12)  # (n_targets=1, n_timesteps=12)

    # Verify values are preserved
    df_sorted = df.sort_values(["item_id", "timestamp"])
    np.testing.assert_array_almost_equal(
        inputs[0]["target"][0], df_sorted[df_sorted["item_id"] == "A"]["target"].values
    )
    np.testing.assert_array_almost_equal(
        inputs[1]["target"][0], df_sorted[df_sorted["item_id"] == "B"]["target"].values
    )


def test_convert_df_with_multiple_targets_preserves_values_and_shape():
    """Test conversion with multiple target columns."""
    df = create_df(series_ids=["A", "B"], n_points=[10, 14], target_cols=["target1", "target2"], freq="h")

    inputs, original_order, prediction_timestamps = convert_df_input_to_list_of_dicts_input(
        df=df,
        future_df=None,
        target_columns=["target1", "target2"],
        prediction_length=5,
    )

    # Verify target arrays have shape (n_targets, n_timesteps)
    assert inputs[0]["target"].shape == (2, 10)
    assert inputs[1]["target"].shape == (2, 14)

    # Verify all target values are preserved for both series
    df_sorted = df.sort_values(["item_id", "timestamp"])
    for i, series_id in enumerate(["A", "B"]):
        series_data = df_sorted[df_sorted["item_id"] == series_id]
        np.testing.assert_array_almost_equal(inputs[i]["target"][0], series_data["target1"].values)
        np.testing.assert_array_almost_equal(inputs[i]["target"][1], series_data["target2"].values)


def test_convert_df_with_past_covariates_includes_them_in_output():
    """Test conversion with past covariates only."""
    df = create_df(
        series_ids=["A", "B"], n_points=[10, 16], target_cols=["target"], covariates=["cov1", "cov2"], freq="h"
    )

    inputs, original_order, prediction_timestamps = convert_df_input_to_list_of_dicts_input(
        df=df,
        future_df=None,
        target_columns=["target"],
        prediction_length=5,
    )

    # Verify output includes past_covariates dictionary
    assert "past_covariates" in inputs[0]
    assert "cov1" in inputs[0]["past_covariates"]
    assert "cov2" in inputs[0]["past_covariates"]

    # Verify covariate values match input for both series
    assert inputs[0]["past_covariates"]["cov1"].shape == (10,)
    assert inputs[0]["past_covariates"]["cov2"].shape == (10,)
    assert inputs[1]["past_covariates"]["cov1"].shape == (16,)
    assert inputs[1]["past_covariates"]["cov2"].shape == (16,)

    # Verify no future_covariates key in output
    assert "future_covariates" not in inputs[0]


def test_convert_df_with_past_and_future_covariates_includes_both():
    """Test conversion with both past and future covariates."""
    df = create_df(series_ids=["A", "B"], n_points=[10, 18], target_cols=["target"], covariates=["cov1"], freq="h")

    forecast_start_times = get_forecast_start_times(df, freq="h")
    future_df = create_future_df(
        forecast_start_times=forecast_start_times,
        series_ids=["A", "B"],
        n_points=[5, 5],
        covariates=["cov1"],
        freq="h",
    )

    inputs, original_order, prediction_timestamps = convert_df_input_to_list_of_dicts_input(
        df=df,
        future_df=future_df,
        target_columns=["target"],
        prediction_length=5,
    )

    # Verify output includes both past_covariates and future_covariates dictionaries for both series
    assert "past_covariates" in inputs[0]
    assert "future_covariates" in inputs[0]
    assert "past_covariates" in inputs[1]
    assert "future_covariates" in inputs[1]

    # Verify all covariate values are preserved with correct shapes
    assert inputs[0]["past_covariates"]["cov1"].shape == (10,)
    assert inputs[0]["future_covariates"]["cov1"].shape == (5,)
    assert inputs[1]["past_covariates"]["cov1"].shape == (18,)
    assert inputs[1]["future_covariates"]["cov1"].shape == (5,)


@pytest.mark.parametrize("freq", ["s", "min", "30min", "h", "D", "W", "ME", "QE", "YE"])
def test_convert_df_generates_prediction_timestamps_with_correct_frequency(freq):
    """Test that prediction timestamps follow the inferred frequency."""
    # Use multiple series with irregular lengths
    df = create_df(series_ids=["A", "B", "C"], n_points=[10, 15, 12], target_cols=["target"], freq=freq)

    inputs, original_order, prediction_timestamps = convert_df_input_to_list_of_dicts_input(
        df=df,
        future_df=None,
        target_columns=["target"],
        prediction_length=5,
    )

    # Verify timestamps for all series
    for series_id in ["A", "B", "C"]:
        # Verify timestamps start after last context timestamp
        last_context_time = df[df["item_id"] == series_id]["timestamp"].max()
        first_pred_time = prediction_timestamps[series_id][0]
        assert first_pred_time > last_context_time

        # Verify timestamps are evenly spaced according to frequency
        pred_times = prediction_timestamps[series_id]
        assert len(pred_times) == 5
        inferred_freq = pd.infer_freq(pred_times)
        assert inferred_freq is not None


def test_convert_df_skips_validation_when_disabled():
    """Test that validate_inputs=False skips validation."""
    df = create_df(series_ids=["A", "B"], n_points=[10, 12], target_cols=["target"], freq="h")

    # Mock validate_df_inputs to verify it's not called when validation is disabled
    with patch("chronos.df_utils.validate_df_inputs") as mock_validate:
        inputs, original_order, prediction_timestamps = convert_df_input_to_list_of_dicts_input(
            df=df,
            future_df=None,
            target_columns=["target"],
            prediction_length=5,
            validate_inputs=False,
        )

        # Verify validate_df_inputs was not called
        mock_validate.assert_not_called()

        # Verify conversion still works
        assert len(inputs) == 2


def test_convert_df_preserves_all_values_with_random_inputs():
    """Generate random dataframe and verify all values are preserved exactly."""
    # Generate random parameters
    n_series = np.random.randint(2, 5)
    n_targets = np.random.randint(1, 4)
    n_past_only_covariates = np.random.randint(1, 3)
    n_future_covariates = np.random.randint(1, 3)
    prediction_length = 5

    series_ids = [f"series_{i}" for i in range(n_series)]
    n_points = [np.random.randint(10, 20) for _ in range(n_series)]
    target_cols = [f"target_{i}" for i in range(n_targets)]
    past_only_covariates = [f"past_cov_{i}" for i in range(n_past_only_covariates)]
    future_covariates = [f"future_cov_{i}" for i in range(n_future_covariates)]
    all_covariates = past_only_covariates + future_covariates

    # Create dataframe with all covariates
    df = create_df(
        series_ids=series_ids, n_points=n_points, target_cols=target_cols, covariates=all_covariates, freq="h"
    )

    # Create future_df with only future covariates (not past-only ones)
    forecast_start_times = get_forecast_start_times(df, freq="h")
    future_df = create_future_df(
        forecast_start_times=forecast_start_times,
        series_ids=series_ids,
        n_points=[prediction_length] * n_series,
        covariates=future_covariates,
        freq="h",
    )

    # Convert to list-of-dicts format
    inputs, original_order, prediction_timestamps = convert_df_input_to_list_of_dicts_input(
        df=df,
        future_df=future_df,
        target_columns=target_cols,
        prediction_length=prediction_length,
    )

    # Verify all target values are preserved exactly
    df_sorted = df.sort_values(["item_id", "timestamp"])
    for i, series_id in enumerate(series_ids):
        series_data = df_sorted[df_sorted["item_id"] == series_id]
        assert inputs[i]["target"].shape == (n_targets, n_points[i])

        for j, target_col in enumerate(target_cols):
            np.testing.assert_array_almost_equal(inputs[i]["target"][j], series_data[target_col].values)

    # Verify all past covariate values are preserved (both past-only and future covariates)
    for i, series_id in enumerate(series_ids):
        series_data = df_sorted[df_sorted["item_id"] == series_id]
        assert "past_covariates" in inputs[i]
        for cov in all_covariates:
            np.testing.assert_array_almost_equal(inputs[i]["past_covariates"][cov], series_data[cov].values)

    # Verify only future covariates are in future_covariates (not past-only ones)
    future_df_sorted = future_df.sort_values(["item_id", "timestamp"])
    for i, series_id in enumerate(series_ids):
        series_future_data = future_df_sorted[future_df_sorted["item_id"] == series_id]
        assert "future_covariates" in inputs[i]
        # Only future covariates should be present
        assert set(inputs[i]["future_covariates"].keys()) == set(future_covariates)
        for cov in future_covariates:
            np.testing.assert_array_almost_equal(inputs[i]["future_covariates"][cov], series_future_data[cov].values)

    # Verify output structure is correct
    assert len(inputs) == n_series
    assert list(original_order) == series_ids
    assert len(prediction_timestamps) == n_series


def test_convert_df_with_freq_and_validate_inputs_raises_error():
    """Test that providing freq with validate_inputs=True raises ValueError."""
    df = create_df(series_ids=["A", "B"], n_points=[10, 12], target_cols=["target"], freq="h")

    with pytest.raises(ValueError, match="freq can only be provided when validate_inputs=False"):
        convert_df_input_to_list_of_dicts_input(
            df=df,
            future_df=None,
            target_columns=["target"],
            prediction_length=5,
            freq="h",
            validate_inputs=True,
        )


@pytest.mark.parametrize("use_future_df", [True, False])
def test_convert_df_with_freq_and_validate_inputs_false(use_future_df):
    """Test that freq works with validate_inputs=False."""
    df = create_df(series_ids=["A", "B"], n_points=[10, 12], target_cols=["target"], covariates=["cov1"], freq="h")
    prediction_length = 5

    future_df = None
    if use_future_df:
        forecast_start_times = get_forecast_start_times(df, freq="h")
        future_df = create_future_df(
            forecast_start_times=forecast_start_times,
            series_ids=["A", "B"],
            n_points=[prediction_length, prediction_length],
            covariates=["cov1"],
            freq="h",
        )

    inputs, original_order, prediction_timestamps = convert_df_input_to_list_of_dicts_input(
        df=df,
        future_df=future_df,
        target_columns=["target"],
        prediction_length=prediction_length,
        freq="h",
        validate_inputs=False,
    )

    assert len(inputs) == 2
    assert len(prediction_timestamps) == 2
    for series_id in ["A", "B"]:
        assert len(prediction_timestamps[series_id]) == prediction_length


@pytest.mark.parametrize("use_future_df", [True, False])
def test_convert_df_with_mismatched_freq_uses_user_provided_freq(use_future_df):
    """Test that user-provided freq overrides data frequency when validate_inputs=False."""
    # Create data with hourly frequency
    data_freq = "h"
    df = create_df(
        series_ids=["A", "B"], n_points=[10, 12], target_cols=["target"], covariates=["cov1"], freq=data_freq
    )
    prediction_length = 5

    # User provides daily frequency (different from data)
    user_freq = "D"

    future_df = None
    if use_future_df:
        # Create future_df with hourly frequency (matching data, not user freq)
        forecast_start_times = get_forecast_start_times(df, freq=data_freq)
        future_df = create_future_df(
            forecast_start_times=forecast_start_times,
            series_ids=["A", "B"],
            n_points=[prediction_length, prediction_length],
            covariates=["cov1"],
            freq=data_freq,
        )

    inputs, original_order, prediction_timestamps = convert_df_input_to_list_of_dicts_input(
        df=df,
        future_df=future_df,
        target_columns=["target"],
        prediction_length=prediction_length,
        freq=user_freq,
        validate_inputs=False,
    )

    # Prediction should work
    assert len(inputs) == 2
    assert len(prediction_timestamps) == 2

    # Forecast timestamps should use user-provided freq (daily), not data freq (hourly)
    for series_id in ["A", "B"]:
        pred_ts = prediction_timestamps[series_id]
        assert len(pred_ts) == prediction_length
        # Verify the frequency matches user-provided freq
        inferred_freq = pd.infer_freq(pred_ts)
        assert inferred_freq == user_freq
