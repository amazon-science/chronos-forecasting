# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from chronos.df_utils import (
    infer_freq_from_df,
    make_future_df,
    normalize_df,
    validate_and_normalize_df,
    validate_df,
)
from test.util import create_df, create_future_df, get_forecast_start_times

# Tests for infer_freq_from_df


@pytest.mark.parametrize("freq", ["s", "min", "30min", "h", "D", "W", "ME", "QE", "YE"])
def test_infer_freq_from_df_recovers_the_data_frequency(freq):
    df = normalize_df(create_df(series_ids=["A", "B"], n_points=[10, 15], freq=freq))
    # pandas may report an anchored alias (e.g. "W" -> "W-SUN"); compare the base offset.
    assert pd.tseries.frequencies.to_offset(infer_freq_from_df(df)) == pd.tseries.frequencies.to_offset(freq)


def test_infer_freq_from_df_ignores_series_with_fewer_than_three_points():
    # Series B has only 2 points; the frequency is pinned down by series A.
    df = normalize_df(create_df(series_ids=["A", "B"], n_points=[10, 2], freq="h"))
    assert pd.tseries.frequencies.to_offset(infer_freq_from_df(df)) == pd.tseries.frequencies.to_offset("h")


def test_infer_freq_from_df_raises_when_no_series_has_three_points():
    df = normalize_df(create_df(series_ids=["A", "B"], n_points=[2, 2], freq="h"))
    with pytest.raises(ValueError, match="no time series has at least 3"):
        infer_freq_from_df(df)


def test_infer_freq_from_df_raises_for_irregular_series():
    df = pd.DataFrame(
        {
            "item_id": ["A"] * 4,
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-04", "2023-01-05"]),
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match="Could not infer frequency for series A"):
        infer_freq_from_df(df)


def test_infer_freq_from_df_raises_when_series_disagree():
    df = pd.DataFrame(
        {
            "item_id": ["A"] * 3 + ["B"] * 3,
            "timestamp": list(pd.date_range("2023-01-01", periods=3, freq="h"))
            + list(pd.date_range("2023-01-01", periods=3, freq="D")),
            "target": [1.0] * 6,
        }
    )
    with pytest.raises(ValueError, match="same frequency"):
        infer_freq_from_df(df)


# Tests for make_future_df


@pytest.mark.parametrize("freq", ["s", "min", "30min", "h", "D", "W", "ME", "QE", "YE"])
def test_make_future_df_generates_correct_timestamps(freq):
    df = normalize_df(create_df(series_ids=["A", "B", "C"], n_points=[10, 15, 12], freq=freq))
    prediction_length = 5

    future = make_future_df(df, prediction_length=prediction_length, freq=freq)

    assert list(future.columns) == ["item_id", "timestamp"]
    assert len(future) == 3 * prediction_length
    for series_id in ["A", "B", "C"]:
        series_future = future[future["item_id"] == series_id]["timestamp"]
        assert len(series_future) == prediction_length
        # Future timestamps must start strictly after the last observed timestamp ...
        last_context_time = df[df["item_id"] == series_id]["timestamp"].max()
        assert series_future.iloc[0] > last_context_time
        # ... and be evenly spaced at the requested frequency.
        assert pd.tseries.frequencies.to_offset(pd.infer_freq(series_future)) == pd.tseries.frequencies.to_offset(freq)


def test_make_future_df_preserves_first_appearance_item_order():
    df = normalize_df(create_df(series_ids=["B", "A"], n_points=[5, 7], freq="h"))
    future = make_future_df(df, prediction_length=3, freq="h")
    # normalize_df keeps first-appearance order (B before A); make_future_df must match it.
    assert future["item_id"].drop_duplicates().tolist() == ["B", "A"]


def test_make_future_df_infers_freq_when_not_provided():
    df = normalize_df(create_df(series_ids=["A", "B"], n_points=[10, 12], freq="D"))
    future = make_future_df(df, prediction_length=4)
    for series_id in ["A", "B"]:
        series_future = future[future["item_id"] == series_id]["timestamp"]
        assert pd.tseries.frequencies.to_offset(pd.infer_freq(series_future)) == pd.tseries.frequencies.to_offset("D")


# Tests for normalize_df


def test_normalize_df_groups_by_id_and_sorts_by_timestamp():
    df = pd.DataFrame(
        {
            "item_id": ["B", "A", "B", "A"],
            "timestamp": pd.to_datetime(["2023-01-02", "2023-01-01", "2023-01-01", "2023-01-02"]),
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )

    out = normalize_df(df)

    # Grouped in first-appearance order (B then A), sorted by timestamp within each group.
    assert out["item_id"].tolist() == ["B", "B", "A", "A"]
    for series_id in ["A", "B"]:
        series = out[out["item_id"] == series_id]["timestamp"]
        assert series.is_monotonic_increasing


def test_normalize_df_coerces_string_timestamps_to_datetime():
    df = pd.DataFrame({"item_id": ["A", "A"], "timestamp": ["2023-01-01", "2023-01-02"], "target": [1.0, 2.0]})
    out = normalize_df(df)
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])


def test_normalize_df_respects_explicit_order():
    df = pd.DataFrame(
        {
            "item_id": ["B", "A", "B", "A"],
            "timestamp": pd.to_datetime(["2023-01-02", "2023-01-01", "2023-01-01", "2023-01-02"]),
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = normalize_df(df, order=np.array(["A", "B"]))
    assert out["item_id"].tolist() == ["A", "A", "B", "B"]


def test_normalize_df_raises_for_ids_absent_from_order():
    df = pd.DataFrame(
        {
            "item_id": ["A", "B"],
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "target": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="ids not present in df"):
        normalize_df(df, order=np.array(["A"]))


def test_normalize_df_does_not_mutate_caller_dataframe():
    df = create_df(series_ids=["B", "A"], n_points=[5, 5], freq="h")
    df["timestamp"] = df["timestamp"].astype(str)  # force non-datetime input
    df_copy = df.copy()

    normalize_df(df)

    pd.testing.assert_frame_equal(df, df_copy)


# Tests for validate_df


def test_validate_df_accepts_valid_inputs():
    df = create_df(series_ids=["A", "B"], n_points=[10, 12], covariates=["cov1"], freq="h")
    future_df = create_future_df(
        get_forecast_start_times(df, "h"), series_ids=["A", "B"], n_points=[5, 5], covariates=["cov1"], freq="h"
    )
    # Should not raise.
    validate_df(df, future_df, ["target"], ["cov1"], prediction_length=5, id_column="item_id", timestamp_column="timestamp")


def test_validate_df_raises_for_missing_required_columns():
    df = pd.DataFrame({"item_id": ["A"], "target": [1.0]})  # no timestamp
    with pytest.raises(ValueError, match="df does not contain all expected columns"):
        validate_df(df, None, ["target"], None, prediction_length=5, id_column="item_id", timestamp_column="timestamp")


def test_validate_df_raises_for_non_numeric_target():
    df = create_df(series_ids=["A"], n_points=[10], freq="h")
    df["target"] = df["target"].astype(str)
    with pytest.raises(ValueError, match="must be numeric"):
        validate_df(df, None, ["target"], None, prediction_length=5, id_column="item_id", timestamp_column="timestamp")


def test_validate_df_raises_for_unknown_known_covariates_names():
    df = create_df(series_ids=["A", "B"], n_points=[10, 12], covariates=["cov1"], freq="h")
    with pytest.raises(ValueError, match="known_covariates_names contains columns not present"):
        validate_df(
            df, None, ["target"], ["does_not_exist"], prediction_length=5, id_column="item_id", timestamp_column="timestamp"
        )


@pytest.mark.parametrize(
    "future_data, error_match",
    [
        # Missing timestamp column
        ({"item_id": ["A"], "cov1": [1.0]}, "future_df does not contain all"),
        # target column present in future_df
        (
            {"item_id": ["A"], "timestamp": ["2023-01-01"], "cov1": [1.0], "target": [1.0]},
            "future_df cannot contain target",
        ),
        # column absent from df
        (
            {"item_id": ["A"], "timestamp": ["2023-01-01"], "cov1": [1.0], "cov2": [1.0]},
            "future_df cannot contain columns not present",
        ),
    ],
)
def test_validate_df_raises_for_malformed_future_df(future_data, error_match):
    df = create_df(series_ids=["A", "B"], n_points=[10, 12], covariates=["cov1"], freq="h")
    future_df = pd.DataFrame(future_data)
    with pytest.raises(ValueError, match=error_match):
        validate_df(
            df, future_df, ["target"], ["cov1"], prediction_length=5, id_column="item_id", timestamp_column="timestamp"
        )


def test_validate_df_raises_when_future_df_has_mismatched_series_ids():
    df = create_df(series_ids=["A", "B"], n_points=[10, 15], covariates=["cov1"], freq="h")
    future_df = create_future_df(
        [get_forecast_start_times(df, "h")[0]], series_ids=["A"], n_points=[5], covariates=["cov1"], freq="h"
    )
    with pytest.raises(ValueError, match="future_df must have the same time series IDs as df"):
        validate_df(
            df, future_df, ["target"], ["cov1"], prediction_length=5, id_column="item_id", timestamp_column="timestamp"
        )


def test_validate_df_raises_when_future_df_has_incorrect_lengths():
    df = create_df(series_ids=["A", "B"], n_points=[10, 13], covariates=["cov1"], freq="h")
    future_df = create_future_df(
        get_forecast_start_times(df, "h"), series_ids=["A", "B"], n_points=[3, 7], covariates=["cov1"], freq="h"
    )
    with pytest.raises(ValueError, match="future_df must contain prediction_length=5 rows per item"):
        validate_df(
            df, future_df, ["target"], ["cov1"], prediction_length=5, id_column="item_id", timestamp_column="timestamp"
        )


# Tests for validate_and_normalize_df (validate then normalize, the entry point for predict_df)


def test_validate_and_normalize_df_returns_normalized_frames():
    df = create_df(series_ids=["B", "A"], n_points=[10, 15], covariates=["cov1"], freq="h")
    future_df = create_future_df(
        get_forecast_start_times(df, "h"), series_ids=["B", "A"], n_points=[5, 5], covariates=["cov1"], freq="h"
    )

    out_df, out_future_df = validate_and_normalize_df(
        df=df, future_df=future_df, target_columns=["target"], prediction_length=5
    )

    # First-appearance order (B before A) preserved in both frames.
    assert out_df["item_id"].drop_duplicates().tolist() == ["B", "A"]
    assert out_future_df["item_id"].drop_duplicates().tolist() == ["B", "A"]
    for series_id in ["A", "B"]:
        assert out_df[out_df["item_id"] == series_id]["timestamp"].is_monotonic_increasing


def test_validate_and_normalize_df_aligns_future_df_order_to_df():
    df = normalize_df(create_df(series_ids=["B", "A"], n_points=[10, 12], covariates=["cov1"], freq="h"))
    forecast_start_times = get_forecast_start_times(df.sort_values("item_id"), "h")
    # future_df arrives in a different item order (A before B) than df (B before A).
    future_df = create_future_df(
        sorted(forecast_start_times), series_ids=["A", "B"], n_points=[5, 5], covariates=["cov1"], freq="h"
    )

    _, out_future_df = validate_and_normalize_df(
        df=df, future_df=future_df, target_columns=["target"], prediction_length=5
    )

    # future_df must be re-ordered to match df's item order.
    assert out_future_df["item_id"].drop_duplicates().tolist() == ["B", "A"]


def test_validate_and_normalize_df_handles_missing_future_df():
    df = create_df(series_ids=["A", "B"], n_points=[10, 12], freq="h")
    out_df, out_future_df = validate_and_normalize_df(
        df=df, future_df=None, target_columns=["target"], prediction_length=5
    )
    assert out_future_df is None
    assert len(out_df) == len(df)
