# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
import torch

from chronos.chronos2.preprocess import (
    _target_encode,
    from_dataframe,
    from_list_of_dicts,
    from_list_of_tensors,
    from_tensor,
)
from test.util import create_df, create_future_df, get_forecast_start_times


# Tests for from_tensor


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_from_tensor_returns_one_prepared_input_per_series(dtype):
    data = torch.rand(3, 2, 10, dtype=dtype)
    prediction_length = 5

    out = from_tensor(data, prediction_length=prediction_length)

    assert len(out) == 3
    for i, prepared in enumerate(out):
        assert prepared["context"].shape == (2, 10)
        assert prepared["context"].dtype == torch.float32
        assert prepared["future_covariates"].shape == (2, prediction_length)
        assert torch.isnan(prepared["future_covariates"]).all()
        assert prepared["n_targets"] == 2
        assert prepared["n_covariates"] == 0
        assert prepared["n_future_covariates"] == 0
        torch.testing.assert_close(prepared["context"], data[i].to(dtype=torch.float32))


def test_from_tensor_accepts_numpy_array():
    data = np.random.randn(2, 1, 8).astype(np.float32)

    out = from_tensor(data, prediction_length=4)

    assert len(out) == 2
    np.testing.assert_array_almost_equal(out[0]["context"].numpy(), data[0])


def test_from_tensor_raises_for_wrong_ndim():
    data = torch.rand(2, 10)
    with pytest.raises(ValueError, match="Expected 3-d tensor"):
        from_tensor(data, prediction_length=5)


# Tests for from_list_of_tensors


def test_from_list_of_tensors_handles_mixed_1d_and_2d():
    data = [torch.rand(10), torch.rand(2, 15), np.random.rand(8).astype(np.float32)]
    prediction_length = 4

    out = from_list_of_tensors(data, prediction_length=prediction_length)

    assert len(out) == 3
    assert out[0]["context"].shape == (1, 10)
    assert out[0]["n_targets"] == 1
    assert out[1]["context"].shape == (2, 15)
    assert out[1]["n_targets"] == 2
    assert out[2]["context"].shape == (1, 8)
    for prepared in out:
        assert prepared["future_covariates"].shape[-1] == prediction_length
        assert torch.isnan(prepared["future_covariates"]).all()
        assert prepared["n_covariates"] == 0
        assert prepared["n_future_covariates"] == 0


def test_from_list_of_tensors_raises_for_3d_element():
    data = [torch.rand(1, 2, 10)]
    with pytest.raises(ValueError, match="Each element should be 1-d or 2-d"):
        from_list_of_tensors(data, prediction_length=5)


# Tests for from_list_of_dicts


def test_from_list_of_dicts_with_target_only():
    data = [{"target": np.random.randn(10).astype(np.float32)}, {"target": np.random.randn(15).astype(np.float32)}]

    out = from_list_of_dicts(data, prediction_length=5)

    assert len(out) == 2
    assert out[0]["context"].shape == (1, 10)
    assert out[1]["context"].shape == (1, 15)
    for prepared, d in zip(out, data):
        np.testing.assert_array_almost_equal(prepared["context"][0].numpy(), d["target"])
        assert prepared["n_targets"] == 1
        assert prepared["n_covariates"] == 0
        assert prepared["n_future_covariates"] == 0


def test_from_list_of_dicts_orders_past_only_before_known_future():
    """context[-n_future_covariates:] must contain known-future rows (Chronos2Dataset invariant)."""
    data = [
        {
            "target": np.random.randn(10).astype(np.float32),
            "past_covariates": {
                "known_a": np.arange(10, dtype=np.float32),  # alphabetically first → should come last
                "past_only_b": np.arange(10, dtype=np.float32) + 100,
            },
            "future_covariates": {"known_a": np.arange(5, dtype=np.float32) + 200},
        }
    ]

    out = from_list_of_dicts(data, prediction_length=5)
    prepared = out[0]

    assert prepared["n_targets"] == 1
    assert prepared["n_covariates"] == 2
    assert prepared["n_future_covariates"] == 1
    # last row of context must be known_a (matches future_covariates last row)
    np.testing.assert_array_almost_equal(prepared["context"][-1].numpy(), np.arange(10, dtype=np.float32))
    np.testing.assert_array_almost_equal(prepared["context"][1].numpy(), np.arange(10, dtype=np.float32) + 100)
    # future_covariates last row contains the known-future values, earlier rows are NaN
    np.testing.assert_array_almost_equal(prepared["future_covariates"][-1].numpy(), np.arange(5, dtype=np.float32) + 200)
    assert torch.isnan(prepared["future_covariates"][:-1]).all()


def test_from_list_of_dicts_with_known_covariates_names_nan_fills_future():
    data = [
        {
            "target": np.random.randn(10).astype(np.float32),
            "past_covariates": {"feat": np.random.randn(10).astype(np.float32)},
        },
        {
            "target": np.random.randn(12).astype(np.float32),
            "past_covariates": {"feat": np.random.randn(12).astype(np.float32)},
        },
    ]

    out = from_list_of_dicts(data, prediction_length=5, known_covariates_names=["feat"])

    for prepared in out:
        assert prepared["n_future_covariates"] == 1
        assert prepared["n_covariates"] == 1
        assert torch.isnan(prepared["future_covariates"]).all()


def test_from_list_of_dicts_translates_none_future_to_known_covariates():
    """{'a': None} in future_covariates should behave like known_covariates_names=['a']."""
    data = [
        {
            "target": np.random.randn(10).astype(np.float32),
            "past_covariates": {"feat": np.random.randn(10).astype(np.float32)},
            "future_covariates": {"feat": None},
        }
    ]

    out = from_list_of_dicts(data, prediction_length=5)

    assert out[0]["n_future_covariates"] == 1
    assert torch.isnan(out[0]["future_covariates"]).all()


def test_from_list_of_dicts_does_not_mutate_caller_data():
    cov = np.random.randn(10).astype(np.float32)
    data = [
        {
            "target": np.random.randn(10).astype(np.float32),
            "past_covariates": {"feat": cov},
            "future_covariates": {"feat": None},
        }
    ]
    snapshot = data[0]["future_covariates"]

    from_list_of_dicts(data, prediction_length=5)

    assert data[0]["future_covariates"] is snapshot
    assert data[0]["future_covariates"]["feat"] is None


def test_from_list_of_dicts_raises_for_unknown_known_covariates_names():
    data = [
        {
            "target": np.random.randn(10).astype(np.float32),
            "past_covariates": {"feat": np.random.randn(10).astype(np.float32)},
        }
    ]
    with pytest.raises(ValueError, match="known_covariates_names must all be"):
        from_list_of_dicts(data, prediction_length=5, known_covariates_names=["does_not_exist"])


def test_from_list_of_dicts_raises_when_future_keys_not_subset_of_past():
    data = [
        {
            "target": np.random.randn(10).astype(np.float32),
            "past_covariates": {"a": np.random.randn(10).astype(np.float32)},
            "future_covariates": {"b": np.random.randn(5).astype(np.float32)},
        }
    ]
    with pytest.raises(ValueError, match="must be a subset"):
        from_list_of_dicts(data, prediction_length=5)


def test_from_list_of_dicts_raises_when_target_too_many_dims():
    data = [{"target": np.random.randn(1, 2, 10).astype(np.float32)}]
    with pytest.raises(ValueError, match="Target must be 1-d or 2-d"):
        from_list_of_dicts(data, prediction_length=5)


def test_from_list_of_dicts_returns_empty_for_empty_input():
    assert from_list_of_dicts([], prediction_length=5) == []


def test_from_list_of_dicts_preserves_numeric_covariate_values():
    rng = np.random.default_rng(42)
    targets = [rng.standard_normal(10).astype(np.float32), rng.standard_normal(12).astype(np.float32)]
    past = [rng.standard_normal(10).astype(np.float32), rng.standard_normal(12).astype(np.float32)]
    future = [rng.standard_normal(5).astype(np.float32), rng.standard_normal(5).astype(np.float32)]

    data = [
        {"target": targets[0], "past_covariates": {"x": past[0]}, "future_covariates": {"x": future[0]}},
        {"target": targets[1], "past_covariates": {"x": past[1]}, "future_covariates": {"x": future[1]}},
    ]

    out = from_list_of_dicts(data, prediction_length=5)

    for i in range(2):
        np.testing.assert_array_almost_equal(out[i]["context"][0].numpy(), targets[i])
        np.testing.assert_array_almost_equal(out[i]["context"][1].numpy(), past[i])
        np.testing.assert_array_almost_equal(out[i]["future_covariates"][1].numpy(), future[i])
        assert torch.isnan(out[i]["future_covariates"][0]).all()  # target row stays NaN


# Tests for from_dataframe


def test_from_dataframe_with_single_target():
    df = create_df(series_ids=["A", "B"], n_points=[10, 12], target_cols=["target"], freq="h")

    out = from_dataframe(df=df, target_columns=["target"], prediction_length=5)

    assert len(out) == 2
    assert out[0]["context"].shape == (1, 10)
    assert out[1]["context"].shape == (1, 12)
    for prepared in out:
        assert prepared["n_targets"] == 1
        assert prepared["n_covariates"] == 0


def test_from_dataframe_with_past_and_future_covariates():
    df = create_df(series_ids=["A", "B"], n_points=[10, 12], target_cols=["target"], covariates=["cov1"], freq="h")
    forecast_start_times = get_forecast_start_times(df, freq="h")
    future_df = create_future_df(
        forecast_start_times=forecast_start_times,
        series_ids=["A", "B"],
        n_points=[5, 5],
        covariates=["cov1"],
        freq="h",
    )

    out = from_dataframe(df=df, target_columns=["target"], prediction_length=5, future_df=future_df)

    for prepared in out:
        assert prepared["n_targets"] == 1
        assert prepared["n_covariates"] == 1
        assert prepared["n_future_covariates"] == 1
        # last row of context = known-future covariate
        assert not torch.isnan(prepared["future_covariates"][-1]).any()
        assert torch.isnan(prepared["future_covariates"][0]).all()


def test_from_dataframe_raises_when_future_df_and_known_covariates_names_both_given():
    df = create_df(series_ids=["A"], n_points=[10], target_cols=["target"], covariates=["cov1"], freq="h")
    future_df = create_future_df(
        forecast_start_times=get_forecast_start_times(df, freq="h"),
        series_ids=["A"],
        n_points=[5],
        covariates=["cov1"],
        freq="h",
    )
    with pytest.raises(ValueError, match="Cannot provide both"):
        from_dataframe(
            df=df,
            target_columns=["target"],
            prediction_length=5,
            future_df=future_df,
            known_covariates_names=["cov1"],
        )


def test_from_dataframe_raises_for_unknown_known_covariates_names():
    df = create_df(series_ids=["A", "B"], n_points=[10, 12], target_cols=["target"], covariates=["cov1"], freq="h")
    with pytest.raises(ValueError, match="known_covariates_names contains columns not present"):
        from_dataframe(
            df=df,
            target_columns=["target"],
            prediction_length=5,
            known_covariates_names=["does_not_exist"],
        )


def test_from_dataframe_does_not_mutate_caller_dataframes():
    df = create_df(series_ids=["A", "B"], n_points=[10, 12], target_cols=["target"], freq="h")
    df["timestamp"] = df["timestamp"].astype(str)  # force non-datetime input
    df_copy = df.copy()

    from_dataframe(df=df, target_columns=["target"], prediction_length=5)

    pd.testing.assert_frame_equal(df, df_copy)


def test_from_dataframe_raises_for_short_series():
    df = create_df(series_ids=["A", "B"], n_points=[10, 2], target_cols=["target"], freq="h")
    with pytest.raises(ValueError, match=">= 3 points"):
        from_dataframe(df=df, target_columns=["target"], prediction_length=5)


def test_from_dataframe_raises_for_non_numeric_target():
    df = create_df(series_ids=["A"], n_points=[10], target_cols=["target"], freq="h")
    df["target"] = df["target"].astype(str)
    with pytest.raises(ValueError, match="must be numeric"):
        from_dataframe(df=df, target_columns=["target"], prediction_length=5)


def test_from_dataframe_with_categorical_covariate_target_encoding():
    """Target encoding with single target should produce finite, per-item-aware encodings."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "item_id": ["A"] * 10 + ["B"] * 10,
            "timestamp": list(pd.date_range(end="2020-01-10", periods=10, freq="D")) * 2,
            "target": rng.standard_normal(20).astype(np.float32),
            "cat": np.array(["x", "y"] * 10),
        }
    )

    out = from_dataframe(df=df, target_columns=["target"], prediction_length=3, known_covariates_names=["cat"])

    assert len(out) == 2
    for prepared in out:
        assert prepared["n_future_covariates"] == 1
        # context past covariate row (encoded) should be finite for all known cat values
        assert torch.isfinite(prepared["context"][1]).all()


# Tests for _target_encode (the core categorical encoder)


def test_target_encode_unseen_future_category_falls_back_to_item_mean():
    """Unseen categories at inference must map to per-item mean, not category 0's encoding."""
    target = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    id_codes = np.array([0, 0, 0, 1, 1, 1])  # 2 items, 3 rows each
    cat_codes = np.array([0, 1, 0, 0, 1, 1])  # categories 'a', 'b'
    n_items = 2
    n_categories = 2  # 'a' and 'b' exist in training

    # future has unseen category mapped to n_categories (the sentinel slot)
    future_id_codes = np.array([0, 1])
    future_cat_codes = np.array([n_categories, n_categories])  # both unseen

    _, encoded_future = _target_encode(
        id_codes=id_codes,
        cat_codes=cat_codes,
        target=target,
        n_items=n_items,
        n_categories=n_categories,
        future_id_codes=future_id_codes,
        future_cat_codes=future_cat_codes,
        smooth=1.0,
    )

    # item 0 mean = (1+2+3)/3 = 2.0; item 1 mean = (4+5+6)/3 = 5.0
    # unseen slot has sums=0, counts=0 → lookup = (1.0 * mean + 0) / (1.0 + 0) = mean
    np.testing.assert_array_almost_equal(encoded_future, [2.0, 5.0], decimal=5)


def test_target_encode_seen_category_uses_smoothed_mean():
    target = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    id_codes = np.array([0, 0, 0, 0])
    cat_codes = np.array([0, 0, 1, 1])  # category 0 has [10, 20], category 1 has [30, 40]
    n_items = 1
    n_categories = 2
    smooth = 1.0

    encoded_past, _ = _target_encode(
        id_codes=id_codes,
        cat_codes=cat_codes,
        target=target,
        n_items=n_items,
        n_categories=n_categories,
        smooth=smooth,
    )

    # item mean = 25.0
    # cat 0: (1.0 * 25.0 + 30.0) / (1.0 + 2) = 55/3 ≈ 18.333
    # cat 1: (1.0 * 25.0 + 70.0) / (1.0 + 2) = 95/3 ≈ 31.667
    np.testing.assert_array_almost_equal(encoded_past, [18.3333, 18.3333, 31.6667, 31.6667], decimal=3)


def test_target_encode_handles_nans_in_target():
    target = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32)
    id_codes = np.array([0, 0, 0, 0])
    cat_codes = np.array([0, 0, 1, 1])

    encoded_past, _ = _target_encode(
        id_codes=id_codes, cat_codes=cat_codes, target=target, n_items=1, n_categories=2, smooth=1.0
    )

    # NaN excluded → item mean = (1+3+4)/3 ≈ 2.667
    # cat 0: (1.0 * 2.667 + 1.0) / (1.0 + 1) = 3.667/2 ≈ 1.833
    # cat 1: (1.0 * 2.667 + 7.0) / (1.0 + 2) = 9.667/3 ≈ 3.222
    np.testing.assert_array_almost_equal(encoded_past, [1.8333, 1.8333, 3.2222, 3.2222], decimal=3)
    assert np.isfinite(encoded_past).all()


def test_target_encode_returns_none_future_when_not_provided():
    target = np.array([1.0, 2.0], dtype=np.float32)
    id_codes = np.array([0, 0])
    cat_codes = np.array([0, 1])

    encoded_past, encoded_future = _target_encode(
        id_codes=id_codes, cat_codes=cat_codes, target=target, n_items=1, n_categories=2
    )

    assert encoded_past.shape == (2,)
    assert encoded_future is None


def test_target_encode_item_with_all_nan_targets_does_not_crash():
    target = np.array([np.nan, np.nan, 1.0, 2.0], dtype=np.float32)
    id_codes = np.array([0, 0, 1, 1])
    cat_codes = np.array([0, 1, 0, 1])

    encoded_past, _ = _target_encode(
        id_codes=id_codes, cat_codes=cat_codes, target=target, n_items=2, n_categories=2, smooth=1.0
    )

    # item 0 has no valid target → item_mean = 0; item 1 mean = 1.5
    assert np.isfinite(encoded_past).all()
    np.testing.assert_array_almost_equal(encoded_past[2:], [(1.0 * 1.5 + 1.0) / 2, (1.0 * 1.5 + 2.0) / 2], decimal=4)


# Round-trip: equivalence with old preprocessing path on a small fixture


def test_from_list_of_dicts_categorical_unseen_future_does_not_leak_category_zero():
    """End-to-end: unseen future category should encode to item mean, not category 0's encoding."""
    rng = np.random.default_rng(123)
    target = rng.standard_normal(20).astype(np.float32)
    past_cat = np.array(["a", "b", "a", "b", "a"] * 4)
    future_cat = np.array(["zzz_unseen"] * 5)

    data = [
        {
            "target": target,
            "past_covariates": {"cat": past_cat},
            "future_covariates": {"cat": future_cat},
        }
    ]

    out = from_list_of_dicts(data, prediction_length=5, use_target_encoding=True)
    prepared = out[0]

    # encoded future for unseen category should equal item mean
    expected_item_mean = float(target.mean())
    np.testing.assert_array_almost_equal(
        prepared["future_covariates"][-1].numpy(),
        np.full(5, expected_item_mean, dtype=np.float32),
        decimal=4,
    )


def test_from_list_of_dicts_ordinal_encoding_when_multivariate():
    """With multi-target, falls back to ordinal encoding (not target encoding)."""
    rng = np.random.default_rng(7)
    data = [
        {
            "target": rng.standard_normal((2, 10)).astype(np.float32),
            "past_covariates": {"cat": np.array(["a", "b"] * 5)},
            "future_covariates": {"cat": np.array(["a", "b", "a", "b", "a"])},
        }
    ]

    out = from_list_of_dicts(data, prediction_length=5, use_target_encoding=True)
    prepared = out[0]

    # ordinal encoding produces integer-like floats {0.0, 1.0}
    past_cat_row = prepared["context"][-1].numpy()
    assert set(np.unique(past_cat_row).tolist()).issubset({0.0, 1.0})
    future_cat_row = prepared["future_covariates"][-1].numpy()
    assert set(np.unique(future_cat_row).tolist()).issubset({0.0, 1.0})


def test_from_list_of_dicts_nan_is_its_own_target_encoded_category():
    """NaN is a category with its own mean: distinct from "x", from the item mean, and matching
    itself across past and future."""
    target = np.array([10.0, 0.0] * 6, dtype=np.float32)  # "x" -> 10, NaN -> 0
    data = [
        {
            "target": target,
            "past_covariates": {"cat": np.array(["x", None] * 6, dtype=object)},
            "future_covariates": {"cat": np.array([None] * 4, dtype=object)},
        }
    ]

    out = from_list_of_dicts(data, prediction_length=4, use_target_encoding=True)
    x_enc, nan_enc = out[0]["context"][-1].numpy()[:2]
    future_row = out[0]["future_covariates"][-1].numpy()

    assert nan_enc < target.mean() < x_enc
    np.testing.assert_array_almost_equal(future_row, np.full(4, nan_enc), decimal=5)


def test_from_list_of_dicts_unseen_future_string_falls_back_to_item_mean():
    """An unseen future string encodes to the item mean, not the NaN category."""
    target = np.array([10.0, 0.0] * 6, dtype=np.float32)
    data = [
        {
            "target": target,
            "past_covariates": {"cat": np.array(["x", None] * 6, dtype=object)},
            "future_covariates": {"cat": np.array(["zzz_unseen", None], dtype=object)},
        }
    ]

    out = from_list_of_dicts(data, prediction_length=2, use_target_encoding=True)
    future_row = out[0]["future_covariates"][-1].numpy()
    nan_enc = out[0]["context"][-1].numpy()[1]

    np.testing.assert_almost_equal(future_row[0], target.mean(), decimal=4)
    np.testing.assert_almost_equal(future_row[1], nan_enc, decimal=4)


def test_from_list_of_dicts_ordinal_encoding_unseen_future_is_nan():
    rng = np.random.default_rng(8)
    data = [
        {
            "target": rng.standard_normal((2, 10)).astype(np.float32),
            "past_covariates": {"cat": np.array(["a", "b"] * 5)},
            "future_covariates": {"cat": np.array(["a", "zzz_unseen", "a", "b", "a"])},
        }
    ]

    out = from_list_of_dicts(data, prediction_length=5, use_target_encoding=True)
    future_cat_row = out[0]["future_covariates"][-1].numpy()

    assert np.isnan(future_cat_row[1])
    assert np.isfinite(future_cat_row[[0, 2, 3, 4]]).all()
