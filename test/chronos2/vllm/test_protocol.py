"""Unit tests for chronos.chronos2.vllm.protocol.forecast Pydantic models."""

import pytest

from chronos.chronos2.vllm.protocol.forecast import (
    ForecastParameters,
    ForecastPrediction,
    TimeSeriesInput,
)


class TestTimeSeriesInput:
    """Tests for TimeSeriesInput Pydantic model."""

    def test_minimal(self):
        ts = TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert ts.target == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert ts.item_id is None
        assert ts.start is None
        assert ts.past_covariates is None
        assert ts.future_covariates is None

    def test_with_item_id(self):
        ts = TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0], item_id="series_1")
        assert ts.item_id == "series_1"

    def test_multivariate_target(self):
        ts = TimeSeriesInput(target=[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
        assert len(ts.target) == 2

    def test_with_covariates(self):
        ts = TimeSeriesInput(
            target=[1.0, 2.0, 3.0, 4.0, 5.0],
            past_covariates={"temp": [10.0, 20.0, 30.0, 40.0, 50.0]},
            future_covariates={"temp": [60.0]},
        )
        assert "temp" in ts.past_covariates
        assert "temp" in ts.future_covariates

    def test_too_short_target_rejected(self):
        with pytest.raises(Exception):
            TimeSeriesInput(target=[1.0, 2.0])

    def test_covariate_length_mismatch_rejected(self):
        with pytest.raises(Exception):
            TimeSeriesInput(
                target=[1.0, 2.0, 3.0, 4.0, 5.0],
                past_covariates={"temp": [10.0, 20.0]},  # wrong length
            )

    def test_future_without_past_rejected(self):
        with pytest.raises(Exception):
            TimeSeriesInput(
                target=[1.0, 2.0, 3.0, 4.0, 5.0],
                future_covariates={"temp": [60.0]},
            )


class TestTimeSeriesInputNanCovariates:
    """Tests for NaN handling in covariate arrays (favorita_stores_1D scenario)."""

    def test_nan_in_string_future_covariates_sanitized(self):
        """NaN floats in string covariate lists should be sanitized to None."""
        ts = TimeSeriesInput(
            target=[1.0, 2.0, 3.0, 4.0, 5.0],
            past_covariates={
                "holiday": ["New Year", "MLK Day", float("nan"), float("nan"), "Easter"]
            },
            future_covariates={"holiday": [float("nan")]},
        )
        # NaN should be converted to None
        assert ts.past_covariates["holiday"][2] is None
        assert ts.past_covariates["holiday"][3] is None
        assert ts.future_covariates["holiday"][0] is None
        # Strings should remain unchanged
        assert ts.past_covariates["holiday"][0] == "New Year"
        assert ts.past_covariates["holiday"][4] == "Easter"

    def test_nan_in_numeric_covariates_preserved(self):
        """NaN in numeric covariates should be sanitized to None too (consistent)."""
        ts = TimeSeriesInput(
            target=[1.0, 2.0, 3.0, 4.0, 5.0],
            past_covariates={"temp": [10.0, float("nan"), 30.0, 40.0, 50.0]},
            future_covariates={"temp": [float("nan")]},
        )
        # NaN floats → None after sanitization
        assert ts.past_covariates["temp"][1] is None
        assert ts.future_covariates["temp"][0] is None

    def test_all_nan_string_covariates(self):
        """All-NaN covariate arrays should be accepted."""
        ts = TimeSeriesInput(
            target=[1.0, 2.0, 3.0, 4.0, 5.0],
            past_covariates={"holiday": [float("nan")] * 5},
            future_covariates={"holiday": [float("nan")]},
        )
        assert all(v is None for v in ts.past_covariates["holiday"])
        assert all(v is None for v in ts.future_covariates["holiday"])

    def test_mixed_string_and_none_accepted(self):
        """Mix of strings and None should be accepted."""
        ts = TimeSeriesInput(
            target=[1.0, 2.0, 3.0, 4.0, 5.0],
            past_covariates={"holiday": ["A", None, "B", None, "C"]},
        )
        assert ts.past_covariates["holiday"] == ["A", None, "B", None, "C"]

    def test_clean_string_covariates_unchanged(self):
        """String covariates without NaN should pass through unchanged."""
        ts = TimeSeriesInput(
            target=[1.0, 2.0, 3.0, 4.0, 5.0],
            past_covariates={"category": ["A", "B", "C", "D", "E"]},
        )
        assert ts.past_covariates["category"] == ["A", "B", "C", "D", "E"]


class TestForecastParameters:
    """Tests for ForecastParameters Pydantic model."""

    def test_defaults(self):
        params = ForecastParameters()
        assert params.prediction_length == 1
        assert params.quantile_levels == [0.1, 0.5, 0.9]
        assert params.freq is None
        assert params.batch_size == 256
        assert params.cross_learning is False

    def test_custom_values(self):
        params = ForecastParameters(
            prediction_length=24,
            quantile_levels=[0.25, 0.5, 0.75],
            batch_size=100,
            cross_learning=True,
        )
        assert params.prediction_length == 24
        assert params.quantile_levels == [0.25, 0.5, 0.75]
        assert params.batch_size == 100
        assert params.cross_learning is True

    def test_invalid_prediction_length(self):
        with pytest.raises(Exception):
            ForecastParameters(prediction_length=0)

    def test_invalid_quantile_level(self):
        with pytest.raises(Exception):
            ForecastParameters(quantile_levels=[0.0, 0.5])

    def test_prediction_length_max(self):
        with pytest.raises(Exception):
            ForecastParameters(prediction_length=2000)


class TestForecastPrediction:
    """Tests for ForecastPrediction Pydantic model."""

    def test_minimal(self):
        pred = ForecastPrediction(mean=[1.0, 2.0, 3.0])
        assert pred.mean == [1.0, 2.0, 3.0]
        assert pred.item_id is None

    def test_with_quantiles(self):
        pred = ForecastPrediction(
            mean=[1.0, 2.0],
            item_id="s1",
            **{"0.1": [0.5, 1.5], "0.9": [1.5, 2.5]},
        )
        assert pred.item_id == "s1"

    def test_multivariate_mean(self):
        pred = ForecastPrediction(mean=[[1.0, 2.0], [3.0, 4.0]])
        assert len(pred.mean) == 2

    def test_extra_fields_allowed(self):
        """ForecastPrediction allows dynamic quantile fields."""
        pred = ForecastPrediction(mean=[1.0], **{"0.5": [1.0], "0.1": [0.5]})
        d = pred.model_dump()
        assert "0.5" in d
        assert "0.1" in d
