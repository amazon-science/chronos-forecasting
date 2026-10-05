"""Unit tests for chronos.chronos2.vllm.protocol.validation."""

import pytest

from chronos.chronos2.vllm.protocol.forecast import ForecastParameters, TimeSeriesInput
from chronos.chronos2.vllm.protocol.validation import (
    validate_cross_series,
    validate_quantile_levels,
    validate_single_series_covariates,
    validate_start_timestamp,
    validate_target,
)


class TestValidateTarget:
    """Tests for validate_target."""

    def test_valid_univariate(self):
        result = validate_target([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(result) == 5

    def test_valid_multivariate(self):
        result = validate_target([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
        assert len(result) == 2

    def test_too_short_univariate(self):
        with pytest.raises(ValueError, match="at least 5"):
            validate_target([1.0, 2.0])

    def test_too_short_multivariate(self):
        with pytest.raises(ValueError, match="at least 5"):
            validate_target([[1.0, 2.0], [3.0, 4.0]])

    def test_inconsistent_multivariate_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            validate_target([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0]])


class TestValidateStartTimestamp:
    """Tests for validate_start_timestamp."""

    def test_none(self):
        assert validate_start_timestamp(None) is None

    def test_valid_date(self):
        assert validate_start_timestamp("2024-01-01") == "2024-01-01"

    def test_valid_datetime(self):
        assert validate_start_timestamp("2024-01-01T12:00:00") == "2024-01-01T12:00:00"

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid start timestamp"):
            validate_start_timestamp("not-a-date")


class TestValidateQuantileLevels:
    """Tests for validate_quantile_levels."""

    def test_valid(self):
        result = validate_quantile_levels([0.1, 0.5, 0.9])
        assert result == [0.1, 0.5, 0.9]

    def test_zero_invalid(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_quantile_levels([0.0, 0.5])

    def test_one_invalid(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_quantile_levels([0.5, 1.0])

    def test_negative_invalid(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_quantile_levels([-0.1])


class TestValidateSingleSeriesCovariates:
    """Tests for validate_single_series_covariates."""

    def test_no_covariates(self):
        validate_single_series_covariates([1.0, 2.0, 3.0, 4.0, 5.0], None, None)

    def test_valid_past_covariates(self):
        target = [1.0, 2.0, 3.0, 4.0, 5.0]
        past = {"temp": [10.0, 20.0, 30.0, 40.0, 50.0]}
        validate_single_series_covariates(target, past, None)

    def test_future_without_past_raises(self):
        target = [1.0, 2.0, 3.0, 4.0, 5.0]
        future = {"temp": [60.0]}
        with pytest.raises(ValueError, match="together"):
            validate_single_series_covariates(target, None, future)

    def test_future_key_not_in_past_raises(self):
        target = [1.0, 2.0, 3.0, 4.0, 5.0]
        past = {"temp": [10.0, 20.0, 30.0, 40.0, 50.0]}
        future = {"wind": [1.0]}
        with pytest.raises(ValueError, match="not in"):
            validate_single_series_covariates(target, past, future)

    def test_past_length_mismatch(self):
        target = [1.0, 2.0, 3.0, 4.0, 5.0]
        past = {"temp": [10.0, 20.0]}
        with pytest.raises(ValueError, match="must match target length"):
            validate_single_series_covariates(target, past, None)

    def test_target_name_forbidden(self):
        target = [1.0, 2.0, 3.0, 4.0, 5.0]
        past = {"target": [10.0, 20.0, 30.0, 40.0, 50.0]}
        with pytest.raises(ValueError, match="not allowed"):
            validate_single_series_covariates(target, past, None)


class TestValidateCrossSeries:
    """Tests for validate_cross_series."""

    def _make_input(self, target=None, item_id=None, start=None, past_cov=None, future_cov=None):
        return TimeSeriesInput(
            target=target or [1.0, 2.0, 3.0, 4.0, 5.0],
            item_id=item_id,
            start=start,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )

    def test_valid_simple(self):
        inputs = [self._make_input(), self._make_input()]
        params = ForecastParameters(prediction_length=3)
        validate_cross_series(inputs, params)

    def test_inconsistent_item_ids(self):
        inputs = [self._make_input(item_id="a"), self._make_input(item_id=None)]
        params = ForecastParameters()
        with pytest.raises(ValueError, match="item_id"):
            validate_cross_series(inputs, params)

    def test_duplicate_item_ids(self):
        inputs = [self._make_input(item_id="a"), self._make_input(item_id="a")]
        params = ForecastParameters()
        with pytest.raises(ValueError, match="unique"):
            validate_cross_series(inputs, params)

    def test_start_without_freq(self):
        inputs = [self._make_input(start="2024-01-01")]
        params = ForecastParameters(prediction_length=1)
        with pytest.raises(ValueError, match="freq"):
            validate_cross_series(inputs, params)

    def test_freq_without_start(self):
        inputs = [self._make_input()]
        params = ForecastParameters(prediction_length=1, freq="D")
        with pytest.raises(ValueError, match="start"):
            validate_cross_series(inputs, params)

    def test_future_cov_length_mismatch(self):
        inputs = [
            self._make_input(
                past_cov={"temp": [1.0, 2.0, 3.0, 4.0, 5.0]},
                future_cov={"temp": [10.0, 20.0]},  # length 2 != prediction_length 3
            )
        ]
        params = ForecastParameters(prediction_length=3)
        with pytest.raises(ValueError, match="prediction_length"):
            validate_cross_series(inputs, params)
