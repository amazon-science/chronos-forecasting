"""Tests for Chronos-2 plugin validation logic."""

import pytest

from chronos.chronos2.vllm.protocol.forecast import (
    ForecastParameters,
    ForecastRequest,
    TimeSeriesInput,
)

# ============================================================================
# TimeSeriesInput — per-input validation
# ============================================================================


class TestTimeSeriesInputTargetValidation:
    """Tests for target field validation."""

    def test_valid_univariate_target(self):
        ts = TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(ts.target) == 5

    def test_valid_multivariate_target(self):
        ts = TimeSeriesInput(target=[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
        assert len(ts.target) == 2

    def test_univariate_target_too_short(self):
        with pytest.raises(ValueError, match="at least 5 observations"):
            TimeSeriesInput(target=[1.0, 2.0, 3.0])

    def test_multivariate_target_too_short(self):
        with pytest.raises(ValueError, match="at least 5 observations"):
            TimeSeriesInput(target=[[1.0, 2.0], [3.0, 4.0]])

    def test_multivariate_inconsistent_lengths(self):
        with pytest.raises(ValueError, match="All target dimensions must have same length"):
            TimeSeriesInput(target=[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])


class TestTimeSeriesInputStartTimestamp:
    """Tests for start timestamp validation."""

    def test_valid_date_format(self):
        ts = TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0], start="2024-01-01")
        assert ts.start == "2024-01-01"

    def test_valid_datetime_format(self):
        ts = TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0], start="2024-01-01T12:00:00")
        assert ts.start == "2024-01-01T12:00:00"

    def test_invalid_timestamp_format(self):
        with pytest.raises(ValueError, match="Invalid start timestamp format"):
            TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0], start="not-a-date")


class TestTimeSeriesInputCovariateValidation:
    """Tests for per-input covariate validation."""

    def test_valid_past_covariates(self):
        ts = TimeSeriesInput(
            target=[1.0, 2.0, 3.0, 4.0, 5.0],
            past_covariates={"feature_a": [1.0, 2.0, 3.0, 4.0, 5.0]},
        )
        assert ts.past_covariates is not None

    def test_past_covariate_length_mismatch(self):
        with pytest.raises(ValueError, match="must match target length"):
            TimeSeriesInput(
                target=[1.0, 2.0, 3.0, 4.0, 5.0],
                past_covariates={"feature_a": [1.0, 2.0, 3.0]},
            )

    def test_target_name_in_past_covariates_rejected(self):
        with pytest.raises(ValueError, match="Covariate with name 'target' is not allowed"):
            TimeSeriesInput(
                target=[1.0, 2.0, 3.0, 4.0, 5.0],
                past_covariates={"target": [1.0, 2.0, 3.0, 4.0, 5.0]},
            )

    def test_target_name_in_future_covariates_rejected(self):
        with pytest.raises(ValueError, match="Covariate with name 'target' is not allowed"):
            TimeSeriesInput(
                target=[1.0, 2.0, 3.0, 4.0, 5.0],
                past_covariates={"feature_a": [1.0, 2.0, 3.0, 4.0, 5.0]},
                future_covariates={"target": [1.0]},
            )

    def test_future_covariates_without_past_rejected(self):
        with pytest.raises(
            ValueError, match="Both 'past_covariates' and 'future_covariates' must be provided"
        ):
            TimeSeriesInput(
                target=[1.0, 2.0, 3.0, 4.0, 5.0],
                future_covariates={"feature_a": [1.0]},
            )

    def test_future_covariate_keys_must_be_subset_of_past(self):
        with pytest.raises(
            ValueError, match="All future covariate keys must be present in past covariates"
        ):
            TimeSeriesInput(
                target=[1.0, 2.0, 3.0, 4.0, 5.0],
                past_covariates={"feature_a": [1.0, 2.0, 3.0, 4.0, 5.0]},
                future_covariates={"feature_b": [1.0]},
            )

    def test_future_covariate_keys_subset_is_valid(self):
        """Past covariates may have more keys than future — that's fine."""
        ts = TimeSeriesInput(
            target=[1.0, 2.0, 3.0, 4.0, 5.0],
            past_covariates={
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_b": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
            future_covariates={"feature_a": [1.0]},
        )
        assert ts.future_covariates is not None

    def test_past_only_covariates_valid(self):
        """Having past_covariates without future_covariates is acceptable."""
        ts = TimeSeriesInput(
            target=[1.0, 2.0, 3.0, 4.0, 5.0],
            past_covariates={"feature_a": [1.0, 2.0, 3.0, 4.0, 5.0]},
        )
        assert ts.past_covariates is not None
        assert ts.future_covariates is None


# ============================================================================
# ForecastParameters validation
# ============================================================================


class TestForecastParameters:
    """Tests for ForecastParameters validation."""

    def test_default_values(self):
        params = ForecastParameters()
        assert params.prediction_length == 1
        assert params.quantile_levels == [0.1, 0.5, 0.9]
        assert params.freq is None
        assert params.batch_size == 256
        assert params.cross_learning is False

    def test_prediction_length_too_large(self):
        with pytest.raises(ValueError):
            ForecastParameters(prediction_length=2000)

    def test_prediction_length_zero(self):
        with pytest.raises(ValueError):
            ForecastParameters(prediction_length=0)

    def test_quantile_out_of_range(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            ForecastParameters(quantile_levels=[0.0, 0.5])

    def test_quantile_one(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            ForecastParameters(quantile_levels=[0.5, 1.0])


# ============================================================================
# Cross-series validation (via validation module)
# ============================================================================


class TestCrossSeriesValidation:
    """Tests for cross-series validation logic in validation module."""

    @staticmethod
    def _validate(inputs, parameters=None):
        """Helper that calls validate_cross_series from the validation module."""
        from chronos.chronos2.vllm.protocol.validation import validate_cross_series

        if parameters is None:
            parameters = ForecastParameters()
        validate_cross_series(inputs, parameters)

    def test_item_id_all_provided(self):
        """All item_ids present — valid."""
        inputs = [
            TimeSeriesInput(target=[1.0] * 5, item_id="A"),
            TimeSeriesInput(target=[2.0] * 5, item_id="B"),
        ]
        self._validate(inputs)  # should not raise

    def test_item_id_none_provided(self):
        """No item_ids at all — valid."""
        inputs = [
            TimeSeriesInput(target=[1.0] * 5),
            TimeSeriesInput(target=[2.0] * 5),
        ]
        self._validate(inputs)  # should not raise

    def test_item_id_partial(self):
        """Some have item_id, some don't — invalid."""
        inputs = [
            TimeSeriesInput(target=[1.0] * 5, item_id="A"),
            TimeSeriesInput(target=[2.0] * 5),
        ]
        with pytest.raises(ValueError, match="item_id.*provided for all time series"):
            self._validate(inputs)

    def test_item_id_not_unique(self):
        """Duplicate item_ids — invalid."""
        inputs = [
            TimeSeriesInput(target=[1.0] * 5, item_id="A"),
            TimeSeriesInput(target=[2.0] * 5, item_id="A"),
        ]
        with pytest.raises(ValueError, match="item_id.*must be unique"):
            self._validate(inputs)

    def test_start_all_provided_with_freq(self):
        """All starts present with freq — valid."""
        inputs = [
            TimeSeriesInput(target=[1.0] * 5, start="2024-01-01"),
            TimeSeriesInput(target=[2.0] * 5, start="2024-02-01"),
        ]
        params = ForecastParameters(freq="D")
        self._validate(inputs, params)  # should not raise

    def test_start_partial(self):
        """Some have start, some don't — invalid."""
        inputs = [
            TimeSeriesInput(target=[1.0] * 5, start="2024-01-01"),
            TimeSeriesInput(target=[2.0] * 5),
        ]
        params = ForecastParameters(freq="D")
        with pytest.raises(ValueError, match="start.*provided for all time series"):
            self._validate(inputs, params)

    def test_start_without_freq(self):
        """start provided but freq missing — invalid."""
        inputs = [
            TimeSeriesInput(target=[1.0] * 5, start="2024-01-01"),
        ]
        params = ForecastParameters()  # freq is None
        with pytest.raises(ValueError, match="freq.*must also be provided"):
            self._validate(inputs, params)

    def test_freq_without_start(self):
        """freq provided but no start on inputs — invalid."""
        inputs = [
            TimeSeriesInput(target=[1.0] * 5),
        ]
        params = ForecastParameters(freq="D")
        with pytest.raises(ValueError, match="start.*must also be provided"):
            self._validate(inputs, params)

    def test_covariate_keys_consistent(self):
        """Same covariate keys on all series — valid."""
        inputs = [
            TimeSeriesInput(
                target=[1.0] * 5,
                past_covariates={"feat": [1.0] * 5},
            ),
            TimeSeriesInput(
                target=[2.0] * 5,
                past_covariates={"feat": [2.0] * 5},
            ),
        ]
        self._validate(inputs)  # should not raise

    def test_covariate_keys_inconsistent(self):
        """Different covariate keys across series — invalid."""
        inputs = [
            TimeSeriesInput(
                target=[1.0] * 5,
                past_covariates={"feat_a": [1.0] * 5},
            ),
            TimeSeriesInput(
                target=[2.0] * 5,
                past_covariates={"feat_b": [2.0] * 5},
            ),
        ]
        with pytest.raises(ValueError, match="same covariates should be provided"):
            self._validate(inputs)

    def test_covariate_some_have_none(self):
        """One series has covariates, other doesn't — invalid."""
        inputs = [
            TimeSeriesInput(
                target=[1.0] * 5,
                past_covariates={"feat": [1.0] * 5},
            ),
            TimeSeriesInput(target=[2.0] * 5),
        ]
        with pytest.raises(ValueError, match="same covariates should be provided"):
            self._validate(inputs)

    def test_future_covariate_length_matches_prediction_length(self):
        """future_covariates length equals prediction_length — valid."""
        inputs = [
            TimeSeriesInput(
                target=[1.0] * 5,
                past_covariates={"feat": [1.0] * 5},
                future_covariates={"feat": [1.0, 2.0, 3.0]},
            ),
        ]
        params = ForecastParameters(prediction_length=3)
        self._validate(inputs, params)  # should not raise

    def test_future_covariate_length_mismatch(self):
        """future_covariates length != prediction_length — invalid."""
        inputs = [
            TimeSeriesInput(
                target=[1.0] * 5,
                past_covariates={"feat": [1.0] * 5},
                future_covariates={"feat": [1.0, 2.0]},
            ),
        ]
        params = ForecastParameters(prediction_length=3)
        with pytest.raises(ValueError, match="must equal prediction_length"):
            self._validate(inputs, params)


# ============================================================================
# ForecastRequest — end-to-end request validation
# ============================================================================


class TestForecastRequest:
    """Tests for ForecastRequest end-to-end validation."""

    def test_valid_minimal_request(self):
        req = ForecastRequest(
            model="chronos-v2",
            data={"inputs": [{"target": [1.0, 2.0, 3.0, 4.0, 5.0]}]},
        )
        assert req.data["inputs"] is not None

    def test_empty_inputs_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            ForecastRequest(model="chronos-v2", data={"inputs": []})

    def test_too_many_inputs_rejected(self):
        many_inputs = [{"target": [1.0] * 5} for _ in range(1025)]
        with pytest.raises(ValueError, match="at most 1024"):
            ForecastRequest(model="chronos-v2", data={"inputs": many_inputs})

    def test_missing_inputs_rejected(self):
        with pytest.raises(ValueError, match="inputs"):
            ForecastRequest(model="chronos-v2", data={"parameters": {}})

    def test_future_covariate_length_cross_validated(self):
        """ForecastRequest validates future_covariates against prediction_length."""
        with pytest.raises(ValueError, match="prediction_length"):
            ForecastRequest(
                model="chronos-v2",
                data={
                    "inputs": [
                        {
                            "target": [1.0] * 5,
                            "past_covariates": {"feat": [1.0] * 5},
                            "future_covariates": {"feat": [1.0, 2.0]},
                        }
                    ],
                    "parameters": {"prediction_length": 5},
                },
            )
