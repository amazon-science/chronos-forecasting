"""Unit tests for chronos.chronos2.vllm.protocol.data_prep."""

import pytest

torch = pytest.importorskip("torch")

from chronos.chronos2.vllm.protocol.data_prep import (  # noqa: E402
    PreparedRequest,
    prepare_request,
)
from chronos.chronos2.vllm.protocol.forecast import (  # noqa: E402
    ForecastParameters,
    TimeSeriesInput,
)


class TestPrepareRequestUnivariate:
    """Tests for prepare_request with univariate inputs."""

    def test_single_series(self):
        inputs = [TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0])]
        params = ForecastParameters(prediction_length=3)
        result = prepare_request(inputs, params)

        assert isinstance(result, PreparedRequest)
        assert len(result.batches) >= 1
        assert len(result.item_ids) == 1
        assert result.item_ids[0] is None

        batch = result.batches[0]
        assert batch.context.shape[0] == 1
        assert batch.context.shape[1] == 5
        assert batch.group_ids.shape == (1,)
        assert batch.target_idx_ranges == [(0, 1)]

    def test_multiple_series(self):
        inputs = [
            TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0], item_id="a"),
            TimeSeriesInput(target=[6.0, 7.0, 8.0, 9.0, 10.0], item_id="b"),
        ]
        params = ForecastParameters(prediction_length=2)
        result = prepare_request(inputs, params)

        # Both series should fit in one batch with default batch_size
        assert len(result.batches) >= 1
        # Count total rows across all batches
        total_rows = sum(b.context.shape[0] for b in result.batches)
        assert total_rows == 2
        assert result.item_ids == ["a", "b"]

    def test_different_lengths_padded(self):
        inputs = [
            TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0]),
            TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        ]
        params = ForecastParameters(prediction_length=1)
        result = prepare_request(inputs, params)

        batch = result.batches[0]
        # Padded to max length = 10
        assert batch.context.shape == (2, 10)
        # First series: right-aligned, left NaN-padded
        assert torch.isnan(batch.context[0, :5]).all()
        assert not torch.isnan(batch.context[0, 5:]).any()

    def test_truncation_to_context_length(self):
        long_target = list(range(100))
        inputs = [TimeSeriesInput(target=long_target)]
        params = ForecastParameters(prediction_length=1)
        result = prepare_request(inputs, params, context_length=20)

        batch = result.batches[0]
        assert batch.context.shape == (1, 20)

    def test_future_covariates_nan_for_targets(self):
        inputs = [TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0])]
        params = ForecastParameters(prediction_length=3)
        result = prepare_request(inputs, params)

        # Target rows should have NaN future covariates
        assert torch.isnan(result.batches[0].future_covariates[0]).all()


class TestPrepareRequestMultivariate:
    """Tests for prepare_request with multivariate inputs."""

    def test_bivariate(self):
        inputs = [TimeSeriesInput(target=[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])]
        params = ForecastParameters(prediction_length=2)
        result = prepare_request(inputs, params)

        batch = result.batches[0]
        # 2 variates = 2 rows
        assert batch.context.shape == (2, 5)
        assert batch.target_idx_ranges == [(0, 2)]
        assert batch.group_ids.tolist() == [0, 0]


class TestPrepareRequestCovariates:
    """Tests for prepare_request with covariates."""

    def test_past_covariates(self):
        inputs = [
            TimeSeriesInput(
                target=[1.0, 2.0, 3.0, 4.0, 5.0],
                past_covariates={"temp": [10.0, 20.0, 30.0, 40.0, 50.0]},
            )
        ]
        params = ForecastParameters(prediction_length=2)
        result = prepare_request(inputs, params)

        batch = result.batches[0]
        # 1 target + 1 covariate = 2 rows
        assert batch.context.shape == (2, 5)
        # Only target row is in target_idx_ranges
        assert batch.target_idx_ranges == [(0, 1)]
        assert batch.group_ids.tolist() == [0, 0]

    def test_future_covariates(self):
        inputs = [
            TimeSeriesInput(
                target=[1.0, 2.0, 3.0, 4.0, 5.0],
                past_covariates={"temp": [10.0, 20.0, 30.0, 40.0, 50.0]},
                future_covariates={"temp": [60.0, 70.0]},
            )
        ]
        params = ForecastParameters(prediction_length=2)
        result = prepare_request(inputs, params)

        batch = result.batches[0]
        # Target row: NaN future covariates
        assert torch.isnan(batch.future_covariates[0]).all()
        # Covariate row: actual future values
        assert batch.future_covariates[1, 0].item() == 60.0
        assert batch.future_covariates[1, 1].item() == 70.0


class TestPrepareRequestCrossLearning:
    """Tests for cross-learning mode."""

    def test_cross_learning_zeros_group_ids(self):
        inputs = [
            TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0]),
            TimeSeriesInput(target=[6.0, 7.0, 8.0, 9.0, 10.0]),
        ]
        params = ForecastParameters(prediction_length=1, cross_learning=True)
        result = prepare_request(inputs, params)

        assert result.batches[0].group_ids.tolist() == [0, 0]

    def test_no_cross_learning_distinct_group_ids(self):
        inputs = [
            TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0]),
            TimeSeriesInput(target=[6.0, 7.0, 8.0, 9.0, 10.0]),
        ]
        params = ForecastParameters(prediction_length=1, cross_learning=False)
        result = prepare_request(inputs, params)

        assert result.batches[0].group_ids.tolist() == [0, 1]


class TestPrepareRequestNanCovariates:
    """Tests for prepare_request handling of NaN-sanitized covariates."""

    def test_numeric_covariates_with_none(self):
        """Numeric covariates with None values should become NaN in tensors."""
        inputs = [
            TimeSeriesInput(
                target=[1.0, 2.0, 3.0, 4.0, 5.0],
                past_covariates={"temp": [10.0, None, 30.0, None, 50.0]},
                future_covariates={"temp": [None, 70.0, None]},
            )
        ]
        params = ForecastParameters(prediction_length=3)
        result = prepare_request(inputs, params)

        batch = result.batches[0]
        # 1 target + 1 covariate = 2 rows
        assert batch.context.shape == (2, 5)
        # Covariate row: None → NaN in context tensor
        assert torch.isnan(batch.context[1, 1])
        assert torch.isnan(batch.context[1, 3])
        assert batch.context[1, 0].item() == 10.0
        assert batch.context[1, 2].item() == 30.0
        # Future covariates: None → NaN
        assert torch.isnan(batch.future_covariates[1, 0])
        assert batch.future_covariates[1, 1].item() == 70.0

    def test_string_covariates_encoded_in_tensors(self):
        """String covariates (like holidays) should be encoded and included in tensors."""
        inputs = [
            TimeSeriesInput(
                target=[1.0, 2.0, 3.0, 4.0, 5.0],
                past_covariates={"holiday": ["A", None, "B", None, "C"]},
            )
        ]
        params = ForecastParameters(prediction_length=2)
        result = prepare_request(inputs, params)

        batch = result.batches[0]
        # 1 target + 1 encoded categorical covariate = 2 rows
        assert batch.context.shape == (2, 5)
        # Encoded values should be finite floats (not NaN) for non-None entries
        assert not torch.isnan(batch.context[1, 0])  # "A" encoded
        assert not torch.isnan(batch.context[1, 2])  # "B" encoded
        assert not torch.isnan(batch.context[1, 4])  # "C" encoded

    def test_categorical_with_future(self):
        """Categorical covariates with future values should be encoded consistently."""
        inputs = [
            TimeSeriesInput(
                target=[1.0, 2.0, 3.0, 4.0, 5.0],
                past_covariates={"holiday": ["A", "B", "A", "B", "A"]},
                future_covariates={"holiday": ["A", "B"]},
            )
        ]
        params = ForecastParameters(prediction_length=2)
        result = prepare_request(inputs, params)

        batch = result.batches[0]
        # 1 target + 1 encoded categorical = 2 rows
        assert batch.context.shape == (2, 5)
        # Future covariates should be filled for the categorical row
        assert not torch.isnan(batch.future_covariates[1, 0])
        assert not torch.isnan(batch.future_covariates[1, 1])


class TestPrepareRequestBatching:
    """Tests for batch output — always a single batch (chunking deferred to model forward)."""

    def test_always_single_batch(self):
        """prepare_request always returns one batch regardless of batch_size parameter."""
        inputs = [TimeSeriesInput(target=[float(i)] * 5) for i in range(5)]
        params = ForecastParameters(prediction_length=1, batch_size=2)
        result = prepare_request(inputs, params)

        # Always 1 batch — model forward() handles chunking if needed
        assert len(result.batches) == 1
        assert result.batches[0].context.shape[0] == 5
        assert len(result.batches[0].target_idx_ranges) == 5

    def test_single_series_single_batch(self):
        inputs = [TimeSeriesInput(target=[1.0, 2.0, 3.0, 4.0, 5.0])]
        params = ForecastParameters(prediction_length=1, batch_size=256)
        result = prepare_request(inputs, params)

        assert len(result.batches) == 1
        assert result.batches[0].context.shape[0] == 1