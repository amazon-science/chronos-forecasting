"""Unit tests for chronos.chronos2.vllm.utils.quantiles and chronos.utils.interpolate_quantiles."""

import pytest

torch = pytest.importorskip("torch")

from chronos.chronos2.vllm.utils.quantiles import select_quantiles  # noqa: E402
from chronos.utils import interpolate_quantiles  # noqa: E402

MODEL_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TestSelectQuantiles:
    """Tests for select_quantiles."""

    def test_subset_selection(self):
        """Requesting a subset of model quantiles should do direct indexing."""
        # Shape: (1, 9_quantiles, 4_horizon)
        pred = torch.randn(1, len(MODEL_QUANTILES), 4)
        quantiles, mean = select_quantiles([pred], MODEL_QUANTILES, [0.1, 0.5, 0.9])

        assert len(quantiles) == 1
        assert len(mean) == 1
        # After swap: (1, 4, 3)
        assert quantiles[0].shape == (1, 4, 3)
        # Mean is median (0.5 = index 4)
        assert mean[0].shape == (1, 4)

    def test_direct_selection_values(self):
        """Verify selected values match expected indices."""
        pred = torch.arange(36, dtype=torch.float32).reshape(1, 9, 4)
        quantiles, mean = select_quantiles([pred], MODEL_QUANTILES, [0.1, 0.9])

        # q=0.1 is index 0, q=0.9 is index 8
        # After swap: pred[..., h, q] → values at (h, q_idx)
        q = quantiles[0]  # (1, 4, 2)
        assert q[0, 0, 0].item() == 0.0  # q=0.1, h=0 → row 0, col 0
        assert q[0, 0, 1].item() == 32.0  # q=0.9, h=0 → row 8, col 0

    def test_interpolation_triggered(self):
        """Non-subset quantile levels should trigger interpolation."""
        pred = torch.randn(1, len(MODEL_QUANTILES), 4)
        quantiles, mean = select_quantiles([pred], MODEL_QUANTILES, [0.15, 0.5])

        assert quantiles[0].shape == (1, 4, 2)

    def test_multiple_predictions(self):
        """Works with multiple prediction tensors."""
        preds = [torch.randn(1, 9, 4) for _ in range(3)]
        quantiles, mean = select_quantiles(preds, MODEL_QUANTILES, [0.5])

        assert len(quantiles) == 3
        assert len(mean) == 3

    def test_multivariate(self):
        """Works with multivariate predictions (2 variates)."""
        pred = torch.randn(2, len(MODEL_QUANTILES), 4)
        quantiles, mean = select_quantiles([pred], MODEL_QUANTILES, [0.1, 0.5, 0.9])

        assert quantiles[0].shape == (2, 4, 3)
        assert mean[0].shape == (2, 4)


class TestInterpolateQuantiles:
    """Tests for chronos.utils.interpolate_quantiles."""

    def test_exact_match(self):
        """Exact match should return the original values."""
        values = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3) for levels [0.1, 0.5, 0.9]
        result = interpolate_quantiles([0.5], [0.1, 0.5, 0.9], values)
        assert result.shape == (1, 1)
        assert result[0, 0].item() == 2.0

    def test_midpoint_interpolation(self):
        """Midpoint between two levels should be the average."""
        values = torch.tensor([[0.0, 10.0]])  # (1, 2) for levels [0.2, 0.8]
        result = interpolate_quantiles([0.5], [0.2, 0.8], values)
        assert abs(result[0, 0].item() - 5.0) < 1e-5

    def test_clamping(self):
        """Query below/above model range should clamp to boundary."""
        values = torch.tensor([[1.0, 2.0, 3.0]])  # levels [0.1, 0.5, 0.9]
        low = interpolate_quantiles([0.01], [0.1, 0.5, 0.9], values)
        high = interpolate_quantiles([0.99], [0.1, 0.5, 0.9], values)
        assert low[0, 0].item() == pytest.approx(1.0, abs=1e-5)  # clamped to 0.1
        assert high[0, 0].item() == pytest.approx(3.0, abs=1e-5)  # clamped to 0.9

    def test_multiple_queries(self):
        """Multiple query levels should produce correct shape."""
        values = torch.tensor([[1.0, 2.0, 3.0]])
        result = interpolate_quantiles([0.1, 0.3, 0.5, 0.9], [0.1, 0.5, 0.9], values)
        assert result.shape == (1, 4)

    def test_batch_dimension(self):
        """Works with batch dimension."""
        values = torch.randn(5, 9)  # 5 time steps, 9 quantiles
        result = interpolate_quantiles([0.25], MODEL_QUANTILES, values)
        assert result.shape == (5, 1)