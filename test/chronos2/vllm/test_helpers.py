"""Unit tests for chronos.chronos2.vllm.utils.helpers."""

import pytest

torch = pytest.importorskip("torch")

from chronos.chronos2.vllm.utils.helpers import empty_prediction, tensor_to_list  # noqa: E402


class TestTensorToList:
    """Tests for tensor_to_list."""

    def test_univariate(self):
        t = torch.tensor([[1.0, 2.0, 3.0]])
        result = tensor_to_list(t)
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)
        assert isinstance(result[0], float)

    def test_multivariate(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = tensor_to_list(t)
        assert result == [[1.0, 2.0], [3.0, 4.0]]
        assert isinstance(result[0], list)

    def test_single_value(self):
        t = torch.tensor([[42.0]])
        assert tensor_to_list(t) == [42.0]

    def test_wrong_dims_raises(self):
        with pytest.raises(AssertionError):
            tensor_to_list(torch.tensor([1.0, 2.0]))  # 1-D


class TestEmptyPrediction:
    """Tests for empty_prediction."""

    def test_basic(self):
        pred = empty_prediction(3, [0.1, 0.5, 0.9])
        assert pred["mean"] == [0.0, 0.0, 0.0]
        assert pred["0.1"] == [0.0, 0.0, 0.0]
        assert pred["0.5"] == [0.0, 0.0, 0.0]
        assert pred["0.9"] == [0.0, 0.0, 0.0]

    def test_single_quantile(self):
        pred = empty_prediction(2, [0.5])
        assert "mean" in pred
        assert "0.5" in pred
        assert len(pred["mean"]) == 2

    def test_empty_quantiles(self):
        pred = empty_prediction(1, [])
        assert pred == {"mean": [0.0]}
