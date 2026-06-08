"""Common helper functions for Chronos-2 vLLM plugin."""

from typing import Any

import torch


def tensor_to_list(tensor: torch.Tensor) -> list[float] | list[list[float]]:
    """Convert 2-D tensor to list (univariate) or list of lists (multivariate).

    Args:
        tensor: shape (n_variates, horizon)

    Returns:
        list[float] if univariate (n_variates == 1), else list[list[float]]
    """
    assert tensor.ndim == 2
    return tensor[0].tolist() if tensor.shape[0] == 1 else [row.tolist() for row in tensor]


def empty_prediction(
    prediction_length: int,
    quantile_levels: list[float],
) -> dict[str, Any]:
    """Return a zero-filled prediction dict for error cases.

    Args:
        prediction_length: number of forecast steps
        quantile_levels: list of quantile levels to include
    """
    zeros = [0.0] * prediction_length
    pred: dict[str, Any] = {"mean": zeros}
    for q in quantile_levels:
        pred[str(q)] = zeros
    return pred
