"""Quantile selection and interpolation utilities."""

import torch

from chronos.utils import interpolate_quantiles


def select_quantiles(
    predictions: list[torch.Tensor],
    model_quantiles: list[float],
    requested_levels: list[float],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Select or interpolate quantiles from model output.

    Args:
        predictions: list of tensors, each shape (..., num_quantiles, horizon)
        model_quantiles: the quantile levels the model was trained on (sorted ascending)
        requested_levels: the quantile levels the caller wants

    Returns:
        (quantiles, mean) where:
          - quantiles: list of tensors, each shape (..., horizon, num_requested_quantiles)
          - mean: list of tensors, each shape (..., horizon) — the median
    """
    # Swap quantile and time axes: [... q h] -> [... h q]
    swapped = [pred.permute(*range(pred.ndim - 2), -1, -2) for pred in predictions]

    if set(requested_levels).issubset(model_quantiles):
        indices = [model_quantiles.index(q) for q in requested_levels]
        quantiles = [pred[..., indices] for pred in swapped]
    else:
        quantiles = [
            interpolate_quantiles(requested_levels, model_quantiles, pred) for pred in swapped
        ]

    # Median as mean (Chronos-2 convention)
    median_idx = model_quantiles.index(0.5) if 0.5 in model_quantiles else len(model_quantiles) // 2
    mean = [pred[..., median_idx] for pred in swapped]

    return quantiles, mean