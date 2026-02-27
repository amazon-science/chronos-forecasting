"""Data preparation for Chronos-2 model input tensors.

Converts validated TimeSeriesInput objects into batched tensors
ready for the model by delegating to ``chronos.chronos2.dataset``.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from chronos.chronos2.vllm.protocol.forecast import ForecastParameters, TimeSeriesInput


@dataclass
class PreparedBatch:
    """A single batch of prepared model input tensors."""

    context: torch.Tensor  # (batch_rows, padded_ctx_len)
    future_covariates: torch.Tensor  # (batch_rows, prediction_length)
    group_ids: torch.Tensor  # (batch_rows,)
    num_output_patches: int
    target_idx_ranges: list[tuple[int, int]]  # per-input-series (start, end) in this batch


@dataclass
class PreparedRequest:
    """All batches and metadata for a single forecast request."""

    batches: list[PreparedBatch]
    item_ids: list[str | None]
    parameters: ForecastParameters


def _timeseries_input_to_dict(ts: TimeSeriesInput) -> dict[str, Any]:
    """Convert a Pydantic TimeSeriesInput to the dict format expected by Chronos2Dataset.

    The dataset expects:
    - ``target``: np.ndarray of shape (history_length,) or (n_variates, history_length)
    - ``past_covariates``: dict[str, np.ndarray]
    - ``future_covariates``: dict[str, np.ndarray]
    """
    target = np.array(ts.target, dtype=np.float32)

    entry: dict[str, Any] = {"target": target}

    if ts.past_covariates:
        past_covs: dict[str, np.ndarray] = {}
        for key, vals in ts.past_covariates.items():
            arr = np.array([v if v is not None else np.nan for v in vals])
            # Keep string arrays as object dtype for categorical encoding
            if any(isinstance(v, str) for v in vals if v is not None):
                arr = np.array([str(v) if v is not None else "nan" for v in vals])
            else:
                arr = arr.astype(np.float32)
            past_covs[key] = arr
        entry["past_covariates"] = past_covs

    if ts.future_covariates:
        future_covs: dict[str, np.ndarray] = {}
        for key, vals in ts.future_covariates.items():
            arr = np.array([v if v is not None else np.nan for v in vals])
            if any(isinstance(v, str) for v in vals if v is not None):
                arr = np.array([str(v) if v is not None else "nan" for v in vals])
            else:
                arr = arr.astype(np.float32)
            future_covs[key] = arr
        entry["future_covariates"] = future_covs

    return entry


def prepare_request(
    inputs: list[TimeSeriesInput],
    parameters: ForecastParameters,
    context_length: int = 8192,
    output_patch_size: int = 16,
) -> PreparedRequest:
    """Convert validated inputs into batched model-ready tensors.

    Delegates to ``Chronos2Dataset`` in TEST mode for data preparation,
    covariate encoding, batching, and group ID construction.

    Args:
        inputs: validated time series inputs
        parameters: forecast parameters
        context_length: maximum context length (from model config)
        output_patch_size: model's output patch size (from model config)

    Returns:
        PreparedRequest with batched tensors and metadata
    """
    # Convert Pydantic models to dicts for Chronos2Dataset
    raw_inputs = [_timeseries_input_to_dict(ts) for ts in inputs]
    item_ids = [ts.item_id for ts in inputs]

    pred_len = parameters.prediction_length

    # Use Chronos2Dataset in TEST mode — handles validation, covariate encoding,
    # left-padding, and group_id construction. We use a very large batch_size
    # to always produce exactly one batch; row-level chunking (if needed for
    # memory) can be handled by the caller or the model's forward pass.
    dataset = Chronos2Dataset(
        inputs=raw_inputs,
        context_length=context_length,
        prediction_length=pred_len,
        batch_size=2**31 - 1,  # large enough to fit all series in one batch
        output_patch_size=output_patch_size,
        mode=DatasetMode.TEST,
        convert_inputs=True,
    )

    batches: list[PreparedBatch] = []
    for batch_dict in dataset:
        group_ids = batch_dict["group_ids"]
        if parameters.cross_learning:
            group_ids = torch.zeros_like(group_ids)

        batches.append(
            PreparedBatch(
                context=batch_dict["context"],
                future_covariates=batch_dict["future_covariates"],
                group_ids=group_ids,
                num_output_patches=batch_dict["num_output_patches"],
                target_idx_ranges=batch_dict["target_idx_ranges"],
            )
        )

    return PreparedRequest(
        batches=batches,
        item_ids=item_ids,
        parameters=parameters,
    )