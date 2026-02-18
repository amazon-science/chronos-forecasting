# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Preprocessing utilities for Chronos-2 datasets.

This module provides functions to prepare raw time series data for training.
"""

from typing import Any, Iterable, Mapping, Sequence, TypedDict, cast

import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from chronos.chronos2.dataset import DatasetMode

TensorOrArray = torch.Tensor | np.ndarray

# Type alias for raw input format
RawTask = Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]


class PreparedTask(TypedDict):
    """A preprocessed time series task ready for model training/inference."""

    context: np.ndarray  # (n_variates, history_length), float32
    future_covariates: np.ndarray  # (n_variates, prediction_length), float32
    n_targets: int
    n_covariates: int
    n_future_covariates: int


def prepare_single_task(
    task: RawTask,
    idx: int,
    prediction_length: int,
) -> PreparedTask:
    """Validate and prepare a single time series task.

    This is the core preprocessing logic extracted from Chronos2Dataset.
    """
    allowed_keys = {"target", "past_covariates", "future_covariates"}

    keys = set(task.keys())
    if not keys.issubset(allowed_keys):
        raise ValueError(
            f"Found invalid keys in element at index {idx}. "
            f"Allowed keys are {allowed_keys}, but found {keys}"
        )
    if "target" not in keys:
        raise ValueError(f"Element at index {idx} does not contain the required key 'target'")

    # Process target
    task_target = task["target"]
    if isinstance(task_target, torch.Tensor):
        # Convert to float32 first for numpy compatibility (handles bfloat16, etc.)
        task_target = task_target.to(torch.float32).numpy()
    task_target = np.asarray(task_target, dtype=np.float32)

    if task_target.ndim > 2:
        raise ValueError(
            "When the input is a list of dicts, the `target` should either be 1-d with shape (history_length,) "
            f" or 2-d with shape (n_variates, history_length). Found element at index {idx} with shape {tuple(task_target.shape)}."
        )
    history_length = task_target.shape[-1]
    task_target = task_target.reshape(-1, history_length)

    # Process past_covariates
    cat_encoders: dict = {}
    task_past_covariates = task.get("past_covariates", {})
    if not isinstance(task_past_covariates, dict):
        raise ValueError(
            f"Found invalid type for `past_covariates` in element at index {idx}. "
            f"Expected dict, but found {type(task_past_covariates)}"
        )

    task_covariates_keys = sorted(task_past_covariates.keys())

    task_future_covariates = task.get("future_covariates", {})
    if not isinstance(task_future_covariates, dict):
        raise ValueError(
            f"Found invalid type for `future_covariates` in element at index {idx}. "
            f"Expected dict, but found {type(task_future_covariates)}"
        )
    task_future_covariates_keys = sorted(task_future_covariates.keys())

    if not set(task_future_covariates_keys).issubset(task_covariates_keys):
        raise ValueError(
            f"Expected keys in `future_covariates` to be a subset of `past_covariates` "
            f"{task_covariates_keys}, but found {task_future_covariates_keys} in element at index {idx}"
        )

    # Ordered: past-only first, then known-future
    task_past_only_keys = [k for k in task_covariates_keys if k not in task_future_covariates_keys]
    task_ordered_covariate_keys = task_past_only_keys + task_future_covariates_keys

    # Process past covariates
    task_past_covariates_list: list[np.ndarray] = []
    for key in task_ordered_covariate_keys:
        tensor = task_past_covariates[key]
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(torch.float32).numpy()
        tensor = np.asarray(tensor)

        # Encode categorical variates
        if not np.issubdtype(tensor.dtype, np.number):
            if task_target.shape[0] == 1:
                cat_encoder = TargetEncoder(target_type="continuous", smooth=1.0)
                X = tensor.astype(str).reshape(-1, 1)
                y = task_target.reshape(-1)
                mask = np.isfinite(y)
                cat_encoder.fit(X[mask], y[mask])
            else:
                cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
                cat_encoder.fit(tensor.astype(str).reshape(-1, 1))
            tensor = cat_encoder.transform(tensor.astype(str).reshape(-1, 1)).reshape(tensor.shape)
            cat_encoders[key] = cat_encoder

        if tensor.ndim != 1 or len(tensor) != history_length:
            raise ValueError(
                f"Individual `past_covariates` must be 1-d with length {history_length}, "
                f"found: {key} with shape {tensor.shape} in element at index {idx}"
            )
        task_past_covariates_list.append(tensor)

    if task_past_covariates_list:
        task_past_covariates_array = np.stack(task_past_covariates_list, axis=0)
    else:
        task_past_covariates_array = np.zeros((0, history_length), dtype=np.float32)

    # Process future covariates
    task_future_covariates_list: list[np.ndarray] = []
    for key in task_ordered_covariate_keys:
        tensor = task_future_covariates.get(key, np.full(prediction_length, np.nan))
        if tensor is None:
            tensor = np.full(prediction_length, np.nan)
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(torch.float32).numpy()
        tensor = np.asarray(tensor)

        if not np.issubdtype(tensor.dtype, np.number):
            cat_encoder = cat_encoders[key]
            tensor = cat_encoder.transform(tensor.astype(str).reshape(-1, 1)).reshape(tensor.shape)

        if tensor.ndim != 1 or len(tensor) != prediction_length:
            raise ValueError(
                f"Individual `future_covariates` must be 1-d with length {prediction_length}, "
                f"found: {key} with shape {tensor.shape} in element at index {idx}"
            )
        task_future_covariates_list.append(tensor)

    if task_future_covariates_list:
        task_future_covariates_array = np.stack(task_future_covariates_list, axis=0)
    else:
        task_future_covariates_array = np.zeros((0, prediction_length), dtype=np.float32)

    task_future_covariates_target_padding = np.full(
        (task_target.shape[0], prediction_length), np.nan, dtype=np.float32
    )

    context = np.concatenate([task_target, task_past_covariates_array], axis=0).astype(np.float32)
    future_covariates = np.concatenate(
        [task_future_covariates_target_padding, task_future_covariates_array], axis=0
    ).astype(np.float32)

    return PreparedTask(
        context=context,
        future_covariates=future_covariates,
        n_targets=task_target.shape[0],
        n_covariates=task_past_covariates_array.shape[0],
        n_future_covariates=len(task_future_covariates_keys),
    )


def prepare_tasks(
    raw_tasks: Iterable[RawTask],
    prediction_length: int,
    min_past: int = 1,
    mode: DatasetMode | str = DatasetMode.TRAIN,
) -> list[PreparedTask]:
    """Prepare multiple time series tasks for training/inference."""
    if isinstance(mode, str):
        mode = DatasetMode(mode)

    tasks: list[PreparedTask] = []

    for idx, raw_task in enumerate(raw_tasks):
        # For non-TEST modes, fix future_covariates
        if mode != DatasetMode.TEST:
            raw_future_covariates = raw_task.get("future_covariates", {})
            if raw_future_covariates:
                raw_future_covariates = cast(dict[str, TensorOrArray | None], raw_future_covariates)
                fixed_future_covariates = {}
                for key, value in raw_future_covariates.items():
                    fixed_future_covariates[key] = (
                        np.full(prediction_length, np.nan) if value is None or len(value) == 0 else value
                    )
                raw_task = {**raw_task, "future_covariates": fixed_future_covariates}

        raw_task = cast(dict[str, TensorOrArray | Mapping[str, TensorOrArray]], raw_task)
        prepared = prepare_single_task(raw_task, idx, prediction_length)

        # Filter by minimum length
        if mode != DatasetMode.TEST and prepared["context"].shape[-1] < min_past + prediction_length:
            continue

        tasks.append(prepared)

    if len(tasks) == 0:
        raise ValueError(
            "The dataset is empty after filtering based on length. "
            "Provide longer time series or reduce min_past/prediction_length."
        )

    return tasks


def validate_prepared_schema(task: Any) -> None:
    """Validate that a task matches the PreparedTask schema."""
    if not isinstance(task, Mapping):
        raise TypeError(
            f"Expected task to be a dict-like, got {type(task).__name__}. "
            "Set convert_inputs=True to preprocess raw inputs."
        )

    required_keys = {"context", "future_covariates", "n_targets", "n_covariates", "n_future_covariates"}
    missing = required_keys - set(task.keys())
    if missing:
        raise TypeError(
            f"Task is missing required keys: {missing}. "
            "Set convert_inputs=True to preprocess raw inputs."
        )

    context = task["context"]
    if not isinstance(context, (np.ndarray, torch.Tensor)) or context.ndim != 2:
        raise TypeError(
            f"Expected 'context' to be 2-d array, got {type(context).__name__} "
            f"with shape {getattr(context, 'shape', 'N/A')}. "
            "Set convert_inputs=True to preprocess raw inputs."
        )
