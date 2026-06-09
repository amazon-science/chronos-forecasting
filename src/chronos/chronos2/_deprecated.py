# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Deprecated input-preparation helpers kept for backwards compatibility.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence, TypeAlias, cast

import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

if TYPE_CHECKING:
    from chronos.chronos2.dataset import DatasetMode

from chronos.chronos2.preprocess import PreparedInput

TensorOrArray: TypeAlias = torch.Tensor | np.ndarray


def _warn(old: str, new: str) -> None:
    warnings.warn(
        f"`{old}` is deprecated and will be removed in a future release. Use `{new}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def validate_and_prepare_single_dict_input(
    raw_input: Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]],
    idx: int,
    prediction_length: int,
) -> PreparedInput:
    """Deprecated. Use ``chronos.chronos2.preprocess.from_list_of_dicts`` instead."""
    _warn("validate_and_prepare_single_dict_input", "chronos.chronos2.preprocess.from_list_of_dicts")

    allowed_keys = {"target", "past_covariates", "future_covariates"}

    keys = set(raw_input.keys())
    if not keys.issubset(allowed_keys):
        raise ValueError(
            f"Found invalid keys in element at index {idx}. Allowed keys are {allowed_keys}, but found {keys}"
        )
    if "target" not in keys:
        raise ValueError(f"Element at index {idx} does not contain the required key 'target'")

    target = raw_input["target"]
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert isinstance(target, torch.Tensor)
    if target.ndim > 2:
        raise ValueError(
            "When the input is a list of dicts, the `target` should either be 1-d with shape (history_length,) "
            f" or 2-d with shape (n_variates, history_length). Found element at index {idx} with shape {tuple(target.shape)}."
        )
    history_length = target.shape[-1]
    target = target.view(-1, history_length)

    cat_encoders: dict = {}
    past_covariates = raw_input.get("past_covariates", {})
    if not isinstance(past_covariates, dict):
        raise ValueError(
            f"Found invalid type for `past_covariates` in element at index {idx}. "
            f'Expected dict with {{"feat_1": tensor_1, "feat_2": tensor_2, ...}}, but found {type(past_covariates)}'
        )

    covariates_keys = sorted(past_covariates.keys())

    future_covariates = raw_input.get("future_covariates", {})
    if not isinstance(future_covariates, dict):
        raise ValueError(
            f"Found invalid type for `future_covariates` in element at index {idx}. "
            f'Expected dict with {{"feat_1": tensor_1, "feat_2": tensor_2, ...}}, but found {type(future_covariates)}'
        )
    future_covariates_keys = sorted(future_covariates.keys())
    if not set(future_covariates_keys).issubset(covariates_keys):
        raise ValueError(
            f"Expected keys in `future_covariates` to be a subset of `past_covariates` {covariates_keys}, "
            f"but found {future_covariates_keys} in element at index {idx}"
        )

    # past-only first, then known-future (so known-future are the last rows)
    past_only_keys = [k for k in covariates_keys if k not in future_covariates_keys]
    ordered_covariate_keys = past_only_keys + future_covariates_keys

    past_covariates_list: list[torch.Tensor] = []
    for key in ordered_covariate_keys:
        tensor = past_covariates[key]
        if isinstance(tensor, np.ndarray):
            if not np.issubdtype(tensor.dtype, np.number):
                if target.shape[0] == 1:
                    cat_encoder = TargetEncoder(target_type="continuous", smooth=1.0)
                    X = tensor.astype(str).reshape(-1, 1)
                    y = target.view(-1).numpy()
                    mask = np.isfinite(y)
                    X = X[mask]
                    y = y[mask]
                    cat_encoder.fit(X, y)
                else:
                    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
                    cat_encoder.fit(tensor.astype(str).reshape(-1, 1))
                tensor = cat_encoder.transform(tensor.astype(str).reshape(-1, 1)).reshape(tensor.shape)
                cat_encoders[key] = cat_encoder
            tensor = torch.from_numpy(tensor)
        assert isinstance(tensor, torch.Tensor)
        if tensor.ndim != 1 or len(tensor) != history_length:
            raise ValueError(
                f"Individual `past_covariates` must be 1-d with length equal to the length of `target` (= {history_length}), "
                f"found: {key} with shape {tuple(tensor.shape)} in element at index {idx}"
            )
        past_covariates_list.append(tensor)
    past_covariates_tensor = (
        torch.stack(past_covariates_list, dim=0)
        if past_covariates_list
        else torch.zeros((0, history_length), device=target.device)
    )

    future_covariates_list: list[torch.Tensor] = []
    for key in ordered_covariate_keys:
        tensor = future_covariates.get(key, torch.full((prediction_length,), fill_value=torch.nan))
        if isinstance(tensor, np.ndarray):
            if not np.issubdtype(tensor.dtype, np.number):
                cat_encoder = cat_encoders[key]
                tensor = cat_encoder.transform(tensor.astype(str).reshape(-1, 1)).reshape(tensor.shape)
            tensor = torch.from_numpy(tensor)
        assert isinstance(tensor, torch.Tensor)
        if tensor.ndim != 1 or len(tensor) != prediction_length:
            raise ValueError(
                f"Individual `future_covariates` must be 1-d with length equal to the {prediction_length=}, "
                f"found: {key} with shape {tuple(tensor.shape)} in element at index {idx}"
            )
        future_covariates_list.append(tensor)
    future_covariates_tensor = (
        torch.stack(future_covariates_list, dim=0)
        if future_covariates_list
        else torch.zeros((0, prediction_length), device=target.device)
    )
    future_covariates_target_padding = torch.full(
        (target.shape[0], prediction_length), fill_value=torch.nan, device=target.device
    )

    context_tensor = torch.cat([target, past_covariates_tensor], dim=0).to(dtype=torch.float32)
    future_covariates_tensor = torch.cat([future_covariates_target_padding, future_covariates_tensor], dim=0).to(
        dtype=torch.float32
    )
    n_targets = target.shape[0]
    n_covariates = past_covariates_tensor.shape[0]
    n_future_covariates = len(future_covariates_keys)

    return PreparedInput(
        context=context_tensor,
        future_covariates=future_covariates_tensor,
        n_targets=n_targets,
        n_covariates=n_covariates,
        n_future_covariates=n_future_covariates,
    )


def prepare_inputs(
    raw_inputs: Iterable[Mapping[str, Any]],
    prediction_length: int,
    min_past: int = 1,
    mode: "DatasetMode | str" = "train",
) -> list[PreparedInput]:
    """Deprecated. Use ``chronos.chronos2.preprocess.from_list_of_dicts`` instead."""
    _warn("prepare_inputs", "chronos.chronos2.preprocess.from_list_of_dicts")

    # Imported lazily to avoid a circular import (dataset.py imports this module).
    from chronos.chronos2.dataset import DatasetMode

    inputs: list[PreparedInput] = []

    for idx, raw_input in enumerate(raw_inputs):
        if mode != DatasetMode.TEST:
            raw_future_covariates = raw_input.get("future_covariates", {})
            if raw_future_covariates:
                raw_future_covariates = cast(dict[str, TensorOrArray | None], raw_future_covariates)
                fixed_future_covariates = {}
                for key, value in raw_future_covariates.items():
                    fixed_future_covariates[key] = (
                        np.full(prediction_length, np.nan) if value is None or len(value) == 0 else value
                    )
                raw_input = {**raw_input, "future_covariates": fixed_future_covariates}

        raw_input = cast(dict[str, TensorOrArray | Mapping[str, TensorOrArray]], raw_input)
        prepared = _prepare_single(raw_input, idx, prediction_length)

        if mode != DatasetMode.TEST and prepared["context"].shape[-1] < min_past + prediction_length:
            continue

        inputs.append(prepared)

    if len(inputs) == 0:
        raise ValueError(
            "The dataset is empty after filtering based on the length of the time series (length >= min_past + prediction_length). "
            "Please provide longer time series or reduce `min_past` or `prediction_length`. "
        )

    return inputs


def _prepare_single(
    raw_input: Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]],
    idx: int,
    prediction_length: int,
) -> PreparedInput:
    """Internal: same logic as the deprecated public function but without the warning."""
    # Suppress the nested deprecation warning so prepare_inputs only emits one.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return validate_and_prepare_single_dict_input(raw_input, idx, prediction_length)


def convert_list_of_tensors_input_to_list_of_dicts_input(
    list_of_tensors: Sequence[TensorOrArray],
) -> list[dict[str, torch.Tensor]]:
    """Deprecated. Use ``chronos.chronos2.preprocess.from_list_of_tensors`` instead."""
    _warn(
        "convert_list_of_tensors_input_to_list_of_dicts_input",
        "chronos.chronos2.preprocess.from_list_of_tensors",
    )

    output: list[dict[str, torch.Tensor]] = []
    for idx, tensor in enumerate(list_of_tensors):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.ndim > 2:
            raise ValueError(
                "When the input is a list of torch tensors or numpy arrays, the elements should either be 1-d with shape (history_length,) "
                f" or 2-d with shape (n_variates, history_length). Found element at index {idx} with shape {tuple(tensor.shape)}."
            )
        length = tensor.shape[-1]
        tensor = tensor.view(-1, length)
        output.append({"target": tensor})

    return output


def convert_tensor_input_to_list_of_dicts_input(tensor: TensorOrArray) -> list[dict[str, torch.Tensor]]:
    """Deprecated. Use ``chronos.chronos2.preprocess.from_tensor`` instead."""
    _warn("convert_tensor_input_to_list_of_dicts_input", "chronos.chronos2.preprocess.from_tensor")

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if tensor.ndim != 3:
        raise ValueError(
            "When the input is a torch tensor or numpy array, it should be 3-d with shape (n_series, n_variates, history_length). "
            f" Found shape: {tuple(tensor.shape)}."
        )

    output: list[dict[str, torch.Tensor]] = []
    for i in range(len(tensor)):
        output.append({"target": tensor[i]})

    return output
