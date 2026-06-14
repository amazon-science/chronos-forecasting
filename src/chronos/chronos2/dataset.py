# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

import math
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence, TypeAlias, cast

import numpy as np
import torch
from torch.utils.data import IterableDataset

from chronos.chronos2 import preprocess
from chronos.chronos2._deprecated import (
    convert_list_of_tensors_input_to_list_of_dicts_input,
    convert_tensor_input_to_list_of_dicts_input,
    prepare_inputs,
    validate_and_prepare_single_dict_input,
)
from chronos.chronos2.preprocess import PreparedInput

__all__ = [
    "Chronos2Dataset",
    "DatasetMode",
    "PreparedInput",
    "convert_fev_window_to_list_of_dicts_input",
    "left_pad_and_cat_2D",
    "validate_prepared_schema",
    # Deprecated re-exports — prefer chronos.chronos2.preprocess.from_* for new code.
    "convert_list_of_tensors_input_to_list_of_dicts_input",
    "convert_tensor_input_to_list_of_dicts_input",
    "prepare_inputs",
    "validate_and_prepare_single_dict_input",
]

if TYPE_CHECKING:
    import datasets
    import fev


TensorOrArray: TypeAlias = torch.Tensor | np.ndarray


def left_pad_and_cat_2D(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Left pads tensors in the list to the length of the longest tensor along the second axis, then concats
    these equal length tensors along the first axis.
    """
    max_len = max(tensor.shape[-1] for tensor in tensors)
    padded = []
    for tensor in tensors:
        n_variates, length = tensor.shape
        if length < max_len:
            padding = torch.full((n_variates, max_len - length), fill_value=torch.nan, device=tensor.device)
            tensor = torch.cat([padding, tensor], dim=-1)
        padded.append(tensor)

    return torch.cat(padded, dim=0)


def validate_prepared_schema(prepared_input: Any) -> None:
    """Validate that an input matches the PreparedInput schema."""
    if not isinstance(prepared_input, Mapping):
        raise TypeError(
            f"Expected input to be a dict-like, got {type(prepared_input).__name__}. "
            "Set convert_inputs=True when calling fit() to preprocess raw inputs."
        )

    required_keys = {"context", "future_covariates", "n_targets", "n_covariates", "n_future_covariates"}
    missing = required_keys - set(prepared_input.keys())
    if missing:
        raise TypeError(
            f"Input is missing required keys: {missing}. Set convert_inputs=True when calling fit() to preprocess raw inputs."
        )

    context = prepared_input["context"]
    if not isinstance(context, torch.Tensor) or context.ndim != 2:
        raise TypeError(
            f"Expected 'context' to be 2-d torch.Tensor, got {type(context).__name__} "
            f"with shape {getattr(context, 'shape', 'N/A')}. "
            "Set convert_inputs=True when calling fit() to preprocess raw inputs."
        )

    future_covariates = prepared_input["future_covariates"]
    if not isinstance(future_covariates, torch.Tensor) or future_covariates.ndim != 2:
        raise TypeError(
            f"Expected 'future_covariates' to be 2-d torch.Tensor, got {type(future_covariates).__name__} "
            f"with shape {getattr(future_covariates, 'shape', 'N/A')}. "
            "Set convert_inputs=True when calling fit() to preprocess raw inputs."
        )

    if context.shape[0] != future_covariates.shape[0]:
        raise ValueError(
            f"Expected 'context' and 'future_covariates' to have the same first dimension, "
            f"got {context.shape[0]} and {future_covariates.shape[0]}. "
            "Set convert_inputs=True when calling fit() to preprocess raw inputs."
        )


def _cast_fev_features(
    past_data: "datasets.Dataset",
    future_data: "datasets.Dataset",
    target_columns: list[str],
    past_dynamic_columns: list[str],
    known_dynamic_columns: list[str],
) -> tuple["datasets.Dataset", "datasets.Dataset"]:
    import datasets

    dynamic_columns = [*past_dynamic_columns, *known_dynamic_columns]
    cat_cols = []
    for col in dynamic_columns:
        item = past_data[0][col]
        if not np.issubdtype(item.dtype, np.number):
            cat_cols.append(col)

    numeric_cols = target_columns + list(set(dynamic_columns) - set(cat_cols))
    past_feature_updates = {col: datasets.Sequence(datasets.Value("float64")) for col in numeric_cols} | {
        col: datasets.Sequence(datasets.Value("string")) for col in cat_cols
    }
    past_data_features = past_data.features
    past_data_features.update(past_feature_updates)
    past_data = past_data.cast(past_data_features)

    future_cat_cols = [k for k in cat_cols if k in known_dynamic_columns]
    future_numeric_cols = list(set(known_dynamic_columns) - set(future_cat_cols))
    future_feature_updates = {col: datasets.Sequence(datasets.Value("float64")) for col in future_numeric_cols} | {
        col: datasets.Sequence(datasets.Value("string")) for col in future_cat_cols
    }
    future_data_features = future_data.features
    future_data_features.update(future_feature_updates)
    future_data = future_data.cast(future_data_features)

    return past_data, future_data


def convert_fev_window_to_list_of_dicts_input(
    window: "fev.EvaluationWindow", as_univariate: bool
) -> tuple[list[dict[str, np.ndarray | dict[str, np.ndarray]]], list[str], list[str], list[str]]:
    import fev

    if as_univariate:
        past_data, future_data = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
        target_columns = ["target"]
        past_dynamic_columns = []
        known_dynamic_columns = []
    else:
        past_data, future_data = window.get_input_data()
        target_columns = window.target_columns
        past_dynamic_columns = window.past_dynamic_columns
        known_dynamic_columns = window.known_dynamic_columns

    past_data, future_data = _cast_fev_features(
        past_data=past_data,
        future_data=future_data,
        target_columns=target_columns,
        past_dynamic_columns=past_dynamic_columns,
        known_dynamic_columns=known_dynamic_columns,
    )

    num_series: int = len(past_data)
    num_past_covariates: int = len(past_dynamic_columns)
    num_future_covariates: int = len(known_dynamic_columns)

    # We use numpy format because torch does not support str covariates
    target_data = past_data.select_columns(target_columns).with_format("numpy")
    # past of past-only and known-future covariates
    dynamic_columns = [*past_dynamic_columns, *known_dynamic_columns]
    past_covariate_data = past_data.select_columns(dynamic_columns).with_format("numpy")
    future_known_data = future_data.select_columns(known_dynamic_columns).with_format("numpy")

    if num_past_covariates + num_future_covariates > 0:
        assert len(past_covariate_data) == num_series
    if num_future_covariates > 0:
        assert len(future_known_data) == num_series

    inputs: list[dict[str, np.ndarray | dict[str, np.ndarray]]] = []
    for idx, target_row in enumerate(target_data):
        target_row = cast(dict, target_row)
        # this assumes that the targets have the same length for multivariate tasks
        target_tensor_i = np.stack([target_row[col] for col in target_columns])
        entry: dict[str, np.ndarray | dict[str, np.ndarray]] = {"target": target_tensor_i}

        if len(dynamic_columns) > 0:
            past_covariate_row = past_covariate_data[idx]
            entry["past_covariates"] = {col: past_covariate_row[col] for col in dynamic_columns}

        if len(known_dynamic_columns) > 0:
            future_known_row = future_known_data[idx]
            entry["future_covariates"] = {col: future_known_row[col] for col in known_dynamic_columns}

        inputs.append(entry)

    return inputs, target_columns, past_dynamic_columns, known_dynamic_columns


class DatasetMode(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Chronos2Dataset(IterableDataset):
    """
    A dataset wrapper for Chronos-2 models.

    Arguments
    ----------
    inputs
        Time series data. Can be either:

        1. Raw inputs (when `convert_inputs=True`, default): A sequence of dictionaries where each
           dictionary may have the following keys:
           - `target` (required): a 1-d or 2-d `torch.Tensor` or `np.ndarray` of shape (history_length,)
             or (n_variates, history_length). Forecasts will be generated for items in `target`.
           - `past_covariates` (optional): a dict of past-only covariates or past values of known future
             covariates.
           - `future_covariates` (optional): a dict of future values of known future covariates.

           All dictionaries must share the same schema: the same `target` shape (`n_variates`) and the same
           `past_covariates` / `future_covariates` keys (the `history_length` may differ across dictionaries).

        2. Pre-processed inputs (when `convert_inputs=False`): A sequence of `PreparedInput` dicts with keys:
           `context`, `future_covariates`, `n_targets`, `n_covariates`, `n_future_covariates`.
           Use the `chronos.chronos2.preprocess.from_*` functions to create pre-processed inputs.
    context_length
        The maximum context length used for training or inference
    prediction_length
        The prediction horizon
    batch_size
        The batch size for training the model. Note that the batch size here means the number of time series,
        including target(s) and covariates, that are input into the model.
    output_patch_size
        The output patch size of the model. This is used to compute the number of patches needed to cover
        `prediction_length`
    min_past
        The minimum number of time steps the context must have during training. All time series shorter than
        `min_past + prediction_length` are filtered out, by default 1
    mode
        `DatasetMode` governing whether to generate training, validation or test samples, by default "train"
    convert_inputs
        If True (default), preprocess raw inputs. If False, inputs are expected to be already preprocessed.
    """

    def __init__(
        self,
        inputs: TensorOrArray | Sequence[TensorOrArray] | Sequence[Mapping[str, Any]] | Sequence[PreparedInput],
        context_length: int,
        prediction_length: int,
        batch_size: int,
        output_patch_size: int,
        min_past: int = 1,
        mode: str | DatasetMode = DatasetMode.TRAIN,
        convert_inputs: bool = True,
    ) -> None:
        super().__init__()
        assert mode in {DatasetMode.TRAIN, DatasetMode.VALIDATION, DatasetMode.TEST}, f"Invalid mode: {mode}"

        self.inputs: Sequence[PreparedInput]
        if convert_inputs:
            if isinstance(inputs, (torch.Tensor, np.ndarray)):
                self.inputs = preprocess.from_tensor(inputs, prediction_length=prediction_length)
            elif (
                isinstance(inputs, Sequence) and len(inputs) > 0 and isinstance(inputs[0], (torch.Tensor, np.ndarray))
            ):
                self.inputs = preprocess.from_list_of_tensors(
                    cast("list[TensorOrArray]", inputs), prediction_length=prediction_length
                )
            else:
                self.inputs = preprocess.from_list_of_dicts(
                    cast(list[dict], inputs), prediction_length=prediction_length
                )
        else:
            validate_prepared_schema(inputs[0])
            self.inputs = cast(Sequence[PreparedInput], inputs)

        if mode != DatasetMode.TEST:
            self.inputs = [x for x in self.inputs if x["context"].shape[-1] >= min_past + prediction_length]
            if len(self.inputs) == 0:
                raise ValueError(
                    "The dataset is empty after filtering based on the length of the time series "
                    "(length >= min_past + prediction_length). Please provide longer time series or "
                    "reduce `min_past` or `prediction_length`."
                )

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.num_output_patches = math.ceil(prediction_length / output_patch_size)
        self.min_past = min_past
        self.mode = mode

    def _construct_slice(self, input_idx: int) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int]:
        prepared = self.inputs[input_idx]
        past_tensor = prepared["context"].clone()  # shape: (n_targets + n_covariates, history_length)
        future_tensor = prepared["future_covariates"].clone()
        n_targets = int(prepared["n_targets"])
        n_covariates = int(prepared["n_covariates"])
        n_future_covariates = int(prepared["n_future_covariates"])
        n_past_only_covariates = n_covariates - n_future_covariates

        full_length = past_tensor.shape[-1]

        if self.mode == DatasetMode.TRAIN:
            # slice a random subsequence from the full series
            slice_idx = np.random.randint(self.min_past, full_length - self.prediction_length + 1)
        elif self.mode == DatasetMode.VALIDATION:
            # slice the last window for validation
            slice_idx = full_length - self.prediction_length
        else:
            # slice the full series for prediction
            slice_idx = full_length

        if slice_idx >= self.context_length:
            # slice series, if it is longer than context_length
            context = past_tensor[:, slice_idx - self.context_length : slice_idx]
        else:
            context = past_tensor[:, :slice_idx]

        # In the TEST mode, we have no target available and the future_covariates can be directly used
        # In the TRAIN and VALIDATION modes, the target and future_covariates need to be constructed from
        # the context_tensor by slicing the appropriate indices which we do below
        if self.mode in [DatasetMode.TRAIN, DatasetMode.VALIDATION]:
            # the first n_targets elements in context_tensor are the targets
            future_target = past_tensor[:, slice_idx : slice_idx + self.prediction_length].clone()
            # mask out all rows corresponding to covariates
            future_target[n_targets:] = torch.nan

            if n_future_covariates > 0:
                # the last n_future_covariates elements in context_tensor are the known covariates
                future_covariates = past_tensor[-n_future_covariates:, slice_idx : slice_idx + self.prediction_length]
            else:
                # zero-length tensor for easy concatenation later
                future_covariates = torch.zeros((0, self.prediction_length))

            # the leading n_targets + n_past_only_covariates elements are masked because the target(s)
            # and past-only covariates are not known into the future
            future_covariates_padding = torch.full(
                (n_targets + n_past_only_covariates, self.prediction_length),
                fill_value=torch.nan,
            )
            future_covariates = torch.cat([future_covariates_padding, future_covariates], dim=0)
        else:
            future_target = None
            future_covariates = future_tensor

        # context: (n_targets + n_covariates, min(context_length, history_length))
        # future_target: (n_targets + n_covariates, prediction_length), the future values of known future covariates
        # are ignored during loss computation
        # future_covariates: (n_targets + n_past_only_covariates + n_future_covariates, prediction_length),
        # the entries corresponding to targets and past-only covariates are NaNs

        return context, future_target, future_covariates, n_targets

    def _build_batch(self, input_indices: list[int]) -> dict[str, torch.Tensor | int | list[tuple[int, int]] | None]:
        """Build a batch from given input indices."""
        batch_context_list = []
        batch_future_target_list = []
        batch_future_covariates_list = []
        batch_group_ids_list = []
        target_idx_ranges: list[tuple[int, int]] = []

        target_start_idx = 0
        for group_id, input_idx in enumerate(input_indices):
            context, future_target, future_covariates, n_targets = self._construct_slice(input_idx)

            group_size = context.shape[0]
            group_ids = torch.full((group_size,), fill_value=group_id)
            batch_context_list.append(context)
            batch_future_target_list.append(future_target)
            batch_future_covariates_list.append(future_covariates)
            batch_group_ids_list.append(group_ids)
            target_idx_ranges.append((target_start_idx, target_start_idx + n_targets))
            target_start_idx += group_size

        return {
            "context": left_pad_and_cat_2D(batch_context_list),
            "future_target": None
            if self.mode == DatasetMode.TEST
            else torch.cat(cast(list[torch.Tensor], batch_future_target_list), dim=0),
            "future_covariates": torch.cat(batch_future_covariates_list, dim=0),
            "group_ids": torch.cat(batch_group_ids_list, dim=0),
            "num_output_patches": self.num_output_patches,
            "target_idx_ranges": target_idx_ranges,
        }

    def _generate_train_batches(self):
        while True:
            current_batch_size = 0
            input_indices = []

            while current_batch_size < self.batch_size:
                input_idx = np.random.randint(len(self.inputs))
                input_indices.append(input_idx)
                current_batch_size += self.inputs[input_idx]["context"].shape[0]

            yield self._build_batch(input_indices)

    def _generate_sequential_batches(self):
        input_idx = 0
        while input_idx < len(self.inputs):
            current_batch_size = 0
            input_indices = []

            while input_idx < len(self.inputs) and current_batch_size < self.batch_size:
                input_indices.append(input_idx)
                current_batch_size += self.inputs[input_idx]["context"].shape[0]
                input_idx += 1

            yield self._build_batch(input_indices)

    def __iter__(self) -> Iterator:
        """
        Generate batches of data for the Chronos-2 model. In training mode, this iterator is infinite.

        Yields
        ------
        dict
            A dictionary containing:
            - context: torch.Tensor of shape (batch_size, context_length) containing input sequences
            - future_target: torch.Tensor of shape (batch_size, prediction_length) containing future target sequences, None in TEST mode
            - future_covariates: torch.Tensor of shape (batch_size, prediction_length) containing known future covariates
            - group_ids: torch.Tensor of shape (batch_size,) containing the group ID for each sequence
            - num_output_patches: int indicating number of patches the model should output to cover prediction_length
            - target_idx_ranges: (only in TEST mode) list of tuples indicating the start & end indices of targets in context
        """
        if self.mode == DatasetMode.TRAIN:
            for batch in self._generate_train_batches():
                batch.pop("target_idx_ranges")
                yield batch
        elif self.mode == DatasetMode.VALIDATION:
            for batch in self._generate_sequential_batches():
                batch.pop("target_idx_ranges")
                yield batch
        else:
            yield from self._generate_sequential_batches()
