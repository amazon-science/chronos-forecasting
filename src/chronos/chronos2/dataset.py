# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

import math
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, NotRequired, Sequence, TypeAlias, TypedDict, cast

import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    import datasets
    import fev


TensorOrArray: TypeAlias = torch.Tensor | np.ndarray


class PreparedInput(TypedDict):
    """A preprocessed time series input ready for model training/inference."""

    context: torch.Tensor  # (n_variates, history_length)
    future_covariates: NotRequired[torch.Tensor]  # (n_variates, prediction_length); only required in TEST mode
    n_targets: int
    n_covariates: int
    n_future_covariates: int


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


def validate_and_prepare_single_dict_input(
    raw_input: Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]], idx: int, prediction_length: int
) -> PreparedInput:
    """Validates and prepares a single dictionary input for Chronos2Model.

    Parameters
    ----------
    raw_input
        A dictionary representing a time series that contains:
        - `target` (required): a 1-d or 2-d `torch.Tensor` or `np.ndarray` of shape (history_length,) or (n_variates, history_length).
        Forecasts will be generated for items in `target`.
        - `past_covariates` (optional): a dict of past-only covariates or past values of known future covariates. The keys of the dict
        must be names of the covariates and values must be 1-d `torch.Tensor` or `np.ndarray` with length equal to the `history_length`
        of `target`.
        - `future_covariates` (optional): a dict of future values of known future covariates. The keys of the dict must be names of the
        covariates and values must be 1-d `torch.Tensor` or `np.ndarray` with length equal to the `prediction_length`. All keys in
        `future_covariates` must be a subset of the keys in `past_covariates`.
    idx
        Index of this input in the list of inputs, used for error messages
    prediction_length
        Number of future time steps to predict, used to validate future covariates

    Returns
    ------
    A PreparedInput containing:
    - context: Concatenated tensor of target and past covariates of shape (group_size, history_length),
        the first `n_targets` items along the first axis contain the target variables and the remaining items contain past-only covariates
        and past values of known future covariates.
    - future_covariates: Tensor of future covariates of shape (group_size, prediction_length). The last `n_future_covariates`
        items along the first axis contain future covariates. All the remaining elements corresponding to target and past-only covariates are NaNs.
    - n_targets: Number of target variables
    - n_covariates: Total number of covariates (sum of past-only and known future covariates)
    - n_future_covariates: Number of known future covariates
    """

    allowed_keys = {"target", "past_covariates", "future_covariates"}

    # validate keys
    keys = set(raw_input.keys())
    if not keys.issubset(allowed_keys):
        raise ValueError(
            f"Found invalid keys in element at index {idx}. Allowed keys are {allowed_keys}, but found {keys}"
        )
    if "target" not in keys:
        raise ValueError(f"Element at index {idx} does not contain the required key 'target'")

    # validate target
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

    # validate past_covariates
    cat_encoders: dict = {}
    past_covariates = raw_input.get("past_covariates", {})
    if not isinstance(past_covariates, dict):
        raise ValueError(
            f"Found invalid type for `past_covariates` in element at index {idx}. "
            f'Expected dict with {{"feat_1": tensor_1, "feat_2": tensor_2, ...}}, but found {type(past_covariates)}'
        )

    # gather keys and ensure known-future keys come last to match downstream assumptions
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

    # create ordered keys: past-only first, then known-future (so known-future are the last rows)
    past_only_keys = [k for k in covariates_keys if k not in future_covariates_keys]
    ordered_covariate_keys = past_only_keys + future_covariates_keys

    past_covariates_list: list[torch.Tensor] = []
    for key in ordered_covariate_keys:
        tensor = past_covariates[key]
        if isinstance(tensor, np.ndarray):
            # apply encoding to categorical variates
            if not np.issubdtype(tensor.dtype, np.number):
                # target encoding, if the target is 1-d
                if target.shape[0] == 1:
                    cat_encoder = TargetEncoder(target_type="continuous", smooth=1.0)
                    X = tensor.astype(str).reshape(-1, 1)
                    y = target.view(-1).numpy()
                    mask = np.isfinite(y)
                    X = X[mask]
                    y = y[mask]
                    cat_encoder.fit(X, y)
                # ordinal encoding, if the target is > 1-d
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

    # validate future_covariates (build rows in the same ordered_covariate_keys order)
    future_covariates_list: list[torch.Tensor] = []
    for key in ordered_covariate_keys:
        # future values of past-only covariates are filled with NaNs
        tensor = future_covariates.get(key, torch.full((prediction_length,), fill_value=torch.nan))
        if isinstance(tensor, np.ndarray):
            # apply encoding to categorical variates
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
    # future values of target series are filled with NaNs
    future_covariates_target_padding = torch.full(
        (target.shape[0], prediction_length), fill_value=torch.nan, device=target.device
    )

    context_tensor = torch.cat([target, past_covariates_tensor], dim=0).to(dtype=torch.float32)
    future_covariates_tensor = torch.cat(
        [future_covariates_target_padding, future_covariates_tensor],
        dim=0,
    ).to(dtype=torch.float32)
    n_targets = target.shape[0]
    n_covariates = past_covariates_tensor.shape[0]
    # number of known-future covariates
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
    """Prepare multiple time series inputs for training/inference.

    This function handles mode-specific preprocessing (e.g., filtering short series)
    and calls validate_and_prepare_single_dict_input for each input.
    """
    inputs: list[PreparedInput] = []

    for idx, raw_input in enumerate(raw_inputs):
        # For non-TEST modes, fix future_covariates (replace None/empty with NaN arrays)
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
        prepared = validate_and_prepare_single_dict_input(raw_input, idx, prediction_length)

        # Filter by minimum length (except in TEST mode)
        if mode != DatasetMode.TEST and prepared["context"].shape[-1] < min_past + prediction_length:
            continue

        inputs.append(prepared)

    if len(inputs) == 0:
        raise ValueError(
            "The dataset is empty after filtering based on the length of the time series (length >= min_past + prediction_length). "
            "Please provide longer time series or reduce `min_past` or `prediction_length`. "
        )

    return inputs


def validate_prepared_schema(prepared_input: Any, mode: "DatasetMode | str") -> None:
    """Validate that an input matches the PreparedInput schema.

    Parameters
    ----------
    prepared_input
        The input to validate
    mode
        Dataset mode. `future_covariates` is only required in TEST mode since it's reconstructed
        from context in TRAIN/VALIDATION modes.
    """
    if not isinstance(prepared_input, Mapping):
        raise TypeError(
            f"Expected input to be a dict-like, got {type(prepared_input).__name__}. "
            "Set convert_inputs=True when calling fit() to preprocess raw inputs."
        )

    # future_covariates is only required in TEST mode (reconstructed from context otherwise)
    required_keys = {"context", "n_targets", "n_covariates", "n_future_covariates"}
    if mode == DatasetMode.TEST:
        required_keys.add("future_covariates")

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

    future_covariates = prepared_input.get("future_covariates")
    if future_covariates is not None:
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


def convert_list_of_tensors_input_to_list_of_dicts_input(
    list_of_tensors: Sequence[TensorOrArray],
) -> list[dict[str, torch.Tensor]]:
    """Convert a list of tensors input format to a list of dictionaries input format.


    Parameters
    ----------
    list_of_tensors
        A sequence of tensors or numpy arrays, where each element represents a time series.
        Each element should be either 1-d with shape (history_length,) or 2-d with shape
        (n_variates, history_length).

    Returns
    -------
    A list of dictionaries, where each dictionary represents a time series and contains:
    - `target`: a 1-d or 2-d torch.Tensor of shape (history_length,) or (n_variates, history_length).
    """

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
    """
    Convert a tensor input format to a list of dictionaries input format.

    Parameters
    ----------
    tensor
        A tensor or numpy array representing multiple time series.
        Should be 3-d with shape (n_series, n_variates, history_length).

    Returns
    -------
    A list of dictionaries, where each dictionary represents a time series and contains:
    - `target`: a 2-d torch.Tensor of shape (n_variates, history_length).
    """

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if tensor.ndim != 3:
        raise ValueError(
            "When the input is a torch tensor or numpy array, it should be 3-d with shape (n_series, n_variates, history_length). "
            f" Found shape: {tuple(tensor.shape)}."
        )

    output: list[dict[str, torch.Tensor]] = []
    n_series = len(tensor)
    for i in range(n_series):
        output.append({"target": tensor[i]})

    return output


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

        2. Pre-processed inputs (when `convert_inputs=False`): A sequence of `PreparedInput` dicts with keys:
           `context`, `future_covariates`, `n_targets`, `n_covariates`, `n_future_covariates`.
           Use `prepare_inputs()` to create pre-processed inputs.
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
                inputs = convert_tensor_input_to_list_of_dicts_input(inputs)
            elif (
                isinstance(inputs, Sequence) and len(inputs) > 0 and isinstance(inputs[0], (torch.Tensor, np.ndarray))
            ):
                inputs = convert_list_of_tensors_input_to_list_of_dicts_input(cast(Sequence[TensorOrArray], inputs))
            self.inputs = prepare_inputs(cast(Iterable[Mapping[str, Any]], inputs), prediction_length, min_past, mode)
        else:
            validate_prepared_schema(inputs[0], mode=mode)
            self.inputs = cast(Sequence[PreparedInput], inputs)

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.num_output_patches = math.ceil(prediction_length / output_patch_size)
        self.min_past = min_past
        self.mode = mode

    def _construct_slice(self, input_idx: int) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int]:
        prepared = self.inputs[input_idx]
        past_tensor = prepared["context"].clone()  # shape: (n_targets + n_covariates, history_length)
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
            future_covariates = prepared["future_covariates"].clone()

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
