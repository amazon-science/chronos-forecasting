# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Preprocessing module for converting various input formats to list[PreparedInput] expected by Chronos2Dataset.
"""

from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch

if TYPE_CHECKING:
    import pandas as pd


class PreparedInput(TypedDict):
    """A preprocessed time series input ready for model training/inference."""

    context: torch.Tensor  # (n_variates, context_length), float32
    future_covariates: torch.Tensor  # (n_variates, prediction_length), float32
    n_targets: int
    n_covariates: int
    n_future_covariates: int


def from_tensor(
    data: "torch.Tensor | np.ndarray",
    prediction_length: int,
) -> list[PreparedInput]:
    """
    Convert 3D tensor to list[PreparedInput].

    All variates are treated as targets (no covariates).

    Parameters
    ----------
    data
        Shape: (n_series, n_variates, context_length)
    prediction_length
        Number of future time steps (for NaN padding in future_covariates)

    Returns
    -------
    list[PreparedInput], one per series
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3-d tensor with shape (n_series, n_variates, context_length), got shape {tuple(data.shape)}"
        )

    data = data.to(dtype=torch.float32)
    n_targets = data.shape[1]

    results: list[PreparedInput] = []
    for i in range(data.shape[0]):
        results.append(
            PreparedInput(
                context=data[i].clone(),
                future_covariates=torch.full((n_targets, prediction_length), fill_value=torch.nan),
                n_targets=n_targets,
                n_covariates=0,
                n_future_covariates=0,
            )
        )
    return results


def from_list_of_tensors(
    data: "list[torch.Tensor | np.ndarray]",
    prediction_length: int,
) -> list[PreparedInput]:
    """
    Convert list of 1D/2D tensors to list[PreparedInput].

    All variates are treated as targets (no covariates).

    Parameters
    ----------
    data
        Each item: (context_length,) or (n_variates, context_length)
    prediction_length
        Number of future time steps

    Returns
    -------
    list[PreparedInput], one per input tensor
    """
    results: list[PreparedInput] = []
    for idx, item in enumerate(data):
        if isinstance(item, np.ndarray):
            item = torch.from_numpy(item)
        if item.ndim > 2:
            raise ValueError(
                f"Each element should be 1-d or 2-d, found shape {tuple(item.shape)} at index {idx}"
            )
        context = item.view(-1, item.shape[-1]).to(dtype=torch.float32)
        n_targets = context.shape[0]
        results.append(
            PreparedInput(
                context=context,
                future_covariates=torch.full((n_targets, prediction_length), fill_value=torch.nan),
                n_targets=n_targets,
                n_covariates=0,
                n_future_covariates=0,
            )
        )
    return results


def from_dataframe(
    df: "pd.DataFrame",
    target_columns: list[str],
    prediction_length: int,
    future_df: "pd.DataFrame | None" = None,
    known_covariates_names: list[str] | None = None,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
    use_target_encoding: bool = True,
    validate_inputs: bool = True,
) -> list[PreparedInput]:
    """
    Convert long-format DataFrame to list[PreparedInput].

    Assumptions (when validate_inputs=False)
    ----------------------------------------
    - df is sorted by (id_column, timestamp_column)
    - future_df (if provided) is sorted by (id_column, timestamp_column)
    - future_df has exactly prediction_length rows per item, same item IDs as df
    - Target columns are numeric; other columns are numeric or categorical

    Parameters
    ----------
    df
        Long-format DataFrame with columns: id_column, timestamp_column, target_columns, covariates
    target_columns
        Column names for target variates
    prediction_length
        Number of future time steps
    future_df
        Optional DataFrame with future covariate values (same id_column, timestamp_column).
        Mutually exclusive with known_covariates_names.
    known_covariates_names
        Optional list of column names that are known-future covariates. Use when future values
        are not available (e.g., during training). Future values will be NaN-filled.
        Mutually exclusive with future_df.
    id_column
        Column name for series ID
    timestamp_column
        Column name for timestamps
    use_target_encoding
        When True (default), use target encoding for categoricals (requires single target).
        When False, use ordinal encoding.
    validate_inputs
        When True (default), validates and sorts dataframes. Set False to skip.

    Returns
    -------
    list[PreparedInput], one per unique item_id (in original order)
    """
    if future_df is not None and known_covariates_names is not None:
        raise ValueError("Cannot provide both future_df and known_covariates_names")

    import pandas as pd
    import pandas.api.types as ptypes

    if validate_inputs:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.sort_values([id_column, timestamp_column])
        if future_df is not None:
            future_df[timestamp_column] = pd.to_datetime(future_df[timestamp_column])
            future_df = future_df.sort_values([id_column, timestamp_column])
        _validate_dataframe(
            df=df,
            future_df=future_df,
            target_columns=target_columns,
            prediction_length=prediction_length,
            id_column=id_column,
            timestamp_column=timestamp_column,
        )

    covariate_columns = [
        c for c in df.columns if c not in {id_column, timestamp_column} and c not in target_columns
    ]

    target = df[target_columns].to_numpy(dtype=np.float32, na_value=np.nan).T

    past_covariates: dict[str, np.ndarray] = {}
    for col in covariate_columns:
        if ptypes.is_numeric_dtype(df[col]):
            past_covariates[col] = df[col].to_numpy(dtype=np.float32, na_value=np.nan)
        else:
            past_covariates[col] = df[col].to_numpy(dtype=str)

    future_covariates: dict[str, np.ndarray | None] = {}
    if future_df is not None:
        for col in covariate_columns:
            if col in future_df.columns:
                # dtype from df (not future_df) is source of truth
                if ptypes.is_numeric_dtype(df[col]):
                    future_covariates[col] = future_df[col].to_numpy(dtype=np.float32, na_value=np.nan)
                else:
                    future_covariates[col] = future_df[col].to_numpy(dtype=str)
    elif known_covariates_names is not None:
        for col in known_covariates_names:
            if col in past_covariates:
                future_covariates[col] = None

    return _build_prepared_inputs(
        target=target,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        series_lengths=df[id_column].value_counts(sort=False).tolist(),
        prediction_length=prediction_length,
        use_target_encoding=use_target_encoding,
    )


def from_list_of_dicts(
    data: list[dict],
    prediction_length: int,
    known_covariates_names: list[str] | None = None,
    use_target_encoding: bool = True,
    validate_inputs: bool = True,
) -> list[PreparedInput]:
    """
    Convert list of dicts to list[PreparedInput].

    Each dict has:
    - "target": np.ndarray, shape (context_length,) or (n_targets, context_length)
    - "past_covariates": optional dict[str, np.ndarray], each shape (context_length,)
    - "future_covariates": optional dict[str, np.ndarray], each shape (prediction_length,)

    Assumptions (when validate_inputs=False)
    ----------------------------------------
    - All dicts have same structure (same keys, same n_targets)
    - All past_covariates have the same column names across dicts
    - future_covariates keys are a subset of past_covariates keys
    - future_covariates arrays have length == prediction_length

    Parameters
    ----------
    data
        List of input dicts
    prediction_length
        Number of future time steps
    known_covariates_names
        Optional list of past_covariates keys that are known into the future.
        Use when future values are not available (e.g., during training).
        Mutually exclusive with "future_covariates" in the dicts.
    use_target_encoding
        When True (default), use target encoding for categoricals (requires single target).
        When False, use ordinal encoding.
    validate_inputs
        When True (default), validates all dicts have consistent structure.

    Returns
    -------
    list[PreparedInput], one per dict
    """
    if validate_inputs:
        _validate_list_of_dicts(data=data, prediction_length=prediction_length)

    if len(data) == 0:
        return []

    has_future_values = "future_covariates" in data[0] and len(data[0]["future_covariates"]) > 0
    if has_future_values and known_covariates_names is not None:
        raise ValueError("Cannot provide both known_covariates_names and future_covariates in dicts")

    past_covariate_keys = sorted(data[0].get("past_covariates", {}).keys())

    # Stack targets: (n_targets, total_context_rows)
    target_arrays = []
    series_lengths = []
    for d in data:
        t = np.asarray(d["target"], dtype=np.float32)
        if t.ndim == 1:
            t = t.reshape(1, -1)
        target_arrays.append(t)
        series_lengths.append(t.shape[-1])
    target = np.concatenate(target_arrays, axis=1)

    past_covariates: dict[str, np.ndarray] = {}
    for key in past_covariate_keys:
        stacked = np.concatenate([np.asarray(d["past_covariates"][key]) for d in data])
        if np.issubdtype(stacked.dtype, np.number):
            past_covariates[key] = stacked.astype(np.float32)
        else:
            past_covariates[key] = stacked.astype(str)

    future_covariates: dict[str, np.ndarray | None] = {}
    if has_future_values:
        for key in sorted(data[0]["future_covariates"].keys()):
            stacked = np.concatenate([np.asarray(d["future_covariates"][key]) for d in data])
            if np.issubdtype(stacked.dtype, np.number):
                future_covariates[key] = stacked.astype(np.float32)
            else:
                future_covariates[key] = stacked.astype(str)
    elif known_covariates_names is not None:
        for col in known_covariates_names:
            if col in past_covariates:
                future_covariates[col] = None

    return _build_prepared_inputs(
        target=target,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        series_lengths=series_lengths,
        prediction_length=prediction_length,
        use_target_encoding=use_target_encoding,
    )


def _build_prepared_inputs(
    target: np.ndarray,
    past_covariates: dict[str, np.ndarray],
    future_covariates: dict[str, np.ndarray | None],
    series_lengths: list[int],
    prediction_length: int,
    use_target_encoding: bool,
) -> list[PreparedInput]:
    """
    Build list[PreparedInput] from stacked arrays. Handles categorical encoding.

    Assumptions
    -----------
    - Arrays are stacked in item order (item 0's rows first, then item 1's, etc.)
    - Categorical columns have str dtype; numeric columns have float32 dtype
    - future_covariates keys are a subset of past_covariates keys
    - Key present in future_covariates = known-future covariate
    - Value is the actual future data (shape: n_series * prediction_length) or None if unavailable

    Parameters
    ----------
    target
        Shape: (n_targets, total_context_rows), dtype float32
    past_covariates
        {name: values} for all covariates (past-only and known-future)
        Each array shape: (total_context_rows,)
    future_covariates
        {name: values_or_None} for known-future covariates.
        Each array shape: (n_series * prediction_length,), or None if values unavailable.
    series_lengths
        Context length of each series (sum = total_context_rows)
    prediction_length
        Number of future time steps
    use_target_encoding
        When True, use target encoding (requires n_targets == 1). When False, use ordinal.

    Returns
    -------
    list[PreparedInput], one per series
    """
    n_series = len(series_lengths)
    n_targets = target.shape[0]
    n_covariates = len(past_covariates)
    n_future_covariates = len(future_covariates)
    nan_future = np.full(n_series * prediction_length, np.nan, dtype=np.float32)

    id_codes = np.repeat(np.arange(n_series), series_lengths)
    future_id_codes = np.repeat(np.arange(n_series), prediction_length)

    encoded_past: list[np.ndarray] = []
    encoded_future: list[np.ndarray] = []

    for key, values in past_covariates.items():
        is_known_future = key in future_covariates
        future_values = future_covariates.get(key)

        if np.issubdtype(values.dtype, np.str_):
            categories = np.unique(values)
            cat_to_code = {cat: i for i, cat in enumerate(categories)}
            n_categories = len(categories)

            past_codes = np.array([cat_to_code[v] for v in values], dtype=np.intp)

            future_codes = None
            if future_values is not None:
                future_codes = np.array(
                    [cat_to_code.get(v, 0) for v in future_values], dtype=np.intp
                )

            if use_target_encoding and n_targets == 1:
                enc_past, enc_future = _target_encode(
                    id_codes=id_codes,
                    cat_codes=past_codes,
                    target=target[0],
                    n_items=n_series,
                    n_categories=n_categories,
                    future_id_codes=future_id_codes if future_codes is not None else None,
                    future_cat_codes=future_codes,
                )
                encoded_past.append(enc_past)
                if is_known_future:
                    encoded_future.append(enc_future if enc_future is not None else nan_future)
            else:
                encoded_past.append(past_codes.astype(np.float32))
                if is_known_future:
                    encoded_future.append(
                        future_codes.astype(np.float32) if future_codes is not None else nan_future
                    )
        else:
            encoded_past.append(values)
            if is_known_future:
                encoded_future.append(future_values if future_values is not None else nan_future)

        if not is_known_future:
            encoded_future.append(nan_future)

    # Split into per-series PreparedInputs using indptr
    indptr = np.concatenate([[0], np.cumsum(series_lengths)]).astype(np.intp)

    results: list[PreparedInput] = []
    for i in range(n_series):
        p_start, p_end = indptr[i], indptr[i + 1]
        f_start, f_end = i * prediction_length, (i + 1) * prediction_length

        target_i = target[:, p_start:p_end]
        past_cov_i = (
            np.stack([arr[p_start:p_end] for arr in encoded_past])
            if encoded_past
            else np.zeros((0, series_lengths[i]), dtype=np.float32)
        )
        future_cov_i = (
            np.stack([arr[f_start:f_end] for arr in encoded_future])
            if encoded_future
            else np.zeros((0, prediction_length), dtype=np.float32)
        )

        context = np.concatenate([target_i, past_cov_i], axis=0)
        target_padding = np.full((n_targets, prediction_length), np.nan, dtype=np.float32)
        future_full = np.concatenate([target_padding, future_cov_i], axis=0)

        results.append(
            PreparedInput(
                context=torch.from_numpy(context).to(dtype=torch.float32),
                future_covariates=torch.from_numpy(future_full).to(dtype=torch.float32),
                n_targets=n_targets,
                n_covariates=n_covariates,
                n_future_covariates=n_future_covariates,
            )
        )

    return results


def _validate_dataframe(
    df: "pd.DataFrame",
    future_df: "pd.DataFrame | None",
    target_columns: list[str],
    prediction_length: int,
    id_column: str,
    timestamp_column: str,
) -> None:
    """
    Validate DataFrame structure. Raises ValueError on failure.

    Checks:
    - Required columns exist
    - Target columns are numeric
    - All series have >= 3 points
    - future_df has same item_ids and exactly prediction_length rows per series
    """
    import pandas.api.types as ptypes

    required = {id_column, timestamp_column} | set(target_columns)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    for col in target_columns:
        if not ptypes.is_numeric_dtype(df[col]):
            raise ValueError(f"Target column '{col}' must be numeric, got dtype {df[col].dtype}")

    series_sizes = df[id_column].value_counts(sort=False)
    short_series = series_sizes[series_sizes < 3]
    if len(short_series) > 0:
        raise ValueError(
            f"All series must have >= 3 points. Found {len(short_series)} series with fewer."
        )

    if future_df is not None:
        if id_column not in future_df.columns:
            raise ValueError(f"future_df is missing required column: {id_column}")

        if not np.array_equal(np.sort(df[id_column].unique()), np.sort(future_df[id_column].unique())):
            raise ValueError("future_df must have the same item IDs as df")

        future_sizes = future_df[id_column].value_counts(sort=False)
        wrong_length = future_sizes[future_sizes != prediction_length]
        if len(wrong_length) > 0:
            raise ValueError(
                f"future_df must have exactly {prediction_length} rows per item. "
                f"Found {len(wrong_length)} items with wrong length."
            )


def _validate_list_of_dicts(
    data: list[dict],
    prediction_length: int,
) -> None:
    """
    Validate list[dict] structure. Raises ValueError on failure.

    Checks:
    - All dicts have "target" key
    - All targets have same n_targets
    - All past_covariates have same column names
    - All future_covariates have same column names and are subset of past_covariates
    - future_covariates have length == prediction_length
    - past_covariates have length == target length
    """
    if len(data) == 0:
        return

    allowed_keys = {"target", "past_covariates", "future_covariates"}
    first_past_keys = sorted(data[0].get("past_covariates", {}).keys())
    first_future_keys = sorted(data[0].get("future_covariates", {}).keys())
    first_target = np.asarray(data[0]["target"])
    first_n_targets = 1 if first_target.ndim == 1 else first_target.shape[0]

    if not set(first_future_keys).issubset(first_past_keys):
        raise ValueError(
            f"future_covariates keys must be a subset of past_covariates keys. "
            f"Got past={first_past_keys}, future={first_future_keys}"
        )

    for idx, d in enumerate(data):
        keys = set(d.keys())
        if not keys.issubset(allowed_keys):
            raise ValueError(f"Invalid keys at index {idx}. Allowed: {allowed_keys}, found: {keys}")
        if "target" not in keys:
            raise ValueError(f"Element at index {idx} is missing required key 'target'")

        target = np.asarray(d["target"])
        if target.ndim > 2:
            raise ValueError(f"Target must be 1-d or 2-d, found shape {tuple(target.shape)} at index {idx}")
        n_targets = 1 if target.ndim == 1 else target.shape[0]
        if n_targets != first_n_targets:
            raise ValueError(
                f"All targets must have same n_targets. Expected {first_n_targets}, got {n_targets} at index {idx}"
            )
        history_length = target.shape[-1]

        past_covariates = d.get("past_covariates", {})
        if not isinstance(past_covariates, dict):
            raise ValueError(f"past_covariates must be a dict at index {idx}, got {type(past_covariates)}")
        if sorted(past_covariates.keys()) != first_past_keys:
            raise ValueError(
                f"All past_covariates must have same keys. Expected {first_past_keys}, "
                f"got {sorted(past_covariates.keys())} at index {idx}"
            )
        for key, val in past_covariates.items():
            val = np.asarray(val)
            if val.ndim != 1 or len(val) != history_length:
                raise ValueError(
                    f"past_covariates['{key}'] must be 1-d with length {history_length}, "
                    f"got shape {tuple(val.shape)} at index {idx}"
                )

        future_covariates = d.get("future_covariates", {})
        if not isinstance(future_covariates, dict):
            raise ValueError(f"future_covariates must be a dict at index {idx}, got {type(future_covariates)}")
        if sorted(future_covariates.keys()) != first_future_keys:
            raise ValueError(
                f"All future_covariates must have same keys. Expected {first_future_keys}, "
                f"got {sorted(future_covariates.keys())} at index {idx}"
            )
        for key, val in future_covariates.items():
            val = np.asarray(val)
            if val.ndim != 1 or len(val) != prediction_length:
                raise ValueError(
                    f"future_covariates['{key}'] must be 1-d with length {prediction_length}, "
                    f"got shape {tuple(val.shape)} at index {idx}"
                )


def _target_encode(
    id_codes: np.ndarray,
    cat_codes: np.ndarray,
    target: np.ndarray,
    n_items: int,
    n_categories: int,
    future_id_codes: np.ndarray | None = None,
    future_cat_codes: np.ndarray | None = None,
    smooth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Per-item target encoding using vectorized bincount operations.

    Computes smoothed mean target value for each (item, category) pair:
        encoded = (smooth * item_mean + category_sum) / (smooth + category_count)

    Assumptions
    -----------
    - id_codes and cat_codes are non-negative integers in [0, n_items) and [0, n_categories)
    - future_id_codes (if provided) are valid item IDs that appear in id_codes
    - future_cat_codes are non-negative integers in [0, n_categories)

    Edge cases
    ----------
    - NaN values in target are excluded from sum/count computations
    - Unseen (item, category) pairs naturally get item_mean via the smoothing formula

    Parameters
    ----------
    id_codes
        Item ID for each row, shape: (n_rows,)
    cat_codes
        Integer category codes, shape: (n_rows,)
    target
        Target values, shape: (n_rows,). May contain NaNs.
    n_items
        Number of unique items
    n_categories
        Number of unique categories
    future_id_codes
        Item ID for each future row, shape: (n_future_rows,). Optional.
    future_cat_codes
        Category codes for future rows, shape: (n_future_rows,). Optional.
    smooth
        Smoothing parameter. Higher values give more weight to item mean vs category mean.

    Returns
    -------
    encoded_past
        Encoded values for past rows, shape: (n_rows,), dtype float32
    encoded_future
        Encoded values for future rows, shape: (n_future_rows,), dtype float32.
        None if future_id_codes and future_cat_codes not provided.
    """
    mask = np.isfinite(target)
    target_masked = np.where(mask, target, 0.0)

    item_sums = np.bincount(id_codes, weights=target_masked * mask, minlength=n_items)
    item_counts = np.bincount(id_codes, weights=mask.astype(float), minlength=n_items)
    item_means = np.divide(item_sums, item_counts, out=np.zeros(n_items), where=item_counts > 0)

    combined_codes = id_codes * n_categories + cat_codes
    sums = np.bincount(combined_codes, weights=target_masked * mask, minlength=n_items * n_categories)
    counts = np.bincount(combined_codes, weights=mask.astype(float), minlength=n_items * n_categories)

    lookup = (smooth * np.repeat(item_means, n_categories) + sums) / (smooth + counts)
    encoded_past = lookup[combined_codes].astype(np.float32)

    encoded_future = None
    if future_id_codes is not None and future_cat_codes is not None:
        encoded_future = lookup[future_id_codes * n_categories + future_cat_codes].astype(np.float32)

    return encoded_past, encoded_future
