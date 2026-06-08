# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Preprocessing module for converting various input formats to list[PreparedInput] expected by Chronos2Dataset.
"""

from typing import TYPE_CHECKING, Any, TypedDict

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
            f"Expected 3-d tensor with shape (n_series, n_variates, history_length), "
            f"got shape {tuple(data.shape)}."
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
                f"Each element should be 1-d or 2-d, with shape (history_length,) or "
                f"(n_variates, history_length). Found element at index {idx} with shape {tuple(item.shape)}."
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
        When True (default), validates schema and sorts both dataframes by (id_column, timestamp_column).
        When False, skips both — caller is responsible for the assumptions listed above.

    Returns
    -------
    list[PreparedInput], one per unique item_id, in the order in which item_ids first appear in `df`.
    """
    if future_df is not None and known_covariates_names is not None:
        raise ValueError("Cannot provide both future_df and known_covariates_names")

    import pandas as pd
    import pandas.api.types as ptypes

    df = _normalize_df(df, id_column, timestamp_column, coerce_timestamps=validate_inputs)
    if future_df is not None:
        future_df = _normalize_df(
            future_df, id_column, timestamp_column,
            coerce_timestamps=validate_inputs,
            order=pd.unique(df[id_column]),
        )

    if validate_inputs:
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
            # Pass object dtype through; _build_prepared_inputs factorizes via pd.factorize.
            past_covariates[col] = df[col].to_numpy()

    future_covariates: dict[str, np.ndarray | None] = {}
    if future_df is not None:
        for col in covariate_columns:
            if col in future_df.columns:
                # dtype from df (not future_df) is the source of truth
                if ptypes.is_numeric_dtype(df[col]):
                    future_covariates[col] = future_df[col].to_numpy(dtype=np.float32, na_value=np.nan)
                else:
                    future_covariates[col] = future_df[col].to_numpy()
    elif known_covariates_names is not None:
        for col in known_covariates_names:
            if col in past_covariates:
                future_covariates[col] = None

    # df is already grouped by id; value_counts(sort=False) returns lengths in that order.
    series_lengths = df[id_column].value_counts(sort=False).tolist()

    return _build_prepared_inputs(
        target=target,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        series_lengths=series_lengths,
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
    if len(data) == 0:
        return []

    if validate_inputs:
        _validate_list_of_dicts(data=data, prediction_length=prediction_length)

    first_future_dict = data[0].get("future_covariates") or {}
    if first_future_dict and known_covariates_names is not None:
        raise ValueError("Cannot provide both known_covariates_names and future_covariates in dicts")

    past_covariate_keys = sorted(data[0].get("past_covariates", {}).keys())
    future_covariate_keys = sorted(first_future_dict.keys())

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

    past_covariates = {key: _stack_covariate(data, "past_covariates", key) for key in past_covariate_keys}

    # Per-key handling: a future-covariate value of None/empty marks the column as
    # known-future-but-values-unavailable (NaN-filled). The validator guarantees that
    # all dicts agree per key, so checking the first dict is sufficient.
    future_covariates: dict[str, np.ndarray | None] = {}
    for key in future_covariate_keys:
        first_value = data[0]["future_covariates"][key]
        if first_value is None or len(first_value) == 0:
            future_covariates[key] = None
        else:
            future_covariates[key] = _stack_covariate(data, "future_covariates", key)

    if known_covariates_names is not None:
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


def _stack_covariate(data: list[dict], group: str, key: str) -> np.ndarray:
    """Concatenate a single covariate column across all dicts. Returns float32 if numeric, else object."""
    stacked = np.concatenate([np.asarray(d[group][key]) for d in data])
    if np.issubdtype(stacked.dtype, np.number):
        return stacked.astype(np.float32)
    return stacked.astype(object)


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

    # past-only first, known-future last (Chronos2Dataset relies on this row order)
    past_only = [k for k in past_covariates if k not in future_covariates]
    known_future = [k for k in past_covariates if k in future_covariates]

    for key in past_only + known_future:
        values = past_covariates[key]
        is_known_future = key in future_covariates
        future_values = future_covariates.get(key)

        if not np.issubdtype(values.dtype, np.number):
            import pandas as pd

            # use_na_sentinel=False keeps NaN/None as a real category (matches user intent:
            # "missing is just another value"). pd.factorize is one C-level pass, no sort.
            past_codes, categories = pd.factorize(values, use_na_sentinel=False)
            past_codes = past_codes.astype(np.intp, copy=False)
            n_categories = len(categories)

            future_codes = None
            if future_values is not None:
                # get_indexer returns -1 for unseen; remap to n_categories (the "unseen" slot).
                future_codes = pd.Index(categories).get_indexer(future_values)
                future_codes = np.where(future_codes < 0, n_categories, future_codes).astype(np.intp)

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
                    if future_codes is not None:
                        encoded = future_codes.astype(np.float32)
                        encoded[future_codes == n_categories] = np.nan
                        encoded_future.append(encoded)
                    else:
                        encoded_future.append(nan_future)
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


def _normalize_df(
    df: "pd.DataFrame",
    id_column: str,
    timestamp_column: str,
    coerce_timestamps: bool,
    order: "np.ndarray | None" = None,
) -> "pd.DataFrame":
    """
    Return a df with rows grouped by id (in first-appearance order, or `order` if given) and
    sorted by timestamp within each group. Optionally coerces the timestamp column to datetime.
    Skips the sort if rows are already in that layout.
    """
    import pandas as pd
    import pandas.api.types as ptypes

    if coerce_timestamps and not ptypes.is_datetime64_any_dtype(df[timestamp_column]):
        df = df.assign(**{timestamp_column: pd.to_datetime(df[timestamp_column])})

    if order is None:
        codes, _ = pd.factorize(df[id_column])
    else:
        codes = pd.Index(order).get_indexer(df[id_column])
        if (codes < 0).any():
            missing = pd.unique(df[id_column][codes < 0])
            raise ValueError(f"future_df has ids not present in df: {list(missing)[:5]}")

    ts = df[timestamp_column].to_numpy()
    code_diff = np.diff(codes)
    grouped = bool(np.all(code_diff >= 0))
    sorted_within = grouped and bool(np.all((np.diff(ts) >= 0) | (code_diff > 0)))
    if not sorted_within:
        perm = np.lexsort([ts, codes])
        df = df.iloc[perm].reset_index(drop=True)
    return df


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
    - Each dict has only allowed keys, and "target" is present
    - past_covariates / future_covariates (when present) are dicts
    - All targets have the same n_targets and are 1-d or 2-d
    - All past_covariates have the same column names across dicts
    - All future_covariates have the same column names and are a subset of past_covariates
    - future_covariates values are None, empty, or 1-d with length == prediction_length
    - For a given future-covariate key, all dicts must agree on availability (None or non-None)
    - past_covariates values are 1-d with length == target's history length
    """
    if len(data) == 0:
        return

    allowed_keys = {"target", "past_covariates", "future_covariates"}

    # First pass: validate types and shapes per element so callers get actionable errors
    # for any malformed dict, regardless of which one it is.
    for idx, d in enumerate(data):
        keys = set(d.keys())
        if not keys.issubset(allowed_keys):
            raise ValueError(
                f"Found invalid keys in element at index {idx}. Allowed keys are {allowed_keys}, but found {keys}"
            )
        if "target" not in keys:
            raise ValueError(f"Element at index {idx} does not contain the required key 'target'")

        target = np.asarray(d["target"])
        if target.ndim > 2:
            raise ValueError(
                f"Target must be 1-d or 2-d (got shape {tuple(target.shape)} at index {idx}). "
                f"When the input is a list of dicts, the `target` should either be 1-d with shape "
                f"(history_length,) or 2-d with shape (n_variates, history_length)."
            )
        history_length = target.shape[-1]

        past_covariates = d.get("past_covariates", {})
        if not isinstance(past_covariates, dict):
            raise ValueError(
                f"Found invalid type for `past_covariates` in element at index {idx}. "
                f'Expected dict with {{"feat_1": tensor_1, ...}}, but found {type(past_covariates)}'
            )
        for key, val in past_covariates.items():
            arr = np.asarray(val)
            if arr.ndim != 1 or len(arr) != history_length:
                raise ValueError(
                    f"Individual `past_covariates` must be 1-d with length equal to the length of `target` "
                    f"(= {history_length}), found: {key} with shape {tuple(arr.shape)} in element at index {idx}"
                )

        future_covariates = d.get("future_covariates", {})
        if not isinstance(future_covariates, dict):
            raise ValueError(
                f"Found invalid type for `future_covariates` in element at index {idx}. "
                f'Expected dict with {{"feat_1": tensor_1, ...}}, but found {type(future_covariates)}'
            )
        for key, val in future_covariates.items():
            if val is None or (hasattr(val, "__len__") and len(val) == 0):
                continue
            arr = np.asarray(val)
            if arr.ndim != 1 or len(arr) != prediction_length:
                raise ValueError(
                    f"Individual `future_covariates` must be 1-d with length equal to the "
                    f"prediction_length={prediction_length}, got shape {tuple(arr.shape)} at index {idx}"
                )

    # Second pass: cross-element consistency (homogeneous schema).
    first_target = np.asarray(data[0]["target"])
    first_n_targets = 1 if first_target.ndim == 1 else first_target.shape[0]
    first_past_keys = sorted(data[0].get("past_covariates", {}).keys())
    first_future_keys = sorted(data[0].get("future_covariates", {}).keys())
    first_future_availability = {
        k: _is_unavailable(data[0]["future_covariates"][k]) for k in first_future_keys
    }

    if not set(first_future_keys).issubset(first_past_keys):
        raise ValueError(
            f"Expected keys in `future_covariates` must be a subset of `past_covariates` {first_past_keys}, "
            f"but found {first_future_keys}"
        )

    for idx, d in enumerate(data):
        target = np.asarray(d["target"])
        n_targets = 1 if target.ndim == 1 else target.shape[0]
        if n_targets != first_n_targets:
            raise ValueError(
                f"All targets must have the same n_targets. Expected {first_n_targets}, got {n_targets} "
                f"at index {idx}. Heterogeneous lists with different target shapes are not supported — "
                f"please loop over the inputs and call the model per-item instead."
            )

        past_keys = sorted(d.get("past_covariates", {}).keys())
        if past_keys != first_past_keys:
            raise ValueError(
                f"All past_covariates must have same keys. Expected {first_past_keys}, "
                f"got {past_keys} at index {idx}. Heterogeneous lists are not supported."
            )

        future_dict = d.get("future_covariates", {})
        if sorted(future_dict.keys()) != first_future_keys:
            raise ValueError(
                f"All future_covariates must have same keys. Expected {first_future_keys}, "
                f"got {sorted(future_dict.keys())} at index {idx}. Heterogeneous lists are not supported."
            )
        for key in first_future_keys:
            if _is_unavailable(future_dict[key]) != first_future_availability[key]:
                raise ValueError(
                    f"All dicts must agree on whether `future_covariates['{key}']` is available "
                    f"(None/empty) or provided. Mismatch at index {idx}."
                )


def _is_unavailable(value: Any) -> bool:
    """A future-covariate value is 'unavailable' if it is None or an empty sequence."""
    return value is None or (hasattr(value, "__len__") and len(value) == 0)


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

    # extra slot for unseen categories — sums/counts stay 0, so lookup → item_mean
    n_slots = n_categories + 1
    combined_codes = id_codes * n_slots + cat_codes
    sums = np.bincount(combined_codes, weights=target_masked * mask, minlength=n_items * n_slots)
    counts = np.bincount(combined_codes, weights=mask.astype(float), minlength=n_items * n_slots)

    lookup = (smooth * np.repeat(item_means, n_slots) + sums) / (smooth + counts)
    encoded_past = lookup[combined_codes].astype(np.float32)

    encoded_future = None
    if future_id_codes is not None and future_cat_codes is not None:
        encoded_future = lookup[future_id_codes * n_slots + future_cat_codes].astype(np.float32)

    return encoded_past, encoded_future
