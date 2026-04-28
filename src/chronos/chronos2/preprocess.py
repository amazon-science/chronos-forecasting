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
    ...


def from_tensor_list(
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
    ...


def from_dataframe(
    df: "pd.DataFrame",
    target_columns: list[str],
    prediction_length: int,
    future_df: "pd.DataFrame | None" = None,
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
        Optional DataFrame with future covariate values (same id_column, timestamp_column)
    id_column
        Column name for series ID
    timestamp_column
        Column name for timestamps
    use_target_encoding
        When True (default), use target encoding for categoricals (requires single target).
        When False, use ordinal encoding.
    validate_inputs
        When True (default), validates dataframes. Set False to skip validation.

    Returns
    -------
    list[PreparedInput], one per unique item_id (in original order)
    """
    ...


def from_dict_list(
    data: list[dict],
    prediction_length: int,
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
    use_target_encoding
        When True (default), use target encoding for categoricals (requires single target).
        When False, use ordinal encoding.
    validate_inputs
        When True (default), validates all dicts have consistent structure.

    Returns
    -------
    list[PreparedInput], one per dict
    """
    ...


def _build_prepared_inputs(
    target: np.ndarray,
    past_covariates: dict[str, np.ndarray],
    future_covariates: dict[str, np.ndarray],
    series_lengths: list[int],
    prediction_length: int,
    use_target_encoding: bool,
) -> list[PreparedInput]:
    """
    Build list[PreparedInput] from stacked arrays. Handles categorical encoding.

    Assumptions
    -----------
    - Arrays are stacked in item order (item 0's rows first, then item 1's, etc.)
    - future_covariates keys are a subset of past_covariates keys
    - Categorical columns have object dtype; numeric columns have float32 dtype

    Parameters
    ----------
    target
        Shape: (n_targets, total_context_rows), dtype float32
    past_covariates
        {name: values} for all covariates (past-only and known-future)
        Each array shape: (total_context_rows,)
    future_covariates
        {name: values} for known-future covariates only
        Each array shape: (n_series * prediction_length,)
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
    ...


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
    - Consistent frequency across series
    - future_df has same item_ids and exactly prediction_length rows per series
    """
    ...


def _validate_dict_list(
    data: list[dict],
    prediction_length: int,
) -> None:
    """
    Validate list[dict] structure. Raises ValueError on failure.

    Checks:
    - All dicts have same keys
    - All targets have same n_targets
    - All past_covariates have same column names
    - All future_covariates have same column names and are subset of past_covariates
    - future_covariates have length == prediction_length
    """
    ...


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
    - future_cat_codes may contain -1 for unseen categories (encoded as NaN)

    Edge cases
    ----------
    - NaN values in target are excluded from sum/count computations
    - Unseen (item, category) pairs get the item mean as fallback (via smoothing formula)
    - Completely unseen categories in future (cat_code=-1) get the item mean

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
        Use -1 for categories not seen in past (will be encoded as NaN).
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
        valid_future = future_cat_codes >= 0
        future_combined = np.where(valid_future, future_id_codes * n_categories + future_cat_codes, 0)
        encoded_future = np.where(
            valid_future,
            lookup[future_combined],
            item_means[future_id_codes]
        ).astype(np.float32)

    return encoded_past, encoded_future
