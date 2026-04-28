# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from chronos.chronos2.dataset import PreparedInput

if TYPE_CHECKING:
    import pandas as pd


def _target_encode(
    id_codes: np.ndarray,
    cat_codes: np.ndarray,
    target: np.ndarray,
    n_items: int,
    n_categories: int,
    smooth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-item target encoding using bincount. Returns (encoded_values, lookup_table)."""
    item_sums = np.bincount(id_codes, weights=target, minlength=n_items)
    item_counts = np.bincount(id_codes, minlength=n_items)
    item_means = item_sums / item_counts

    combined_codes = id_codes * n_categories + cat_codes
    sums = np.bincount(combined_codes, weights=target, minlength=n_items * n_categories)
    counts = np.bincount(combined_codes, minlength=n_items * n_categories)

    lookup = (smooth * np.repeat(item_means, n_categories) + sums) / (smooth + counts)
    return lookup[combined_codes].astype(np.float32), lookup.reshape(n_items, n_categories)


def convert_df_to_prepared_inputs(
    df: "pd.DataFrame",
    target_columns: list[str],
    prediction_length: int,
    future_df: "pd.DataFrame | None" = None,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
    categorical_encoding: Literal["target", "ordinal"] = "target",
) -> list[PreparedInput]:
    """Convert long-format DataFrame to list[PreparedInput] efficiently."""
    import pandas as pd

    df = df.sort_values([id_column, timestamp_column])
    id_codes, id_categories = pd.factorize(df[id_column], sort=False)
    n_items = len(id_categories)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(id_codes, minlength=n_items))])

    # Covariate columns: past-only first, then known-future
    all_covariate_columns = sorted(set(df.columns) - {id_column, timestamp_column} - set(target_columns))
    known_future_columns = sorted([c for c in all_covariate_columns if future_df is not None and c in future_df.columns])
    covariate_columns = [c for c in all_covariate_columns if c not in known_future_columns] + known_future_columns
    categorical_columns = [c for c in covariate_columns if not pd.api.types.is_numeric_dtype(df[c])]

    use_target_encoding = categorical_encoding == "target" and len(target_columns) == 1
    target_values = df[target_columns[0]].values if use_target_encoding else None

    # Encode categorical columns
    encoded_categoricals: dict[str, np.ndarray] = {}
    encoding_lookups: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # (lookup_table, categories)

    for col in categorical_columns:
        cat_codes, categories = pd.factorize(df[col], sort=False)
        if use_target_encoding:
            encoded_categoricals[col], lookup = _target_encode(
                id_codes, cat_codes, target_values, n_items, len(categories)
            )
            encoding_lookups[col] = (lookup, categories)
        else:
            encoded_categoricals[col] = np.where(cat_codes >= 0, cat_codes, np.nan).astype(np.float32)
            encoding_lookups[col] = (None, categories)

    # Build context array: (n_targets + n_covariates, n_rows)
    context_arrays = [df[target_columns].to_numpy(dtype=np.float32).T]
    for col in covariate_columns:
        if col in categorical_columns:
            context_arrays.append(encoded_categoricals[col])
        else:
            context_arrays.append(df[col].to_numpy(dtype=np.float32))
    context_full = np.vstack(context_arrays)

    # Build future covariate array if provided
    future_covariates_full = None
    future_indptr = None
    if future_df is not None and known_future_columns:
        future_df = future_df.sort_values([id_column, timestamp_column])
        future_id_codes = pd.Categorical(future_df[id_column], categories=id_categories).codes
        future_indptr = np.concatenate([[0], np.cumsum(np.bincount(future_id_codes, minlength=n_items))])

        future_arrays = []
        for col in known_future_columns:
            if col not in categorical_columns:
                future_arrays.append(future_df[col].to_numpy(dtype=np.float32))
            else:
                lookup, categories = encoding_lookups[col]
                future_cat_codes = pd.Categorical(future_df[col], categories=categories).codes
                if use_target_encoding:
                    encoded = np.where(future_cat_codes >= 0, lookup[future_id_codes, future_cat_codes], np.nan)
                else:
                    encoded = np.where(future_cat_codes >= 0, future_cat_codes, np.nan)
                future_arrays.append(encoded.astype(np.float32))
        future_covariates_full = np.vstack(future_arrays)

    # Assemble PreparedInputs
    n_targets = len(target_columns)
    n_covariates = len(covariate_columns)
    n_future_covariates = len(known_future_columns)
    nan_padding = np.full((n_targets + n_covariates - n_future_covariates, prediction_length), np.nan, dtype=np.float32)

    inputs = []
    for i in range(n_items):
        context = context_full[:, indptr[i]:indptr[i + 1]]

        if future_covariates_full is not None:
            future_covariates = np.vstack([
                nan_padding,
                future_covariates_full[:, future_indptr[i]:future_indptr[i + 1]]
            ])
        else:
            future_covariates = np.full((n_targets + n_covariates, prediction_length), np.nan, dtype=np.float32)

        inputs.append(PreparedInput(
            context=torch.from_numpy(context.copy()),
            future_covariates=torch.from_numpy(future_covariates.copy()),
            n_targets=n_targets,
            n_covariates=n_covariates,
            n_future_covariates=n_future_covariates,
        ))

    return inputs
