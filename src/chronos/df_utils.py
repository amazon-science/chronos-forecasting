# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>


import warnings

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

__all__ = [
    "infer_freq_from_df",
    "make_future_df",
    "normalize_df",
    "validate_df",
    "validate_and_normalize_df",
]


def infer_freq_from_df(
    df: pd.DataFrame,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> str:
    """
    Infer the (shared) frequency of the time series in a normalized df.

    ``pd.infer_freq`` requires at least 3 observations, so series shorter than that are
    skipped (they cannot pin down a frequency on their own). Every series with >= 3 points
    must have a regular, inferrable frequency, and all such series must agree. Raises
    ValueError if any qualifying series is irregular, if they disagree, or if no series has
    >= 3 points — in the last case the caller must provide an explicit ``freq``.

    Assumes ``df`` is already grouped by id (e.g. via ``normalize_df``).
    """
    series_lengths = df[id_column].value_counts(sort=False).to_list()
    item_ids = df[id_column].to_numpy()
    timestamp_index = pd.DatetimeIndex(df[timestamp_column])

    freqs: set[str] = set()
    start_idx = 0
    for length in series_lengths:
        if length >= 3:
            freq = pd.infer_freq(timestamp_index[start_idx : start_idx + length])
            if freq is None:
                raise ValueError(f"Could not infer frequency for series {item_ids[start_idx]}")
            freqs.add(freq)
        start_idx += length

    if len(freqs) > 1:
        raise ValueError("All time series must have the same frequency")
    if not freqs:
        raise ValueError(
            "Could not infer frequency: no time series has at least 3 regularly-spaced observations. "
            "Please provide an explicit `freq`."
        )
    return freqs.pop()


def make_future_df(
    df: pd.DataFrame,
    prediction_length: int,
    freq: "str | None" = None,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Build the forecast-horizon timestamps for each series in a normalized df.

    For each item, generates the timestamps for the next ``prediction_length`` steps
    after the item's last observed timestamp. Returns a long-format DataFrame with
    columns ``[id_column, timestamp_column]`` and ``n_series * prediction_length`` rows,
    in the df's first-appearance item order with timestamps ascending within each item.

    Assumes ``df`` is already normalized (grouped by id in first-appearance order, sorted
    by timestamp within each group, e.g. via ``normalize_df``). If ``freq`` is None, it is
    inferred via ``infer_freq_from_df``.
    """
    if freq is None:
        freq = infer_freq_from_df(df, id_column=id_column, timestamp_column=timestamp_column)

    series_lengths = df[id_column].value_counts(sort=False).to_list()
    indptr = np.concatenate([[0], np.cumsum(series_lengths)]).astype("int64")
    last_idx = indptr[1:] - 1
    last_ts = pd.DatetimeIndex(df[timestamp_column].to_numpy()[last_idx])  # (n_series,)
    item_ids = df[id_column].to_numpy()[indptr[:-1]]  # first-appearance order

    offset = pd.tseries.frequencies.to_offset(freq)
    with warnings.catch_warnings():
        # Silence PerformanceWarning for non-vectorized offsets (e.g. BusinessDay)
        # https://github.com/pandas-dev/pandas/blob/95624ca2e99b0/pandas/core/arrays/datetimes.py#L822
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        future_ts = np.stack([last_ts + step * offset for step in range(1, prediction_length + 1)], axis=1).ravel()

    return pd.DataFrame(
        {
            id_column: np.repeat(item_ids, prediction_length),
            timestamp_column: pd.DatetimeIndex(future_ts),
        }
    )


def normalize_df(
    df: pd.DataFrame,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
    order: "np.ndarray | None" = None,
) -> pd.DataFrame:
    """
    Return a df with the timestamp column coerced to datetime, rows grouped by id (in
    first-appearance order, or `order` if given), and sorted by timestamp within each group.
    Skips the sort if rows are already in that layout.
    """
    if not ptypes.is_datetime64_any_dtype(df[timestamp_column]):
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


def validate_df(
    df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    target_columns: list[str],
    known_covariates_names: list[str] | None,
    prediction_length: int,
    id_column: str,
    timestamp_column: str,
) -> None:
    """
    Validate DataFrame structure. Raises ValueError on failure.

    Checks:
    - Required columns exist
    - Target columns are numeric
    - known_covariates_names are all covariate columns
    - future_df has the expected columns (id + timestamp, no targets, no columns absent from df)
    - future_df has the same item_ids and exactly prediction_length rows per series
    """
    required = {id_column, timestamp_column} | set(target_columns)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df does not contain all expected columns. Missing columns: {missing}")

    for col in target_columns:
        if not ptypes.is_numeric_dtype(df[col]):
            raise ValueError(f"Target column '{col}' must be numeric, got dtype {df[col].dtype}")

    if known_covariates_names is not None:
        covariate_columns = set(df.columns) - {id_column, timestamp_column} - set(target_columns)
        unknown = set(known_covariates_names) - covariate_columns
        if unknown:
            raise ValueError(f"known_covariates_names contains columns not present in df: {unknown}")

    if future_df is not None:
        missing_future = {id_column, timestamp_column} - set(future_df.columns)
        if missing_future:
            raise ValueError(f"future_df does not contain all expected columns. Missing columns: {missing_future}")

        targets_in_future = [c for c in future_df.columns if c in target_columns]
        if targets_in_future:
            raise ValueError(f"future_df cannot contain target columns. Target columns found: {targets_in_future}")

        extra_future = [c for c in future_df.columns if c not in df.columns]
        if extra_future:
            raise ValueError(f"future_df cannot contain columns not present in df. Extra columns: {extra_future}")

        if not np.array_equal(np.sort(df[id_column].unique()), np.sort(future_df[id_column].unique())):
            raise ValueError("future_df must have the same time series IDs as df")

        future_sizes = future_df[id_column].value_counts(sort=False)
        wrong_length = future_sizes[future_sizes != prediction_length]
        if len(wrong_length) > 0:
            raise ValueError(
                f"future_df must contain prediction_length={prediction_length} rows per item. "
                f"Found {len(wrong_length)} items with a different number of rows."
            )


def validate_and_normalize_df(
    df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    target_columns: list[str],
    prediction_length: int,
    known_covariates_names: list[str] | None = None,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Validate and normalize df (and future_df) for the DataFrame-based prediction paths.

    Runs ``validate_df`` then ``normalize_df`` so that the returned frames are grouped by id
    (first-appearance order) and sorted by timestamp within each group. This is the single
    source of truth for df preparation; callers that have already prepared their inputs pass
    them straight to ``from_data_frame``/``make_future_df`` without repeating this step.
    """
    validate_df(
        df=df,
        future_df=future_df,
        target_columns=target_columns,
        known_covariates_names=known_covariates_names,
        prediction_length=prediction_length,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    df = normalize_df(df, id_column=id_column, timestamp_column=timestamp_column)
    if future_df is not None:
        future_df = normalize_df(
            future_df, id_column=id_column, timestamp_column=timestamp_column, order=pd.unique(df[id_column])
        )
    return df, future_df


def validate_df_inputs(*args, **kwargs):
    raise RuntimeError(
        "`validate_df_inputs` has been deprecated. "
        "Please use `chronos.df_utils.validate_df` and `chronos.df_utils.normalize_df` instead."
    )


def convert_df_input_to_list_of_dicts_input(
    df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    target_columns: list[str],
    prediction_length: int,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
    validate_inputs: bool = True,
    freq: str | None = None,
) -> tuple[list[dict[str, np.ndarray | dict[str, np.ndarray]]], None, None]:
    # We only keep the implementation around for compatibility with AutoGluon
    # https://github.com/autogluon/autogluon/blob/v1.5.0/timeseries/src/autogluon/timeseries/models/chronos/chronos2.py#L314-L320
    # For all other users, raise RuntimeError and redirect to the new method
    if validate_inputs:
        raise RuntimeError(
            "`convert_df_input_to_list_of_dicts_input` has been deprecated. "
            "Please use `chronos.chronos2.preprocess.from_data_frame` instead."
        )

    # Original order of time series IDs and series lengths (df is grouped by id after normalize_df)
    series_lengths = df[id_column].value_counts(sort=False).to_list()

    # Convert to list of dicts format
    inputs: list[dict[str, np.ndarray | dict[str, np.ndarray]]] = []

    indptr = np.concatenate([[0], np.cumsum(series_lengths)]).astype("int64")
    target_array = df[target_columns].to_numpy().T  # Shape: (n_targets, len(df))

    past_covariates_dict = {
        col: df[col].to_numpy() for col in df.columns if col not in [id_column, timestamp_column] + target_columns
    }
    future_covariates_dict = {}
    if future_df is not None:
        for col in future_df.columns.drop([id_column, timestamp_column]):
            future_covariates_dict[col] = future_df[col].to_numpy()

    for i in range(len(series_lengths)):
        start_idx, end_idx = indptr[i], indptr[i + 1]
        future_start_idx, future_end_idx = i * prediction_length, (i + 1) * prediction_length
        task: dict[str, np.ndarray | dict[str, np.ndarray]] = {"target": target_array[:, start_idx:end_idx]}

        if len(past_covariates_dict) > 0:
            task["past_covariates"] = {col: values[start_idx:end_idx] for col, values in past_covariates_dict.items()}
            if len(future_covariates_dict) > 0:
                task["future_covariates"] = {
                    col: values[future_start_idx:future_end_idx] for col, values in future_covariates_dict.items()
                }
        inputs.append(task)
    return inputs, None, None
