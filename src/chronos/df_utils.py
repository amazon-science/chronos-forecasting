# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>


import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _validate_df_types_and_cast(
    df: "pd.DataFrame",
    future_df: "pd.DataFrame | None",
    target_columns: list[str],
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> tuple["pd.DataFrame", "pd.DataFrame | None"]:
    import pandas as pd

    astype_dict = {}
    future_astype_dict = {}
    for col in df.columns.drop([id_column, timestamp_column]):
        col_dtype = df[col].dtype
        if col in target_columns and not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"All target columns must be numeric but got {col=} with dtype={col_dtype}")

        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or isinstance(col_dtype, pd.CategoricalDtype)
        ):
            astype_dict[col] = "category"
        elif pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            astype_dict[col] = "float32"
        else:
            raise ValueError(
                f"All columns must contain numeric, object, category, string, or bool dtype but got {col=} with dtype={col_dtype}"
            )

        if future_df is not None and col in future_df.columns:
            if future_df[col].dtype != col_dtype:
                raise ValueError(
                    f"Column {col} in future_df has dtype {future_df[col].dtype} but column in df has dtype {col_dtype}"
                )
            future_astype_dict[col] = astype_dict[col]

    df = df.astype(astype_dict, copy=True)
    if future_df is not None:
        future_df = future_df.astype(future_astype_dict, copy=True)

    return df, future_df


def validate_df_inputs(
    df: "pd.DataFrame",
    future_df: "pd.DataFrame | None",
    target_columns: list[str],
    prediction_length: int,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> tuple["pd.DataFrame", "pd.DataFrame | None", str, list[int], np.ndarray]:
    """
    Validates and prepares dataframe inputs

    Parameters
    ----------
    df
        Input dataframe containing time series data with columns:
        - id_column: Identifier for each time series
        - timestamp_column: Timestamps for each observation
        - target_columns: One or more target variables to forecast
        - Additional columns are treated as covariates
    future_df
        Optional dataframe containing future covariate values with columns:
        - id_column: Identifier for each time series
        - timestamp_column: Future timestamps
        - Subset of covariate columns from df
    target_columns
        Names of target columns to forecast
    prediction_length
        Number of future time steps to predict
    id_column
        Name of column containing time series identifiers
    timestamp_column
        Name of column containing timestamps

    Returns
    -------
    A tuple containing:
    - Validated and sorted input dataframe
    - Validated and sorted future dataframe (if provided)
    - Inferred frequency of the time series
    - List of series lengths from input dataframe
    - Original order of time series IDs

    Raises
    ------
    ValueError
        If validation fails for:
        - Missing required columns
        - Invalid data types
        - Inconsistent frequencies
        - Insufficient data points
        - Mismatched series between df and future_df
        - Invalid future_df lengths
    """

    import pandas as pd

    required_cols = [id_column, timestamp_column] + target_columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"df does not contain all expected columns. Missing columns: {missing_cols}")

    if future_df is not None:
        future_required_cols = [id_column, timestamp_column]
        missing_future_cols = [col for col in future_required_cols if col not in future_df.columns]
        targets_in_future = [col for col in future_df.columns if col in target_columns]
        extra_future_cols = [col for col in future_df.columns if col not in df.columns]
        if missing_future_cols:
            raise ValueError(
                f"future_df does not contain all expected columns. Missing columns: {missing_future_cols}"
            )
        if targets_in_future:
            raise ValueError(
                f"future_df cannot contain target columns. Target columns found in future_df: {targets_in_future}"
            )
        if extra_future_cols:
            raise ValueError(f"future_df cannot contain columns not present in df. Extra columns: {extra_future_cols}")

    df, future_df = _validate_df_types_and_cast(
        df, future_df, id_column=id_column, timestamp_column=timestamp_column, target_columns=target_columns
    )

    # Get the original order of time series IDs
    original_order = df[id_column].unique()

    # Sort and prepare df
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values([id_column, timestamp_column])

    # Get series lengths
    series_lengths = df[id_column].value_counts(sort=False).to_list()

    def validate_freq(timestamps: pd.DatetimeIndex, series_id: str):
        freq = pd.infer_freq(timestamps)
        if not freq:
            raise ValueError(f"Could not infer frequency for series {series_id}")
        return freq

    # Validate each series
    all_freqs = []
    start_idx = 0
    timestamp_index = pd.DatetimeIndex(df[timestamp_column])
    for length in series_lengths:
        if length < 3:
            series_id = df[id_column].iloc[start_idx]
            raise ValueError(
                f"Every time series must have at least 3 data points, found {length=} for series {series_id}"
            )
        timestamps = timestamp_index[start_idx : start_idx + length]
        series_id = df[id_column].iloc[start_idx]
        all_freqs.append(validate_freq(timestamps, series_id))
        start_idx += length

    if len(set(all_freqs)) > 1:
        raise ValueError("All time series must have the same frequency")

    inferred_freq = all_freqs[0]

    # Sort future_df if provided and validate its series lengths
    future_series_lengths = None
    if future_df is not None:
        future_df[timestamp_column] = pd.to_datetime(future_df[timestamp_column])
        future_df = future_df.sort_values([id_column, timestamp_column])

        # Validate that future_df contains all series from df
        context_ids = set(df[id_column].unique())
        future_ids = set(future_df[id_column].unique())
        if context_ids != future_ids:
            raise ValueError("future_df must contain the same time series IDs as df")

        future_series_lengths = future_df[id_column].value_counts(sort=False)
        if (future_series_lengths != prediction_length).any():
            invalid_series = future_series_lengths[future_series_lengths != prediction_length]
            raise ValueError(
                f"future_df must contain {prediction_length=} values for each series, "
                f"but found series with different lengths: {invalid_series.to_dict()}"
            )

    return df, future_df, inferred_freq, series_lengths, original_order


def convert_df_input_to_list_of_dicts_input(
    df: "pd.DataFrame",
    future_df: "pd.DataFrame | None",
    target_columns: list[str],
    prediction_length: int,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
    validate_inputs: bool = True,
    freq: str | None = None,
) -> tuple[list[dict[str, np.ndarray | dict[str, np.ndarray]]], np.ndarray, dict[str, "pd.DatetimeIndex"]]:
    """
    Convert from dataframe input format to a list of dictionaries input format.

    Parameters
    ----------
    df
        Input dataframe containing time series data with columns:
        - id_column: Identifier for each time series
        - timestamp_column: Timestamps for each observation
        - target_columns: One or more target variables to forecast
        - Additional columns are treated as covariates
    future_df
        Optional dataframe containing future covariate values with columns:
        - id_column: Identifier for each time series
        - timestamp_column: Future timestamps
        - Subset of covariate columns from df
    target_columns
        Names of target columns to forecast
    prediction_length
        Number of future time steps to predict
    id_column
        Name of column containing time series identifiers
    timestamp_column
        Name of column containing timestamps
    validate_inputs
        [ADVANCED] When True (default), validates dataframes before prediction. Setting to False removes the
        validation overhead, but may silently lead to wrong predictions if data is misformatted. When False, you
        must ensure: (1) all dataframes are sorted by (id_column, timestamp_column); (2) future_df (if provided)
        has the same item IDs as df with exactly prediction_length rows of future timestamps per item; (3) all
        timestamps are regularly spaced (e.g., with hourly frequency).
    freq
        Frequency string for timestamp generation (e.g., "h", "D", "W"). Can only be used
        when validate_inputs=False. When provided, skips frequency inference from the data.

    Returns
    -------
    A tuple containing:
    - Time series converted to list of dictionaries format
    - Original order of time series IDs
    - Dictionary mapping series IDs to future time index
    """

    import pandas as pd

    if freq is not None and validate_inputs:
        raise ValueError(
            "freq can only be provided when validate_inputs=False. "
            "When using freq with validate_inputs=False, you must ensure: "
            "(1) all dataframes are sorted by (id_column, timestamp_column);  "
            "(2) future_df (if provided) has the same item IDs as df with exactly "
            "prediction_length rows of future timestamps per item; "
            "(3) all timestamps are regularly spaced."
        )

    if validate_inputs:
        df, future_df, freq, series_lengths, original_order = validate_df_inputs(
            df,
            future_df=future_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_columns=target_columns,
            prediction_length=prediction_length,
        )
    else:
        # Get the original order of time series IDs
        original_order = df[id_column].unique()

        # Get series lengths
        series_lengths = df[id_column].value_counts(sort=False).to_list()

        # If freq is not provided, infer from the first series with >= 3 points
        if freq is None:
            timestamp_index = pd.DatetimeIndex(df[timestamp_column])
            start_idx = 0
            for length in series_lengths:
                if length < 3:
                    start_idx += length
                    continue
                timestamps = timestamp_index[start_idx : start_idx + length]
                freq = pd.infer_freq(timestamps)
                break

            assert freq is not None, "validate_inputs is False, but could not infer frequency from the dataframe"

    # Convert to list of dicts format
    inputs: list[dict[str, np.ndarray | dict[str, np.ndarray]]] = []
    prediction_timestamps: dict[str, pd.DatetimeIndex] = {}

    indptr = np.concatenate([[0], np.cumsum(series_lengths)]).astype("int64")
    target_array = df[target_columns].to_numpy().T  # Shape: (n_targets, len(df))
    last_ts = pd.DatetimeIndex(df[timestamp_column].iloc[indptr[1:] - 1])  # Shape: (n_series,)
    offset = pd.tseries.frequencies.to_offset(freq)
    with warnings.catch_warnings():
        # Silence PerformanceWarning for non-vectorized offsets https://github.com/pandas-dev/pandas/blob/95624ca2e99b0/pandas/core/arrays/datetimes.py#L822
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        # Generate all prediction timestamps at once by stacking offsets into shape (n_series * prediction_length)
        prediction_timestamps_array = pd.DatetimeIndex(
            np.dstack([last_ts + step * offset for step in range(1, prediction_length + 1)]).ravel()
        )

    past_covariates_dict = {
        col: df[col].to_numpy() for col in df.columns if col not in [id_column, timestamp_column] + target_columns
    }
    future_covariates_dict = {}
    if future_df is not None:
        for col in future_df.columns.drop([id_column, timestamp_column]):
            future_covariates_dict[col] = future_df[col].to_numpy()
        if validate_inputs:
            if (pd.DatetimeIndex(future_df[timestamp_column]) != pd.DatetimeIndex(prediction_timestamps_array)).any():
                raise ValueError(
                    "future_df timestamps do not match the expected prediction timestamps. "
                    "You can disable this check by setting `validate_inputs=False`"
                )

    for i in range(len(series_lengths)):
        start_idx, end_idx = indptr[i], indptr[i + 1]
        future_start_idx, future_end_idx = i * prediction_length, (i + 1) * prediction_length

        series_id = df[id_column].iloc[start_idx]
        prediction_timestamps[series_id] = prediction_timestamps_array[future_start_idx:future_end_idx]
        task: dict[str, np.ndarray | dict[str, np.ndarray]] = {"target": target_array[:, start_idx:end_idx]}

        if len(past_covariates_dict) > 0:
            task["past_covariates"] = {col: values[start_idx:end_idx] for col, values in past_covariates_dict.items()}
            if len(future_covariates_dict) > 0:
                task["future_covariates"] = {
                    col: values[future_start_idx:future_end_idx] for col, values in future_covariates_dict.items()
                }
        inputs.append(task)

    assert len(inputs) == len(series_lengths)

    return inputs, original_order, prediction_timestamps
