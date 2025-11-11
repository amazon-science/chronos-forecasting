# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>


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
) -> tuple["pd.DataFrame", "pd.DataFrame | None", "pd.Timedelta", list[int], list[int] | None, np.ndarray]:
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
    - List of series lengths from future dataframe (if provided)
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

    def validate_freq(timestamps: pd.Series, series_id: str):
        freq = pd.infer_freq(timestamps)
        if not freq:
            raise ValueError(f"Could not infer frequency for series {series_id}")
        return freq

    # Validate each series
    all_freqs = []
    start_idx = 0
    for length in series_lengths:
        if length < 3:
            series_id = df.iloc[start_idx][id_column]
            raise ValueError(
                f"Every time series must have at least 3 data points, found {length=} for series {series_id}"
            )

        series_data = df.iloc[start_idx : start_idx + length]
        timestamps = series_data[timestamp_column]
        series_id = series_data.iloc[0][id_column]
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

        future_series_lengths = future_df[id_column].value_counts(sort=False).to_list()

        # Validate future series lengths match prediction_length
        future_start_idx = 0
        for future_length in future_series_lengths:
            future_series_data = future_df.iloc[future_start_idx : future_start_idx + future_length]
            future_timestamps = future_series_data[timestamp_column]
            future_series_id = future_series_data.iloc[0][id_column]
            if future_length != prediction_length:
                raise ValueError(
                    f"Future covariates all time series must have length {prediction_length}, got {future_length} for series {future_series_id}"
                )
            if future_length < 3 or inferred_freq != validate_freq(future_timestamps, future_series_id):
                raise ValueError(
                    f"Future covariates must have the same frequency as context, found series {future_series_id} with a different frequency"
                )
            future_start_idx += future_length

        assert len(series_lengths) == len(future_series_lengths)

    return df, future_df, inferred_freq, series_lengths, future_series_lengths, original_order


def convert_df_input_to_list_of_dicts_input(
    df: "pd.DataFrame",
    future_df: "pd.DataFrame | None",
    target_columns: list[str],
    prediction_length: int,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
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

    Returns
    -------
    A tuple containing:
    - Time series converted to list of dictionaries format
    - Original order of time series IDs
    - Dictionary mapping series IDs to future time index
    """

    import pandas as pd

    df, future_df, freq, series_lengths, future_series_lengths, original_order = validate_df_inputs(
        df,
        future_df=future_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_columns=target_columns,
        prediction_length=prediction_length,
    )

    # Convert to list of dicts format
    inputs: list[dict[str, np.ndarray | dict[str, np.ndarray]]] = []
    prediction_timestamps: dict[str, pd.DatetimeIndex] = {}
    start_idx: int = 0
    future_start_idx: int = 0

    for i, length in enumerate(series_lengths):
        series_data = df.iloc[start_idx : start_idx + length]
        # Extract target(s)
        target_data = series_data[target_columns].to_numpy().T  # Shape: (n_targets, history_length)
        task: dict[str, np.ndarray | dict[str, np.ndarray]] = {"target": target_data}

        # Generate future timestamps
        series_id = series_data.iloc[0][id_column]
        last_timestamp = series_data[timestamp_column].iloc[-1]
        future_ts = pd.date_range(start=last_timestamp, periods=prediction_length + 1, freq=freq)[1:]
        prediction_timestamps[series_id] = future_ts

        # Handle covariates if present
        covariate_cols = [
            col for col in series_data.columns if col not in [id_column, timestamp_column] + target_columns
        ]

        if covariate_cols:
            past_covariates = {col: series_data[col].to_numpy() for col in covariate_cols}
            task["past_covariates"] = past_covariates

            # Handle future covariates
            if future_df is not None:
                assert future_series_lengths is not None
                future_length = future_series_lengths[i]
                future_data = future_df.iloc[future_start_idx : future_start_idx + future_length]
                assert future_data[timestamp_column].iloc[0] == future_ts[0], (
                    f"the first timestamp in future_df must be the first forecast timestamp, found mismatch "
                    f"({future_data[timestamp_column].iloc[0]} != {future_ts[0]}) in series {series_id}"
                )

                if len(future_data) > 0:
                    future_covariates = {
                        col: future_data[col].to_numpy() for col in covariate_cols if col in future_data.columns
                    }
                    if future_covariates:
                        task["future_covariates"] = future_covariates
                future_start_idx += future_length

        inputs.append(task)
        start_idx += length

    assert len(inputs) == len(series_lengths)

    return inputs, original_order, prediction_timestamps
