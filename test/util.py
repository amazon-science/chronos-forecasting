from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch


def validate_tensor(a: torch.Tensor, shape: Tuple[int, ...], dtype: Optional[torch.dtype] = None) -> None:
    assert isinstance(a, torch.Tensor)
    assert a.shape == shape

    if dtype is not None:
        assert a.dtype == dtype



def create_df(series_ids=["A", "B"], n_points=[10, 10], target_cols=["target"], covariates=None, freq="h"):
    """Helper to create test context DataFrames."""
    series_dfs = []
    for series_id, length in zip(series_ids, n_points):
        series_data = {"item_id": series_id, "timestamp": pd.date_range(end="2001-10-01", periods=length, freq=freq)}
        for target_col in target_cols:
            series_data[target_col] = np.random.randn(length)
        if covariates:
            for cov in covariates:
                series_data[cov] = np.random.randn(length)
        series_dfs.append(pd.DataFrame(series_data))
    return pd.concat(series_dfs, ignore_index=True)


def create_future_df(forecast_start_times: list, series_ids=["A", "B"], n_points=[5, 5], covariates=None, freq="h"):
    """Helper to create test future DataFrames."""
    series_dfs = []
    for series_id, length, start in zip(series_ids, n_points, forecast_start_times):
        series_data = {"item_id": series_id, "timestamp": pd.date_range(start=start, periods=length, freq=freq)}
        if covariates:
            for cov in covariates:
                series_data[cov] = np.random.randn(length)
        series_dfs.append(pd.DataFrame(series_data))
    return pd.concat(series_dfs, ignore_index=True)


def get_forecast_start_times(df, freq="h"):
    context_end_times = df.groupby("item_id")["timestamp"].max()
    forecast_start_times = [pd.date_range(end_time, periods=2, freq=freq)[-1] for end_time in context_end_times]

    return forecast_start_times