# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Caner Turkmen <atturkm@amazon.com>, Abdul Fatir Ansari <ansarnd@amazon.com>, Lorenzo Stella <stellalo@amazon.com>
# Original source:
# https://github.com/autogluon/autogluon/blob/f57beb26cb769c6e0d484a6af2b89eab8aee73a8/timeseries/src/autogluon/timeseries/models/chronos/pipeline/base.py


import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    import datasets
    import fev
    from transformers import PreTrainedModel


from .df_utils import make_future_df, normalize_df
from .utils import left_pad_and_stack_1D


class ForecastType(Enum):
    SAMPLES = "samples"
    QUANTILES = "quantiles"


class PipelineRegistry(type):
    REGISTRY: Dict[str, "PipelineRegistry"] = {}

    def __new__(cls, name, bases, attrs):
        """See, https://github.com/faif/python-patterns."""
        new_cls = type.__new__(cls, name, bases, attrs)
        if name is not None:
            cls.REGISTRY[name] = new_cls

        return new_cls


class BaseChronosPipeline(metaclass=PipelineRegistry):
    """
    Abstract base class for Chronos pretrained time series forecasting pipelines.

    This class defines the common interface for all Chronos models. The package provides
    multiple pipeline implementations with different forecasting approaches and architectures:

    - [ChronosPipeline][chronos.chronos.ChronosPipeline]: Sample-based forecasting with scaling and quantization based tokenization
    - [ChronosBoltPipeline][chronos.chronos_bolt.ChronosBoltPipeline]: Quantile-based forecasting with patching
    - [Chronos2Pipeline][chronos.chronos2.pipeline.Chronos2Pipeline] (recommended): Quantile-based forecasting with support for multivariate and covariate-informed forecasting

    Each subclass implements the abstract methods and properties defined here,
    potentially with different parameter signatures and return types depending
    on the model architecture and forecasting approach.
    """

    forecast_type: ForecastType
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}

    def __init__(self, inner_model: "PreTrainedModel"):
        """
        Initialize the base pipeline with a pretrained model.

        Parameters
        ----------
        inner_model
            A HuggingFace transformers PreTrainedModel that serves as the
            underlying forecasting model (e.g., T5ForConditionalGeneration)
        """
        # for easy access to the inner HF-style model
        self.inner_model = inner_model

    @property
    def model_context_length(self) -> int:
        """
        Maximum number of time steps the model can use as context.

        This is an abstract property that must be implemented by subclasses.

        Returns
        -------
        int
            Maximum context length supported by the model
        """
        raise NotImplementedError()

    @property
    def model_prediction_length(self) -> int:
        """
        Default prediction horizon for the model.

        This is an abstract property that must be implemented by subclasses.

        Returns
        -------
        int
            Default prediction horizon
        """
        raise NotImplementedError()

    def _prepare_and_validate_context(self, context: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(context, list):
            context = left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2

        return context

    def predict(self, inputs: Union[torch.Tensor, List[torch.Tensor]], prediction_length: Optional[int] = None):
        """
        Generate forecasts for the given time series.

        This is an abstract method that must be implemented by subclasses.
        Each subclass may have different parameters and return types depending
        on the model architecture and forecasting approach. Predictions are
        typically returned in fp32 on the CPU.

        Parameters
        ----------
        inputs
            Input time series. Can be a 1D tensor (single series), a list
            of 1D tensors (multiple series of varying lengths), or a 2D tensor
            where the first dimension is batch size. For 2D tensors, use
            left-padding with torch.nan to align series of different lengths.
        prediction_length
            Number of time steps to forecast. If not provided, defaults to
            the model's default prediction length.

        Returns
        -------
        torch.Tensor
            Forecasts tensor. The shape and interpretation depend on the
            subclass's forecast_type (samples or quantiles).

        Notes
        -----
        Subclasses may extend this interface with additional parameters
        specific to their forecasting approach. Refer to specific subclass
        documentation for complete parameter lists and return value details.
        """
        raise NotImplementedError()

    def predict_quantiles(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate quantile and mean forecasts for given time series.

        This is an abstract method that must be implemented by subclasses.
        Each subclass may have different parameters depending on the model
        architecture. Predictions are typically returned in fp32 on the CPU.

        Parameters
        ----------
        inputs
            Input time series. Can be a 1D tensor (single series), a list
            of 1D tensors (multiple series of varying lengths), or a 2D tensor
            where the first dimension is batch size. For 2D tensors, use
            left-padding with torch.nan to align series of different lengths.
        prediction_length
            Number of time steps to forecast. If not provided, defaults to
            the model's default prediction length.
        quantile_levels
            List of quantile levels to compute, each between 0 and 1.
            Default is [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
        **kwargs
            Additional keyword arguments that may be used by subclass implementations.

        Returns
        -------
        torch.Tensor
            Tensor of quantile forecasts with shape
            (batch_size, prediction_length, num_quantiles)
        torch.Tensor
            Tensor of mean (point) forecasts with shape
            (batch_size, prediction_length)

        Notes
        -----
        Subclasses may extend this interface with additional parameters
        specific to their forecasting approach. Refer to specific subclass
        documentation for complete parameter lists and implementation details.
        """
        raise NotImplementedError()

    def predict_df(
        self,
        df: pd.DataFrame,
        *,
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        target: str = "target",
        prediction_length: int | None = None,
        quantile_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        batch_size: int = 256,
        validate_inputs: bool = True,
        freq: str | None = None,
        **predict_kwargs,
    ) -> pd.DataFrame:
        """
        Generate forecasts for time series data in a pandas DataFrame.

        This method provides a convenient interface for forecasting on long-format
        pandas DataFrames containing multiple time series. It handles data conversion,
        batching, and result formatting automatically.

        Parameters
        ----------
        df
            Time series data in long format with an id column, a timestamp, and one target column.
            Any other columns, if present, will be ignored
        id_column
            The name of the column which contains the unique time series identifiers, by default "item_id"
        timestamp_column
            The name of the column which contains timestamps, by default "timestamp"
            All time series in the dataframe must have regular timestamps with the same frequency (no gaps)
        target
            The name of the column which contains the target variables to be forecasted, by default "target"
        prediction_length
            Number of steps to predict for each time series
        quantile_levels
            Quantile levels to compute
        batch_size
            The number of time series to predict in a single forward pass, by default 256
        validate_inputs
            [ADVANCED] When True (default), validates dataframes before prediction. Setting to False removes the
            validation overhead, but may silently lead to wrong predictions if data is misformatted. When False, you
            must ensure: (1) all dataframes are sorted by (id_column, timestamp_column); (2) future_df (if provided)
            has the same item IDs as df with exactly prediction_length rows of future timestamps per item; (3) all
            timestamps are regularly spaced (e.g., with hourly frequency).
        freq
            Frequency string for timestamp generation (e.g., "h", "D", "W"). When provided, skips
            frequency inference from the data and uses this frequency to generate the forecast timestamps.
            Note: the provided `freq` is used as-is and is not checked against the data, even when
            validate_inputs=True.
        **predict_kwargs
            Additional arguments passed to predict_quantiles

        Returns
        -------
        pd.DataFrame
            Forecast results in long format with the following columns:

            - Column named by id_column: Time series identifiers
            - Column named by timestamp_column: Future timestamps
            - "target_name": Name of the forecasted target variable
            - "predictions": Point forecasts (mean predictions)
            - One column per quantile level (e.g., "0.1", "0.5", "0.9")

        Raises
        ------
        ImportError
            If pandas is not installed.
        ValueError
            If target is not a string (multivariate forecasting not supported).

        Notes
        -----
        This method requires pandas to be installed. Install with `pip install pandas`.

        The method internally converts the DataFrame to tensor format, generates
        forecasts using predict_quantiles, and converts results back to DataFrame format.

        Subclasses may have additional parameters or behavior. Refer to specific
        subclass documentation for implementation details.
        """
        if not isinstance(target, str):
            raise ValueError(
                f"Expected `target` to be str, but found {type(target)}. {self.__class__.__name__} only supports univariate forecasting."
            )

        if prediction_length is None:
            prediction_length = self.model_prediction_length

        if target not in df.columns:
            raise ValueError(f"df does not contain the target column '{target}'")

        # Group by item (first-appearance order) and sort within group; covariates are ignored here.
        if validate_inputs:
            df = normalize_df(df, id_column=id_column, timestamp_column=timestamp_column)

        series_lengths = df[id_column].value_counts(sort=False).to_list()
        target_values = df[target].to_numpy()
        indptr = np.concatenate([[0], np.cumsum(series_lengths)]).astype("int64")
        context = [torch.tensor(target_values[indptr[i] : indptr[i + 1]]) for i in range(len(series_lengths))]

        future = make_future_df(
            df, prediction_length, freq=freq, id_column=id_column, timestamp_column=timestamp_column
        )

        # Generate forecasts in batches of at most `batch_size` series to bound memory usage.
        quantiles_all = []
        mean_all = []
        for start in range(0, len(context), batch_size):
            quantiles, mean = self.predict_quantiles(
                inputs=context[start : start + batch_size],
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                limit_prediction_length=False,
                **predict_kwargs,
            )
            quantiles_all.append(quantiles.numpy())
            mean_all.append(mean.numpy())

        quantiles_np = np.concatenate(quantiles_all, axis=0)  # [n_series, horizon, num_quantiles]
        mean_np = np.concatenate(mean_all, axis=0)  # [n_series, horizon]

        # `future` has prediction_length rows per item, in the same item order as the predictions,
        # so it lines up with `mean` / `quantiles` directly (single target, no per-variate repeat).
        result = future.copy()
        result["target_name"] = target
        result["predictions"] = mean_np.ravel()

        quantiles_flat = quantiles_np.reshape(len(result), len(quantile_levels))
        for q_idx, q_level in enumerate(quantile_levels):
            result[str(q_level)] = quantiles_flat[:, q_idx]

        return result

    def predict_fev(
        self, task: "fev.Task", batch_size: int = 32, **kwargs
    ) -> tuple[list["datasets.DatasetDict"], float]:
        """
        Generate predictions for evaluation on a fev benchmark task.

        This method provides integration with the fev (Forecasting Evaluation)
        library for standardized benchmark evaluation. It handles batching,
        timing, and formatting predictions according to the task requirements.

        Parameters
        ----------
        task
            A fev.Task object defining the benchmark evaluation task, including
            the dataset, horizon, quantile levels, and evaluation metric.
        batch_size
            Number of time series to process in each batch during inference.
            Larger batch sizes may improve throughput but require more memory.
            Default is 32.
        **kwargs
            Additional keyword arguments forwarded to the predict_quantiles method.
            These may include model-specific parameters.

        Returns
        -------
        list[DatasetDict]
            List of DatasetDict objects, one for each evaluation window in the task.
            Each DatasetDict contains predictions formatted according to fev requirements.
        float
            Total inference time in seconds across all windows, excluding data
            loading and preprocessing time.

        Raises
        ------
        ImportError
            If the fev library is not installed.

        Notes
        -----
        This method requires the fev library to be installed. Install with
        `pip install fev`.
        """
        try:
            import fev
        except ImportError:
            raise ImportError("fev is required for predict_fev. Please install it with `pip install fev`.")

        # `predict_df` puts `predict_quantiles`'s `mean` in the "predictions" column. For point-forecast metrics
        # the mean is the right target; for the others the median (0.5 quantile) is, so we request it and swap it in.
        use_median_point_forecast = task.eval_metric not in ["MSE", "RMSE", "RMSSE"]
        quantile_levels = task.quantile_levels.copy()
        if use_median_point_forecast and 0.5 not in quantile_levels:
            quantile_levels.append(0.5)

        predictions_per_window = []
        inference_time_s = 0.0
        for window in task.iter_windows():
            # Base pipelines are univariate, so we always split multivariate targets and drop covariates.
            past_df, _, _ = self._fev_window_to_df(window, as_univariate=True)

            start_time = time.monotonic()
            forecast_df = self.predict_df(
                past_df,
                id_column=window.id_column,
                timestamp_column=window.timestamp_column,
                target="target",
                prediction_length=task.horizon,
                quantile_levels=quantile_levels,
                batch_size=batch_size,
                **kwargs,
            )
            inference_time_s += time.monotonic() - start_time

            if use_median_point_forecast:
                forecast_df["predictions"] = forecast_df["0.5"]

            predictions_per_window.append(
                fev.utils.convert_forecast_df_to_predictions(
                    forecast_df,
                    horizon=task.horizon,
                    quantile_levels=task.quantile_levels,
                    target_columns=task.target_columns,
                )
            )
        return predictions_per_window, inference_time_s

    @staticmethod
    def _fev_window_to_df(
        window: "fev.EvaluationWindow", as_univariate: bool
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
        """Convert a fev evaluation window into the (past_df, future_df, target_columns) inputs for `predict_df`."""
        import fev

        past_df, future_df, _ = fev.convert_input_data(window, adapter="pandas", as_univariate=as_univariate)

        if as_univariate:
            # `as_univariate=True` splits multivariate targets into separate univariate series. The adapter keeps
            # the covariate columns, so we drop them here to predict each target independently and ignore covariates.
            past_df = past_df[[window.id_column, window.timestamp_column, "target"]]
            future_df = None
            target_columns = ["target"]
        else:
            # The pandas adapter's future_df only contains the known-future covariates; pass None when there are none.
            if not window.known_dynamic_columns:
                future_df = None
            target_columns = list(window.target_columns)

        return past_df, future_df, target_columns

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *model_args,
        force_s3_download=False,
        **kwargs,
    ):
        """
        Load a pretrained Chronos pipeline from various sources.

        This class method loads a pretrained model from a local path, S3 bucket,
        or the HuggingFace Hub. It automatically detects the appropriate pipeline
        class based on the model configuration and instantiates it.

        Parameters
        ----------
        pretrained_model_name_or_path
            Path or identifier for the pretrained model. Can be:
            - A local directory path containing model files
            - An S3 URI (s3://bucket/prefix)
            - A HuggingFace Hub model identifier (e.g., "amazon/chronos-t5-small")
        *model_args
            Additional positional arguments passed to the model constructor.
        force_s3_download
            When True, forces re-downloading from S3 even if cached locally.
            Only applicable for S3 URIs. Default is False.
        **kwargs
            Additional keyword arguments passed to AutoConfig and the model
            constructor. Common options include:
            - torch_dtype: Data type for model weights ("auto", "float32", "bfloat16")
            - device_map: Device placement strategy for model layers
            - Other transformers AutoConfig and AutoModel arguments

        Returns
        -------
        BaseChronosPipeline
            An instance of the appropriate pipeline subclass (ChronosPipeline,
            ChronosBoltPipeline, or Chronos2Pipeline) based on the model configuration.

        Raises
        ------
        ValueError
            If the configuration is not a valid Chronos config or if the
            specified pipeline class is not recognized.
        ImportError
            If required dependencies are not installed.

        Notes
        -----
        The method reads the model configuration to determine which pipeline
        class to instantiate. The configuration must contain either a
        `chronos_pipeline_class` or `chronos_config` attribute.

        For S3 URIs, the model is first downloaded to a local cache directory
        before loading.

        The torch_dtype parameter can be specified as a string ("float32", "bfloat16")
        or as a torch dtype object. When set to "auto", the dtype is determined
        from the model configuration.
        """
        if str(pretrained_model_name_or_path).startswith("s3://"):
            from .boto_utils import cache_model_from_s3

            try:
                local_model_path = cache_model_from_s3(
                    str(pretrained_model_name_or_path), force_download=force_s3_download
                )
            except ImportError as e:
                raise ImportError(
                    "Loading models from s3:// URIs requires boto3. "
                    "Install the optional dependencies with: "
                    "pip install 'chronos-forecasting[extras]'"
                ) from e
            return cls.from_pretrained(local_model_path, *model_args, **kwargs)

        from packaging import version
        from transformers import AutoConfig
        from transformers import __version__ as transformers_version

        # `dtype` kwarg was introduced in transformers 4.56; older versions use `torch_dtype`
        _dtype_kwarg = "dtype" if version.parse(transformers_version) >= version.parse("4.56.0") else "torch_dtype"

        # Handle both torch_dtype (deprecated in transformers v5) and dtype arguments
        torch_dtype_value = kwargs.pop("torch_dtype", None)
        dtype_value = kwargs.pop("dtype", None)
        resolved_dtype = torch_dtype_value or dtype_value or "auto"
        if resolved_dtype != "auto" and isinstance(resolved_dtype, str):
            resolved_dtype = cls.dtypes[resolved_dtype]
        kwargs[_dtype_kwarg] = resolved_dtype

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        is_valid_config = hasattr(config, "chronos_pipeline_class") or hasattr(config, "chronos_config")

        if not is_valid_config:
            raise ValueError("Not a Chronos config file")

        pipeline_class_name = getattr(config, "chronos_pipeline_class", "ChronosPipeline")
        class_ = PipelineRegistry.REGISTRY.get(pipeline_class_name)
        if class_ is None:
            raise ValueError(f"Trying to load unknown pipeline class: {pipeline_class_name}")

        return class_.from_pretrained(  # type: ignore[attr-defined]
            pretrained_model_name_or_path, *model_args, **kwargs
        )
