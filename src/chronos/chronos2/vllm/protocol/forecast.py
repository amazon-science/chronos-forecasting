import math
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel
except ImportError:
    OpenAIBaseModel = BaseModel  # type: ignore[misc,assignment]

from chronos.chronos2.vllm.protocol.validation import (
    MAX_NUM_TIME_SERIES,
    validate_quantile_levels,
    validate_single_series_covariates,
    validate_start_timestamp,
    validate_target,
)


class TimeSeriesInput(BaseModel):
    """Input time series data for forecasting."""

    target: list[float] | list[list[float]] = Field(
        ...,
        description="Historical time series values. "
        "1-D array for univariate, 2-D array for multivariate",
    )

    item_id: str | None = Field(default=None, description="Unique identifier for the time series")

    start: str | None = Field(
        default=None,
        description="Start timestamp in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
    )

    # Values can be NaN (numeric) or None/NaN mixed in string arrays.
    past_covariates: dict[str, list[Union[float, str, None]]] | None = Field(
        default=None,
        description=(
            "Dictionary of past covariate arrays (numeric or categorical). "
            "Each array must match the length of the target"
        ),
    )

    # Values can be NaN (numeric) or None/NaN mixed in string arrays.
    future_covariates: dict[str, list[Union[float, str, None]]] | None = Field(
        default=None,
        description="Dictionary of known future covariate arrays. "
        "Keys must be a subset of past_covariates. "
        "Each array must match prediction_length",
    )

    @field_validator("past_covariates", "future_covariates", mode="before")
    @classmethod
    def _sanitize_nan_in_covariates(
        cls, v: dict[str, list[Any]] | None
    ) -> dict[str, list[Any]] | None:
        """Convert NaN floats to None in covariate arrays.

        Datasets often encode missing categorical values as float NaN.
        Pydantic rejects NaN in string-typed lists, so we normalize
        NaN → None before validation.
        """
        if v is None:
            return v
        sanitized: dict[str, list[Any]] = {}
        for key, arr in v.items():
            sanitized[key] = [None if isinstance(x, float) and math.isnan(x) else x for x in arr]
        return sanitized

    @field_validator("target")
    @classmethod
    def _validate_target(
        cls, v: list[float] | list[list[float]]
    ) -> list[float] | list[list[float]]:
        return validate_target(v)

    @field_validator("start")
    @classmethod
    def _validate_start(cls, v: str | None) -> str | None:
        return validate_start_timestamp(v)

    @model_validator(mode="after")
    def _validate_covariates(self) -> "TimeSeriesInput":
        validate_single_series_covariates(self.target, self.past_covariates, self.future_covariates)
        return self


class ForecastParameters(BaseModel):
    """Parameters for time series forecasting."""

    prediction_length: int = Field(
        default=1, ge=1, le=1024, description="Number of future steps to forecast"
    )

    quantile_levels: list[float] = Field(
        default=[0.1, 0.5, 0.9],
        description="Quantile levels for uncertainty quantification. "
        "Each value must be between 0 and 1 (exclusive)",
    )

    freq: str | None = Field(
        default=None,
        description="Pandas frequency string (e.g., 'D' for daily, 'H' for hourly). "
        "Required if 'start' is provided in inputs",
    )

    batch_size: int = Field(
        default=256,
        ge=1,
        description="Internal row batch size for model inference. "
        "Controls how many rows (series + covariates) are processed "
        "in a single model forward pass. Rows are chunked internally "
        "respecting series boundaries via group_ids.",
    )

    cross_learning: bool = Field(
        default=False,
        description="Enable information sharing across time series in batch",
    )

    @field_validator("quantile_levels")
    @classmethod
    def _validate_quantiles(cls, v: list[float]) -> list[float]:
        return validate_quantile_levels(v)


class ForecastRequest(OpenAIBaseModel):
    """Request format for time series forecasting via pooling API."""

    model: str = Field(..., description="Model name to use for forecasting")

    task: Literal["forecast"] = Field(
        default="forecast", description="Task type, must be 'forecast'"
    )

    data: dict[str, Any] = Field(
        ...,
        description=("Forecast request data containing 'inputs' and optional 'parameters'"),
    )

    @field_validator("data")
    @classmethod
    def validate_data_structure(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate the data field contains required structure."""
        if "inputs" not in v:
            raise ValueError("data must contain 'inputs' field")

        if not isinstance(v["inputs"], list):
            raise ValueError("data.inputs must be a list")

        if len(v["inputs"]) == 0:
            raise ValueError("data.inputs cannot be empty")

        if len(v["inputs"]) > MAX_NUM_TIME_SERIES:
            raise ValueError(
                f"data.inputs may contain at most {MAX_NUM_TIME_SERIES} time series "
                f"(received {len(v['inputs'])})"
            )

        # Validate each input as TimeSeriesInput
        validated_inputs = []
        for i, ts_input in enumerate(v["inputs"]):
            try:
                validated_input = TimeSeriesInput(**ts_input)
                validated_inputs.append(validated_input)
            except Exception as e:
                raise ValueError(f"Invalid time series input at index {i}: {e}") from e

        # Validate parameters if present
        validated_params = None
        if "parameters" in v and v["parameters"] is not None:
            try:
                validated_params = ForecastParameters(**v["parameters"])
            except Exception as e:
                raise ValueError(f"Invalid parameters: {e}") from e

        # Cross-validate future_covariates length with prediction_length
        if validated_params is not None:
            prediction_length = validated_params.prediction_length
            for i, ts_input in enumerate(validated_inputs):
                if ts_input.future_covariates is not None:
                    for key, values in ts_input.future_covariates.items():
                        if len(values) != prediction_length:
                            raise ValueError(
                                f"Input {i}: future_covariate '{key}' length "
                                f"({len(values)}) must match prediction_length "
                                f"({prediction_length})"
                            )

        return v


class ForecastPrediction(BaseModel):
    """Single time series forecast result with quantile forecasts."""

    mean: list[float] | list[list[float]] = Field(
        ..., description="Point forecast (mean). Shape matches input target"
    )

    item_id: str | None = Field(default=None, description="Echoed from input if provided")

    start: str | None = Field(default=None, description="Start timestamp of forecast horizon")

    class Config:
        extra = "allow"  # Allow dynamic quantile fields like "0.1", "0.5", "0.9"


class ForecastResponse(OpenAIBaseModel):
    """Response format for time series forecasting."""

    request_id: str = Field(..., description="Request identifier")

    created_at: int = Field(..., description="Unix timestamp when the response was created")

    data: dict[str, list[ForecastPrediction]] = Field(
        ..., description="Forecast results with 'predictions' key"
    )
