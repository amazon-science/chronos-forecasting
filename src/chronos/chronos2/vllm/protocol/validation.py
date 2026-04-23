"""Centralized validation for Chronos-2 forecast requests.

All validation rules live here so that both Pydantic model validators
(in ``protocol.forecast``) and the IOProcessor can delegate to a single
source of truth.  The design mirrors the SageMaker endpoint's
``validate_payload`` / ``validate_covariates`` utilities.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chronos.chronos2.vllm.protocol.forecast import (  # noqa: F811
        ForecastParameters,
        TimeSeriesInput,
    )

# ---------------------------------------------------------------------------
# Constants — match the SageMaker endpoint constraints
# TODO: Get some of the values from model config
# ---------------------------------------------------------------------------
MIN_TARGET_LENGTH: int = 5
MAX_PREDICTION_LENGTH: int = 1024
MAX_NUM_TIME_SERIES: int = 1024


# ---------------------------------------------------------------------------
# Per-input validators (called from Pydantic field/model validators)
# ---------------------------------------------------------------------------


def validate_target(
    target: list[float] | list[list[float]],
) -> list[float] | list[list[float]]:
    """Validate minimum target length and multivariate row consistency.

    Raises ``ValueError`` if:
    - The target is empty.
    - The target has fewer than ``MIN_TARGET_LENGTH`` observations.
    - For multivariate targets, rows have inconsistent lengths.
    """
    if not target:
        raise ValueError("Target must not be empty")
    if isinstance(target[0], list):
        first_len = len(target[0])
        if first_len < MIN_TARGET_LENGTH:
            raise ValueError(
                f"Target must contain at least {MIN_TARGET_LENGTH} "
                f"observations (received {first_len})"
            )
        for i, dim in enumerate(target[1:], start=1):
            if len(dim) != first_len:  # type: ignore[arg-type]
                raise ValueError(
                    f"All target dimensions must have same length. "
                    f"Dimension 0 has {first_len} observations, "
                    f"dimension {i} has {len(dim)}"  # type: ignore[arg-type]
                )
    else:
        if len(target) < MIN_TARGET_LENGTH:
            raise ValueError(
                f"Target must contain at least {MIN_TARGET_LENGTH} "
                f"observations (received {len(target)})"
            )
    return target


def validate_start_timestamp(start: str | None) -> str | None:
    """Validate that *start* is a valid ISO-8601 string (or ``None``)."""
    if start is not None:
        try:
            datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(
                f"Invalid start timestamp format: {start}. "
                f"Expected ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
            ) from e
    return start


def validate_quantile_levels(quantile_levels: list[float]) -> list[float]:
    """Validate that every quantile level is in the open interval (0, 1)."""
    for q in quantile_levels:
        if not (0 < q < 1):
            raise ValueError(f"Quantile levels must be between 0 and 1 (exclusive), got {q}")
    return quantile_levels


def validate_single_series_covariates(
    target: list[float] | list[list[float]],
    past_covariates: dict[str, list] | None,
    future_covariates: dict[str, list] | None,
) -> None:
    """Validate covariate constraints for a **single** time series.

    Raises ``ValueError`` if any of the following are violated:

    - ``'target'`` is used as a covariate name.
    - ``future_covariates`` is provided without ``past_covariates``.
    - Future covariate keys are not a subset of past covariate keys.
    - Past covariate array lengths do not match the target length.
    """
    if not target:
        raise ValueError("Target must not be empty")
    target_len = len(target[0]) if isinstance(target[0], list) else len(target)

    # 'target' must not be used as a covariate name
    for label, covariates in [
        ("past_covariates", past_covariates),
        ("future_covariates", future_covariates),
    ]:
        if covariates is not None and "target" in covariates:
            raise ValueError("Covariate with name 'target' is not allowed")

    # future_covariates requires past_covariates
    if future_covariates is not None and past_covariates is None:
        raise ValueError(
            "Both 'past_covariates' and 'future_covariates' must be provided "
            "together. Got 'future_covariates' without 'past_covariates'"
        )

    # future keys ⊆ past keys
    if past_covariates is not None and future_covariates is not None:
        past_keys = set(past_covariates.keys())
        future_keys = set(future_covariates.keys())
        if not future_keys.issubset(past_keys):
            extra = future_keys - past_keys
            raise ValueError(
                f"All future covariate keys must be present in past covariates. "
                f"Keys {extra} are in 'future_covariates' but not in "
                f"'past_covariates'"
            )

    # past covariate lengths must match target length
    if past_covariates is not None:
        for key, values in past_covariates.items():
            if len(values) != target_len:
                raise ValueError(
                    f"Past covariate '{key}' length ({len(values)}) "
                    f"must match target length ({target_len})"
                )


# ---------------------------------------------------------------------------
# Cross-series validators (called from IOProcessor.parse_request)
# ---------------------------------------------------------------------------


def validate_cross_series(
    inputs: list[TimeSeriesInput],
    parameters: ForecastParameters,
) -> None:
    """Validate constraints that span across multiple time series.

    Mirrors the SageMaker endpoint's ``validate_payload`` and
    ``validate_covariates`` utilities.

    Raises ``ValueError`` if any of the following are violated:

    - ``item_id`` provided for some but not all inputs.
    - ``item_id`` values are not unique.
    - ``start`` provided for some but not all inputs.
    - ``start`` is provided without ``freq`` (or vice-versa).
    - Covariate keys are not identical across all series.
    - ``future_covariates`` array lengths don't match ``prediction_length``.
    """
    _validate_item_ids(inputs)
    _validate_start_freq(inputs, parameters)
    _validate_covariate_consistency(inputs)
    _validate_future_covariate_lengths(inputs, parameters)


# -- helpers (private) -------------------------------------------------------


def _validate_item_ids(inputs: list[TimeSeriesInput]) -> None:
    item_ids = [ts.item_id for ts in inputs]
    has_none = any(x is None for x in item_ids)
    has_value = any(x is not None for x in item_ids)
    if has_none and has_value:
        raise ValueError(
            "If 'item_id' is provided for at least one time series in "
            "'inputs', it should be provided for all time series"
        )
    if has_value and len(item_ids) != len(set(item_ids)):
        raise ValueError("'item_id' must be unique for all time series in 'inputs'")


def _validate_start_freq(
    inputs: list[TimeSeriesInput],
    parameters: ForecastParameters,
) -> None:
    starts = [ts.start for ts in inputs]
    has_none = any(x is None for x in starts)
    has_value = any(x is not None for x in starts)
    if has_none and has_value:
        raise ValueError(
            "If 'start' is provided for at least one time series in "
            "'inputs', it should be provided for all time series"
        )
    if has_value and parameters.freq is None:
        raise ValueError(
            "If 'start' is provided, then 'freq' must also be provided " "in 'parameters'"
        )
    if parameters.freq is not None and not has_value:
        raise ValueError(
            "If 'freq' is provided in 'parameters', then 'start' must "
            "also be provided for all time series in 'inputs'"
        )


def _validate_covariate_consistency(inputs: list[TimeSeriesInput]) -> None:
    key_sets: list[frozenset[str] | None] = []
    for ts in inputs:
        if ts.past_covariates is not None:
            key_sets.append(frozenset(ts.past_covariates.keys()))
        else:
            key_sets.append(None)

    if len(set(key_sets)) > 1:
        raise ValueError(
            "If 'past_covariates' and 'future_covariates' are provided "
            "for at least one time series in 'inputs', the same "
            "covariates should be provided for all time series"
        )


def _validate_future_covariate_lengths(
    inputs: list[TimeSeriesInput],
    parameters: ForecastParameters,
) -> None:
    prediction_length = parameters.prediction_length
    for i, ts in enumerate(inputs):
        if ts.future_covariates is not None:
            for key, values in ts.future_covariates.items():
                if len(values) != prediction_length:
                    raise ValueError(
                        f"Input {i}: length of future covariate '{key}' "
                        f"({len(values)}) must equal prediction_length "
                        f"({prediction_length})"
                    )
