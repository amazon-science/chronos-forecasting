"""IOProcessor for Chronos-2 time series forecasting.

Thin orchestrator that delegates to:
  - protocol.data_prep: tensor preparation from validated inputs (via Chronos2Dataset)
  - protocol.validation: cross-series validation
  - multimodal.MODALITY: MM prompt construction
  - utils: quantile selection, helpers

Flow:
  1. parse_request  → validate inputs via Pydantic models
  2. pre_process    → prepare_request() → timeseries MM prompts (one per batch)
  3. post_process   → select_quantiles() → per-series predictions
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams

from chronos.chronos2.vllm.multimodal import MODALITY
from chronos.chronos2.vllm.protocol.data_prep import PreparedRequest, prepare_request
from chronos.chronos2.vllm.protocol.forecast import (
    ForecastParameters,
    ForecastPrediction,
    TimeSeriesInput,
)
from chronos.chronos2.vllm.protocol.validation import validate_cross_series
from chronos.chronos2.vllm.utils.helpers import empty_prediction, tensor_to_list
from chronos.chronos2.vllm.utils.quantiles import select_quantiles

logger = init_logger(__name__)


@dataclass
class _PostProcessInfo:
    """Lightweight cache of data needed for post_process — avoids storing full tensors."""

    item_ids: list[str | None]
    parameters: ForecastParameters
    target_idx_ranges: list[list[tuple[int, int]]]  # per-batch list of (start, end) ranges


class Chronos2IOProcessor(IOProcessor[dict, dict]):
    """IOProcessor for Chronos-2 time series forecasting."""

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        config = vllm_config.model_config.hf_config
        config.is_encoder_decoder = False
        # Set projection_dim=0 so vLLM uses IdentityPooler (no projection layer).
        # This avoids requiring --hf-overrides '{"projection_dim": 0}' on the CLI.
        if not hasattr(config, "projection_dim") or config.projection_dim != 0:
            config.projection_dim = 0

        cc = getattr(config, "chronos_config", {})
        self._chronos_config = (
            cc if isinstance(cc, dict) else (cc.__dict__ if hasattr(cc, "__dict__") else {})
        )
        self._context_length = self._chronos_config.get("context_length", 8192)
        self._output_patch_size = self._chronos_config.get("output_patch_size", 16)
        self._model_quantiles = self._chronos_config.get(
            "quantiles",
            [
                0.01,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                0.99,
            ],
        )
        self._request_info: dict[str, _PostProcessInfo] = {}
        logger.info("Initialized Chronos2IOProcessor")

    # -----------------------------------------------------------
    # Request parsing
    # -----------------------------------------------------------

    def parse_request(self, request: Any) -> dict:
        data = request.data if hasattr(request, "data") else request
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")
        if "inputs" not in data:
            raise ValueError("Request must contain 'inputs' field")
        raw_inputs = data["inputs"]
        if not isinstance(raw_inputs, list) or len(raw_inputs) == 0:
            raise ValueError("'inputs' must be a non-empty list")

        validated_inputs = []
        for i, ts in enumerate(raw_inputs):
            try:
                validated_inputs.append(TimeSeriesInput(**ts))
            except Exception as e:
                raise ValueError(f"Invalid input at index {i}: {e}") from e

        try:
            validated_params = ForecastParameters(**data.get("parameters", {}))
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

        validate_cross_series(validated_inputs, validated_params)
        return {"inputs": validated_inputs, "parameters": validated_params}

    # -----------------------------------------------------------
    # Pre-process: inputs → MM prompts
    # -----------------------------------------------------------

    def pre_process(
        self,
        prompt: dict,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> PromptType | Sequence[PromptType]:
        prepared = prepare_request(
            inputs=prompt["inputs"],
            parameters=prompt["parameters"],
            context_length=self._context_length,
            output_patch_size=self._output_patch_size,
        )

        # Cache only lightweight post-processing metadata
        if request_id is not None:
            self._request_info[request_id] = _PostProcessInfo(
                item_ids=prepared.item_ids,
                parameters=prepared.parameters,
                target_idx_ranges=[batch.target_idx_ranges for batch in prepared.batches],
            )

        # Build one MM prompt per batch (Chronos2Dataset handles batch splitting)
        prompts: list[PromptType] = []
        for batch in prepared.batches:
            prompts.append(
                {
                    "prompt_token_ids": [1],
                    "multi_modal_data": {
                        MODALITY: {
                            "context": batch.context,
                            "future_covariates": batch.future_covariates,
                            "group_ids": batch.group_ids,
                            "num_output_patches": batch.num_output_patches,
                        }
                    },
                }
            )

        return prompts if len(prompts) > 1 else prompts[0]

    # -----------------------------------------------------------
    # Post-process: model output → predictions
    # -----------------------------------------------------------

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs: Any,
    ) -> dict:
        info = self._request_info.pop(request_id, None) if request_id else None
        if info is None:
            logger.error("No pending request for request_id=%s", request_id)
            return {"predictions": []}

        parameters = info.parameters
        item_ids = info.item_ids

        try:
            # Collect predictions across all batches, trimmed to exact prediction_length
            pred_len = parameters.prediction_length
            all_predictions: list[torch.Tensor] = []
            for batch_idx, output in enumerate(model_output):
                tensor = output.outputs.data
                while tensor.ndim > 3:
                    tensor = tensor.squeeze(0)

                batch_ranges = info.target_idx_ranges[batch_idx]
                for start, end in batch_ranges:
                    all_predictions.append(tensor[start:end, :, :pred_len])

            quantiles_out, mean_out = select_quantiles(
                all_predictions, self._model_quantiles, parameters.quantile_levels
            )

            result: dict[str, Any] = {"predictions": [], "request_id": request_id}
            for i, (q_tensor, m_tensor) in enumerate(zip(quantiles_out, mean_out)):
                pred: dict[str, Any] = {"mean": tensor_to_list(m_tensor)}
                for q, q_vals in zip(parameters.quantile_levels, q_tensor.unbind(dim=-1)):
                    pred[str(q)] = tensor_to_list(q_vals)
                if i < len(item_ids) and item_ids[i] is not None:
                    pred["item_id"] = item_ids[i]
                result["predictions"].append(pred)

        except Exception as e:
            logger.error("Failed to decode predictions: %s", e, exc_info=True)
            result = {
                "predictions": [
                    empty_prediction(parameters.prediction_length, parameters.quantile_levels)
                    for _ in item_ids
                ],
                "request_id": request_id,
            }

        return result

    # -----------------------------------------------------------
    # Response formatting
    # -----------------------------------------------------------

    def validate_or_generate_params(self, params: Any = None) -> PoolingParams:
        return PoolingParams(task="plugin")

    def output_to_response(self, plugin_output: dict) -> IOProcessorResponse:
        validated = []
        for pred in plugin_output.get("predictions", []):
            try:
                validated.append(ForecastPrediction(**pred).model_dump(exclude_none=True))
            except Exception as e:
                logger.warning("Failed to validate prediction: %s", e)
                validated.append(pred)
        return IOProcessorResponse(
            request_id=plugin_output.get("request_id"),
            created_at=int(time.time()),
            data={"predictions": validated},
        )