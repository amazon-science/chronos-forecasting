"""Chronos-2 vLLM model wrapper.

Thin wrapper around the existing ``chronos.chronos2.model.Chronos2Model``
that plugs into vLLM's multimodal (MM) interface. No model architecture
is duplicated — all computation is delegated to the upstream implementation.

Architecture:
  pre_process  → IOProcessor prepares context/covariates/group_ids tensors
               → Chronos2Dataset handles batching and group_id construction
               → returns MM prompt(s) with timeseries data
  forward()    → receives pre-batched context/future_covariates/group_ids as kwargs
               → delegates to chronos.chronos2.model.Chronos2Model
  pooler       → IdentityPooler passes output through unchanged
  post_process → IOProcessor selects/interpolates quantiles
"""

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import IdentityPooler
from vllm.model_executor.models.interfaces import (
    IsAttentionFree,
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.interfaces_base import attn_type
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils import length_from_prompt_token_ids_or_embeds

from chronos.chronos2.model import Chronos2Model

from .multimodal import (
    MODALITY,
    ChronosInputBuilder,
    ChronosMultiModalProcessor,
    ChronosProcessingInfo,
)

logger = init_logger(__name__)


@attn_type("attention_free")
@MULTIMODAL_REGISTRY.register_processor(
    ChronosMultiModalProcessor,
    info=ChronosProcessingInfo,
    dummy_inputs=ChronosInputBuilder,
)
class Chronos2ForForecasting(nn.Module, IsAttentionFree, SupportsMultiModal):
    """Chronos-2 forecasting model for vLLM.

    Delegates all computation to ``chronos.chronos2.model.Chronos2Model``.
    Receives pre-batched tensors (context, future_covariates, group_ids)
    directly in forward() via the timeseries MM pipeline. Batching is
    handled upstream by ``Chronos2Dataset`` in the IOProcessor.
    """

    supports_multimodal_raw_input_only = True
    is_pooling_model = True

    # Required by VllmModel interface
    packed_modules_mapping: dict[str, Any] = {}
    supported_lora_modules: list[str] = []
    embedding_modules: dict[str, str] = {}
    embedding_padding_modules: list[str] = []

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith(MODALITY):
            return None
        raise ValueError(f"Only {MODALITY} modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        config.is_encoder_decoder = False
        if not hasattr(config, "projection_dim") or config.projection_dim != 0:
            config.projection_dim = 0
        vllm_config.model_config.hf_text_config = None

        self.config = config
        self.model_name = vllm_config.model_config.model
        self.d_model = getattr(config, "d_model", 768)

        cc = getattr(config, "chronos_config", {})
        self.chronos_config = (
            cc if isinstance(cc, dict) else (cc.__dict__ if hasattr(cc, "__dict__") else {})
        )
        self.output_patch_size = self.chronos_config.get("output_patch_size", 16)
        self.max_output_patches = self.chronos_config.get("max_output_patches", 64)

        # Instantiate the upstream Chronos2Model — reuses all existing layers
        self.model = Chronos2Model(config)

        self.pooler = IdentityPooler()

        logger.info(
            "Initialized Chronos2ForForecasting (d_model=%d, delegating to chronos.chronos2.model)",
            self.d_model,
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        """Return empty embeddings — Chronos-2 has no token vocabulary."""
        return torch.empty((input_ids.shape[0], 0))

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run Chronos-2 inference by delegating to the upstream model.

        Receives pre-batched context, future_covariates, group_ids from
        the timeseries MM pipeline. Batching is handled upstream by
        Chronos2Dataset in the IOProcessor.
        """
        input_len = length_from_prompt_token_ids_or_embeds(input_ids, inputs_embeds)

        context: torch.Tensor | None = kwargs.get("context")  # type: ignore[assignment]
        future_covariates: torch.Tensor | None = kwargs.get("future_covariates")  # type: ignore[assignment]
        group_ids: torch.Tensor | None = kwargs.get("group_ids")  # type: ignore[assignment]
        num_output_patches: int | None = kwargs.get("num_output_patches")  # type: ignore[assignment]

        if context is None:
            # Warmup/profiling pass — return zeros
            return torch.zeros(input_len, 0, device=positions.device, dtype=torch.float32)

        # Determine num_output_patches from preprocessor or fall back to computing from future_covariates
        if num_output_patches is None:
            prediction_length = future_covariates.shape[1] if future_covariates is not None else 0
            if prediction_length == 0:
                prediction_length = self.output_patch_size * self.max_output_patches
            num_output_patches = int(math.ceil(prediction_length / self.output_patch_size))
            num_output_patches = min(num_output_patches, self.max_output_patches)

        prediction_length = num_output_patches * self.output_patch_size

        # Pad or trim future_covariates to match output_size
        if future_covariates is not None:
            if prediction_length > future_covariates.shape[1]:
                pad_size = prediction_length - future_covariates.shape[1]
                pad_tensor = torch.full(
                    (future_covariates.shape[0], pad_size),
                    fill_value=float("nan"),
                    device=future_covariates.device,
                )
                future_covariates = torch.cat([future_covariates, pad_tensor], dim=1)
            else:
                future_covariates = future_covariates[:, :prediction_length]

        # Delegate to the upstream Chronos2Model — single forward pass
        model_kwargs: dict[str, Any] = {
            "context": context,
            "num_output_patches": num_output_patches,
        }
        if group_ids is not None:
            model_kwargs["group_ids"] = group_ids
        if future_covariates is not None:
            model_kwargs["future_covariates"] = future_covariates

        output = self.model(**model_kwargs)
        batch_prediction = output.quantile_preds[..., :prediction_length]

        # Expand to match input_len for vLLM pipeline compatibility.
        # IdentityPooler passes through unchanged; post_process squeezes.
        hidden_states = batch_prediction[None].expand(
            input_len, *(-1 for _ in range(batch_prediction.ndim))
        )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights into the upstream Chronos2Model.

        Prefix checkpoint names with 'model.' to match our wrapping.
        Uses AutoWeightsLoader for robust weight loading.
        """
        prefixed = [(f"model.{name}", tensor) for name, tensor in weights]
        loader = AutoWeightsLoader(self)
        return loader.load_weights(prefixed)