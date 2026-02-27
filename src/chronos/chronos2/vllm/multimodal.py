"""Multimodal boilerplate for the "timeseries" modality in vLLM.

Provides the MM processing pipeline classes that route timeseries
dict data (context, future_covariates, group_ids) through vLLM's
multimodal infrastructure.
"""

import hashlib
import time
from typing import Any, Mapping, Sequence

import torch
from transformers import BatchFeature
from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalInputs,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptUpdate,
)

# The modality name used throughout the MM pipeline.
MODALITY = "timeseries"

# Field names expected in the timeseries MM data dict.
REQUIRED_FIELDS = frozenset({"context", "future_covariates", "group_ids", "num_output_patches"})


def _field_config() -> dict[str, MultiModalFieldConfig]:
    """Shared field config for all timeseries fields."""
    return {
        "context": MultiModalFieldConfig.shared(MODALITY, batch_size=1),
        "future_covariates": MultiModalFieldConfig.shared(MODALITY, batch_size=1),
        "group_ids": MultiModalFieldConfig.shared(MODALITY, batch_size=1),
        "num_output_patches": MultiModalFieldConfig.shared(MODALITY, batch_size=1),
    }


# -------------------------------------------------------------------
# Processing info: tells vLLM what modalities we support
# -------------------------------------------------------------------


class ChronosProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {MODALITY: None}

    def get_data_parser(self) -> MultiModalDataParser:
        return ChronosMultiModalDataParser()

    def build_data_parser(self) -> MultiModalDataParser:
        return ChronosMultiModalDataParser()

    @property  # type: ignore[override]
    def data_parser(self) -> MultiModalDataParser:
        if not hasattr(self, "_data_parser"):
            self._data_parser = ChronosMultiModalDataParser()
        return self._data_parser


# -------------------------------------------------------------------
# Dummy input builder: provides profiling data for GPU warmup
# -------------------------------------------------------------------


class ChronosInputBuilder(BaseDummyInputsBuilder[ChronosProcessingInfo]):
    """Provides dummy data for vLLM's GPU profiling/warmup pass."""

    def __init__(self, info: ChronosProcessingInfo):
        super().__init__(info)

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
        **kwargs: Any,
    ) -> MultiModalDataDict:
        return {
            MODALITY: {
                "context": torch.ones(100, 2048, dtype=torch.float32),
                "future_covariates": torch.ones(100, 1024, dtype=torch.float32),
                "group_ids": torch.zeros(100, dtype=torch.long),
                "num_output_patches": 64,
            }
        }


# -------------------------------------------------------------------
# Data parser: routes timeseries dict data through the MM pipeline
# -------------------------------------------------------------------


class ChronosMultiModalDataParser(MultiModalDataParser):
    """Parses timeseries dict data for vLLM's MM pipeline."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _parse_timeseries_data(
        self,
        data: dict[str, torch.Tensor],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality=MODALITY,
                required_fields=REQUIRED_FIELDS,
                fields_factory=lambda _: _field_config(),
            )
        return None

    def _get_subparsers(self) -> Mapping[str, Any]:
        return {MODALITY: self._parse_timeseries_data}

    def parse_mm_data(self, mm_data: MultiModalDataDict, **kwargs: Any) -> MultiModalDataItems:
        if MODALITY not in mm_data:
            mm_data = {MODALITY: mm_data}

        ts_data = mm_data[MODALITY]
        items = self._parse_timeseries_data(ts_data)
        if items is None:
            raise ValueError("Failed to parse timeseries data")

        return MultiModalDataItems({MODALITY: items})


# -------------------------------------------------------------------
# Processor: converts parsed data into MultiModalInputs
# -------------------------------------------------------------------


class ChronosMultiModalProcessor(BaseMultiModalProcessor):
    """Processes timeseries MM data into MultiModalInputs for vLLM."""

    def __init__(
        self,
        info: ChronosProcessingInfo,
        dummy_inputs: BaseDummyInputsBuilder[ChronosProcessingInfo],
        *,
        cache: MultiModalProcessorOnlyCache | None = None,
    ) -> None:
        super().__init__(info=info, dummy_inputs=dummy_inputs, cache=cache)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _field_config()

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        return []

    def apply(self, *args: Any, **kwargs: Any) -> MultiModalInputs:
        # Handle both old and new vLLM calling conventions:
        # Old: apply(prompt, mm_items, hf_processor_mm_kwargs, ...)
        # New: apply(processor_inputs, timing_ctx)
        # Internal (from get_dummy_mm_inputs): apply(prompt=..., mm_items=..., ...)

        ts_data: dict[str, Any] = {}

        if args and hasattr(args[0], "prompt"):
            # New vLLM: first arg is ProcessorInputs dataclass
            processor_inputs = args[0]
            mm_items = getattr(processor_inputs, "mm_items", None) or getattr(
                processor_inputs, "mm_data_items", None
            )
            if mm_items is not None and isinstance(mm_items, MultiModalDataItems):
                if MODALITY in mm_items:
                    ts_items = mm_items[MODALITY]
                    ts_data = ts_items.data if hasattr(ts_items, "data") else {}
        else:
            # Old vLLM / direct call: extract from positional/keyword args
            mm_items = args[1] if len(args) > 1 else kwargs.get("mm_items")
            mm_data = kwargs.get("mm_data")

            if mm_items is not None and isinstance(mm_items, MultiModalDataItems):
                if MODALITY in mm_items:
                    ts_items = mm_items[MODALITY]
                    ts_data = ts_items.data if hasattr(ts_items, "data") else {}
            elif mm_data is not None and isinstance(mm_data, dict):
                ts_data = mm_data.get(MODALITY, mm_data)

        mm_placeholders = {MODALITY: [PlaceholderRange(offset=0, length=0)]}

        # Build MultiModalKwargsItems directly to ensure proper modality keying.
        # from_hf_inputs + BatchFeature can produce empty modalities when
        # the data doesn't match the expected shared field batch structure.
        field_config = _field_config()
        mm_item_dict: dict[str, MultiModalFieldElem] = {}
        for key, config in field_config.items():
            tensor = ts_data.get(key) if isinstance(ts_data, dict) else None
            if tensor is not None:
                mm_item_dict[key] = MultiModalFieldElem(
                    data=tensor,
                    field=config.field,
                )

        mm_kwargs_items = MultiModalKwargsItems({MODALITY: [MultiModalKwargsItem(mm_item_dict)]})

        # Unique hash per request (required by vLLM v0.16+)
        ts_hash = hashlib.sha256(
            str(id(ts_data)).encode() + str(time.monotonic()).encode()
        ).hexdigest()[:16]

        return MultiModalInputs(
            type="multimodal",
            prompt_token_ids=[1],
            mm_kwargs=mm_kwargs_items,
            mm_hashes={MODALITY: [ts_hash]},
            mm_placeholders=mm_placeholders,
        )
