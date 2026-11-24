# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/vllm_plugin/src/olmoe3_vllm_plugin/vision.py
"""vLLM executor for whole-model (vision + OLMoE3 LM) exports, language-model-only.

An ``olmo3moe_vision`` export holds the complete multimodal model; vLLM serves only its
language model. With ``language_model_only=True`` (every modality limit is zero) this class
builds exactly the text executor :class:`olmoe3_vllm_plugin.model.Olmo3MoeForCausalLM` from
``hf_config.text_config`` and loads the ``language_model.*`` tensors, leaving the
``vision_tower.*`` / ``connector.*`` tensors on disk. Any other mode is refused: vision
inference is not implemented.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence

import torch
from olmoe3_vllm_plugin.model import Olmo3MoeForCausalLM
from olmoe3_vllm_plugin.multimodal_weights import (
    MultimodalWeightStream,
    check_language_model_loaded,
)
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
)
from vllm.sequence import IntermediateTensors

logger = logging.getLogger(__name__)

TEXT_ARCHITECTURE = "Olmo3MoeForCausalLM"
LANGUAGE_MODEL_ONLY_MESSAGE = (
    "Olmo3MoeVisionForConditionalGeneration serves the language model only: pass "
    "language_model_only=True (olmo-eval: provider.kwargs.language_model_only=true). "
    "Vision inference is not implemented in the OLMoE3 vLLM plugin."
)


def _text_vllm_config(vllm_config: VllmConfig) -> VllmConfig:
    """The outer config re-pointed at the language model, as the text executor expects it."""
    return vllm_config.with_hf_config(
        vllm_config.model_config.hf_config.text_config, architectures=[TEXT_ARCHITECTURE]
    )


class Olmo3MoeVisionProcessingInfo(BaseProcessingInfo):
    """Declares the image modality so vLLM honors ``language_model_only`` for it."""

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}


class Olmo3MoeVisionDummyInputsBuilder(BaseDummyInputsBuilder[Olmo3MoeVisionProcessingInfo]):
    """Only consulted when an image limit is positive, which this executor refuses."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        raise NotImplementedError(LANGUAGE_MODEL_ONLY_MESSAGE)

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int], mm_options=None):
        raise NotImplementedError(LANGUAGE_MODEL_ONLY_MESSAGE)


class Olmo3MoeVisionMultiModalProcessor(BaseMultiModalProcessor[Olmo3MoeVisionProcessingInfo]):
    """Constructed at engine init even in text-only mode; never applied to image inputs."""

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs: Mapping[str, object]):
        raise NotImplementedError(LANGUAGE_MODEL_ONLY_MESSAGE)

    def _get_prompt_updates(
        self, mm_items, hf_processor_mm_kwargs: Mapping[str, object], out_mm_kwargs
    ) -> Sequence:
        raise NotImplementedError(LANGUAGE_MODEL_ONLY_MESSAGE)


@MULTIMODAL_REGISTRY.register_processor(
    Olmo3MoeVisionMultiModalProcessor,
    info=Olmo3MoeVisionProcessingInfo,
    dummy_inputs=Olmo3MoeVisionDummyInputsBuilder,
)
class Olmo3MoeVisionForConditionalGeneration(
    nn.Module, SupportsMultiModal, HasInnerState, IsHybrid, SupportsPP
):
    """The whole-model export's architecture, serving its OLMoE3 language model."""

    packed_modules_mapping = Olmo3MoeForCausalLM.packed_modules_mapping

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        return "<image>" if modality.startswith("image") else None

    # vLLM calls the hybrid-cache classmethods on this class with the OUTER config, whose
    # hf_config is the vision wrapper; the text executor reads its own fields.
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return Olmo3MoeForCausalLM.get_mamba_state_shape_from_config(_text_vllm_config(vllm_config))

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return Olmo3MoeForCausalLM.get_mamba_state_dtype_from_config(_text_vllm_config(vllm_config))

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Olmo3MoeForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        mm_config = vllm_config.model_config.multimodal_config
        language_model_only = bool(
            mm_config is not None
            and (
                getattr(mm_config, "language_model_only", False)
                or mm_config.get_limit_per_prompt("image") == 0
            )
        )
        if not language_model_only:
            raise NotImplementedError(LANGUAGE_MODEL_ONLY_MESSAGE)
        self.config = config
        self.language_model_only = True
        # No vision_tower / connector modules: nothing to initialize, nothing for vLLM's strict
        # loader to expect, and the forward pass is the text executor's.
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=[TEXT_ARCHITECTURE],
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def embed_multimodal(self, **kwargs: object):
        raise NotImplementedError(LANGUAGE_MODEL_ONLY_MESSAGE)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load the ``language_model.*`` tensors into the text executor.

        ``vision_tower.*`` / ``connector.*`` tensors are counted and skipped; any other name
        raises. Returns the loaded names in this module's namespace so vLLM's loader can prove
        no language-model parameter was left uninitialized.
        """
        stream = MultimodalWeightStream(weights)
        loaded = self.language_model.load_weights(stream)
        expected = [name for name, _ in self.language_model.named_parameters()]
        result = check_language_model_loaded(loaded, expected)
        logger.info(
            "Olmo3MoeVision language-model-only load: %s; %d parameter(s) initialized",
            stream.summary(),
            len(result),
        )
        return result


__all__ = [
    "LANGUAGE_MODEL_ONLY_MESSAGE",
    "Olmo3MoeVisionDummyInputsBuilder",
    "Olmo3MoeVisionForConditionalGeneration",
    "Olmo3MoeVisionMultiModalProcessor",
    "Olmo3MoeVisionProcessingInfo",
]
