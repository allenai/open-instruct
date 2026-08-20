# Copyright 2025 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multimodal (vision-language) plugins for SFT.

Assistant label spans are derived from character offsets over the rendered chat template
(``dataset_transformation._tokenize_tulu_sft_with_assistant_labels``), which is text-only. To keep
that machinery unchanged, each ``<image>`` placeholder is expanded into the model's real image
tokens *before* the chat template runs; afterwards the conversation is plain text again.

Pixels are never stored in the dataset cache -- the tokenized dataset carries image paths, and the
collator calls :meth:`MultiModalPlugin.get_mm_inputs` at batch time.

Adding a model family means a subclass implementing ``image_token_counts`` and
``format_image_tokens``, registered in ``MM_PLUGIN_REGISTRY`` by ``config.model_type``.
"""

import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image
from PIL.Image import Image as ImageObject

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

# The sentinel we expect inside message content.
IMAGE_PLACEHOLDER = "<image>"
# Dataset column holding the (unprocessed) images for a row.
IMAGES_KEY = "images"

# Pixel budget applied *before* the model's own image processor runs.
DEFAULT_IMAGE_MAX_PIXELS = 768 * 768
DEFAULT_IMAGE_MIN_PIXELS = 32 * 32

ImageInput = str | bytes | dict[str, Any] | ImageObject


def attach_processor_config(
    processor: Any, image_max_pixels: int | None = None, image_min_pixels: int | None = None
) -> Any:
    """Stash the pixel budget on the processor object.

    Keeping it here means tokenization and collation read identical values without threading extra
    kwargs through both; a mismatch would silently produce the wrong number of image tokens.
    """
    if processor is None:
        return None
    processor.image_max_pixels = image_max_pixels if image_max_pixels is not None else DEFAULT_IMAGE_MAX_PIXELS
    processor.image_min_pixels = image_min_pixels if image_min_pixels is not None else DEFAULT_IMAGE_MIN_PIXELS
    return processor


@dataclass
class MultiModalPlugin:
    """Base plugin. Subclasses implement the two model-specific pieces."""

    image_token: str = ""
    # Some families (Qwen3.5) raise rather than infer M-RoPE positions without a modality map.
    requires_mm_token_type_ids: bool = False

    # ------------------------------------------------------------------ validation
    def validate_messages(self, messages: list[dict[str, Any]], images: list[ImageInput]) -> None:
        """Placeholder count must equal the number of images, or the forward pass dies obscurely."""
        num_placeholders = sum(str(message.get("content", "")).count(IMAGE_PLACEHOLDER) for message in messages)
        if num_placeholders != len(images):
            raise ValueError(
                f"Found {num_placeholders} '{IMAGE_PLACEHOLDER}' placeholders but {len(images)} images. "
                f"They must match exactly. Roles: {[m.get('role') for m in messages]}"
            )

    # ------------------------------------------------------------------ image loading
    def _preprocess_image(self, image: ImageObject, image_max_pixels: int, image_min_pixels: int) -> ImageObject:
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            image = image.resize((max(1, int(image.width * resize_factor)), max(1, int(image.height * resize_factor))))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def regularize_images(self, images: list[ImageInput], processor: Any) -> list[ImageObject]:
        """Open, RGB-convert and resize images into the configured pixel budget.

        Must be deterministic: tokenization derives the image token count from this output and the
        collator recomputes pixels from it, so any divergence silently corrupts training.
        """
        image_max_pixels = getattr(processor, "image_max_pixels", DEFAULT_IMAGE_MAX_PIXELS)
        image_min_pixels = getattr(processor, "image_min_pixels", DEFAULT_IMAGE_MIN_PIXELS)

        results = []
        for image in images:
            if isinstance(image, str):
                loaded = Image.open(image)
            elif isinstance(image, bytes):
                loaded = Image.open(BytesIO(image))
            elif isinstance(image, ImageObject):
                loaded = image
            elif isinstance(image, dict):
                # `datasets.Image(decode=False)` yields {"bytes": ..., "path": ...}
                if image.get("bytes") is not None:
                    loaded = Image.open(BytesIO(image["bytes"]))
                elif image.get("path") is not None:
                    loaded = Image.open(image["path"])
                else:
                    raise ValueError(f"Image dict has neither 'bytes' nor 'path': {sorted(image)}")
            else:
                raise ValueError(f"Unsupported image input type: {type(image)}")

            results.append(self._preprocess_image(loaded, image_max_pixels, image_min_pixels))

        return results

    # ------------------------------------------------------------------ model specific
    def image_token_counts(self, images: list[ImageObject], processor: Any) -> list[int]:
        """Number of image tokens the model emits for each (already regularized) image."""
        raise NotImplementedError

    def format_image_tokens(self, num_tokens: int) -> str:
        """Wrap ``num_tokens`` repetitions of the image token in whatever the model expects."""
        raise NotImplementedError

    # ------------------------------------------------------------------ main entry points
    def process_messages(
        self, messages: list[dict[str, Any]], images: list[ImageInput], processor: Any
    ) -> list[dict[str, Any]]:
        """Replace each ``<image>`` placeholder with the model's real image-token block."""
        self.validate_messages(messages, images)
        if not images:
            return messages

        regularized = self.regularize_images(images, processor)
        counts = self.image_token_counts(regularized, processor)

        out_messages = []
        image_idx = 0
        for message in messages:
            content = str(message.get("content", ""))
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, self.format_image_tokens(counts[image_idx]), 1)
                image_idx += 1
            out_messages.append({**message, "content": content})

        return out_messages

    def get_mm_inputs(self, images: list[ImageInput], processor: Any) -> dict[str, Any]:
        """Produce the model's vision kwargs (``pixel_values``, ``image_grid_thw``, ...)."""
        if not images:
            return {}
        regularized = self.regularize_images(images, processor)
        return dict(processor.image_processor(regularized, return_tensors="pt"))

    def dummy_image(self) -> ImageObject:
        """A tiny white image, used to keep the vision tower in the graph for all-text batches."""
        return Image.new("RGB", (64, 64), (255, 255, 255))

    def build_mm_token_type_ids(self, input_ids: Any, processor: Any, tokenizer: Any) -> Any:
        """Per-token modality map: text 0, image 1, video 2.

        Qwen3.5 needs this to place 3-D M-RoPE positions and raises if it is missing, because we
        build ``input_ids`` with the tokenizer rather than the full processor.
        """
        import torch  # noqa: PLC0415  -- keep torch out of the import path for dataset-only use

        if hasattr(processor, "create_mm_token_type_ids"):
            return torch.tensor(processor.create_mm_token_type_ids(input_ids.tolist()), dtype=torch.long)

        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        return (input_ids == image_token_id).to(torch.long)


@dataclass
class Qwen2VLPlugin(MultiModalPlugin):
    """Qwen2-VL / Qwen2.5-VL / Qwen3-VL.

    Token count is ``prod(image_grid_thw) // merge_size**2``, delimited by ``<|vision_start|>`` /
    ``<|vision_end|>``.
    """

    image_token: str = "<|image_pad|>"
    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    def image_token_counts(self, images: list[ImageObject], processor: Any) -> list[int]:
        image_processor = processor.image_processor
        merge_length = int(image_processor.merge_size) ** 2
        # Running the real image processor rather than reimplementing `smart_resize` keeps the count
        # correct against whatever transformers version is installed. It costs a second pass over the
        # images at dataset-build time, which is cached. Deriving the grid analytically from the
        # image dimensions would avoid that.
        grid_thw = image_processor(images, return_tensors="pt")["image_grid_thw"]
        return [int(grid.prod()) // merge_length for grid in grid_thw]

    def format_image_tokens(self, num_tokens: int) -> str:
        return f"{self.vision_bos_token}{self.image_token * num_tokens}{self.vision_eos_token}"


@dataclass
class Qwen3_5Plugin(Qwen2VLPlugin):
    """Qwen3.5.

    Image handling is identical to Qwen2-VL. The one difference that matters for training is that
    `Qwen3_5ForConditionalGeneration.forward` raises instead of inferring M-RoPE positions when
    `mm_token_type_ids` is absent.
    """

    requires_mm_token_type_ids: bool = True


# Keyed by `AutoConfig.model_type`. Only families that are actually tested belong here: a plugin
# whose token count is off by one trains without error and silently degrades the model.
MM_PLUGIN_REGISTRY: dict[str, type[MultiModalPlugin]] = {
    "qwen2_vl": Qwen2VLPlugin,
    "qwen2_5_vl": Qwen2VLPlugin,
    "qwen3_vl": Qwen2VLPlugin,
    "qwen3_vl_moe": Qwen2VLPlugin,
    "qwen3_5": Qwen3_5Plugin,
    "qwen3_5_moe": Qwen3_5Plugin,
}


@dataclass
class CompositeModuleKeys:
    """Module-name fragments identifying the parts of a composite VLM.

    Matched as substrings against ``model.named_parameters()``, so they are robust to the
    ``model.`` / ``model.model.`` prefix differences between transformers versions.

    The vision keys enumerate the tower's *sub*-modules rather than the tower itself: on Qwen the
    projector (``visual.merger``) lives inside ``visual``, so naming the whole tower would make
    ``--freeze_vision_tower`` silently freeze the projector too.
    """

    vision_model_keys: list[str]
    projector_keys: list[str]
    language_model_keys: list[str]
    lora_conflict_keys: list[str]
    # Module whose `gradient_checkpointing` flag governs the vision tower, matched on the last
    # path component (e.g. "model.visual" -> "visual").
    vision_root_keys: list[str]


_QWEN2_VL_MODULES = CompositeModuleKeys(
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    projector_keys=["visual.merger"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
    vision_root_keys=["visual"],
)
_QWEN3_VL_MODULES = CompositeModuleKeys(
    vision_model_keys=["visual.pos_embed", "visual.patch_embed", "visual.blocks", "visual.deepstack_merger_list"],
    projector_keys=["visual.merger"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
    vision_root_keys=["visual"],
)

_QWEN3_5_MODULES = CompositeModuleKeys(
    vision_model_keys=["visual.pos_embed", "visual.patch_embed", "visual.blocks"],
    projector_keys=["visual.merger"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
    vision_root_keys=["visual"],
)

COMPOSITE_MODULES: dict[str, CompositeModuleKeys] = {
    "qwen2_vl": _QWEN2_VL_MODULES,
    "qwen2_5_vl": _QWEN2_VL_MODULES,
    "qwen3_vl": _QWEN3_VL_MODULES,
    "qwen3_vl_moe": _QWEN3_VL_MODULES,
    # No deepstack merger modules, despite `deepstack_visual_indexes` in the config.
    "qwen3_5": _QWEN3_5_MODULES,
    "qwen3_5_moe": _QWEN3_5_MODULES,
}


def configure_visual_config(config: Any) -> Any:
    """Hoist `hidden_size` from the text sub-config to the top level.

    A composite VLM config keeps `hidden_size` under `config.text_config`. DeepSpeed ZeRO-3 reads
    `model.config.hidden_size` to fill the `auto` entries in the ds config and hard-errors when it
    is missing.
    """
    text_config = getattr(config, "text_config", None)
    if text_config is not None and not getattr(config, "hidden_size", None):
        hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is not None:
            logger.info(f"Hoisting hidden_size={hidden_size} from text_config for DeepSpeed ZeRO-3.")
            config.hidden_size = hidden_size
    return config


def disable_vision_gradient_checkpointing(model: Any, model_type: str) -> list[str]:
    """Turn gradient checkpointing off for the vision tower. Returns the modules changed.

    Required when the vision tower is frozen under ZeRO-3, which releases a frozen parameter after
    the forward pass and never re-gathers it. Activation recomputation still needs it, so the
    recompute sees shape-[0] tensors and torch raises ``CheckpointError``. Checkpointing a frozen
    tower buys nothing anyway: there is no backward pass through it to trade compute for.
    """
    keys = COMPOSITE_MODULES.get(model_type)
    if keys is None:
        return []

    disabled = []
    for name, module in model.named_modules():
        # Match the vision root *and everything nested under it*: on Qwen that is `visual` plus all
        # 32 `visual.blocks.N`, and the blocks are where the recompute actually happens.
        path_parts = name.split(".")
        if any(root in path_parts for root in keys.vision_root_keys) and getattr(
            module, "gradient_checkpointing", False
        ):
            module.gradient_checkpointing = False
            disabled.append(name)
    return disabled


def get_frozen_module_keys(
    model_type: str, freeze_vision_tower: bool, freeze_multi_modal_projector: bool, freeze_language_model: bool
) -> list[str]:
    """Module-name fragments that should have ``requires_grad=False``."""
    keys = COMPOSITE_MODULES.get(model_type)
    if keys is None:
        return []

    frozen: list[str] = []
    if freeze_vision_tower:
        frozen.extend(keys.vision_model_keys)
    if freeze_multi_modal_projector:
        frozen.extend(keys.projector_keys)
    if freeze_language_model:
        frozen.extend(keys.language_model_keys)
    return frozen


def filter_lora_target_modules(
    model: Any, model_type: str, target_modules: list[str], frozen_keys: list[str]
) -> list[str]:
    """Expand bare LoRA target names into full module names, dropping vision-side matches.

    Qwen's vision blocks have their own ``gate_proj`` / ``up_proj`` / ``down_proj``, so the default
    target list would make ``--use_lora`` quietly train the vision encoder.
    """
    keys = COMPOSITE_MODULES.get(model_type)
    if keys is None:
        return target_modules

    forbidden = list(frozen_keys) + list(keys.lora_conflict_keys)
    module_names = [
        name
        for name, _ in model.named_modules()
        if any(target in name for target in target_modules) and not any(bad in name for bad in forbidden)
    ]
    return module_names


def get_mm_plugin(model_type: str, processor: Any = None) -> MultiModalPlugin | None:
    """Return the plugin for a ``model_type``, or None if the model is text-only/unsupported.

    When the processor exposes an ``image_token`` we take it from there rather than trusting the
    subclass default, so a checkpoint with a non-standard token still lines up.
    """
    plugin_cls = MM_PLUGIN_REGISTRY.get(model_type)
    if plugin_cls is None:
        return None

    plugin = plugin_cls()
    processor_image_token = getattr(processor, "image_token", None)
    if processor_image_token and processor_image_token != plugin.image_token:
        logger.warning(
            f"Processor reports image_token={processor_image_token!r}, overriding the plugin default "
            f"{plugin.image_token!r} for model_type={model_type}."
        )
        plugin.image_token = processor_image_token
    return plugin


def is_multimodal_model_type(model_type: str) -> bool:
    return model_type in MM_PLUGIN_REGISTRY
