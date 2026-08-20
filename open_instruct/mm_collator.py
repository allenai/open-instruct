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
"""Collator for multimodal SFT.

The tokenized dataset carries image paths/bytes, not pixels. This collator pops them off the
features, runs the model's image processor once per batch, and merges the resulting vision kwargs
into the batch. Packing and explicit mrope positions are out of scope; see
``docs/algorithms/multimodal_sft.md``.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import DataCollatorForSeq2Seq

from open_instruct import logger_utils
from open_instruct.mm_plugin import IMAGE_PLACEHOLDER, IMAGES_KEY, MultiModalPlugin

logger = logger_utils.setup_logger(__name__)

MASKED_TOKEN_VALUE = -100


@dataclass
class MultiModalDataCollator:
    """Pad text features and attach vision inputs.

    Args:
        tokenizer: the text tokenizer (used for padding and for encoding the dummy-image block).
        processor: the ``AutoProcessor``, carrying the image processor and pixel budget.
        plugin: the model-family plugin that knows how to expand and process images.
        model: passed through to ``DataCollatorForSeq2Seq`` for ``prepare_decoder_input_ids``.
        compute_dtype: floating point vision inputs are cast to this, to match the model weights.
        inject_dummy_image: add a throwaway image to batches that have none. Required under
            ZeRO-3/FSDP: if a rank's batch never touches the vision tower, its parameters get no
            gradient and the collective hangs.
    """

    tokenizer: Any
    processor: Any
    plugin: MultiModalPlugin
    model: Any = None
    padding: str = "longest"
    compute_dtype: torch.dtype = torch.bfloat16
    inject_dummy_image: bool = True
    _base_collator: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._base_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model, padding=self.padding)

    def _dummy_image_token_ids(self) -> tuple[list[int], list[Any]]:
        """Token ids for one dummy image, plus the image itself."""
        dummy_images = [self.plugin.dummy_image()]
        expanded = self.plugin.process_messages(
            [{"role": "user", "content": IMAGE_PLACEHOLDER}], dummy_images, self.processor
        )
        token_ids = self.tokenizer.encode(expanded[0]["content"], add_special_tokens=False)
        return token_ids, dummy_images

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_images: list[Any] = []
        for feature in features:
            images = feature.pop(IMAGES_KEY, None) or []
            batch_images.extend(images)
            # `index` is bookkeeping from the dataset cache; it must not reach the model forward.
            feature.pop("index", None)

        if not batch_images and self.inject_dummy_image:
            # Append the dummy block to the first example. attention_mask 0 and label -100 keep it
            # out of both attention and the loss; it exists purely so the vision tower runs.
            dummy_token_ids, batch_images = self._dummy_image_token_ids()
            feature = features[0]
            device = feature["input_ids"].device if torch.is_tensor(feature["input_ids"]) else None
            dummy = torch.tensor(dummy_token_ids, dtype=torch.long, device=device)

            feature["input_ids"] = torch.cat([_as_tensor(feature["input_ids"], device), dummy])
            feature["attention_mask"] = torch.cat(
                [_as_tensor(feature["attention_mask"], device), torch.zeros_like(dummy)]
            )
            feature["labels"] = torch.cat(
                [_as_tensor(feature["labels"], device), torch.full_like(dummy, MASKED_TOKEN_VALUE)]
            )

        batch = self._base_collator(features)

        mm_inputs = self.plugin.get_mm_inputs(batch_images, self.processor)
        for key, value in mm_inputs.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                value = value.to(self.compute_dtype)
            batch[key] = value

        if self.plugin.requires_mm_token_type_ids:
            # Computed last, so it lines up with the final padded input_ids -- including any dummy
            # image tokens appended above.
            batch["mm_token_type_ids"] = self.plugin.build_mm_token_type_ids(
                batch["input_ids"], self.processor, self.tokenizer
            )

        return batch


def _as_tensor(value: Any, device: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return torch.tensor(value, dtype=torch.long, device=device)
