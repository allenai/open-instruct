"""Immutable still-image inputs and exact processor-expanded prompt accounting.

The initial contract is the qualified Olmo KDA document prompt, with no chat
template or stochastic augmentation. Pixel bytes, processor code/config and
expanded token layout are all part of the policy's conditioning identity.
"""

import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path

from PIL import Image

from open_instruct.miles.datasets import jigsaw
from open_instruct.miles.errors import InputError


def image_row(row: dict) -> bool:
    metadata = row.get("metadata") or {}
    prompt = row.get("input")
    return (
        "images" in row
        or (isinstance(metadata, dict) and "vision" in metadata)
        or (isinstance(prompt, str) and "<|image|>" in prompt)
    )


def load_processor(checkpoint: Path):
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") != "olmo3moe_vision":
        raise InputError("Image preparation currently requires the qualified olmo3moe_vision processor")
    try:
        module = importlib.import_module("olmo_sglang.vision.processing")
    except ImportError as error:
        raise InputError(
            "Install the pinned olmo-sglang vision processor and OLMo-core; tokenizer fallback is unsafe"
        ) from error
    return module.OlmoVisionProcessor.from_pretrained(checkpoint)


def processor_files(processor) -> list[Path]:
    """Include transitive image preprocessing code as well as its wrapper."""
    wrapper = inspect.getmodule(type(processor))
    return sorted(
        {
            Path(inspect.getfile(obj)).resolve()
            for obj in (
                wrapper,
                wrapper.preprocess_image_molmo2,
                wrapper.build_image_token_ids,
                wrapper.validate_image_tokenizer,
            )
        }
    )


def read_images(row: dict, inputs: dict) -> list[Image.Image]:
    vision = (row.get("metadata") or {}).get("vision")
    if not isinstance(vision, dict) or vision.get("schema_version") != 1 or vision.get("prompt_format") != "document":
        raise InputError("Image rows require metadata.vision schema_version=1 and prompt_format=document")
    descriptors, paths = vision.get("images"), row.get("images")
    if not isinstance(descriptors, list) or not descriptors or not isinstance(paths, list):
        raise InputError("Image rows require nonempty images and matching vision descriptors")
    if paths != [d.get("path") for d in descriptors if isinstance(d, dict)]:
        raise InputError("Image paths/order differ from the immutable descriptors")
    decoded = []
    for descriptor in descriptors:
        path = Path(descriptor["path"])
        if not path.is_absolute():
            raise InputError("Image references must be absolute worker-visible paths")
        try:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != descriptor.get("sha256"):
                raise InputError(f"Image bytes changed: {path}")
            with Image.open(io.BytesIO(raw)) as source:
                image = source.convert("RGB")
        except (OSError, ValueError) as error:
            raise InputError(f"Cannot decode image {path}: {error}") from error
        if image.size != (descriptor.get("width"), descriptor.get("height")) or jigsaw.pixels_sha256(
            image
        ) != descriptor.get("pixels_sha256"):
            raise InputError(f"Image pixels/geometry changed: {path}")
        inputs[str(path)] = digest
        decoded.append(image)
    return decoded


def tensor_digest(tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return jigsaw.sha256(jigsaw.encoded({"shape": array.shape, "dtype": str(array.dtype)}) + array.tobytes())


def process_row(row: dict, processor, inputs: dict) -> tuple[list[int], list]:
    images = read_images(row, inputs)
    tokenizer = processor.tokenizer
    if tokenizer.eos_token_id is None:
        raise InputError("Document image prompts require an EOS token")
    unexpanded = [tokenizer.eos_token_id, *tokenizer.encode(row["input"], add_special_tokens=False)]
    try:
        ids, prepared = processor.prepare(unexpanded, images)
    except (TypeError, ValueError, NotImplementedError) as error:
        raise InputError(f"Image processor rejected the input: {error}") from error
    if len(prepared) != len(images):
        raise InputError("Image processor dropped an image")
    fingerprints = {}
    for path in processor_files(processor):
        digest = jigsaw.sha256(path.read_bytes())
        inputs[str(path)] = digest
        fingerprints[path.name] = digest
    state = {
        "placeholder_token_ids": unexpanded,
        "expanded_prompt_token_ids": ids,
        "processor_code": fingerprints,
        "processor_config_sha256": jigsaw.sha256(jigsaw.encoded(processor.config)),
        "images": [
            {
                "patches_sha256": tensor_digest(image.patches),
                "pooling_sha256": tensor_digest(image.pooling),
                "token_ids": image.token_ids,
                "offsets": [list(pair) for pair in image.offsets],
                "crop_count": image.patches.shape[1],
                "patch_tensor_bytes": image.patches.numel() * image.patches.element_size(),
            }
            for image in prepared
        ],
    }
    existing = row["metadata"]["vision"].get("prepared")
    if existing is not None and existing != state:
        raise InputError("Prepared image layout/features changed under the selected processor")
    row["metadata"]["vision"]["prepared"] = state
    return ids, prepared


def prepare_row(row: dict, processor, inputs: dict) -> list[int]:
    return process_row(row, processor, inputs)[0]


def prompt_identity(row: dict) -> str:
    if not image_row(row):
        return row["input"]
    return jigsaw.sha256(
        jigsaw.encoded(
            {
                "input": row["input"],
                "pixels": [image["pixels_sha256"] for image in row["metadata"]["vision"]["images"]],
            }
        )
    )
