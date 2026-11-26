"""Export a committed MoE DCP checkpoint to resumable Hugging Face shards."""

import json
import math
import os
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path

import torch
from olmo_core.config import Config
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_state_dict
from olmo_core.nn.moe.v2 import olmo3
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_core.nn.transformer.config import TransformerConfig
from safetensors import safe_open
from safetensors.torch import save_file

from open_instruct import logger_utils
from open_instruct.miles.infrastructure.artifacts import atomic_json

logger = logger_utils.setup_logger(__name__)


def layer_number(name, prefix):
    return int(name.removeprefix(prefix).split(".", 1)[0]) if name.startswith(prefix) else -1


class CheckpointWeights:
    """Resolve Core optimizer masters and read only the layer being converted."""

    def __init__(self, model, checkpoint, existing_layers):
        self.checkpoint = checkpoint
        self.existing_layers = existing_layers
        self.targets = model.state_dict()
        self.layers = defaultdict(dict)
        specs = get_checkpoint_metadata(checkpoint).state_dict_metadata
        for name, target in self.targets.items():
            candidates = (f"{name}.main", f"module.{name}.main", name, f"module.{name}", f"model_buffer.{name}")
            matches = [key for key in candidates if key in specs]
            if len(matches) != 1 or specs[matches[0]].size.numel() != target.numel():
                raise ValueError(f"Missing or ambiguous native weight {name}: {matches}")
            key = matches[0]
            self.layers[layer_number(name, "blocks.")][name] = (key, specs[key].size, target.dtype)
        self.loaded_layer = None
        self.loaded = {}

    def __getitem__(self, name):
        layer = layer_number(name, "blocks.")
        if layer in self.existing_layers:
            return self.targets[name]
        if layer != self.loaded_layer:
            started = time.monotonic()
            self.loaded.clear()
            entries = self.layers[layer]
            values = {key: torch.empty(shape, dtype=dtype) for key, shape, dtype in entries.values()}
            load_state_dict(str(self.checkpoint), values, thread_count=8)
            self.loaded = {
                name: values[key].reshape(self.targets[name].shape) for name, (key, _, _) in entries.items()
            }
            self.loaded_layer = layer
            logger.info("Read checkpoint layer %d in %.2fs", layer, time.monotonic() - started)
        return self.loaded[name]


def inventory(path):
    with safe_open(path, framework="pt", device="cpu") as file:
        return {
            name: (tuple(file.get_slice(name).get_shape()), file.get_slice(name).get_dtype())
            for name in file.keys()  # noqa: SIM118 - safe_open does not implement iteration.
        }


def dtype_code(dtype):
    return {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}[dtype]


def shard_name(layer):
    return f"model-layer-{layer + 1:03d}.safetensors"


def export(native: Path, tokenizer: Path, output: Path, update: int):
    manifest = json.loads((native / "complete.json").read_text())
    if manifest["clock"]["completed_steps"] != update:
        raise ValueError(f"Checkpoint {native} does not contain update {update}")
    if (output / ".complete").is_file():
        return
    config = Olmo3MoeConfig.from_dict(manifest["hf_config"])
    model_config = Config.from_dict(manifest["model_config"])
    if not isinstance(model_config, TransformerConfig):
        raise ValueError(f"Expected a transformer model config, found {type(model_config).__name__}")
    model = model_config.build(init_device="meta")
    dense_config = olmo3._config_for_native_dense_layout(model, config)
    source = olmo3._GatheredMoEState(model, dense_config)
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
        "added_tokens.json",
    ):
        if (tokenizer / name).is_file():
            shutil.copyfile(tokenizer / name, output / name)
    config.save_pretrained(output)
    existing = {-1, *range(config.num_hidden_layers)}
    committed = {layer for layer in existing if (output / shard_name(layer)).is_file()}
    source.state = CheckpointWeights(model, native / "model", committed)
    bucket, expected, weights = {}, {}, {}
    active_layer = None
    total_size = 0

    def commit(layer):
        if not expected:
            return
        final = output / shard_name(layer)
        if layer not in committed:
            started = time.monotonic()
            staging = output / f".{final.name}.{uuid.uuid4().hex}.partial"
            save_file(bucket, staging, metadata={"format": "pt"})
            if inventory(staging) != expected:
                raise ValueError(f"Exported tensor inventory differs for {staging}")
            with staging.open("rb") as stream:
                os.fsync(stream.fileno())
            staging.replace(final)
            logger.info("Wrote %s bytes to %s in %.2fs", final.stat().st_size, final, time.monotonic() - started)
        elif inventory(final) != expected:
            raise ValueError(f"Existing shard differs from checkpoint inventory: {final}")
        expected.clear()
        bucket.clear()

    for name, tensor in olmo3.iter_olmo3moe_state_to_hf(dense_config, source):
        layer = layer_number(name, "model.layers.")
        if layer != active_layer:
            commit(active_layer)
            active_layer = layer
        dtype = tensor.dtype
        if layer not in committed:
            bucket[name] = tensor.detach().to(dtype).contiguous().clone()
        expected[name] = (tuple(tensor.shape), dtype_code(dtype))
        weights[name] = shard_name(layer)
        total_size += math.prod(tensor.shape) * torch.empty((), dtype=dtype).element_size()
    commit(active_layer)
    atomic_json(
        output / "model.safetensors.index.json", {"metadata": {"total_size": total_size}, "weight_map": weights}
    )
    atomic_json(output / ".complete", {"update": update, "checkpoint": str(native)})
