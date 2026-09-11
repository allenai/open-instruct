"""Stream Core DCP FP32 masters or reconstructed model values in exact HF layout."""

from collections.abc import Mapping
from pathlib import Path

import torch
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
from olmo_core.nn import attention
from olmo_core.nn.hf import convert
from olmo_core.nn.moe.v2 import olmo3

from open_instruct.miles import fla_compat


class CoreCheckpointState(Mapping):
    def __init__(self, path, hf_config, *, category):
        if category not in ("fp32_masters", "reconstructed_model_storage"):
            raise ValueError("Unknown Core checkpoint category")
        self.path = Path(path)
        self.category = category
        self.metadata = get_checkpoint_metadata(str(path))
        # Meta construction establishes native parameter shape and storage dtype
        # without allocating model weights or executing any forward operation.
        if "linear_attention" in hf_config.layer_types:
            fla_compat.install_kda_triton_compat()
        config = olmo3.build_olmo3_moe_config_from_hf_config(
            hf_config, attention_backend=attention.AttentionBackendName.torch
        )
        model = config.build(init_device="meta")
        self.hf_config = olmo3._config_for_native_dense_layout(model, hf_config)
        self.parameters = {name: (tuple(value.shape), value.dtype) for name, value in model.named_parameters()}
        checkpoint_keys = {
            name.removeprefix("module.").removesuffix(".main"): name
            for name in self.metadata.state_dict_metadata
            if name.endswith(".main")
        }
        if set(checkpoint_keys) != set(self.parameters):
            raise ValueError("Core checkpoint master inventory differs from the exact native factory")
        self.keys = checkpoint_keys
        self.virtual = {}
        q_size = hf_config.num_attention_heads * hf_config.head_dim
        kv_size = hf_config.num_key_value_heads * hf_config.head_dim
        for name in self.parameters:
            saved = self.metadata.state_dict_metadata[self.keys[name]]
            shape, _ = self.parameters[name]
            if saved.properties.dtype != torch.float32 or saved.size.numel() != torch.Size(shape).numel():
                raise ValueError(f"Core master shape or dtype differs: {name}")
            if name.endswith(".attention.w_qkv.weight"):
                start = 0
                for suffix, size in (("w_q", q_size), ("w_k", kv_size), ("w_v", kv_size)):
                    self.virtual[name.replace("w_qkv", suffix)] = (name, start, size)
                    start += size
                if shape[0] != start:
                    raise ValueError("Native fused QKV shape differs from HF head geometry")
            else:
                self.virtual[name] = (name, None, None)
        self.loaded_key = None
        self.loaded_value = None

    def __iter__(self):
        return iter(self.virtual)

    def __len__(self):
        return len(self.virtual)

    def __getitem__(self, key):
        native, start, size = self.virtual[key]
        if self.loaded_key != native:
            self.loaded_value = None
            tensor = next(load_keys(str(self.path), [self.keys[native]]))
            shape, dtype = self.parameters[native]
            if not torch.is_tensor(tensor) or tensor.dtype != torch.float32 or tensor.device.type != "cpu":
                raise ValueError(f"Expected CPU FP32 master: {native}")
            tensor = tensor.reshape(shape)
            if self.category == "reconstructed_model_storage":
                tensor = tensor.to(dtype=dtype)
            self.loaded_key, self.loaded_value = native, tensor
        tensor = self.loaded_value
        return tensor if start is None else tensor.narrow(0, start, size)

    def stream(self):
        yield from convert.iter_olmo3moe_state_to_hf(self.hf_config, self)
