"""Stable model API shared by the MILES actor and both Core trainer backends."""

import json
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import transformers
from olmo_core import config as core_config
from olmo_core.nn import transformer
from olmo_core.nn.transformer import config as transformer_config

from open_instruct.miles.timing import startup_stage


def _backend(kind):
    return import_module(f"open_instruct.miles.{kind}_models")


def __getattr__(name):
    # Preserve the former public subclass without importing the specialized
    # trainer when a dense model is selected.
    if name == "HFInitializedMoETrainModule":
        return _backend("moe").HFInitializedMoETrainModule
    raise AttributeError(name)


def model_config_from_hf(hf: Any, options: Any):
    if options.model_config:
        document = json.loads(Path(options.model_config).read_text())
        config = core_config.Config.from_dict(document.get("model", document))
        if not isinstance(config, (transformer.TransformerConfig, transformer_config.OLMoDDPModelConfig)):
            raise ValueError("core.model_config must describe a Core transformer")
        return config
    return _backend("moe" if hf.model_type == "olmo3moe" else "standard").model_config_from_hf(hf, options)


class MetricSink:
    """The small trainer-facing surface used by Core's optimizer/metric methods."""

    global_step = 0
    global_train_tokens_seen = 0

    def __init__(self):
        self.metrics = {}

    def record_metric(self, name, value, *args, namespace=None, **kwargs):
        key = f"{namespace}/{name}" if namespace else name
        self.metrics[key] = float(value.detach().item() if isinstance(value, torch.Tensor) else value)


def build_train_module(args, source=None):
    source = source or args.hf_checkpoint
    # AutoConfig cannot identify the custom MoE family until it is registered.
    # Read the descriptor first so ordinary Olmo 3 never loads the MoE backend.
    descriptor, _ = transformers.PretrainedConfig.get_config_dict(source)
    if descriptor.get("model_type") == "olmo3moe":
        _backend("moe").register_hf_classes()
    hf = transformers.AutoConfig.from_pretrained(source, trust_remote_code=True)
    config = model_config_from_hf(hf, args.olmo_core)
    config.init_seed = args.seed
    kind = "moe" if isinstance(config, transformer_config.OLMoDDPModelConfig) else "standard"
    backend = _backend(kind)
    backend.validate_training_options(args)
    if kind == "moe":
        backend.prepare_model_config(config, hf, args.olmo_core)
    model = config.build(init_device="meta")
    forward_capacity = args.olmo_core.packing_max_tokens or args.olmo_core.max_sequence_length
    common = dict(
        model=model,
        rank_microbatch_size=forward_capacity,
        max_sequence_length=forward_capacity,
        compile_model=False,
        device=torch.device("cuda", torch.cuda.current_device()),
        max_grad_norm=args.clip_grad,
    )
    optim = dict(
        lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps, weight_decay=args.weight_decay
    )
    with startup_stage(args, "hf_read"):
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            source, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
    with startup_stage(args, "native_model_optimizer_build", device=torch.cuda):
        module = backend.build_train_module(
            args, common=common, optim=optim, hf_config=hf, hf_state=hf_model.state_dict()
        )
    module._trainer = MetricSink()
    module._miles_model_backend = kind
    del hf_model
    return module, hf, config


def iter_export_state(module, hf, *, stream_moe=True, fused_experts=False):
    """Yield HF weights through the selected trainer's native conversion path.

    ``fused_experts`` yields routed experts stacked per layer in the serving
    engine's layout, for publication only; HF exports keep per-expert slices.
    """
    kind = getattr(module, "_miles_model_backend", "moe" if hf.model_type == "olmo3moe" else "standard")
    yield from _backend(kind).iter_export_state(module, hf, stream_moe=stream_moe, fused_experts=fused_experts)


def export_state(module, hf):
    return {name: value.detach().cpu() for name, value in iter_export_state(module, hf)}


def _module_backend(module):
    kind = getattr(module, "_miles_model_backend", None)
    if kind is None:
        # Compatibility for train modules constructed outside this façade.
        kind = "moe" if hasattr(module, "save_state_dict_direct") else "standard"
    return _backend(kind)


def save_native(module, path):
    return _module_backend(module).save_native(module, path)


def load_native(module, path, *, optim=True):
    _module_backend(module).load_native(module, path, optim=optim)


def replay_context(module, batch, *, enabled):
    """Enter the selected trainer's routing context only when replay is enabled."""
    if not enabled:
        return nullcontext()
    return _module_backend(module).replay_context(module, batch)


def save_hf_config(hf, path):
    if hf.model_type == "olmo3":
        _backend("standard").save_hf_config(hf, path)
    else:
        hf.save_pretrained(path)
