# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/transformers_plugin/src/olmoe3_transformers_plugin/convert_checkpoint.py
# open-instruct change: --eos-token-ids (default <|im_end|>, <|endoftext|>) rewrites generation_config.json after
# both the text and the multimodal export, as scripts/train/debug/convert_moe_checkpoint_to_hf.py does.
"""Convert an OLMoE3 KDA + EMo OLMo Core checkpoint to Hugging Face format.

A multimodal checkpoint (``MultimodalLMConfig``: that language model plus an olmo-core vision
tower and connector) is exported as one whole-model directory; see :mod:`.multimodal`.
"""

from __future__ import annotations

import copy
import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp

from olmo_core.config import DType
from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader
from olmo_core.nn.hf import convert_checkpoint_to_hf, load_config
try:
    from olmo_core.nn.hf.convert_checkpoint import _normalize_legacy_latent_moe_config
except ImportError:  # olmo-core #885 moved it (same logic) to nn.transformer.config
    from olmo_core.nn.transformer.config import (
        normalize_legacy_config_dict as _normalize_legacy_latent_moe_config,
    )
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.utils import prepare_cli_environment

from .multimodal import is_multimodal_model_config

log = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--checkpoint-input-path", required=True)
    parser.add_argument("-o", "--huggingface-output-dir", required=True)
    parser.add_argument("-s", "--max-sequence-length", type=int)
    parser.add_argument(
        "-t",
        "--tokenizer",
        required=True,
        help="Explicit tokenizer used to create the training data; never inferred.",
    )
    parser.add_argument("--tokenizer-revision", help="Immutable HF tokenizer revision.")
    parser.add_argument("--skip-validation", dest="validate", action="store_false")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=torch.device)
    parser.add_argument("--validation-device", type=torch.device)
    parser.add_argument("--dtype", type=DType, default=DType.bfloat16)
    # Chat SFT exports end a turn with <|im_end|> (100265). The converter derives generation_config's
    # eos_token_id from the tokenizer config (<|endoftext|>, 100257, only), and vLLM then runs past every
    # <|im_end|> (H271: BFCL tool calls ran to the token budget). A base/midtraining export passes 100257.
    parser.add_argument(
        "--eos-token-ids",
        type=int,
        nargs="+",
        default=list(CHAT_EOS_TOKEN_IDS),
        help="generation_config.json eos_token_id list (default: <|im_end|>, <|endoftext|>)",
    )
    return parser.parse_args()


CHAT_EOS_TOKEN_IDS = (100265, 100257)  # <|im_end|>, <|endoftext|>


def write_generation_eos(output_dir: str | Path, eos_token_ids: list[int]) -> None:
    """Set generation_config.json's eos_token_id (the stop list vLLM and HF generate use)."""
    path = Path(output_dir) / "generation_config.json"
    generation_config = json.loads(path.read_text())
    generation_config["eos_token_id"] = list(eos_token_ids)
    path.write_text(json.dumps(generation_config, indent=2) + "\n")
    log.info(f"generation_config.json eos_token_id = {list(eos_token_ids)}")


def _use_fla_kda_kernels(value: Any) -> None:
    """Build KDA with FLA's kernels instead of kernel-fun's experimental ones.

    Kernel choice doesn't change the parameters. The exported HF model runs FLA, so this also
    makes validation compare like with like, and conversion doesn't need kernel-fun.
    """

    if isinstance(value, dict):
        if value.get("use_experimental_kernels"):
            log.info("Using FLA KDA kernels (not experimental kernels) for conversion")
            value["use_experimental_kernels"] = False
        for child in value.values():
            _use_fla_kda_kernels(child)
    elif isinstance(value, list):
        for child in value:
            _use_fla_kda_kernels(child)


def _use_torch_attention(value: Any) -> None:
    """Build full attention with the torch backend instead of the training one (e.g. FA4).

    The backend doesn't change the parameters, so conversion doesn't need FlashAttention
    installed. olmo-core only does this itself when validating.
    """

    if isinstance(value, dict):
        if str(value.get("_CLASS_", "")).endswith(".AttentionConfig") and value.get(
            "backend"
        ) not in (None, "torch"):
            log.info(f"Using torch attention (not {value['backend']}) for conversion")
            value["backend"] = "torch"
            if "use_flash" in value:
                value["use_flash"] = False
        for child in value.values():
            _use_torch_attention(child)
    elif isinstance(value, list):
        for child in value:
            _use_torch_attention(child)


def _resolve_main_param_key(name: str, checkpoint_keys: set[str]) -> str | None:
    """Match OLMoDDPTrainModule's resolution of a parameter's fp32 master copy."""

    for key in (f"module.{name}.main", f"{name}.main"):
        if key in checkpoint_keys:
            return key
    return None


def load_ddp_main_param_state_dict(
    checkpoint_input_path: str, model_config: dict[str, Any]
) -> dict[str, torch.Tensor] | None:
    """Rebuild the model state dict from an OLMoDDP optimizer-only checkpoint.

    OLMoDDP checkpoints store each parameter only as its optimizer's flattened fp32 master
    copy (``module.<param>.main``), with no ``model.*`` entries, which olmo-core's generic
    loader expects. Returns ``None`` for checkpoints that do carry model entries.
    """

    model_and_optim = Path(checkpoint_input_path) / "model_and_optim"
    # olmo-core's reader also handles metadata written by older torch versions.
    reader = RemoteFileSystemReader(str(model_and_optim))
    checkpoint_keys = set(reader.read_metadata().state_dict_metadata)
    if any(key.startswith("model.") for key in checkpoint_keys):
        return None
    if not any(key.endswith(".main") for key in checkpoint_keys):
        raise RuntimeError(f"{model_and_optim} has neither model.* nor .main entries")

    config = TransformerConfig.from_dict(copy.deepcopy(model_config))
    model = config.build(init_device="meta")
    model.to_empty(device=torch.device("cpu"))

    to_load: dict[str, torch.Tensor] = {}
    missing = []
    for name, param in model.named_parameters():
        key = _resolve_main_param_key(name, checkpoint_keys)
        if key is None:
            missing.append(name)
            continue
        # DCP loads in place, so the flattened view fills the parameter itself.
        to_load[key] = param.data.view(-1)
    if missing:
        raise RuntimeError(f"No .main master copy for parameters: {missing[:10]}")

    log.info(f"Loading {len(to_load)} fp32 master parameters from '{model_and_optim}'")
    dcp.load(to_load, storage_reader=reader)
    return {key: value.detach() for key, value in model.state_dict().items()}


def resolve_tokenizer_config(experiment_config: dict) -> dict | None:
    """Locate the tokenizer config, which sits in different places per workload.

    Single-source stages serialize ``dataset`` as one config with a ``tokenizer`` key. The
    merged MT+LC stage serializes it as a *list* of composable sources, each carrying its own
    copy (and its data loader carries one too). Require every copy to agree rather than
    picking one: a mismatched tokenizer diverges silently. Ported from the dense mainline
    converter (dd0202c8).
    """
    dataset = experiment_config.get("dataset")
    if isinstance(dataset, dict) and "tokenizer" in dataset:
        return dataset["tokenizer"]

    found: list[dict] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "tokenizer" and isinstance(value, dict):
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(dataset)
    if not found:
        walk(experiment_config.get("data_loader"))
    if not found:
        return None
    if any(tokenizer != found[0] for tokenizer in found[1:]):
        raise RuntimeError(
            f"Checkpoint config has {len(found)} disagreeing tokenizer configs; "
            "refusing to guess which one trained the model"
        )
    return found[0]


def convert_multimodal_checkpoint(args, experiment_config: dict[str, Any]) -> None:
    """Export a ``MultimodalLMConfig`` checkpoint as one whole-model HF directory.

    The language model is exported by :func:`convert_checkpoint_to_hf`, the text path, from
    the checkpoint's ``lm.*`` tensors and ``lm`` config, so ``language_model.*`` is byte for
    byte what a text-only export of that LM produces (and passes the text path's validation).
    The vision tower and connector tensors, the combined config and the wrapper code are then
    added around it, and the wrapper's text forward is validated against the olmo-core
    language model with the same logit check.
    """
    from olmo_core.data.tokenizer import TokenizerConfig
    from olmo_core.nn.hf.convert_checkpoint import validate_conversion

    from .multimodal import (
        check_multimodal_export_tensors,
        load_multimodal_state_dict,
        split_multimodal_state_dict,
        write_multimodal_export,
    )

    model_config = experiment_config["model"]
    tokenizer_config = resolve_tokenizer_config(experiment_config)
    if tokenizer_config is None:
        raise RuntimeError("Checkpoint config does not identify its tokenizer")
    _normalize_legacy_latent_moe_config(model_config)
    _use_fla_kda_kernels(model_config)
    _use_torch_attention(model_config)

    state_dict = load_multimodal_state_dict(args.checkpoint_input_path, model_config)
    lm_state, vision_state, connector_state = split_multimodal_state_dict(state_dict)
    del state_dict
    log.info(
        f"Multimodal checkpoint: {len(lm_state)} LM, {len(vision_state)} vision and "
        f"{len(connector_state)} connector tensors"
    )

    convert_checkpoint_to_hf(
        original_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=copy.deepcopy(model_config["lm"]),
        tokenizer_config_dict=copy.deepcopy(tokenizer_config),
        model_state_dict=lm_state,
        dtype=args.dtype,
        max_sequence_length=args.max_sequence_length,
        tokenizer_id=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        validate=args.validate,
        debug=args.debug,
        device=args.device,
        validation_device=args.validation_device or args.device,
    )

    write_multimodal_export(
        args.huggingface_output_dir,
        vision_state_dict=vision_state,
        connector_state_dict=connector_state,
        model_config=model_config,
        experiment_config=experiment_config,
        dtype=args.dtype,
    )
    check_multimodal_export_tensors(args.huggingface_output_dir)
    log.info(f"Wrote the multimodal export to '{args.huggingface_output_dir}'")

    if args.validate:
        from . import register

        register()
        log.info("Validating the exported wrapper's text forward against the olmo-core LM")
        model = TransformerConfig.from_dict(copy.deepcopy(model_config["lm"])).build(
            init_device="meta"
        )
        model.to_empty(device=args.device or torch.device("cpu"))
        model.load_state_dict(lm_state)
        validate_conversion(
            args.huggingface_output_dir,
            model,
            TokenizerConfig.from_dict(copy.deepcopy(tokenizer_config)).vocab_size,
            debug=False,
            dtype=args.dtype,
            device=args.validation_device or args.device,
        )
        log.info("Wrapper validation completed successfully")


def main() -> None:
    args = parse_args()
    experiment_config = load_config(args.checkpoint_input_path)
    if experiment_config is None:
        raise RuntimeError("Experiment config not found; cannot convert checkpoint")
    model_config = experiment_config.get("model")
    tokenizer_config = resolve_tokenizer_config(experiment_config)
    if model_config is None or tokenizer_config is None:
        raise RuntimeError("Checkpoint config must contain model and dataset.tokenizer")
    if is_multimodal_model_config(model_config):
        convert_multimodal_checkpoint(args, experiment_config)
        write_generation_eos(args.huggingface_output_dir, args.eos_token_ids)
        return
    _normalize_legacy_latent_moe_config(model_config)
    _use_fla_kda_kernels(model_config)
    _use_torch_attention(model_config)

    convert_checkpoint_to_hf(
        original_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=model_config,
        tokenizer_config_dict=tokenizer_config,
        model_state_dict=load_ddp_main_param_state_dict(
            args.checkpoint_input_path, model_config
        ),
        dtype=args.dtype,
        max_sequence_length=args.max_sequence_length,
        tokenizer_id=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        validate=args.validate,
        debug=args.debug,
        device=args.device,
        validation_device=args.validation_device or args.device,
    )
    write_generation_eos(args.huggingface_output_dir, args.eos_token_ids)


if __name__ == "__main__":
    prepare_cli_environment()
    main()
