"""Convert an olmo-core MoE v2 checkpoint (KDA / OLMoE3) to HuggingFace format.

Thin wrapper around olmo-core's own converter, mirroring
`ladders/olmoe3/transformers_plugin/convert_checkpoint.py` in scaling-ladders.
Requires olmo-core on `akshitab/emo_modularity` or later: earlier revisions raise
NotImplementedError for peri-LN checkpoints, which these are.

The model config comes from a file rather than the checkpoint's own config.json,
because the training config is the one whose blocks match the saved weights. Pass
the olmo-core training config JSON the run was launched with (it needs the `model`
and `dataset.tokenizer` sections); none ships in this repo.

    uv run python scripts/train/debug/convert_moe_checkpoint_to_hf.py \
        -i <ckpt>/step172 -o <ckpt>/hf_step172 \
        -c <run's training config>.json

For KDA checkpoints add `--skip-validation --device cpu` (see the caveat below).
Input and output must be local or WEKA paths: `-i` is joined with pathlib and the
EOS step writes to a local directory, so remote (`gs://`) paths are not supported.

Validation caveat: strict logit validation FAILS (~92-99% mismatch, abs diff
~1.5) on every OLMoE3 latent-KDA 1.2B checkpoint, including a known-good
midtrain checkpoint -- the two forward implementations disagree at this scale regardless
of checkpoint, config, or the OLMO_HF_* parity env vars (bisected 2026-08-20).
The scaling-ladders production workload passes --skip-validation --device cpu;
strict validation only ever passed on the 275M/480M ladder models. Use
--skip-validation for KDA and validate functionally with an eval instead
(CPU and CUDA conversions were verified bit-identical over the full 35 GB).

generation_config.json gets eos_token_id [100265, 100257] (<|im_end|>, <|endoftext|>) by default: the
chat stop list for the default Olmo 3 chat tokenizer, as chat SFT exports need. A tokenizer with other ids, or a
base/midtrain export that should stop only at <|endoftext|>, passes --eos-token-ids explicitly (e.g. 100257).

Multimodal (Olmo 3.5 vision) SFT checkpoints: the language model alone is exported (hf_text, for text evals).
When the config's `model` is a MultimodalLMConfig (`lm` / `vision` / `connector`), the `lm` config is converted from
the checkpoint's `lm.*` weights (`module.lm.<p>.main`, `frozen_model.lm.<p>`, `model_buffer.lm.<b>`). Two inputs:

    # an OLMo-core DCP step dir (GPU run, or a TPU run converted to OLMo-core); -c is the step's own config.json
    python scripts/train/debug/convert_moe_checkpoint_to_hf.py -i <run>/step10962 -c <run>/step10962/config.json \
        -o <out>/hf_text -t <promoted olmo35 tokenizer dir>
    # a consolidated export (model.safetensors + olmo_core_config.json, OLMo-core's export_multimodal_olmoddp.py)
    python scripts/train/debug/convert_moe_checkpoint_to_hf.py --consolidated-export <out>/olmo-core-export \
        -o <out>/hf_text -t <promoted olmo35 tokenizer dir>

For these, -s defaults to the run's train_module.max_sequence_length, the training-only kernel/attention choices
(kernel-fun KDA, flash attention) are swapped for FLA / torch (no parameter changes), and the tokenizer config is
found wherever the run serialized it. Text-only checkpoints behave exactly as before.
See scripts/export/olmo35_mm/README.md for the whole multimodal export.

Note for GDN-based MoE hybrids (not KDA): olmo-core's `is_olmo_hybrid_model`
returns True for them and routes to the dense-hybrid exporter, which crashes on
the missing `feed_forward` attribute. KDA is unaffected because
`KimiDeltaAttention` is not a `GatedDeltaNet` subclass.
"""

import argparse
import json
import pathlib
import tempfile

import torch
from olmo_core.config import DType
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
from olmo_core.nn.hf import convert_checkpoint_to_hf
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.utils import prepare_cli_environment
from safetensors.torch import load_file
from torch.distributed.checkpoint import state_dict as dist_cp_sd

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def load_ddp_main_params(
    model_and_optim_dir: str, model_config: dict, work_dir: str, key_prefix: str = ""
) -> dict | None:
    """Gather model weights from an OLMoDDP checkpoint, or return None for a conventional one.

    ``OLMoDDPTrainModule`` stores the authoritative weights as the fused optimizer's fp32
    ``module.<parameter>.main`` tensors, not under ``model.<parameter>``. olmo-core on
    ``akshitab/emo_modularity`` (f2cf93839) taught ``convert_checkpoint_to_hf`` that layout
    (``_load_ddp_optimizer_model_state``); the Olmo 3.5 hero lineage (89e7dcb7) never got it,
    so every hero SFT conversion died with "Missing key in checkpoint state_dict:
    model.embeddings.weight" (2026-09-16). This is that function,
    ported: the same key candidates, the converter then takes the gathered state dict.

    ``key_prefix`` ("lm." for a multimodal checkpoint) selects one submodule's weights: ``model_config`` is then
    that submodule's config, and frozen parameters (``frozen_model.<prefix><name>``) are accepted too. With a prefix
    the weights are always gathered here, whatever the layout (OLMoDDP masters or ``model.*``).
    """
    checkpoint_keys = set(get_checkpoint_metadata(model_and_optim_dir).state_dict_metadata)
    # A prefixed (multimodal) load always gathers: the converter's own loader knows no ``lm.*`` names, and a TPU
    # run converted to OLMo-core stores ``model.lm.<name>`` rather than OLMoDDP masters.
    if not key_prefix and not any(key.endswith(".main") for key in checkpoint_keys):
        return None
    logger.info("OLMoDDP checkpoint layout (<param>.main); gathering weights through the fused optimizer keys")
    model = TransformerConfig.from_dict(model_config).build(init_device="meta")
    model = model.to_empty(device="cpu")
    keys_to_load: list[str] = []
    destinations: list[tuple[str, torch.Tensor]] = []
    missing: list[str] = []
    for name, param in model.named_parameters():
        full_name = f"{key_prefix}{name}"
        candidates = (
            f"model.{full_name}",
            f"model.module.{full_name}",
            full_name,
            f"module.{full_name}",
            f"{full_name}.main",
            f"module.{full_name}.main",
            f"frozen_model.{full_name}",
        )
        key = next((k for k in candidates if k in checkpoint_keys), None)
        if key is None:
            missing.append(name)
            continue
        keys_to_load.append(key)
        destinations.append((name, param))
    if missing:
        raise RuntimeError("DDP checkpoint is missing weights for model parameters: " + ", ".join(missing))
    loaded = load_keys(model_and_optim_dir, keys_to_load, work_dir=work_dir)
    with torch.no_grad():
        for key, (name, destination), value in zip(keys_to_load, destinations, loaded, strict=True):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Checkpoint value '{key}' is not a tensor")
            if value.numel() != destination.numel():
                raise RuntimeError(
                    f"Checkpoint value '{key}' has {value.numel()} elements, model parameter '{name}' has {destination.numel()}"
                )
            destination.copy_(value.reshape(destination.shape).to(destination.dtype))
        for name, buffer in model.named_buffers():
            buffer_candidates = (f"model_buffer.{key_prefix}{name}", f"model.{key_prefix}{name}")
            key = next((k for k in buffer_candidates if k in checkpoint_keys), None)
            if key is not None:
                value = next(load_keys(model_and_optim_dir, [key], work_dir=work_dir))
                buffer.copy_(value.reshape(buffer.shape).to(buffer.dtype))
    options = dist_cp_sd.StateDictOptions(full_state_dict=True, cpu_offload=True)
    return dist_cp_sd.get_model_state_dict(model, options=options)


CHAT_EOS_TOKEN_IDS = (100265, 100257)  # <|im_end|>, <|endoftext|>
MULTIMODAL_SUBMODULES = ("lm", "vision", "connector")


def is_multimodal_model_config(model_config: dict) -> bool:
    """Whether a run config's ``model`` is an OLMo-core ``MultimodalLMConfig`` (LM + vision tower + connector)."""
    if str(model_config.get("_CLASS_", "")).rsplit(".", 1)[-1] == "MultimodalLMConfig":
        return True
    return all(isinstance(model_config.get(name), dict) for name in MULTIMODAL_SUBMODULES)


def resolve_tokenizer_config(experiment_config: dict) -> dict | None:
    """``dataset.tokenizer``, or the one tokenizer config every data source agrees on.

    From allenai/scaling-ladders@59534d7b ladders/olmoe3/transformers_plugin/.../convert_checkpoint.py: multi-source
    runs serialize a copy per source (and in the data loader); disagreeing copies are an error, never a guess.
    """
    dataset = experiment_config.get("dataset")
    if isinstance(dataset, dict) and isinstance(dataset.get("tokenizer"), dict):
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
            f"{len(found)} disagreeing tokenizer configs; refusing to guess which one trained the model"
        )
    return found[0]


def use_inference_kernels(value) -> None:
    """Build KDA with FLA (not kernel-fun) and attention with torch (not e.g. flash_4), in place.

    Neither choice holds parameters, so the conversion needs no training-only kernels and validation compares the
    same kernels the exported HF model runs (scaling-ladders' _use_fla_kda_kernels / _use_torch_attention).
    """
    if isinstance(value, dict):
        if value.get("use_experimental_kernels"):
            value["use_experimental_kernels"] = False
        if str(value.get("_CLASS_", "")).endswith(".AttentionConfig") and value.get("backend") not in (None, "torch"):
            value["backend"] = "torch"
            if "use_flash" in value:
                value["use_flash"] = False
        for child in value.values():
            use_inference_kernels(child)
    elif isinstance(value, list):
        for child in value:
            use_inference_kernels(child)


def language_model_from_consolidated_export(export_dir: pathlib.Path) -> tuple[dict, dict]:
    """(run config, LM state dict) of a consolidated multimodal export: ``lm.*`` tensors, prefix stripped.

    The export is OLMo-core's ``export_multimodal_olmoddp.py`` layout: ``model.safetensors`` (the ``MultimodalLM``
    state dict) next to ``olmo_core_config.json`` (the run's config.json).
    """
    payload = json.loads((export_dir / "olmo_core_config.json").read_text())
    state = load_file(str(export_dir / "model.safetensors"))
    unexpected = sorted(k for k in state if not k.startswith(tuple(f"{m}." for m in MULTIMODAL_SUBMODULES)))
    if unexpected:
        raise RuntimeError(f"{export_dir} is not a multimodal export; unexpected tensors {unexpected[:5]}")
    lm_state = {name[len("lm.") :]: tensor for name, tensor in state.items() if name.startswith("lm.")}
    if not lm_state:
        raise RuntimeError(f"{export_dir}/model.safetensors holds no lm.* tensors")
    logger.info("consolidated export: %d of %d tensors are the language model", len(lm_state), len(state))
    return payload, lm_state


def write_generation_eos(output_dir: pathlib.Path, eos_token_ids: list[int]) -> None:
    """Set generation_config.json's eos_token_id to the chat stop list; the converter derives it from config.json."""
    path = output_dir / "generation_config.json"
    generation_config = json.loads(path.read_text())
    generation_config["eos_token_id"] = list(eos_token_ids)
    path.write_text(json.dumps(generation_config, indent=2) + "\n")
    logger.info("generation_config.json eos_token_id = %s", eos_token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-i", "--checkpoint-input-path", help="OLMo-core step dir (model_and_optim/ + config).")
    source.add_argument(
        "--consolidated-export",
        type=pathlib.Path,
        help="Multimodal export dir (model.safetensors + olmo_core_config.json); exports its language model.",
    )
    parser.add_argument("-o", "--huggingface-output-dir", required=True)
    parser.add_argument("-c", "--config", help="Training config json (model + tokenizer). Required with -i.")
    parser.add_argument("-t", "--tokenizer", default="allenai/olmo-3-tokenizer-instruct-dev")
    parser.add_argument(
        "-s",
        "--max-sequence-length",
        type=int,
        default=None,
        help="Default 8192; for a multimodal run, its train_module.max_sequence_length.",
    )
    parser.add_argument("--skip-validation", dest="validate", action="store_false")
    # The chat models end a turn with <|im_end|> (100265), not <|endoftext|> (100257). Without it in
    # generation_config.json, servers that take stop tokens from there (vLLM's OpenAI server) never stop the turn:
    # an A4B SFT export without it repeated its tool call up to the token budget on BFCL.
    # Delete this override once the olmo-core pin's save_hf_model writes the chat stop list itself.
    parser.add_argument(
        "--eos-token-ids",
        type=int,
        nargs="+",
        default=list(CHAT_EOS_TOKEN_IDS),
        help="generation_config.json eos_token_id list (default: <|im_end|>, <|endoftext|>)",
    )
    # Defaults to CUDA, not CPU: validation runs both implementations, and the
    # KDA/fla Triton kernels reject CPU tensors ("Pointer argument (at 0) cannot be
    # accessed from Triton"). On CPU the weights still get written, so the failure
    # looks like a converted-but-unvalidated checkpoint -- the exact state that let
    # a 38%-logit-error conversion through on Olmo-Hybrid-7B. Do not silence it.
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    args = parser.parse_args()

    if args.consolidated_export is not None:
        if args.config is not None:
            parser.error("--consolidated-export carries its own olmo_core_config.json; drop -c")
        payload, model_state_dict = language_model_from_consolidated_export(args.consolidated_export)
        if not is_multimodal_model_config(payload["model"]):
            raise SystemExit(f"{args.consolidated_export} is not a multimodal (MultimodalLMConfig) export")
    else:
        if args.config is None:
            parser.error("-c/--config is required with -i")
        payload = json.loads(pathlib.Path(args.config).read_text())
        model_state_dict = None
    model_config = payload["model"]
    multimodal = is_multimodal_model_config(model_config)
    if multimodal:
        model_config = model_config["lm"]
        use_inference_kernels(model_config)
        tokenizer_config = resolve_tokenizer_config(payload)
        max_sequence_length = args.max_sequence_length or payload.get("train_module", {}).get("max_sequence_length")
        logger.info("multimodal run config: exporting its language model (lm.*) only")
    else:
        tokenizer_config = payload.get("dataset", {}).get("tokenizer")
        max_sequence_length = args.max_sequence_length
    if tokenizer_config is None:
        raise SystemExit(f"{args.config or args.consolidated_export} has no tokenizer config")
    max_sequence_length = max_sequence_length or 8192

    source = args.checkpoint_input_path or str(args.consolidated_export)
    logger.info("converting %s -> %s", source, args.huggingface_output_dir)
    if model_state_dict is None:
        # load_keys is a generator, but load_ddp_main_params consumes it fully inside this block and returns
        # materialized CPU tensors, so work_dir can go before the conversion. Keep the consumption inside the block.
        with tempfile.TemporaryDirectory() as work_dir:
            model_state_dict = load_ddp_main_params(
                str(pathlib.Path(args.checkpoint_input_path) / "model_and_optim"),
                model_config,
                work_dir,
                key_prefix="lm." if multimodal else "",
            )
    convert_checkpoint_to_hf(
        original_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=model_config,
        tokenizer_config_dict=tokenizer_config,
        model_state_dict=model_state_dict,
        dtype=DType.bfloat16,
        max_sequence_length=max_sequence_length,
        tokenizer_id=args.tokenizer,
        validate=args.validate,
        device=args.device,
        validation_device=args.device,
    )
    write_generation_eos(pathlib.Path(args.huggingface_output_dir), args.eos_token_ids)
    logger.info("CONVERSION_OK")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
