"""Convert an olmo-core MoE v2 checkpoint (KDA / OLMoE3) to HuggingFace format.

Thin wrapper around olmo-core's own converter, mirroring
`ladders/olmoe3/transformers_plugin/convert_checkpoint.py` in scaling-ladders.
Requires olmo-core on `akshitab/emo_modularity` or later: earlier revisions raise
NotImplementedError for peri-LN checkpoints, which these are.

The model config comes from a file rather than the checkpoint's own config.json,
because the training config is the one whose blocks match the saved weights (see
scripts/train/debug/make_kda_sft_config.py).

    uv run python scripts/train/debug/convert_moe_checkpoint_to_hf.py \
        -i <ckpt>/step172 -o <ckpt>/hf_step172 \
        -c scripts/train/debug/kda_mt_sft.json

Validation caveat: strict logit validation FAILS (~92-99% mismatch, abs diff
~1.5) on every OLMoE3 latent-KDA 1.2B checkpoint, including Jacob's known-good
midtrain -- the two forward implementations disagree at this scale regardless
of checkpoint, config, or the OLMO_HF_* parity env vars (bisected 2026-08-20).
The scaling-ladders production workload passes --skip-validation --device cpu;
strict validation only ever passed on the 275M/480M ladder models. Use
--skip-validation for KDA and validate functionally with an eval instead
(CPU and CUDA conversions were verified bit-identical over the full 35 GB).

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
from torch.distributed.checkpoint import state_dict as dist_cp_sd

from open_instruct import export_chat_template, logger_utils

logger = logger_utils.setup_logger(__name__)


def load_ddp_main_params(model_and_optim_dir: str, model_config: dict, work_dir: str) -> dict | None:
    """Gather model weights from an OLMoDDP checkpoint, or return None for a conventional one.

    ``OLMoDDPTrainModule`` stores the authoritative weights as the fused optimizer's fp32
    ``module.<parameter>.main`` tensors, not under ``model.<parameter>``. olmo-core on
    ``akshitab/emo_modularity`` (f2cf93839) taught ``convert_checkpoint_to_hf`` that layout
    (``_load_ddp_optimizer_model_state``); the Olmo 3.5 hero lineage (89e7dcb7) never got it,
    so every hero SFT conversion died with "Missing key in checkpoint state_dict:
    model.embeddings.weight" (01M2N8CRTAZHXD6605MSY9615M, 2026-09-16). This is that function,
    ported: the same key candidates, the converter then takes the gathered state dict.
    """
    checkpoint_keys = set(get_checkpoint_metadata(model_and_optim_dir).state_dict_metadata)
    if not any(key.endswith(".main") for key in checkpoint_keys):
        return None
    logger.info("OLMoDDP checkpoint layout (<param>.main); gathering weights through the fused optimizer keys")
    model = TransformerConfig.from_dict(model_config).build(init_device="meta")
    model = model.to_empty(device="cpu")
    keys_to_load: list[str] = []
    destinations: list[tuple[str, torch.Tensor]] = []
    missing: list[str] = []
    for name, param in model.named_parameters():
        candidates = (
            f"model.{name}",
            f"model.module.{name}",
            name,
            f"module.{name}",
            f"{name}.main",
            f"module.{name}.main",
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
            key = f"model_buffer.{name}"
            if key in checkpoint_keys:
                value = next(load_keys(model_and_optim_dir, [key], work_dir=work_dir))
                buffer.copy_(value.reshape(buffer.shape).to(buffer.dtype))
    options = dist_cp_sd.StateDictOptions(full_state_dict=True, cpu_offload=True)
    return dist_cp_sd.get_model_state_dict(model, options=options)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--checkpoint-input-path", required=True)
    parser.add_argument("-o", "--huggingface-output-dir", required=True)
    parser.add_argument("-c", "--config", required=True, help="Training config json (model + dataset.tokenizer).")
    parser.add_argument("-t", "--tokenizer", default="allenai/olmo-3-tokenizer-instruct-dev")
    parser.add_argument(
        "--export-chat-template",
        type=pathlib.Path,
        help="Jinja file to install after export; requires --tokenizer to be a saved training tokenizer directory.",
    )
    parser.add_argument("-s", "--max-sequence-length", type=int, default=8192)
    parser.add_argument("--skip-validation", dest="validate", action="store_false")
    # Defaults to CUDA, not CPU: validation runs both implementations, and the
    # KDA/fla Triton kernels reject CPU tensors ("Pointer argument (at 0) cannot be
    # accessed from Triton"). On CPU the weights still get written, so the failure
    # looks like a converted-but-unvalidated checkpoint -- the exact state that let
    # a 38%-logit-error conversion through on Olmo-Hybrid-7B. Do not silence it.
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    args = parser.parse_args()
    template = export_chat_template.read_export_chat_template(args.export_chat_template)
    reference_tokenizer = None
    if template is not None:
        tokenizer_file = pathlib.Path(args.tokenizer) / "tokenizer.json"
        if not tokenizer_file.is_file():
            parser.error(
                "--export-chat-template requires --tokenizer to name a saved directory containing tokenizer.json"
            )
        # Snapshot before conversion: OLMo-core may prefer a checkpoint-local
        # tokenizer over tokenizer_id, or Transformers may rebuild its backend.
        reference_tokenizer = json.loads(tokenizer_file.read_text(encoding="utf-8"))
        if not isinstance(reference_tokenizer, dict):
            parser.error(f"Expected a JSON object in {tokenizer_file}")

    payload = json.loads(pathlib.Path(args.config).read_text())
    model_config = payload["model"]
    tokenizer_config = payload.get("dataset", {}).get("tokenizer")
    if tokenizer_config is None:
        raise SystemExit(f"{args.config} has no dataset.tokenizer section")

    logger.info("converting %s -> %s", args.checkpoint_input_path, args.huggingface_output_dir)
    with tempfile.TemporaryDirectory() as work_dir:
        model_state_dict = load_ddp_main_params(
            str(pathlib.Path(args.checkpoint_input_path) / "model_and_optim"), model_config, work_dir
        )
    convert_checkpoint_to_hf(
        original_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=model_config,
        tokenizer_config_dict=tokenizer_config,
        model_state_dict=model_state_dict,
        dtype=DType.bfloat16,
        max_sequence_length=args.max_sequence_length,
        tokenizer_id=args.tokenizer,
        validate=args.validate,
        device=args.device,
        validation_device=args.device,
    )
    if reference_tokenizer is not None:
        exported_tokenizer = json.loads(
            (pathlib.Path(args.huggingface_output_dir) / "tokenizer.json").read_text(encoding="utf-8")
        )
        if exported_tokenizer != reference_tokenizer:
            raise RuntimeError(
                "Exported tokenizer.json differs from the saved training tokenizer. "
                "Check OLMo-core's checkpoint-local tokenizer precedence and Transformers serialization. "
                "The export is not qualified for RL; the chat template was not installed."
            )
    export_chat_template.install_export_chat_template(args.huggingface_output_dir, template)
    logger.info("CONVERSION_OK")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
