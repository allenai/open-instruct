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

Note for GDN-based MoE hybrids (not KDA): olmo-core's `is_olmo_hybrid_model`
returns True for them and routes to the dense-hybrid exporter, which crashes on
the missing `feed_forward` attribute. KDA is unaffected because
`KimiDeltaAttention` is not a `GatedDeltaNet` subclass.
"""

import argparse
import json
import pathlib

import torch
from olmo_core.config import DType
from olmo_core.nn.hf import convert_checkpoint_to_hf
from olmo_core.utils import prepare_cli_environment

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--checkpoint-input-path", required=True)
    parser.add_argument("-o", "--huggingface-output-dir", required=True)
    parser.add_argument("-c", "--config", required=True, help="Training config json (model + dataset.tokenizer).")
    parser.add_argument("-t", "--tokenizer", default="allenai/olmo-3-tokenizer-instruct-dev")
    parser.add_argument("-s", "--max-sequence-length", type=int, default=8192)
    parser.add_argument("--skip-validation", dest="validate", action="store_false")
    # Defaults to CUDA, not CPU: validation runs both implementations, and the
    # KDA/fla Triton kernels reject CPU tensors ("Pointer argument (at 0) cannot be
    # accessed from Triton"). On CPU the weights still get written, so the failure
    # looks like a converted-but-unvalidated checkpoint -- the exact state that let
    # a 38%-logit-error conversion through on Olmo-Hybrid-7B. Do not silence it.
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    args = parser.parse_args()

    payload = json.loads(pathlib.Path(args.config).read_text())
    model_config = payload["model"]
    tokenizer_config = payload.get("dataset", {}).get("tokenizer")
    if tokenizer_config is None:
        raise SystemExit(f"{args.config} has no dataset.tokenizer section")

    logger.info("converting %s -> %s", args.checkpoint_input_path, args.huggingface_output_dir)
    convert_checkpoint_to_hf(
        original_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=model_config,
        tokenizer_config_dict=tokenizer_config,
        dtype=DType.bfloat16,
        max_sequence_length=args.max_sequence_length,
        tokenizer_id=args.tokenizer,
        validate=args.validate,
        device=args.device,
        validation_device=args.device,
    )
    logger.info("CONVERSION_OK")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
