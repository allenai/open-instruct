#!/usr/bin/env python
"""Convert a text-only Qwen3.5 SFT checkpoint into a vLLM-loadable form.

Why this exists
---------------
Some Qwen3.5 SFT checkpoints (e.g. open-instruct saves) are written with
``architectures = ["Qwen3_5ForCausalLM"]`` and ``model_type = "qwen3_5_text"``
(a *flat* text config, weights under ``model.language_model.*`` + ``lm_head``).
vLLM defines a ``Qwen3_5ForCausalLM`` class but DOES NOT register it (checked
0.19.1 -> 0.22.0); only the multimodal ``Qwen3_5ForConditionalGeneration`` is
registered. So vLLM refuses to load such a checkpoint:

    Model architectures ['Qwen3_5ForCausalLM'] are not supported for now.

The official Qwen3.5 dense checkpoints (``Qwen/Qwen3.5-9B``,
``hamishivi/Qwen3.5-9B``) instead ship ``Qwen3_5ForConditionalGeneration`` /
``model_type = "qwen3_5"`` (a multimodal wrapper: nested ``text_config`` +
``vision_config``, weights ``model.language_model.*`` + ``model.visual.*`` +
``lm_head``). vLLM's weight mapper maps ``model.language_model.`` ->
``language_model.model.`` etc., so the text weights of the two layouts are
byte-for-byte interchangeable — the *only* things the SFT checkpoint is missing
are the vision-tower weights and the canonical (multimodal) config/processor.

This script produces a NEW checkpoint that vLLM loads via the registered CG
class, by combining:
  * text weights (``model.language_model.*`` + ``lm_head``) from --src
  * vision/mtp weights (``model.visual.*`` + ``mtp.*``) from --donor
  * canonical CG config + image/video processor configs from --donor
  * tokenizer + chat template from --src (the SFT tokenizer/template)
The vision tower is never exercised for text-only search/eval; it just has to be
present so vLLM's weight loader doesn't error on missing params.

The source checkpoint is treated as READ-ONLY; output goes to a separate dir.

Usage
-----
    python scripts/general_agent/utils/convert_qwen35_causallm_to_cg.py \
        --src /weka/.../drtulu_sft_qwen35_9b_v1_sanitized_full_reasoning \
        --donor hamishivi/Qwen3.5-9B \
        --out  /weka/.../drtulu_sft_qwen35_9b_v1_sanitized_full_reasoning_cg

--donor may be a local dir or an HF repo id (downloaded via snapshot_download).
The donor MUST be the same base architecture/size as --src (same text dims).
"""

import argparse
import glob
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

# Weights that belong to the text model (taken from --src). Everything else in
# the donor (model.visual.*, mtp.*) is kept as-is.
TEXT_PREFIXES = ("model.language_model.",)
TEXT_EXACT = ("lm_head.weight",)

# Config fields that must match between the src (flat) text config and the
# donor's nested text_config, or the donor is the wrong base model.
CRITICAL_DIMS = ("hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads")

# Files copied from the donor (canonical CG config + multimodal processor).
DONOR_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
)
# Files copied from the src (the SFT tokenizer + chat template).
SRC_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def is_text_key(key: str) -> bool:
    return key in TEXT_EXACT or any(key.startswith(p) for p in TEXT_PREFIXES)


def resolve_donor(donor: str) -> str:
    """Return a local directory for the donor (download if it's an HF repo id)."""
    if os.path.isdir(donor):
        return donor
    from huggingface_hub import snapshot_download

    print(f">>> Donor '{donor}' is not a local dir; downloading via HF hub...")
    return snapshot_download(donor)


def load_src_text_tensors(src_dir: str):
    """Open all src safetensors; return {key: safe_open_handle} for text keys."""
    files = sorted(glob.glob(os.path.join(src_dir, "*.safetensors")))
    if not files:
        sys.exit(f"!!! no *.safetensors found in {src_dir}")
    key_to_handle = {}
    handles = []
    for f in files:
        h = safe_open(f, framework="pt")
        handles.append(h)
        for k in h.keys():
            key_to_handle[k] = h
    return key_to_handle, handles


def check_configs(src_dir: str, donor_dir: str) -> None:
    src_cfg = json.load(open(os.path.join(src_dir, "config.json")))
    donor_cfg = json.load(open(os.path.join(donor_dir, "config.json")))
    donor_text = donor_cfg.get("text_config", donor_cfg)
    mismatches = []
    for k in CRITICAL_DIMS:
        s, d = src_cfg.get(k), donor_text.get(k)
        if s is not None and d is not None and s != d:
            mismatches.append((k, s, d))
    if mismatches:
        for k, s, d in mismatches:
            print(f"!!! config dim mismatch {k}: src={s} donor={d}")
        sys.exit("!!! donor is not the same base as src; aborting.")
    print("    config dims match between src and donor text_config:",
          {k: src_cfg.get(k) for k in CRITICAL_DIMS})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="Source Qwen3_5ForCausalLM / qwen3_5_text checkpoint dir (read-only).")
    ap.add_argument("--donor", default="hamishivi/Qwen3.5-9B",
                    help="Canonical CG donor: local dir or HF repo id (default: hamishivi/Qwen3.5-9B).")
    ap.add_argument("--out", required=True, help="Output dir for the converted checkpoint.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite --out if it already exists.")
    args = ap.parse_args()

    src = os.path.realpath(args.src)
    out = os.path.realpath(args.out)
    if out == src:
        sys.exit("!!! --out must differ from --src (never overwrite the original).")
    if os.path.exists(out):
        if not args.force:
            sys.exit(f"!!! --out exists: {out} (use --force to overwrite).")
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    donor = resolve_donor(args.donor)
    print(f">>> src   : {src}")
    print(f">>> donor : {donor}")
    print(f">>> out   : {out}")

    check_configs(src, donor)

    # Donor weight layout (sharded, with an index).
    donor_index_path = os.path.join(donor, "model.safetensors.index.json")
    if not os.path.exists(donor_index_path):
        sys.exit(f"!!! donor has no model.safetensors.index.json ({donor}); "
                 "expected a sharded canonical checkpoint.")
    weight_map = json.load(open(donor_index_path))["weight_map"]
    shards = sorted(set(weight_map.values()))

    # Source text tensors.
    src_text, _handles = load_src_text_tensors(src)
    print(f">>> src text tensors: {len(src_text)}")

    donor_text_keys = {k for k in weight_map if is_text_key(k)}
    missing_in_src = sorted(donor_text_keys - set(src_text))
    extra_in_src = sorted(set(src_text) - donor_text_keys)
    if missing_in_src:
        sys.exit(f"!!! src is missing {len(missing_in_src)} text tensors the donor "
                 f"expects, e.g. {missing_in_src[:5]}")
    if extra_in_src:
        print(f"    note: src has {len(extra_in_src)} tensors not in donor text set "
              f"(ignored), e.g. {extra_in_src[:5]}")

    # Merge shard-by-shard: src text weights win, donor vision/mtp weights kept.
    swapped = kept = 0
    for shard in shards:
        out_tensors = {}
        with safe_open(os.path.join(donor, shard), framework="pt") as st:
            for k in st.keys():
                if k in src_text:
                    ours = src_text[k].get_tensor(k)
                    dshape = tuple(st.get_slice(k).get_shape())
                    if tuple(ours.shape) != dshape:
                        sys.exit(f"!!! shape mismatch for {k}: src={tuple(ours.shape)} donor={dshape}")
                    out_tensors[k] = ours.contiguous()
                    swapped += 1
                else:
                    out_tensors[k] = st.get_tensor(k).contiguous()
                    kept += 1
        save_file(out_tensors, os.path.join(out, shard), metadata={"format": "pt"})
        print(f"    wrote {shard}: {len(out_tensors)} tensors")
        del out_tensors

    if swapped != len(donor_text_keys):
        sys.exit(f"!!! expected to swap {len(donor_text_keys)} text tensors, swapped {swapped}")
    print(f">>> swapped(src text)={swapped}  kept(donor vision/mtp)={kept}")

    # Index + canonical CG config/processor from donor; tokenizer/template from src.
    shutil.copy(donor_index_path, os.path.join(out, "model.safetensors.index.json"))
    for f in DONOR_FILES:
        p = os.path.join(donor, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out, f))
            print(f"    copied donor:{f}")
    for f in SRC_FILES:
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out, f))
            print(f"    copied src:{f}")

    # Sanity: the output config must declare the registered CG architecture.
    out_cfg = json.load(open(os.path.join(out, "config.json")))
    arch = out_cfg.get("architectures")
    print(f">>> output architectures: {arch}  model_type: {out_cfg.get('model_type')}")
    if arch != ["Qwen3_5ForConditionalGeneration"]:
        print("!!! WARNING: output architecture is not Qwen3_5ForConditionalGeneration; "
              "vLLM may not load it. Check the donor.")
    print(f">>> DONE. Converted checkpoint at: {out}")
    print("    Serve with a CLEAN --served-model-name (a path containing 'ada', e.g.")
    print("    'oe-adapt', trips the client's commercial-API detector).")


if __name__ == "__main__":
    main()
