"""Check an HF export's KDA weights against the OLMo-core checkpoint it was converted from, exactly.

Five tensors per KDA layer can be float32 in a training forward: `A_log`, `dt_bias` and the `q/k/v_conv1d` weights.
An export must store the value the forward used (see convert_moe_checkpoint_to_hf.py, `--kda-decay`). This reads
those tensors from the export's safetensors and from the checkpoint and compares every element:

    uv run python scripts/train/debug/check_kda_export.py \
        --checkpoint <ckpt>/step172 --hf-dir <ckpt>/hf_step172 --kda-decay fp32

- `--kda-decay fp32`: all five are float32 in the export and equal the checkpoint's float32 values.
- `--kda-decay bf16-grid`: `A_log` and `dt_bias` are float32, the convolution weights bfloat16, and all five equal
  the checkpoint's values rounded to the bfloat16 grid.

It is a mechanism check: it passes or fails on equality, prints one line per tensor that differs, writes the
result as JSON with `--output`, and exits 1 on any difference. It reads files only. It does not show what a
serving stack holds after loading the export.
"""

import argparse
import json
import pathlib
import re
import sys
import tempfile

import torch
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
from safetensors import safe_open

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

DECAY = ("A_log", "dt_bias")
CONVS = ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")
MODES = ("bf16-grid", "fp32")


def kda_layers(hf_dir: pathlib.Path) -> list[int]:
    """Indices of the export's KDA layers, from its config.json."""
    layer_types = json.loads((hf_dir / "config.json").read_text())["layer_types"]
    return [index for index, kind in enumerate(layer_types) if kind == "linear_attention"]


def checkpoint_kda_layers(checkpoint: pathlib.Path) -> list[int]:
    """Indices of the checkpoint's KDA layers: the blocks that store an ``attention.A_log``."""
    stored = get_checkpoint_metadata(str(checkpoint / "model_and_optim")).state_dict_metadata
    found = (re.search(r"(?:^|\.)blocks\.(\d+)\.attention\.A_log(?:\.main)?$", key) for key in stored)
    return sorted({int(match[1]) for match in found if match})


def read_export(hf_dir: pathlib.Path, layers: list[int]) -> dict[tuple[int, str], torch.Tensor]:
    """The five KDA tensors of each layer, as stored in the export's safetensors shards."""
    found: dict[tuple[int, str], torch.Tensor] = {}
    for shard in sorted(hf_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as tensors:
            for name in tensors.keys():  # noqa: SIM118 - safe_open is not a mapping
                for layer in layers:
                    for weight in (*DECAY, *CONVS):
                        if f".layers.{layer}." in name and name.endswith(f".{weight}"):
                            if (layer, weight) in found:
                                raise ValueError(f"layer {layer} has two tensors ending in {weight}")
                            found[layer, weight] = tensors.get_tensor(name)
    missing = sorted(set((layer, weight) for layer in layers for weight in (*DECAY, *CONVS)) - set(found))
    if missing:
        raise KeyError(f"KDA tensors missing from the export: {missing}")
    return found


def read_checkpoint(checkpoint: pathlib.Path, layers: list[int]) -> dict[tuple[int, str], torch.Tensor]:
    """The same tensors from an OLMo-core checkpoint, in either layout (model.<name> or OLMoDDP's <name>.main)."""
    model_and_optim = str(checkpoint / "model_and_optim")
    stored = set(get_checkpoint_metadata(model_and_optim).state_dict_metadata)
    wanted: dict[tuple[int, str], str] = {}
    for layer in layers:
        for weight in (*DECAY, *CONVS):
            name = f"blocks.{layer}.attention.{weight}"
            # OLMoDDP's float32 masters first, in every mode. The converter reads them first only for `fp32`; in the
            # other modes it reads `model.<name>` first, so where a checkpoint holds both and they disagree on the
            # bfloat16 grid, this check fails rather than follow whatever the converter picked.
            candidates = (
                f"{name}.main",
                f"module.{name}.main",
                f"model.{name}",
                f"model.module.{name}",
                name,
                f"module.{name}",
            )
            key = next((k for k in candidates if k in stored), None)
            if key is None:
                raise KeyError(f"checkpoint has no weight for {name}")
            wanted[layer, weight] = key
    with tempfile.TemporaryDirectory() as work_dir:
        loaded = list(load_keys(model_and_optim, list(wanted.values()), work_dir=work_dir))
    return dict(zip(wanted, loaded, strict=True))


def compare(
    exported: dict[tuple[int, str], torch.Tensor], source: dict[tuple[int, str], torch.Tensor], kda_decay: str
) -> list[dict]:
    """One record per tensor that breaks the rule for ``kda_decay``; empty when the export is right."""
    if kda_decay not in MODES:
        raise ValueError(f"kda_decay must be one of {MODES}, got {kda_decay!r}")
    failures = []
    for (layer, weight), tensor in sorted(exported.items()):
        stored = source[layer, weight]
        master = stored.to(torch.float32)
        want_dtype = torch.float32 if kda_decay == "fp32" or weight in DECAY else torch.bfloat16
        want = master if kda_decay == "fp32" else master.to(torch.bfloat16).to(torch.float32)
        got = tensor.to(torch.float32).reshape(-1)
        problem = None
        if tensor.dtype != want_dtype:
            problem = f"stored as {tensor.dtype}, expected {want_dtype}"
        elif got.numel() != want.numel():
            problem = f"{got.numel()} elements, the checkpoint has {want.numel()}"
        elif tensor.dim() != (1 if weight in DECAY else 3):
            # What the HF loader needs: a vector per decay weight, a Conv1d weight (channels, 1, kernel) per convolution.
            problem = f"shape {tuple(tensor.shape)} is not a {'vector' if weight in DECAY else 'Conv1d weight'}"
        elif stored.dim() == tensor.dim() and stored.shape != tensor.shape:
            # A checkpoint may store a tensor flattened (the converter reshapes it), so only like ranks are compared.
            problem = f"shape {tuple(tensor.shape)}, the checkpoint has {tuple(stored.shape)}"
        elif not torch.equal(got, want.reshape(-1)):
            differ = got != want.reshape(-1)
            problem = f"{int(differ.sum())} of {got.numel()} elements differ, max |diff| {float((got - want.reshape(-1)).abs().max()):.3e}"
        if problem:
            failures.append({"layer": layer, "weight": weight, "problem": problem})
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path, help="OLMo-core checkpoint step directory")
    parser.add_argument("--hf-dir", required=True, type=pathlib.Path, help="the HF export converted from it")
    parser.add_argument("--kda-decay", required=True, choices=MODES, help="the mode the export was written with")
    parser.add_argument("--output", type=pathlib.Path, help="write the result here as JSON")
    args = parser.parse_args()

    layers = kda_layers(args.hf_dir)
    expected = checkpoint_kda_layers(args.checkpoint)
    if not layers or layers != expected:
        # Nothing, or not everything, would be compared: an error, not a pass.
        raise SystemExit(
            f"KDA layers differ or are absent: the export's config.json lists {layers}, the checkpoint has {expected}"
        )
    failures = compare(read_export(args.hf_dir, layers), read_checkpoint(args.checkpoint, layers), args.kda_decay)
    result = {
        "checkpoint": str(args.checkpoint),
        "hf_dir": str(args.hf_dir),
        "kda_decay": args.kda_decay,
        "kda_layers": len(layers),
        "tensors": len(layers) * len((*DECAY, *CONVS)),
        "failures": failures,
        "ok": not failures,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    for failure in failures:
        logger.error("layer %(layer)d %(weight)s: %(problem)s", failure)
    logger.info(
        "KDA_EXPORT_%s: %d tensors in %d KDA layers",
        "OK" if not failures else "MISMATCH",
        result["tensors"],
        len(layers),
    )
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
