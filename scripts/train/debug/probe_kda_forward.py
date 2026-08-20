"""Isolate a single KDA layer to separate a kernel bug from memory pressure.

Training Jacob's OLMoE3 latent-KDA checkpoint dies with
`CUBLAS_STATUS_NOT_INITIALIZED` inside `olmo_core/nn/attention/kda.py`'s forward
on the first step. That error is ambiguous: cuBLAS allocates its handle and
workspace outside PyTorch's caching allocator, so it can mean either a genuine
kernel/version problem or an out-of-memory that never surfaces as a PyTorch OOM.

This builds ONE KimiDeltaAttention layer from the checkpoint's own config and
runs forward at increasing sequence lengths on a single GPU, using a few GB
rather than a few hundred. Read the result as:

* fails even at seq 128  -> kernel/version problem; memory is irrelevant
* succeeds small, fails long -> genuinely a size/memory limit
* succeeds throughout -> the layer is fine in isolation; the failure needs the
  full model's memory footprint, which supports the memory reading

    uv run python scripts/train/debug/probe_kda_forward.py \
        --config scripts/train/debug/kda_mt_sft.json
"""

import argparse
import json
import pathlib

import torch

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/train/debug/kda_mt_sft.json")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 2048, 8192])
    args = parser.parse_args()



    payload = json.loads(pathlib.Path(args.config).read_text())
    model_section = payload["model"]
    d_model = model_section["d_model"]
    mixer_config = model_section["block"]["sequence_mixer"]

    import importlib.metadata as metadata

    logger.info("torch=%s", torch.__version__)
    logger.info("fla=%s", metadata.version("flash-linear-attention"))
    logger.info("cuda=%s device=%s", torch.version.cuda, torch.cuda.get_device_name(0))

    from olmo_core.nn.attention.kda import KimiDeltaAttentionConfig

    config = KimiDeltaAttentionConfig.from_dict({k: v for k, v in mixer_config.items() if k != "type"})
    layer = config.build(
        d_model=d_model, layer_idx=0, n_layers=model_section["n_layers"], init_device="cuda"
    ).to("cuda")
    n_params = sum(p.numel() for p in layer.parameters())
    logger.info("built one KDA layer: %s params, d_model=%d", f"{n_params:,}", d_model)

    for seq_len in args.seq_lens:
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(1, seq_len, d_model, device="cuda", dtype=torch.bfloat16)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = layer(x)
            peak = torch.cuda.max_memory_allocated() / 2**30
            logger.info("seq %5d: OK    out=%s peak=%.2f GiB", seq_len, tuple(out.shape), peak)
        except Exception as exc:  # noqa: BLE001 - the failure mode is the result
            peak = torch.cuda.max_memory_allocated() / 2**30
            logger.info("seq %5d: FAIL  peak=%.2f GiB  %s: %s", seq_len, peak, type(exc).__name__, str(exc)[:200])
            break

    logger.info("PROBE_DONE")


if __name__ == "__main__":
    main()
