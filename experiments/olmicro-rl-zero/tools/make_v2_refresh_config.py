"""Generate the v2 refresh RL config for an Olmicro long-context final (--size, default 2.7b) from the v1 Arm B config by asserted edits.

Changes vs v1 B (approved 2026-11-18): long-context (stage3) checkpoint; long-context base-model filtered data (--data);
publication_mode refresh with max_weight_staleness 12 (Michael's maintained refresh templates); response cap --cap
(32k default); persistent Triton compiler cache under oe-adapt-default (tmp-30d); 2 waves of producer samples; 128 running requests per engine; SGLang memory fraction 0.6 so the KV
pool fits 128 x context; min_runtime 8h; core.engine_drain_timeout 1800 s and the FLA config cache env (added after
the 32k/8k/16k launches, so configs generated now differ from those three); --size picks the model, Core config and a
default trainer/inference split (2+6 for 275m/450m/810m, 4+4 for 1.4b/2.7b); 512 in-flight samples at every size (the
2.7B 4+4 value), so off-policy lag per update matches across sizes.

Usage: python make_v2_refresh_config.py RUNS_DIR --data /path/to/filtered/dataset [--size 1.4b] [--cap 32768]
"""

import argparse
from pathlib import Path

V1B = "olmicro-2.7b-math-rl-v1b-passband-20261118"
M = "/weka/oe-adapt-default/jacobm/olmicro/models/olmicro-2.7b-base"
MODELS = "/weka/oe-adapt-default/jacobm/olmicro/models"
# Long-context (stage3) final per size, and the default trainer + inference split of one 8-GPU node.
SIZES = {
    "275m": ("stage3-step0047684", 2, 6),
    "450m": ("stage3-step0047684", 2, 6),
    "810m": ("stage3-step0023842", 2, 6),
    "1.4b": ("stage3-step0023842", 4, 4),
    "2.7b": ("stage3-step0023842", 4, 4),
}
IN_FLIGHT = 512
V1B_DATA = "/weka/oe-adapt-default/jacobm/olmicro/data/dolci-math-base-r1-v1-passband-base16"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=32768)
    parser.add_argument("--name")
    parser.add_argument("--size", choices=sorted(SIZES), default="2.7b")
    parser.add_argument("--trainer-gpus", type=int)
    parser.add_argument("--inference-gpus", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout instead of writing")
    args = parser.parse_args()
    revision, default_trainer, default_inference = SIZES[args.size]
    args.trainer_gpus = args.trainer_gpus or default_trainer
    args.inference_gpus = args.inference_gpus or default_inference
    model = f"{MODELS}/olmicro-{args.size}-base"
    if args.trainer_gpus + args.inference_gpus != 8:
        raise SystemExit("trainer + inference GPUs must equal the 8-GPU node")
    if not (args.data / "train.jsonl").is_file():
        raise SystemExit(f"missing {args.data}/train.jsonl")
    name = args.name or (f"olmicro-2.7b-lc-math-rl-v2-refresh-{args.cap // 1024}k-20261118" if args.size == "2.7b"
                         else f"olmicro-{args.size}-lc-math-rl-refresh-{args.cap // 1024}k")
    context = args.cap + 512
    packing = max(16384, -(-context // 8192) * 8192)  # smallest multiple of 8,192 covering the context (40,960 at 32k, as qualified)
    source = (args.runs / f"{V1B}.toml").read_text()
    body = source[source.index("schema_version") :]
    edits = [
        (V1B, name),
        (V1B_DATA, str(args.data)),
        ("# allenai/olmicro-2.7b-base@stage2-step0011921 (midtrain final), SHA256SUMS verified 2026-11-18.",
         f"# allenai/olmicro-{args.size}-base@{revision} (long-context final), SHA256SUMS verified 2026-11-18."),
        (f'source = "{M}/stage2-step0011921"', f'source = "{model}/{revision}"'),
        ("# allenai/olmicro-2.7b-base HF bucket stage2/checkpoints/step0011921/config.json, SHA256SUMS verified.",
         f"# allenai/olmicro-{args.size}-base HF bucket stage3/checkpoints/{revision.split('-')[1]}/config.json, SHA256SUMS verified."),
        (f'model_config = "{M}/olmo-core/stage2-step0011921/config.json"', f'model_config = "{model}/olmo-core/{revision}/config.json"'),
        ('min_runtime = "2h"', 'min_runtime = "8h"'),
        ("packing_max_tokens = 16384", f"packing_max_tokens = {packing}"),
        ("max_response_length = 7680\nmax_context_length = 8192", f"max_response_length = {args.cap}\nmax_context_length = {context}"),
        ("eval_max_response_len = 7680", f"eval_max_response_len = {args.cap}"),
        ('publication_mode = "barrier"',
         'publication_mode = "refresh"\n'
         "# Before each eval, refresh drains in-flight generation; the default 180 s deadline failed the 32k run's\n"
         "# update-100 eval at ~10k-token responses (2026-11-18). Allow long responses to finish.\n"
         "engine_drain_timeout = 1800.0"),
        ("checkpoint_keep_last = 3",
         "checkpoint_keep_last = 3\n# Keep every eval-interval checkpoint for later broader evals (each ~31 GB at 2.7B).\ncheckpoint_keep_every = 50"),
        ('MILES_HYBRID_REFERENCE_BACKBONE = "0"\n',
         'MILES_HYBRID_REFERENCE_BACKBONE = "0"\n'
         "# FLA config cache (runs/warmup-fix-notes-20261118.md): fixed causal_conv1d/l2norm configs instead of re-autotuning\n"
         "# for every pack length. Validated 2026-11-18 (identical loss, cold packs 355 s -> 63 s); print flag shows what still tunes.\n"
         'FLA_CACHE_MODE = "default"\n'
         'FLA_CONFIG_DIR = "/weka/oe-adapt-default/jacobm/olmicro/fla-configs/b300-fla0.5.2-v1"\n'
         'TRITON_PRINT_AUTOTUNING = "1"\n'),
        ("max_weight_staleness = 1", "max_weight_staleness = 12"),
        ("async_max_concurrent_samples = 256", f"async_max_concurrent_samples = {IN_FLIGHT}"),
        ("[trainer]\n# Native checkpoints only resume on the same trainer world size; keep this fixed for the run.\ngpus = 4",
         f"[trainer]\n# Native checkpoints only resume on the same trainer world size; keep this fixed for the run.\ngpus = {args.trainer_gpus}"),
        ("# Native serving is TP1 only; scale generation with engines. The 450M run was generation-bound.\ngpus = 4",
         f"# Native serving is TP1 only; scale generation with engines. The 450M run was generation-bound.\ngpus = {args.inference_gpus}"),
        ("sglang_cuda_graph_max_bs_decode = 64", "sglang_cuda_graph_max_bs_decode = 128"),
        ("sglang_max_running_requests = 64", "sglang_max_running_requests = 128"),
        ("sglang_server_concurrency = 64", "sglang_server_concurrency = 128"),
        ("sglang_mem_fraction_static = 0.35", "sglang_mem_fraction_static = 0.6"),
    ]
    for old, new in edits:
        count = body.count(old)
        if count < 1 or (old not in (V1B, V1B_DATA) and count != 1):
            raise SystemExit(f"expected exactly one occurrence of {old!r} (found {count})")
        body = body.replace(old, new)
    # Persistent Triton compiler cache on a mount we already have (the default root is on oe-training-default, and
    # a job mounting it stalled before start on 2026-11-18). FA4/CuTe compiles are not persisted by MILES.
    body += '\n[compiler_cache]\nshared_root = "/weka/oe-adapt-default/jacobm/olmicro/compiler-cache/tmp-30d"\n'
    header = (
        f"# v2: RL-zero on the Olmicro {args.size} long-context final ({revision}) with mixed-policy refresh publication\n"
        f"# (TIS-corrected, max_weight_staleness 12 as in the maintained refresh templates), {args.cap:,}-token responses,\n"
        f"# long-context base-model pass-band filtered data {args.data.name}. Otherwise the v1 recipe: R1 template,\n"
        f"# 32 prompts x 8 samples, lr 1e-6, 400 updates, {args.trainer_gpus} trainer + {args.inference_gpus} inference GPUs, {IN_FLIGHT} samples in flight,\n"
        f"# 4 eval samples per prompt.\n"
    )
    text = header + body
    if args.dry_run:
        print(text)
        return
    path = args.runs / f"{name}.toml"
    if path.exists() and path.read_text() != text:
        raise SystemExit(f"refusing to overwrite a different {path}")
    path.write_text(text)
    print("wrote", path)


if __name__ == "__main__":
    main()
