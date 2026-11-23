"""Native lifecycle smoke for an Olmicro size's long-context final, from the passing 2.7B smoke-b by asserted edits.

Changes vs smoke-b: model + Core config for --size (stage3 final); R1 data/stop; refresh publication (staleness 12,
engine_drain_timeout 1800); FLA config-cache env; graph/eager probe on (OI_NATIVE_GRAPH_PROBE; rerun with --no-probe
after a near-tie-only greedy flip); 8,192-token responses. Topology stays 2 trainer + 2 inference (native_smoke.py
requires checkpoint world_size 2).

Usage: python make_size_smoke_config.py RUNS_DIR --size 1.4b [--no-probe] [--suffix b] [--dry-run]
"""

import argparse
from pathlib import Path

O = "/weka/oe-adapt-default/jacobm/olmicro"
REVISIONS = {"275m": "stage3-step0047684", "450m": "stage3-step0047684", "810m": "stage3-step0023842", "1.4b": "stage3-step0023842"}
SMOKE = "olmicro-2.7b-math-smoke-20261118-b"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--size", choices=sorted(REVISIONS), required=True)
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rev = REVISIONS[args.size]
    name = f"olmicro-{args.size}-lc-smoke-20261118" + (f"-{args.suffix}" if args.suffix else "")
    model = f"{O}/models/olmicro-{args.size}-base"
    source = (args.runs / f"{SMOKE}.toml").read_text()
    body = source[source.index("schema_version"):]
    edits = [
        (SMOKE, name),
        (f'source = "{O}/models/olmicro-2.7b-base/stage2-step0011921"', f'source = "{model}/{rev}"'),
        ("# allenai/olmicro-2.7b-base@stage2-step0011921 (midtrain final), SHA256SUMS verified 2026-11-18.",
         f"# allenai/olmicro-{args.size}-base@{rev} (long-context final), SHA256SUMS verified 2026-11-18."),
        ('OI_NATIVE_LIFECYCLE_SMOKE = "1"\n', 'OI_NATIVE_LIFECYCLE_SMOKE = "1"\n' + ("" if args.no_probe else 'OI_NATIVE_GRAPH_PROBE = "1"\n')),
        (f'OI_NATIVE_SMOKE_EVAL_SOURCE = "{O}/data/dolci-math-base-boxed-v1/eval-smoke.jsonl"',
         f'OI_NATIVE_SMOKE_EVAL_SOURCE = "{O}/data/dolci-math-base-r1-v1/eval-smoke.jsonl"'),
        ('MILES_HYBRID_REFERENCE_BACKBONE = "0"\n',
         'MILES_HYBRID_REFERENCE_BACKBONE = "0"\n'
         "# FLA config cache (runs/warmup-fix-notes-20261118.md); print flag shows what still autotunes.\n"
         'FLA_CACHE_MODE = "default"\n'
         f'FLA_CONFIG_DIR = "{O}/fla-configs/b300-fla0.5.2-v1"\n'
         'TRITON_PRINT_AUTOTUNING = "1"\n'),
        (f'prompt_data = "{O}/data/dolci-math-base-boxed-v1/train.jsonl"', f'prompt_data = "{O}/data/dolci-math-base-r1-v1/train.jsonl"'),
        (f'reward_config = "{O}/data/dolci-math-base-boxed-v1/verifiers.json"', f'reward_config = "{O}/data/dolci-math-base-r1-v1/verifiers.json"'),
        ("packing_max_tokens = 8192", "packing_max_tokens = 16384"),
        ("# Planned first-run lengths (512 prompt + 4096 response), so the smoke exercises them.\nmax_response_length = 4096\nmax_context_length = 4608",
         "# 512 prompt + 8,192 response: long enough to exercise refresh continuation, short enough for a quick smoke.\nmax_response_length = 8192\nmax_context_length = 8704"),
        ("eval_max_response_len = 4096", "eval_max_response_len = 8192"),
        ("# allenai/olmicro-2.7b-base HF bucket stage2/checkpoints/step0011921/config.json, SHA256SUMS verified.",
         f"# allenai/olmicro-{args.size}-base HF bucket stage3/checkpoints/{rev.split('-')[1]}/config.json, SHA256SUMS verified."),
        (f'model_config = "{O}/models/olmicro-2.7b-base/olmo-core/stage2-step0011921/config.json"', f'model_config = "{model}/olmo-core/{rev}/config.json"'),
        ('publication_mode = "barrier"', 'publication_mode = "refresh"\nengine_drain_timeout = 1800.0'),
        ("diagnostic_interval = 1", "# Refresh requires 0; full publication audits run at startup instead.\ndiagnostic_interval = 0"),
        ("max_weight_staleness = 1", "max_weight_staleness = 12"),
        ('# Stop a base model that finishes and starts a new "Question:"; MathVerifier scores the last \\boxed{}.\nrollout_stop = ["\\nQuestion:"]',
         '# R1 template (tools/format_probe.py): stop when the base model starts a new "User:" turn.\nrollout_stop = ["\\nUser:"]'),
    ]
    for old, new in edits:
        count = body.count(old)
        if count < 1 or (old != SMOKE and count != 1):
            raise SystemExit(f"expected exactly one occurrence of {old!r} (found {count})")
        body = body.replace(old, new)
    header = (
        f"# Native {args.size} long-context MILES/Core/SGLang lifecycle smoke (tools/make_size_smoke_config.py from {SMOKE}).\n"
        f"# Gates: {'graph/eager probe, ' if not args.no_probe else ''}initial weight equality, finite nonzero gradients,\n"
        "# checkpoint 5 restored in a separate process for update 6. Refresh publication, R1 data, FLA config cache.\n"
        "# trainer.gpus must stay 2: native_smoke.py requires checkpoint world_size == 2.\n"
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
