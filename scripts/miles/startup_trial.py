"""Two fresh Ray lifetimes on one allocation, cold and restored Triton caches."""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.miles import exercise_controls
from scripts.miles.prepare_gsm8k_parity import verify_preparation


def prepare(campaign, root):
    verify_preparation(campaign)
    root.mkdir(parents=True, exist_ok=False)
    for arm in ("cold", "restored"):
        output = root / arm
        output.mkdir()
        config = exercise_controls.configuration(campaign, output, "sync", 2)
        config = dataclasses.replace(
            config,
            core=dataclasses.replace(
                config.core,
                compiler_cache=True,
                compiler_cache_root=str(root / "cache/tmp-7d"),
                compiler_cache_restore=arm == "restored",
                compiler_cache_diagnostics=True,
                max_sequence_length=1024,
            ),
        )
        config.miles.update(
            rollout_batch_size=2,
            n_samples_per_prompt=2,
            global_batch_size=4,
            rollout_max_response_len=256,
            rollout_max_context_len=1024,
            rollout_max_prompt_len=768,
            rollout_temperature=0.0,
            sglang_context_length=1024,
            sglang_log_level="info",
            use_wandb=False,
        )
        exercise_controls.write_config(output / "run.toml", config)


def compare(root):
    result = {
        "arms": {},
        "limits": [
            "Same allocation: filesystem page cache may warm between arms; HF I/O is timed separately.",
            "Two updates, short greedy responses, full SFT weights, EP2 trainer and TP1 serving.",
            "Compiler artifacts only; CUDA graphs and in-memory Python JIT state are rebuilt.",
        ],
    }
    for arm in ("cold", "restored"):
        metrics = root / arm / "metrics"
        cached = json.loads((metrics / "compiler-cache.json").read_text())
        if not cached["success"] or len(cached["workers"]) != 3:
            raise ValueError(f"{arm}: incomplete cache worker lifecycle")
        stages = [json.loads(s) for s in (metrics / "driver_timing.jsonl").read_text().splitlines()]
        if any(not row["passed"] for row in stages):
            raise ValueError(f"{arm}: failed stage")
        ranks = {
            p.name: [json.loads(s) for s in p.read_text().splitlines()] for p in metrics.glob("startup_rank*.jsonl")
        }
        result["arms"][arm] = dict(
            cache=cached,
            driver_stages=stages,
            rank_stages=ranks,
            elapsed=json.loads((root / arm / "elapsed.json").read_text()),
        )
    cold = {w["slot"]: w for w in result["arms"]["cold"]["cache"]["workers"]}
    warm = {w["slot"]: w for w in result["arms"]["restored"]["cache"]["workers"]}
    if cold.keys() != warm.keys():
        raise ValueError("Worker slots changed")
    for slot, value in warm.items():
        if value["restore"]["status"] != "hit" or value["activity"].get("group_hit", 0) <= 0:
            raise ValueError(f"{slot}: restored artifacts were not consumed")
        if value["activity"].get("put", 0) >= cold[slot]["activity"].get("put", 0):
            raise ValueError(f"{slot}: compiler writes did not decrease")
    result["passed"] = True
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(campaign, root):
    prepare(campaign, root)
    for arm in ("cold", "restored"):
        environment = {**os.environ}
        # Other compiler families stay cold and separate; this isolates Triton reuse.
        for variable, family in [
            ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
            ("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", "fa4"),
            ("TILELANG_CACHE_DIR", "tilelang"),
        ]:
            environment[variable] = f"/tmp/startup-{root.name}-{arm}/{family}"
        with (root / arm / "run.log").open("w") as stream:
            subprocess.run(
                [sys.executable, "-m", "scripts.miles.startup_trial", "worker", str(root / arm)],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    compare(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker", "compare"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        exercise_controls.train_cli(args.root)
    elif args.command == "compare":
        compare(args.root)
    else:
        if args.campaign is None:
            parser.error("run requires --campaign")
        run(args.campaign, args.root)


if __name__ == "__main__":
    main()
