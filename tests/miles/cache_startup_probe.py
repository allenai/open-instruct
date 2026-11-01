"""Two fresh EP1 trainer processes, one optimizer step each, identical inputs."""

import argparse
import collections
import dataclasses
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from miles.utils.compiler_cache import startup_cache, tilelang_cache

from open_instruct.miles.configuration.config import CoreConfig, RunConfig


def configuration(root):
    return RunConfig(
        CoreConfig(
            expert_parallel_size=1,
            attention_backend="torch",
            max_sequence_length=128,
            activation_checkpointing=False,
            diagnostic_interval=1,
            router_aux_loss_weight=0.01,
            router_z_loss_weight=1e-5,
        ),
        dict(
            hf_checkpoint=str(root / "hf"),
            global_batch_size=4,
            rollout_batch_size=1,
            n_samples_per_prompt=4,
            num_rollout=2,
            actor_num_gpus_per_node=1,
            debug_train_only=True,
            save=str(root / "ep1-combined-ac0"),
            rollout_global_dataset=True,
            prompt_data=str(root / "prompts.jsonl"),
            lr=1e-4,
            clip_grad=1e9,
            use_rollout_routing_replay=True,
            use_miles_router=True,
        ),
    )


def child(root, phase):
    run = root / phase
    config = configuration(run)
    # expected.toml intentionally names every non-default fixture setting. The
    # production contract checks full dataclass equality before constructing the trainer.
    fields = {k: v for k, v in dataclasses.asdict(config.core).items() if v is not None}
    lines = ["[core]"] + [f"{k} = {json.dumps(v)}" for k, v in fields.items()]
    lines += ["", "[miles]"] + [f"{k} = {json.dumps(v)}" for k, v in config.miles.items()]
    (run / "expected.toml").write_text("\n".join(lines) + "\n")
    args = SimpleNamespace(
        **config.miles,
        olmo_core=dataclasses.replace(
            config.core, compiler_cache_root=str(root / "shared"), compiler_cache_diagnostics=True
        ),
    )
    application = Path(importlib.util.find_spec("open_instruct").submodule_search_locations[0]).parent
    startup_cache.prepare(args, application_root=application)
    descriptor = dict(
        args.olmo_core_startup_cache,
        slot="trainer-engine-actor-0-0",
        report_dir=str(run / "reports"),
        tilelang_runtime=tilelang_cache.compiler_identity(),
    )
    (run / "reports").mkdir()
    os.environ[startup_cache.ENV] = json.dumps(descriptor)
    startup_cache.setup_worker()
    report = json.loads(next((run / "reports").glob("*.json")).read_text())
    started = time.monotonic()
    try:
        importlib.import_module("cache_core_contract").run(run)
    finally:
        startup_cache.archive_activity(report, Path(report["local"]))
    seconds = time.monotonic() - started
    published = startup_cache.publish_worker(report, retain_local=True)
    events = [
        json.loads(line) for p in Path(report["local"]).glob("activity-*.jsonl") for line in p.read_text().splitlines()
    ]
    summary = dict(
        phase=phase,
        trainer_seconds=seconds,
        restore=report["restore"],
        fingerprint=report["fingerprint"],
        publish=published.get("publish"),
        tilelang=published.get("tilelang"),
        events=dict(collections.Counter(e["event"] for e in events)),
        compile_seconds=sum(e.get("compiler_microseconds", 0) for e in events if e["event"] == "compile_miss") / 1e6,
    )
    (run / "cache-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase:
        child(root, args.phase)
        return
    root.mkdir(parents=True)
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29671",
        TILELANG_JIT_DIAGNOSTICS="1",
        TILELANG_KERNEL_CACHE_USE_LIB_STAMP="1",
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import ep_contract; from pathlib import Path; import sys; ep_contract.bootstrap(Path(sys.argv[1]))",
            str(root / "fixture"),
        ],
        check=True,
        env=environment,
        timeout=180,
    )
    for phase in ("cold", "warm"):
        shutil.copytree(root / "fixture", root / phase)
        subprocess.run(
            [sys.executable, __file__, "--output", str(root), "--phase", phase],
            check=True,
            env=environment,
            timeout=900,
        )
    torch = importlib.import_module("torch")
    cold = torch.load(root / "cold/observation.pt", weights_only=True)
    warm = torch.load(root / "warm/observation.pt", weights_only=True)
    torch.testing.assert_close(cold, warm, rtol=0, atol=0)
    cold_state = torch.load(root / "cold/stress-ep1-rank0.pt", weights_only=True)
    warm_state = torch.load(root / "warm/stress-ep1-rank0.pt", weights_only=True)
    torch.testing.assert_close(cold_state, warm_state, rtol=0, atol=0)
    summaries = [json.loads((root / p / "cache-summary.json").read_text()) for p in ("cold", "warm")]
    assert summaries[0]["events"].get("compile_miss", 0) > 0, summaries
    assert summaries[1]["restore"]["status"] == "hit", summaries
    assert summaries[1]["events"].get("compile_miss", 0) == 0, summaries
    (root / "summary.json").write_text(
        json.dumps({"numerical_checks": "bitwise_equal", "phases": summaries}, indent=2) + "\n"
    )
    print("TRAINER_CACHE_PROBE_PASSED", flush=True)


if __name__ == "__main__":
    main()
