"""Run a native Olmicro smoke in two separate training processes and prove resume."""

import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def nonzero_updates(log):
    updates = []
    for match in re.finditer(r"Core optimizer step (\d+): (\{[^\n]+\})", log):
        metrics = ast.literal_eval(match[2])
        norm = metrics.get("train/grad_norm", 0.0)
        if math.isfinite(norm) and norm > 0:
            updates.append(int(match[1]))
    return sorted(set(updates))


def graph_probe(document, root):
    probe = "/opt/core-rl/sources/sglang/test/manual/olmicro/probe_backend.py"
    reports = []
    for mode in ("eager", "graph"):
        output = root / f"{mode}-probe.json"
        command = [
            sys.executable,
            probe,
            "--backend",
            "sglang",
            "--model",
            document["model"]["source"],
            "--attention-backend",
            document["miles"]["sglang_attention_backend"],
            "--output",
            str(output),
        ]
        if mode == "graph":
            command.append("--cuda-graph")
        subprocess.run(command, check=True)
        reports.append(str(output))
    subprocess.run(
        [sys.executable, probe, "--compare", *reports, "--output", str(root / "graph-eager-parity.json")], check=True
    )
    parity = json.loads((root / "graph-eager-parity.json").read_text())
    if not all(parity["greedy_exact_matches"]) or max(c["max_logprob_error"] for c in parity["cases"]) > 1e-3:
        raise RuntimeError("Graph/eager probe requires identical greedy tokens and logprob error <= 0.001")


def checkpoint(root, update):
    checkpoints = root / "checkpoints"
    rollout = update - 1
    manifest = json.loads((checkpoints / "core" / f"rollout_{rollout:07d}" / "complete.json").read_text())
    cursor = checkpoints / "rollout" / f"global_dataset_state_dict_{rollout}.pt"
    if hashlib.sha256(cursor.read_bytes()).hexdigest() != manifest["cursor_sha256"]:
        raise RuntimeError("Smoke checkpoint cursor differs from its committed manifest")
    if manifest["clock"]["completed_steps"] != update or manifest["world_size"] != 2:
        raise RuntimeError(f"Unexpected native checkpoint boundary: {manifest['clock']}")
    return manifest


def main():
    config = Path(sys.argv[1])
    document = json.loads(config.read_text())
    stop = document["miles"]["debug_exit_after_rollout"]
    total = document["training"]["num_rollouts"]
    if stop < 1 or total != stop + 1:
        raise ValueError("Native smoke expects one update after the debug-stop checkpoint")
    root = Path(document["output"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    if document["launch"]["env"].get("OI_NATIVE_GRAPH_PROBE") == "1":
        graph_probe(document, root)
    # Only the small evaluation slice is changed; training uses the real GSM8K data.
    source = Path(document["launch"]["env"]["OI_NATIVE_SMOKE_EVAL_SOURCE"])
    target = Path(document["data"]["eval_prompt_data"][1])
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = source.read_text().splitlines()
    target.write_text("\n".join(rows[:8]) + "\n")
    nonzero = []
    for attempt in range(2):
        log_path = root / f"native-smoke-process-{attempt}.log"
        with (
            log_path.open("w") as log,
            subprocess.Popen(
                [sys.executable, "-m", "open_instruct.miles", "train", str(config)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ) as process,
        ):
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError(f"Smoke training failed; inspect {log_path}")
        nonzero.append(nonzero_updates(log_path.read_text()))
        if not nonzero[-1]:
            raise RuntimeError(f"Smoke process {attempt} had no verified nonzero-gradient update")
        state = json.loads((root / "workflow.json").read_text())
        result = state["result"]
        if attempt == 0:
            checkpoint(root, stop)
            if state["status"] != "stopped" or result["completed_rollout_ids"] != list(range(stop)):
                raise RuntimeError(f"First smoke process did not stop at checkpoint {stop}: {state}")
            logger.info("NATIVE_SMOKE_CHECKPOINT_COMMITTED update=%s; starting separate resume process", stop)
        else:
            manifest = checkpoint(root, total)
            if (
                state["status"] != "complete"
                or result["start_rollout_id"] != stop
                or result["completed_rollout_ids"] != [stop]
            ):
                raise RuntimeError(f"Second smoke process did not restore and advance: {state}")
            receipt = {
                "status": "passed",
                "restored_update": stop,
                "completed_updates": total,
                "nonzero_updates_by_process": nonzero,
                "clock": manifest["clock"],
            }
            (root / "native-smoke-complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
            logger.info("NATIVE_SMOKE_RESUME_PASSED %s", receipt)


if __name__ == "__main__":
    main()
