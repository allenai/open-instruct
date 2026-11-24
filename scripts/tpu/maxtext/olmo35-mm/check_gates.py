"""Read the two pre-run gates of a TPU multimodal SFT run from its gate run's pod logs.

Gate 1, step-1 CE: the TPU gate run (../k8s/olmo35-mm-sft-4x4x8-gate: EMo document pool fixed at 512) logs
``LOSS-PARTS step=1 ce=...``; the reference is an OLMo-core forward of the same global batch with the pool at 512,
``step1_eval.json`` (``eval_mode_ce``) from a GPU run with ``--recipe.batch_hash_dir=<dir>
--recipe.step1_reference=true`` (allenai/OLMo-core@fd9bba1ba BatchHashCallback; eval mode routes with the eval pool,
512 = every expert), or ``--reference-ce``. Pass: |TPU - reference| <= --tolerance (1e-3). The CE is not comparable,
and the gate fails, when the TPU's first step dropped routed tokens (``MOE-DROPPED step=0 dropped>0``, or without that
line ``REQUIRED_RBF step=0 max`` above ``--ragged-buffer-factor``).

Gate 2, batch order: every ``MM-BATCH-HASH {json}`` record (MaxText ``olmo_mm_batch_hash_steps``) for steps 1..--steps
equals the GPU run's ``rank{R}.jsonl`` record of the same (step, rank) in its ``all`` hash, and every (step, rank) the
GPU logged is present. Without --gpu-hashes the records are only checked for consistency (hosts that read the same
rank must agree) and completeness.

    python check_gates.py --tpu-logs LOG_DIR_OR_FILES... [--gpu-hashes DIR] [--gpu-step1 step1_eval.json |
        --reference-ce X] [--steps 20] [--ranks 32] [--out gates.json]

Prints a JSON summary; exits 0 only if every gate it could check passed and both gates were checked.
"""

import argparse
import glob
import json
import os
import re
import sys

LOSS_PARTS = re.compile(r"LOSS-PARTS step=(\d+) ce=([-+0-9.eE]+)")
REQUIRED_RBF = re.compile(r"REQUIRED_RBF step=(\d+) max=([-+0-9.eE]+)")
MOE_DROPPED = re.compile(r"MOE-DROPPED step=(\d+) dropped=(\d+) routed=(\d+)")
BATCH_HASH = re.compile(r"MM-BATCH-HASH (\{.*\})\s*$")


def log_files(paths):
    files = []
    for path in paths:
        if os.path.isdir(path):
            files += sorted(glob.glob(os.path.join(path, "**", "*.log*"), recursive=True))
        else:
            files.append(path)
    return files


def read_tpu_logs(paths):
    """LOSS-PARTS CE by step, first-step MoE probes, and MM-BATCH-HASH records by (step, rank)."""
    ce, rbf, dropped, hashes, conflicts = {}, {}, {}, {}, []
    for path in log_files(paths):
        with open(path, errors="replace") as f:
            for line in f:
                if m := LOSS_PARTS.search(line):
                    ce.setdefault(int(m[1]), float(m[2]))
                elif m := REQUIRED_RBF.search(line):
                    step = int(m[1])
                    rbf[step] = max(rbf.get(step, 0.0), float(m[2]))
                elif m := MOE_DROPPED.search(line):
                    step = int(m[1])
                    dropped[step] = max(dropped.get(step, 0), int(m[2]))
                elif m := BATCH_HASH.search(line):
                    record = json.loads(m[1])
                    key = (int(record["step"]), int(record["rank"]))
                    old = hashes.setdefault(key, record["hashes"])
                    if old != record["hashes"]:
                        conflicts.append(list(key))
    return {"ce": ce, "rbf": rbf, "dropped": dropped, "hashes": hashes, "conflicts": conflicts}


def read_gpu_hashes(directory):
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "rank*.jsonl"))):
        with open(path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    out[(int(record["step"]), int(record["rank"]))] = record["hashes"]
    return out


def ce_gate(tpu, reference_ce, tolerance, ragged_buffer_factor):
    res = {"tpu_step1_ce": tpu["ce"].get(1), "reference_ce": reference_ce, "tolerance": tolerance}
    # MaxText's REQUIRED_RBF / MOE-DROPPED lines count steps from 0; LOSS-PARTS counts from 1.
    first_dropped = tpu["dropped"].get(0)
    first_rbf = tpu["rbf"].get(0)
    res["first_step_dropped_tokens"] = first_dropped
    res["first_step_required_rbf"] = first_rbf
    if first_dropped is not None:
        drops = first_dropped > 0
    elif first_rbf is not None and ragged_buffer_factor is not None:
        drops = first_rbf > ragged_buffer_factor
    else:
        drops = None
    res["first_step_drops"] = drops
    if res["tpu_step1_ce"] is None or reference_ce is None:
        res["pass"] = None
        res["why"] = "no LOSS-PARTS step=1 line" if res["tpu_step1_ce"] is None else "no reference CE given"
        return res
    res["abs_diff"] = abs(res["tpu_step1_ce"] - reference_ce)
    if drops:
        res["pass"] = False
        res["why"] = "the TPU's first step dropped routed tokens; raise ragged_buffer_factor for the gate"
    else:
        res["pass"] = res["abs_diff"] <= tolerance
    return res


def hash_gate(tpu, gpu, steps, ranks):
    window = {k: v for k, v in tpu["hashes"].items() if 1 <= k[0] <= steps}
    res = {"steps": steps, "tpu_records": len(window), "tpu_conflicts": tpu["conflicts"]}
    expected = {(s, r) for s in range(1, steps + 1) for r in range(ranks)} if ranks else set()
    if gpu is not None:
        gpu = {k: v for k, v in gpu.items() if 1 <= k[0] <= steps}
        expected |= set(gpu)
        res["gpu_records"] = len(gpu)
        compared = sorted(set(gpu) & set(window))
        mismatched = [list(k) for k in compared if gpu[k]["all"] != window[k]["all"]]
        res["compared"] = len(compared)
        res["equal"] = len(compared) - len(mismatched)
        res["mismatched"] = mismatched[:50]
        res["mismatched_keys"] = sorted(
            {name for k in compared for name in gpu[k] if name != "all" and gpu[k].get(name) != window[k].get(name)}
        )
    missing = sorted(expected - set(window))
    res["missing_on_tpu"] = [list(k) for k in missing[:50]]
    res["missing_count"] = len(missing)
    ok = not res["tpu_conflicts"] and not missing and bool(window)
    if gpu is not None:
        ok = ok and not res["mismatched"] and res["compared"] > 0
    res["pass"] = ok
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--tpu-logs", nargs="+", required=True, help="pod log files or directories (e.g. watch_jobset.sh's)"
    )
    p.add_argument("--gpu-hashes", help="the GPU run's batch_hash_dir (rank*.jsonl)")
    p.add_argument("--gpu-step1", help="the GPU run's step1_eval.json (eval_mode_ce)")
    p.add_argument("--reference-ce", type=float, help="reference step-1 CE instead of --gpu-step1")
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--ranks", type=int, default=32, help="OLMo-core loader ranks (OLMO_MM_LOADER_WORLD); 0 = no check")
    p.add_argument(
        "--ragged-buffer-factor",
        type=float,
        default=3.0,
        help="the gate run's ragged_buffer_factor, used when it logged no MOE-DROPPED line",
    )
    p.add_argument("--out", help="also write the summary here")
    a = p.parse_args(argv)

    reference = a.reference_ce
    if a.gpu_step1:
        with open(a.gpu_step1) as f:
            reference = float(json.load(f)["eval_mode_ce"])
    tpu = read_tpu_logs(a.tpu_logs)
    gpu = read_gpu_hashes(a.gpu_hashes) if a.gpu_hashes else None
    summary = {
        "step1_ce": ce_gate(tpu, reference, a.tolerance, a.ragged_buffer_factor),
        "batch_hashes": hash_gate(tpu, gpu, a.steps, a.ranks),
    }
    summary["pass"] = (
        summary["step1_ce"]["pass"] is True and summary["batch_hashes"]["pass"] is True and gpu is not None
    )
    text = json.dumps(summary, indent=1)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print("GATES-PASS" if summary["pass"] else "GATES-FAIL")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
