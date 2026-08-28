"""Generate a Training Observatory registry entry for one Terminal-RL run.

Derives the full provenance automatically:

- W&B runs whose name matches ``<exp_name>__<seed>__<timestamp>`` in the
  configured project; each distinct ``beaker_workload_id`` becomes one launch
  (initial + resumes, ordered by experiment-ULID creation time), carrying its
  surviving W&B run id.
- Rollout attempt prefixes found in the shared Weka dump, bucketed to launches
  by comparing attempt start timestamps against launch creation times. The
  emitted rollout binding uses the repo-local ``rl_rollouts/<id>`` symlink-farm
  path plus a ``source:`` declaration for viewer.tools.link_rollouts.
- Checkpoint directories under the checkpoint root, per attempt.
- TB2.1 / TBlite evaluation rows joined from the terminal-eval tracking CSVs
  (``--csv-model`` names the sheet's model_name), with per-row eval Beaker
  experiment links. Trial totals are inferred per row (k=5 vs k=1 protocols).

Usage (from the repo root, in the open-instruct uv env):

    uv run --no-sync python -m viewer.tools.bootstrap_registry \
        --id q35-9b-dppo-repro-4node-64k \
        --exp-name swerl_qwen35_9b_dppo_repro_4node_64k \
        --title "Qwen3.5-9B tmax-15k DPPO repro 4 nodes 64k" \
        --tag model="Qwen3.5-9B" --tag loss="DPPO" \
        --csv-model swerl-qwen35-9b-dppo-4n64k

Then run ``python -m viewer.tools.link_rollouts`` and validate with the
registry tests. The generated YAML is a starting point: notes, tags, and
manual corrections belong in the file afterwards.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from importlib import import_module
from pathlib import Path

import yaml

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROLLOUT_SOURCE = "/weka/oe-adapt-default/allennlp/deletable_rollouts"
DEFAULT_CHECKPOINT_ROOT = "/weka/oe-adapt-default/allennlp/deletable_checkpoint/shashankg"
DEFAULT_CSVS = [
    "/weka/nora-default/shashankg/code/tmax/scripts/beaker/terminalbench_combined_evals.csv",
    "/weka/nora-default/shashankg/code/tmax/scripts/beaker/opd_experiments_evals.csv",
    "/weka/nora-default/shashankg/code/tmax/scripts/beaker/cuda13_lr_ab_evals.csv",
]
# (benchmark name, CSV column prefix, candidate trial totals; k=5 first)
BENCHMARKS = [
    ("terminal-bench-2.1", "tb21", (445, 89)),
    ("tblite-2.0", "tblite", (500, 100)),
]
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid_seconds(ulid: str) -> float:
    milliseconds = 0
    for char in ulid[:10]:
        milliseconds = milliseconds * 32 + CROCKFORD.index(char)
    return milliseconds / 1000.0


def wandb_launches(project_path: str, exp_name: str) -> list[dict]:
    """One row per beaker experiment, each carrying its surviving W&B run."""
    api = import_module("wandb").Api()
    pattern = re.compile(rf"^{re.escape(exp_name)}__\d+__\d+$")
    rows = []
    for run in api.runs(project_path, filters={"display_name": {"$regex": f"^{re.escape(exp_name)}__"}}):
        if not pattern.match(run.name or ""):
            continue
        config = run.config or {}
        rows.append(
            {
                "wandb_id": run.id,
                "name": run.name,
                "started": int(run.name.rsplit("__", 1)[1]),
                "state": run.state,
                "beaker": str(config.get("beaker_workload_id") or "") or None,
                "checkpoint_state_dir": str(config.get("checkpoint_state_dir") or "") or None,
                "training_step": (run.summary or {}).get("training_step"),
            }
        )
    by_beaker: dict[str | None, list[dict]] = {}
    for row in sorted(rows, key=lambda item: item["started"]):
        by_beaker.setdefault(row["beaker"], []).append(row)
    launches = []
    for beaker, members in by_beaker.items():
        chosen = members[-1]
        if len(members) > 1:
            logger.warning(
                "%s: beaker %s has %d W&B runs; registering the last (%s)",
                exp_name,
                beaker,
                len(members),
                chosen["wandb_id"],
            )
        created = ulid_seconds(beaker) if beaker else float(chosen["started"])
        launches.append({**chosen, "created": created})
    launches.sort(key=lambda item: item["created"])
    return launches


def rollout_attempts(source: Path, exp_name: str) -> list[str]:
    prefix = f"{exp_name}__"
    seen = set()
    with os.scandir(source) as entries:
        for entry in entries:
            if not entry.name.startswith(prefix) or not entry.name.endswith(".jsonl"):
                continue
            base = entry.name.split("_metadata")[0].split("_rollouts")[0].split("_filtered")[0]
            base = base.split("_trainer_logprobs")[0]
            if re.fullmatch(rf"{re.escape(exp_name)}__\d+__\d+", base):
                seen.add(base)
    return sorted(seen, key=lambda name: int(name.rsplit("__", 1)[1]))


def bucket_attempts(attempts: list[str], launches: list[dict]) -> dict[int, list[str]]:
    """Assign each attempt to the newest launch created at or before it started."""
    buckets: dict[int, list[str]] = {index: [] for index in range(len(launches))}
    for attempt in attempts:
        started = int(attempt.rsplit("__", 1)[1])
        owner = 0
        for index, launch in enumerate(launches):
            if started >= launch["created"] - 300:  # clock slack: mason launches within minutes
                owner = index
        buckets[owner].append(attempt)
    return buckets


def checkpoint_map(root: Path, exp_name: str) -> dict[int, str]:
    """step -> newest attempt directory holding an exact step_<N> checkpoint."""
    out: dict[int, str] = {}
    if not root.is_dir():
        return out
    for directory in sorted(root.glob(f"{exp_name}__*_checkpoints")):
        for entry in directory.iterdir():
            if entry.name.startswith("step_") and entry.name[5:].isdigit():
                out[int(entry.name[5:])] = str(entry)
    return out


def infer_total(rate: float, candidates: tuple[int, ...]) -> tuple[int, int]:
    """Pick the trial count whose grid the reported pass rate sits on."""
    best = min(candidates, key=lambda total: abs(rate * total - round(rate * total)))
    return round(rate * best), best


def csv_evaluations(csv_paths: list[str], model_name: str) -> list[dict]:
    rows: dict[tuple[str, int], dict] = {}
    for path in csv_paths:
        if not Path(path).is_file():
            continue
        with open(path) as handle:
            for record in csv.DictReader(handle):
                if record.get("model_name") != model_name or not (record.get("step") or "").strip():
                    continue
                step = int(record["step"])
                for benchmark, column, totals in BENCHMARKS:
                    rate_text = (record.get(f"{column}_pass@1") or "").strip()
                    if not rate_text:
                        continue
                    correct, total = infer_total(float(rate_text), totals)
                    url = (record.get(f"{column}_beaker_url") or "").strip()
                    row = {"benchmark": benchmark, "step": step, "correct": correct, "total": total}
                    if url:
                        row["beaker_experiment"] = url.rstrip("/").rsplit("/", 1)[-1]
                    key = (benchmark, step)
                    # Some sheets carry both k=1 and k=5 rows for one step;
                    # keep the higher-trial protocol.
                    if key not in rows or rows[key]["total"] < total:
                        rows[key] = row
    return sorted(rows.values(), key=lambda row: (row["benchmark"], row["step"]))


def find_script(exp_name: str) -> str | None:
    matches = []
    scripts_root = REPO_ROOT / "scripts"
    for path in scripts_root.rglob("*.sh"):
        try:
            if exp_name in path.read_text(encoding="utf-8", errors="ignore"):
                matches.append(path.relative_to(REPO_ROOT))
        except OSError:
            continue
    if len(matches) == 1:
        return str(matches[0])
    if matches:
        logger.warning("%s: %d candidate scripts, leaving script unset: %s", exp_name, len(matches), matches)
    return None


def build_entry(args: argparse.Namespace) -> dict:
    launches = wandb_launches(args.wandb_project, args.exp_name)
    if not launches:
        raise SystemExit(f"No W&B runs named {args.exp_name}__<seed>__<ts> in {args.wandb_project}")
    attempts = rollout_attempts(Path(args.rollout_source), args.exp_name)
    buckets = bucket_attempts(attempts, launches)
    checkpoints = checkpoint_map(Path(args.checkpoint_root), args.exp_name)
    evaluations = csv_evaluations(args.csv, args.csv_model) if args.csv_model else []
    script = args.script or find_script(args.exp_name)

    entry: dict = {
        "schema_version": 1,
        "kind": "training",
        "id": args.id,
        "title": args.title or args.exp_name,
        "classification": args.classification or ("evaluated" if evaluations else "substantive"),
        "visibility": args.visibility,
    }
    if args.tag:
        entry["tags"] = dict(pair.split("=", 1) for pair in args.tag)
    if args.note:
        entry["note"] = args.note
    entry["wandb"] = {"run_id": launches[0]["wandb_id"]}

    launch_rows = []
    for index, launch in enumerate(launches):
        row: dict = {
            "id": "initial" if index == 0 else f"resume-{index}",
            "relation": "initial" if index == 0 else "resume",
        }
        if script:
            row["script"] = script
        if launch["beaker"]:
            row["beaker_experiment"] = launch["beaker"]
        if launch.get("checkpoint_state_dir"):
            row["checkpoint_state_dir"] = launch["checkpoint_state_dir"]
        if index > 0:
            row["wandb"] = {"run_id": launch["wandb_id"]}
        binding: dict = {"path": f"rl_rollouts/{args.id}", "source": args.rollout_source}
        if buckets[index]:
            binding["attempts"] = buckets[index]
        else:
            binding["configured_only"] = True
            binding["attempts"] = []
        row["rollouts"] = [binding]
        launch_rows.append(row)
    entry["launches"] = launch_rows

    artifacts: dict = {}
    progress_steps = [int(l["training_step"]) for l in launches if isinstance(l["training_step"], (int, float))]
    furthest = max([*progress_steps, *checkpoints.keys()], default=0)
    artifacts["furthest_step"] = furthest
    if checkpoints:
        artifacts["checkpoints"] = [{"step": step, "path": checkpoints[step]} for step in sorted(checkpoints)]
    if evaluations:
        for row in evaluations:
            if row["step"] in checkpoints:
                row["checkpoint"] = {"step": row["step"], "path": checkpoints[row["step"]]}
        artifacts["evaluations"] = evaluations
        primary = [row for row in evaluations if row["benchmark"] == BENCHMARKS[0][0]]
        if primary:
            best = max(primary, key=lambda row: row["correct"] / row["total"])
            artifacts["best_evaluation"] = {
                key: best[key] for key in ("benchmark", "step", "correct", "total", "checkpoint") if key in best
            }
    if checkpoints:
        last_step = max(checkpoints)
        artifacts["latest_checkpoint"] = {"step": last_step, "path": checkpoints[last_step]}
    entry["artifacts"] = artifacts
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True, help="Stable registry id (also the YAML filename)")
    parser.add_argument("--exp-name", required=True, help="exp_name used by the trainer (attempt prefix)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--note", default=None)
    parser.add_argument("--tag", action="append", default=[], help="key=value; repeatable")
    parser.add_argument("--classification", default=None, choices=[None, "evaluated", "substantive", "smoke"])
    parser.add_argument("--visibility", default="default", choices=["default", "archive", "hidden"])
    parser.add_argument("--csv-model", default=None, help="model_name in the terminal-eval tracking CSVs")
    parser.add_argument("--csv", action="append", default=list(DEFAULT_CSVS))
    parser.add_argument("--script", default=None, help="repo-relative launch script (auto-detected if unique)")
    parser.add_argument("--wandb-project", default="ai2-llm/oe-general-agents")
    parser.add_argument("--rollout-source", default=DEFAULT_ROLLOUT_SOURCE)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--registry", default=str(REPO_ROOT / "viewer" / "registry"))
    parser.add_argument("--force", action="store_true", help="overwrite an existing registry file")
    args = parser.parse_args()

    target = Path(args.registry) / "trainings" / f"{args.id}.yaml"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists; pass --force to overwrite")
    entry = build_entry(args)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    total_attempts = sum(len(l["rollouts"][0].get("attempts") or []) for l in entry["launches"])
    print(
        f"wrote {target}: {len(entry['launches'])} launches, {total_attempts} attempts, "
        f"{len(entry['artifacts'].get('evaluations') or [])} eval rows, furthest step {entry['artifacts']['furthest_step']}"
    )


if __name__ == "__main__":
    main()
