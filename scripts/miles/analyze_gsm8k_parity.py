"""Audit trusted GSM8K campaign dumps on CPU and compare descriptive learning curves.

The .pt inputs are trusted artifacts produced by this campaign. They are loaded
with torch's full pickle loader; do not point this command at third-party dumps.
Raw responses remain in their original files; output contains hashes and scores.
"""

import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

import torch
from matplotlib import pyplot as plt

from open_instruct.ground_truth_utils import GSM8KVerifier

UPDATES = 100
EVAL_STEPS = (0, 20, 40, 60, 80, 100)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def identity(row):
    return row["metadata"]["prepared_sample_id"]


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [identity(row) for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: prepared prompt IDs must be nonempty and unique")
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["id"]].append(row["correct"])
    count = len(rows)
    mixed = sum(min(scores) != max(scores) for scores in groups.values())
    return {
        "samples": count,
        "accuracy": sum(row["correct"] for row in rows) / count if count else None,
        "mean_reward": sum(row["correct"] for row in rows) / count if count else None,
        "mean_response_tokens": sum(row["response_tokens"] for row in rows) / count if count else None,
        "truncation_rate": sum(row["truncated"] for row in rows) / count if count else None,
        "at_response_cap": sum(row["at_response_cap"] for row in rows),
        "prompt_groups": len(groups),
        "mixed_reward_groups": mixed,
        "mixed_reward_group_fraction": mixed / len(groups) if groups else None,
    }


def audit_dump(path, prepared_rows, *, version, multiplicity, response_cap=4096, token_proofs=None):
    expected = {identity(row): row for row in prepared_rows}
    expected_counts = Counter({key: multiplicity for key in expected})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    errors, records, seen = [], [], Counter()
    verifier = GSM8KVerifier()
    expected_rollout_id = int(path.stem.removeprefix("eval_"))
    if payload.get("rollout_id") != expected_rollout_id:
        errors.append("dump rollout_id differs from filename")
    for index, sample in enumerate(payload["samples"]):
        prefix = f"sample {index}"
        try:
            key = identity(sample)
        except (KeyError, TypeError):
            errors.append(f"{prefix}: missing prepared prompt ID")
            continue
        if key not in expected:
            errors.append(f"{prefix}: prompt ID outside expected membership")
            continue
        seen[key] += 1
        row = expected[key]
        if sample["prompt"] != row["input"] or sample["label"] != row["label"]:
            errors.append(f"{prefix}: prompt or label differs from prepared data")
        if sample["metadata"].get("verifiers") != row["metadata"].get("verifiers"):
            errors.append(f"{prefix}: verifier specification differs from prepared data")
        versions = sample.get("weight_versions")
        if not versions or any(str(value) != str(version) for value in versions):
            errors.append(f"{prefix}: missing or unexpected policy version (expected {version})")
        length = sample["response_length"]
        if not isinstance(length, int) or not 0 < length <= response_cap or length > len(sample["tokens"]):
            errors.append(f"{prefix}: invalid response token length")
            continue
        if token_proofs is not None:
            prompt_ids = list(sample["tokens"][:-length])
            encoded = (json.dumps(prompt_ids, indent=2, sort_keys=True) + "\n").encode()
            proof = token_proofs[key]
            if hashlib.sha256(encoded).hexdigest() != proof["token_ids_sha256"]:
                errors.append(f"{prefix}: prompt token IDs differ from preparation")
        logprobs = sample.get("rollout_log_probs")
        if logprobs is None or len(logprobs) != length or not all(math.isfinite(value) for value in logprobs):
            errors.append(f"{prefix}: response log probabilities missing, wrong length, or nonfinite")
        status = sample["status"]
        if status not in ("completed", "truncated") or sample.get("remove_sample", False):
            errors.append(f"{prefix}: unsuccessful or removed generation")
        # Reconstruct the target from immutable preparation, bypassing the reward
        # bridge and the sample's possibly corrupted target entirely.
        score = verifier([], sample["response"], row["label"]).score
        stored = sample["reward"]
        if not isinstance(stored, int | float) or not math.isfinite(stored) or score != stored:
            errors.append(f"{prefix}: stored reward differs from direct GSM8K verification")
        records.append(
            dict(
                id=key,
                correct=int(score),
                response_tokens=length,
                truncated=status == "truncated",
                at_response_cap=length == response_cap,
                response_sha256=hashlib.sha256(sample["response"].encode()).hexdigest(),
            )
        )
    if seen != expected_counts:
        errors.append("prompt membership or sample multiplicity differs from expected batch")
    return dict(
        file=path.name,
        sha256=digest(path),
        policy_version=version,
        valid=not errors,
        errors=errors,
        summary=summarize(records),
        samples=records,
    )


def evidence_rows(path):
    if not path.is_file():
        return None, [f"missing {path.name}"]
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        return None, [f"cannot read {path.name}: {error}"]
    if not all(isinstance(row, dict) for row in rows):
        return None, [f"{path.name} records must be objects"]
    return rows, []


def publication_summary(path):
    rows, errors = evidence_rows(path)
    if rows is None:
        return {"available": path.is_file(), "valid": False, "errors": errors}
    versions = [row.get("version") for row in rows]
    if versions != list(range(UPDATES + 1)) or any(type(version) is not int for version in versions):
        errors.append(f"publication versions must be exactly 0..{UPDATES} in order")
    repeated = sum(bool(row.get("repeated_version", False)) for row in rows)
    if repeated:
        errors.append("unexpected repeated-version publications")
    durations = [row.get("total_seconds") for row in rows]
    valid_durations = all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in durations)
    if not valid_durations:
        errors.append("publication durations must be finite and nonnegative")
    return {
        "available": True,
        "valid": not errors,
        "errors": errors,
        "sha256": digest(path),
        "count": len(rows),
        "versions": versions,
        "total_seconds": sum(durations) if valid_durations else None,
        "mean_seconds": sum(durations) / len(durations) if durations and valid_durations else None,
        "max_seconds": max(durations, default=None) if valid_durations else None,
        "repeated_versions": repeated,
    }


def optimizer_summary(path):
    rows, errors = evidence_rows(path)
    if rows is None:
        return {"available": path.is_file(), "valid": False, "errors": errors}
    steps = [row.get("step") for row in rows if row.get("event") == "optimizer"]
    if steps != list(range(1, UPDATES + 1)) or any(type(step) is not int for step in steps):
        errors.append(f"optimizer records must be exactly 1..{UPDATES} in order")
    return {
        "available": True,
        "valid": not errors,
        "errors": errors,
        "sha256": digest(path),
        "count": len(steps),
        "steps": steps,
    }


def check_core_completion(root, directory, report):
    for key, filename, summarizer in (
        ("publication", "publication.jsonl", publication_summary),
        ("optimizer", "training_contract_rank0.jsonl", optimizer_summary),
    ):
        report[key] = summarizer(root / "metrics" / filename)
        report["errors"].extend(f"{key}: {error}" for error in report[key]["errors"])
    expected_names = {f"{step}.pt" for step in range(UPDATES)} | {
        f"eval_{step - 1 if step else 0}.pt" for step in EVAL_STEPS
    }
    actual_names = {path.name for path in directory.glob("*.pt")}
    report["rollout_file_count"] = len(actual_names)
    if actual_names != expected_names:
        report["errors"].append("Core rollout file set differs from the frozen protocol")
    completion = root / "completion.json"
    if not completion.is_file():
        report["errors"].append("missing Core completion.json")
        return
    try:
        record = json.loads(completion.read_text())
    except (OSError, ValueError) as error:
        report["errors"].append(f"cannot read Core completion.json: {error}")
        return
    report["completion"] = record
    if not isinstance(record, dict) or record.get("completed") is not True:
        report["errors"].append("Core completion must explicitly report completed=true")
    expected = {
        "optimizer_steps": UPDATES,
        "publication_count": UPDATES + 1,
        "rollout_files": UPDATES + len(EVAL_STEPS),
    }
    for key, count in expected.items():
        if not isinstance(record, dict) or type(record.get(key)) is not int or record.get(key) != count:
            report["errors"].append(f"Core completion {key} must equal {count}")


def arm_directory(backend, megatron_directory="megatron"):
    if (
        not megatron_directory
        or Path(megatron_directory).name != megatron_directory
        or megatron_directory in (".", "..")
    ):
        raise ValueError("megatron_directory must be one directory name beneath the campaign root")
    return megatron_directory if backend == "megatron" else "core"


def audit(root, backend, *, megatron_directory="megatron"):
    selected_directory = arm_directory(backend, megatron_directory)
    preparation = json.loads((root / "preparation.json").read_text())
    if not {"train.jsonl", "eval.jsonl"}.issubset(preparation["files"]):
        raise ValueError("Preparation manifest is missing shared dataset hashes")
    for name, expected_hash in preparation["files"].items():
        if digest(root / name) != expected_hash:
            raise ValueError(f"Preparation artifact changed: {name}")
    token_proofs = {
        row["prepared_sample_id"]: row for partition in preparation["partitions"].values() for row in partition["rows"]
    }
    training, evaluation = read_rows(root / "train.jsonl"), read_rows(root / "eval.jsonl")
    if set(map(identity, training)) & set(map(identity, evaluation)):
        raise ValueError("Training and held-out prompt IDs overlap")
    # The sync Megatron updater increments before its initial publication.
    # Core publishes completed optimizer steps directly. Never infer this shift
    # from the observed data, which could conceal a stale publication.
    version_offset = 0 if backend == "core" else 1
    directory = root / selected_directory / ("rollouts" if backend == "core" else "rollout_data")
    report = {
        "schema_version": 1,
        "backend": backend,
        "artifact_directory": selected_directory,
        "version_offset": version_offset,
        "preparation_sha256": digest(root / "preparation.json"),
        "prepared_sha256": {name: digest(root / f"{name}.jsonl") for name in ("train", "eval")},
        "training": [],
        "evaluation": [],
        "errors": [],
        "interpretation": "One run per backend; descriptive comparison, without significance or learning-rate conclusions.",
    }
    for rollout in range(UPDATES):
        selected = [training[(rollout * 4 + offset) % len(training)] for offset in range(4)]
        path = directory / f"{rollout}.pt"
        if not path.is_file():
            report["errors"].append(f"missing training dump {path.name}")
            continue
        entry = audit_dump(path, selected, version=rollout + version_offset, multiplicity=4, token_proofs=token_proofs)
        entry["completed_steps_before_update"] = rollout
        report["training"].append(entry)
    for step in EVAL_STEPS:
        path = directory / f"eval_{step - 1 if step else 0}.pt"
        if not path.is_file():
            report["errors"].append(f"missing evaluation dump {path.name}")
            continue
        entry = audit_dump(path, evaluation, version=step + version_offset, multiplicity=1, token_proofs=token_proofs)
        entry["completed_steps"] = step
        report["evaluation"].append(entry)
    report["training_summary"] = summarize([sample for row in report["training"] for sample in row["samples"]])
    for key in ("prompt_groups", "mixed_reward_groups"):
        report["training_summary"][key] = sum(row["summary"][key] for row in report["training"])
    groups = report["training_summary"]["prompt_groups"]
    report["training_summary"]["mixed_reward_group_fraction"] = (
        report["training_summary"]["mixed_reward_groups"] / groups if groups else None
    )
    if backend == "core":
        check_core_completion(root / selected_directory, directory, report)
    else:
        report["completion_evidence"] = (
            "Rollout/evaluation evidence only; confirm final training log and Beaker exit status separately. "
            "Megatron does not emit the Core completion manifest."
        )
    report["valid"] = not report["errors"] and all(row["valid"] for row in report["training"] + report["evaluation"])
    return report


def within_backend_transitions(report):
    endpoints = []
    for step in (0, UPDATES):
        matches = [entry for entry in report["evaluation"] if entry["completed_steps"] == step]
        if len(matches) != 1:
            return None, [f"{report['backend']}: missing unique transition endpoint {step}"]
        rows = matches[0]["samples"]
        index = {row["id"]: row for row in rows}
        if len(index) != len(rows):
            return None, [f"{report['backend']}: duplicate transition prompt IDs at {step}"]
        endpoints.append(index)
    initial, final = endpoints
    if initial.keys() != final.keys() or not initial:
        return None, [f"{report['backend']}: transition endpoint membership differs or is empty"]
    pairs = [
        {
            "id": key,
            "initial_correct": initial[key]["correct"],
            "final_correct": final[key]["correct"],
            "initial_at_response_cap": initial[key]["at_response_cap"],
            "final_at_response_cap": final[key]["at_response_cap"],
            "initial_response_tokens": initial[key]["response_tokens"],
            "final_response_tokens": final[key]["response_tokens"],
        }
        for key in sorted(initial)
    ]
    scores = Counter((pair["initial_correct"], pair["final_correct"]) for pair in pairs)
    caps = Counter((pair["initial_at_response_cap"], pair["final_at_response_cap"]) for pair in pairs)
    return {
        "initial_step": 0,
        "final_step": UPDATES,
        "questions": len(pairs),
        "correctness": {
            "correct_to_correct": scores[1, 1],
            "correct_to_incorrect": scores[1, 0],
            "incorrect_to_correct": scores[0, 1],
            "incorrect_to_incorrect": scores[0, 0],
            "net_correct_change": scores[0, 1] - scores[1, 0],
        },
        "response_cap": {
            "initial_count": caps[True, True] + caps[True, False],
            "final_count": caps[True, True] + caps[False, True],
            "at_cap_both": caps[True, True],
            "left_cap": caps[True, False],
            "entered_cap": caps[False, True],
            "below_cap_both": caps[False, False],
        },
        "pairs": pairs,
        "interpretation": "Within-run paired question transitions; descriptive counts, without a significance claim.",
    }, []


def compare(core, megatron):
    errors = []
    if not core["valid"] or not megatron["valid"]:
        errors.append("At least one arm failed its independent audit")
    if (
        core["prepared_sha256"] != megatron["prepared_sha256"]
        or core["preparation_sha256"] != megatron["preparation_sha256"]
    ):
        errors.append("Arms were audited against different prepared data")
    curves = []
    for step in EVAL_STEPS:
        arms = []
        for report in (core, megatron):
            rows = [row for row in report["evaluation"] if row["completed_steps"] == step]
            arms.append(rows[0] if len(rows) == 1 else None)
        if any(row is None for row in arms):
            errors.append(f"Missing unique evaluation at completed step {step}")
            continue
        left, right = [{row["id"]: row for row in arm["samples"]} for arm in arms]
        if any(len(arm["samples"]) != len(index) for arm, index in zip(arms, (left, right), strict=True)):
            errors.append(f"Duplicate evaluation prompt IDs at completed step {step}")
            continue
        if left.keys() != right.keys():
            errors.append(f"Evaluation membership differs at completed step {step}")
            continue
        pairs = [
            dict(id=key, core_correct=left[key]["correct"], megatron_correct=right[key]["correct"])
            for key in sorted(left)
        ]
        cells = Counter((pair["core_correct"], pair["megatron_correct"]) for pair in pairs)
        curves.append(
            dict(
                completed_steps=step,
                core=arms[0]["summary"],
                megatron=arms[1]["summary"],
                core_minus_megatron_accuracy=arms[0]["summary"]["accuracy"] - arms[1]["summary"]["accuracy"],
                both_correct=cells[1, 1],
                core_only_correct=cells[1, 0],
                megatron_only_correct=cells[0, 1],
                neither_correct=cells[0, 0],
                pairs=pairs,
            )
        )
    transitions = {}
    for report in (core, megatron):
        transition, transition_errors = within_backend_transitions(report)
        errors.extend(transition_errors)
        if transition is not None:
            transitions[report["backend"]] = transition
    gains = {}
    if curves and curves[0]["completed_steps"] == 0 and curves[-1]["completed_steps"] == UPDATES:
        gains = {
            backend: curves[-1][backend]["accuracy"] - curves[0][backend]["accuracy"]
            for backend in ("core", "megatron")
        }
    return dict(
        schema_version=1,
        valid=not errors,
        errors=errors,
        learning_curves=curves,
        accuracy_gain_0_to_100=gains,
        within_backend_0_to_100=transitions,
        artifact_directories={
            "core": core.get("artifact_directory", "core"),
            "megatron": megatron.get("artifact_directory", "megatron"),
        },
        interpretation="Descriptive single-pair comparison; no statistical significance or learning-rate claim.",
    )


def _metric_dict(text):
    """Decode logged numeric dictionaries without executing Python representations."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Read individual values so an unrelated tensor/numpy repr cannot discard
        # otherwise ordinary numeric timer fields. Calls are never evaluated.
        expression = ast.parse(text, mode="eval").body
        if not isinstance(expression, ast.Dict):
            raise ValueError("Expected a metric dictionary") from None
        result = {}
        for key, value in zip(expression.keys, expression.values, strict=True):
            try:
                result[ast.literal_eval(key)] = ast.literal_eval(value)
            except (ValueError, TypeError):
                continue
        return result


def _duration_summary(points):
    values = [point["seconds"] for point in points]
    return dict(
        count=len(values),
        sum_seconds=sum(values),
        mean_seconds=sum(values) / len(values) if values else None,
        median_seconds=median(values) if values else None,
        min_seconds=min(values, default=None),
        max_seconds=max(values, default=None),
    )


def parse_timing_log(path, *, warmup_updates=5):
    """Extract indexed durations; initial eval/setup never enters warm phase means.

    Rollout and trainer `perf` dictionaries can share a rollout ID. Merge their
    distinct keys, deduplicate repeated log lines, and reject conflicting values.
    Unindexed rounded Timer lines are retained separately; never assign them a
    rollout ID by position, because partial logs would shift that assignment.
    """
    if warmup_updates < 0:
        raise ValueError("warmup_updates must be nonnegative")
    points, conflicts, warnings, timestamps, timers = {}, set(), [], [], defaultdict(list)
    completion_seconds = None
    generation_ends, boundary_conflicts = {}, set()
    for number, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw.replace(r"\u001b", "\x1b"))
        stamp = re.search(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2}(?:\.\d+)?)", line)
        if stamp:
            timestamps.append(datetime.fromisoformat(stamp.group(1) + "T" + stamp.group(2)).timestamp())
        timer = re.search(r"Timer ([a-z_]+) end \(elapsed: ([0-9.]+)s\)", line)
        if timer:
            timers[timer.group(1)].append(float(timer.group(2)))
        match = re.search(r"\bperf (\d+): (\{.*\})", line)
        core = re.search(r"Core optimizer step (\d+): (\{.*\})", line)
        publication = re.search(r"Core weight publication: (\{.*\})", line)
        completed = re.search(r"GSM8K_PARITY_CORE_COMPLETED (\{.*\})", line)
        found = match or core or publication or completed
        if found is None:
            continue
        try:
            data = _metric_dict(found.group(2) if match or core else found.group(1))
        except (SyntaxError, ValueError, TypeError):
            warnings.append(f"line {number}: could not decode metric dictionary")
            continue
        additions = []
        if match:
            index = int(match.group(1))
            # Use the manager's own clock, not a Beaker ingestion timestamp or
            # a trainer timestamp. Duplicate/restarted boundaries are ambiguous.
            native_stamp = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) rollout_manager\]", line)
            if "perf/rollout_time" in data and native_stamp:
                manager = re.search(r"\(RolloutManager pid=([^)]*)\)", line)
                boundary = (
                    datetime.fromisoformat(native_stamp.group(1)).timestamp(),
                    manager.group(1) if manager else None,
                )
                if index in generation_ends and generation_ends[index] != boundary:
                    boundary_conflicts.add(index)
                    warnings.append(f"line {number}: conflicting collection boundary at rollout {index}")
                else:
                    generation_ends[index] = boundary
            for field, phase in (
                ("perf/rollout_time", "generation"),
                ("perf/actor_train_time", "training"),
                ("perf/update_weights_time", "publication"),
                ("perf/log_probs_time", "scoring"),
            ):
                if field in data:
                    additions.append((phase, index, data[field], "miles_perf"))
        elif core and "train/step_seconds" in data:
            additions.append(("training", int(core.group(1)) - 1, data["train/step_seconds"], "core_step"))
        elif publication and not data.get("repeated_version", False):
            additions.append(("publication", data["version"], data["total_seconds"], "core_publication"))
        elif completed:
            completion_seconds = data.get("elapsed_seconds")
        for phase, index, seconds, source in additions:
            key = (phase, index)
            if not isinstance(seconds, float | int) or not math.isfinite(seconds) or seconds < 0:
                warnings.append(f"line {number}: invalid {phase} duration")
                continue
            point = dict(index=index, seconds=seconds, source=source)
            if key in points and points[key]["seconds"] != seconds:
                conflicts.add(key)
                warnings.append(f"line {number}: conflicting {phase} duration at index {index}")
            else:
                points[key] = point
    for key in conflicts:
        points.pop(key, None)
    for index in boundary_conflicts:
        generation_ends.pop(index, None)
    excluded_eval_intervals = []
    for index, (end, manager) in sorted(generation_ends.items()):
        if index - 1 not in generation_ends:
            continue  # A partial log is not evidence for any missing boundary.
        if index in EVAL_STEPS:
            excluded_eval_intervals.append([index - 1, index])
            continue
        start, previous_manager = generation_ends[index - 1]
        if end <= start or manager != previous_manager:
            warnings.append(f"rollouts {index - 1}->{index}: nonmonotonic clock or changed manager")
            continue
        points["collection_boundary_cycle", index - 1] = dict(
            index=index - 1, next_rollout=index, seconds=end - start, source="rollout_manager_native_timestamps"
        )
    phases = {}
    for phase in ("generation", "collection_boundary_cycle", "training", "publication", "scoring"):
        selected = [point for (name, _), point in sorted(points.items()) if name == phase]
        warm = [point for point in selected if point["index"] >= warmup_updates]
        phases[phase] = dict(all=_duration_summary(selected), warm=_duration_summary(warm), points=selected)
    return dict(
        log_sha256=digest(path),
        warnings=warnings,
        warmup_updates_excluded=warmup_updates,
        phases=phases,
        collection_boundaries_observed=len(generation_ends),
        excluded_eval_intervals=excluded_eval_intervals,
        diagnostic_phase_scopes={
            "training": "Core step_seconds and MILES actor_train both exclude pre-update scoring, but differ in "
            "instrumentation and rank reduction. Retained for diagnosis, not plotted as equivalent phases.",
            "publication": "Backend-specific actor publication scopes; retained for diagnosis.",
        },
        rounded_unindexed_timers={
            name: _duration_summary([{"seconds": x} for x in values]) for name, values in timers.items()
        },
        run_elapsed_seconds=completion_seconds,
        observed_log_span_seconds=max(timestamps) - min(timestamps) if timestamps else None,
        scope="Collection-boundary cycles run from generation-end N to generation-end N+1: "
        "update/publication N plus generation N+1 and orchestration. Missing boundaries are never inferred. "
        "Scheduled-eval crossings are excluded; final update99 is not covered by cycles. "
        "Generation includes rewards and rollout debug-dump time. "
        "Warm phase means exclude indices below warmup_updates and all eval/setup. "
        "Phase sums are not total runtime. Log span is only the observed timestamp span. "
        "Core training includes contract checks; MILES actor_train timing has different instrumentation. "
        "MILES publication perf index is the publication preceding that rollout (final publication may be absent).",
    )


def comparable_timing(timing):
    """Compare only common measured warm indices, never unlike trainer scopes."""
    result = {}
    for phase in ("generation", "collection_boundary_cycle"):
        arms = []
        for backend in ("core", "megatron"):
            report = timing.get(backend, {})
            warmup = report.get("warmup_updates_excluded", 5)
            points = report.get("phases", {}).get(phase, {}).get("points", [])
            arms.append({point["index"]: point for point in points if point["index"] >= warmup})
        indices = sorted(arms[0].keys() & arms[1].keys())
        result[phase] = {
            "indices": indices,
            **{
                backend: _duration_summary([arm[index] for index in indices])
                for backend, arm in zip(("core", "megatron"), arms, strict=True)
            },
        }
    return result


def plot_comparison(report, path):
    """Write an exportable figure; one pair supports descriptive curves only."""
    curves = report["learning_curves"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for backend, label in (("core", "OLMo-core"), ("megatron", "Megatron")):
        axes[0].plot(
            [row["completed_steps"] for row in curves],
            [row[backend]["accuracy"] for row in curves],
            marker="o",
            label=label,
        )
    axes[0].set(xlabel="Completed optimizer updates", ylabel="Held-out accuracy", ylim=(0, 1))
    axes[0].legend()
    timing = comparable_timing(report.get("timing", {}))
    phases = ("generation", "collection_boundary_cycle")
    for offset, backend in ((-0.18, "core"), (0.18, "megatron")):
        values = [timing[phase][backend]["mean_seconds"] for phase in phases]
        valid = [(index, value) for index, value in enumerate(values) if value is not None]
        axes[1].bar([index + offset for index, _ in valid], [value for _, value in valid], width=0.36, label=backend)
    axes[1].set(
        xticks=range(2),
        xticklabels=("Generation\n(includes rewards/dump)", "Collection-boundary\ncycle"),
        ylabel="Warm mean seconds (matched observed indices)",
        title="Operational cadence; bars are not additive",
    )
    axes[1].legend()
    figure.suptitle("One run per backend: descriptive learning and timing comparison")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("root", type=Path)
    audit_parser.add_argument("--backend", choices=("core", "megatron"), required=True)
    audit_parser.add_argument("--output", type=Path)
    audit_parser.add_argument("--megatron-directory", default="megatron")
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("root", type=Path)
    compare_parser.add_argument("--output", type=Path)
    compare_parser.add_argument("--megatron-directory", default="megatron")
    compare_parser.add_argument("--core-log", type=Path)
    compare_parser.add_argument("--megatron-log", type=Path)
    compare_parser.add_argument("--warmup-updates", type=int, default=5)
    compare_parser.add_argument("--plot", type=Path)
    compare_parser.add_argument("--core-allocated-seconds", type=float)
    compare_parser.add_argument("--megatron-allocated-seconds", type=float)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(args.root, args.backend, megatron_directory=args.megatron_directory)
        output = args.output or args.root / arm_directory(args.backend, args.megatron_directory) / "audit.json"
    else:
        result = compare(
            *[
                json.loads((args.root / arm_directory(backend, args.megatron_directory) / "audit.json").read_text())
                for backend in ("core", "megatron")
            ]
        )
        output = args.output or args.root / "comparison.json"
        result["timing"] = {
            backend: parse_timing_log(path, warmup_updates=args.warmup_updates)
            for backend, path in (("core", args.core_log), ("megatron", args.megatron_log))
            if path
        }
        result["comparable_timing"] = comparable_timing(result["timing"])
        result["allocated_runtime_seconds"] = {}
        for backend, seconds in (("core", args.core_allocated_seconds), ("megatron", args.megatron_allocated_seconds)):
            if seconds is not None:
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError("Allocated durations must be positive finite seconds from Beaker job metadata")
                result["allocated_runtime_seconds"][backend] = seconds
        if args.plot and result["valid"]:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            plot_comparison(result, args.plot)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"valid": result["valid"], "report": str(output)}))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
