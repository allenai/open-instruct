"""Independently audit a bounded, fresh GSM8K researcher-workflow execution on CPU.

No GPU or Ray imports. Full audits require torch for restricted loading of the
workflow's retained .pt samples and open-instruct's GSM8K verifier. This checks
one optimizer step per collection, standard rank-strided distribution, and
per-step diagnostics; it is not a general restart or multi-step auditor.
"""

import argparse
import hashlib
import importlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_counters(
    contracts, publications, stages, *, updates, batch_size, world, diagnostic_interval=1, equality=True, max_lag=1
):
    """Check exact clocks and full-rank coverage, rather than relying on process exit."""
    require(diagnostic_interval == 1, "This bounded auditor requires per-step diagnostics")
    expected_steps = list(range(1, updates + 1))
    result = {}
    for rank in range(world):
        rows = contracts[str(rank)]
        require(not any(row.get("event") == "optimizer_rejected" for row in rows), "Rejected optimizer event")
        steps = [row for row in rows if row.get("event") == "optimizer"]
        require([row.get("step") for row in steps] == expected_steps, f"Rank {rank}: missing/repeated optimizer steps")
        for index, step in enumerate(steps):
            require(step.get("rank") == rank and step.get("rollout_id") == index, "Rank/rollout identity mismatch")
            require(step.get("optimizer_skipped") is False, "Skipped optimizer step")
            normal = step["normalization"]
            require(
                normal["samples"] == batch_size and normal["world_size"] == world, "Wrong normalization batch/world"
            )
            require(step["local_microbatches"] == batch_size // world, "Wrong rank microbatch count")
            require(step["published_step"] == index, "Optimizer consumed an unexpected publication boundary")
            versions = step["local_behavior_versions"]
            require(
                versions and all(type(version) is int and 0 <= index - version <= max_lag for version in versions),
                "Trainer behavior versions exceed lag bound",
            )
            require(finite(step["local_policy_objective"]), "Nonfinite policy objective")
            require(
                all(finite(value) for value in step["local_auxiliary_objective"].values()),
                "Nonfinite auxiliary objective",
            )
            for field, norm in (
                ("local_pre_optimizer_gradients", "local_l2"),
                ("sampled_model_updates", "sampled_update_l2"),
            ):
                groups = step.get(field)
                require(isinstance(groups, dict) and bool(groups), f"Missing {field}")
                require(
                    all(finite(group.get(norm)) and group[norm] >= 0 for group in groups.values()),
                    f"Nonfinite {field}",
                )
                require(sum(group[norm] for group in groups.values()) > 0, f"No nonzero {field}")
                if field == "local_pre_optimizer_gradients":
                    require(
                        all(group.get("missing_parameters") == 0 for group in groups.values()),
                        "Missing parameter gradients",
                    )
            require(all(finite(lr) and lr > 0 for lr in step["lr_used"]), "Invalid learning rate")
        result[str(rank)] = steps
    expected_publications = [(0, False)]
    for version in expected_steps:
        expected_publications.append((version, False))
        if equality:
            expected_publications.append((version, True))
    require(
        [(row.get("version"), row.get("repeated_version")) for row in publications] == expected_publications,
        "Publication/diagnostic sequence mismatch",
    )
    require(
        stages
        and all(row.get("passed") is True and finite(row.get("seconds")) and row["seconds"] >= 0 for row in stages),
        "Missing or failed driver stage",
    )
    cycles = []
    for rollout in range(updates):
        cycle = [
            row
            for row in stages
            if row.get("rollout_id") == rollout and row.get("stage") in ("generation_wait", "training", "publication")
        ]
        require(
            Counter(row["stage"] for row in cycle) == Counter({"generation_wait": 1, "training": 1, "publication": 1}),
            "Incomplete/repeated driver cycle",
        )
        cycles.append(sum(row["seconds"] for row in cycle))
    evaluation = [row for row in stages if row.get("stage") == "evaluation"]
    require(
        any(row.get("details", {}).get("phase") == "initial" and row["rollout_id"] == 0 for row in evaluation),
        "Missing initial blocking evaluation timing",
    )
    require(
        any(
            row.get("details", {}).get("phase") == "periodic" and row["rollout_id"] == updates - 1
            for row in evaluation
        ),
        "Missing final blocking evaluation timing",
    )
    return {
        "optimizer_by_rank": result,
        "cycle_seconds": cycles,
        "mean_cycle_seconds": mean(cycles),
        "evaluation_timings": evaluation,
        "publication_count": len(publications),
    }


def prepared_rows(path):
    rows = read_jsonl(path)
    identities = [row["metadata"]["prepared_sample_id"] for row in rows]
    require(rows and len(identities) == len(set(identities)), "Prepared IDs must be unique and nonempty")
    return dict(zip(identities, rows, strict=True))


def audit_samples(
    samples,
    prepared,
    *,
    rollout,
    version_limit,
    multiplicity,
    group_count,
    consumed,
    groups_seen,
    response_cap,
    verifier,
    evaluation=False,
):
    """Verify membership, exact prompt tokens, rewards and per-prompt policy purity."""
    counts, versions, groups = Counter(), {}, {}
    records = []
    for sample in samples:
        key = sample["metadata"]["prepared_sample_id"]
        require(key in prepared, "Unknown prepared prompt")
        row = prepared[key]
        require(sample["prompt"] == row["input"] and sample["label"] == row["label"], "Prepared prompt/label mismatch")
        require(
            sample["metadata"].get("verifiers") == row["metadata"].get("verifiers"), "Verifier target/weight mismatch"
        )
        group = key if evaluation else sample.get("group_index")
        require(
            (evaluation or (type(group) is int and group >= 0)) and group not in groups_seen,
            "Missing/repeated group identity",
        )
        require(group not in groups or groups[group] == key, "Merged prompt identities")
        groups[group] = key
        counts[group] += 1
        raw = sample.get("weight_versions")
        require(isinstance(raw, list) and raw, "Missing behavior policy version")
        require(
            all(
                (type(value) is int and value >= 0) or (isinstance(value, str) and value.isascii() and value.isdigit())
                for value in raw
            ),
            "Invalid behavior policy version",
        )
        values = {int(value) for value in raw}
        require(len(values) == 1, "Response mixes policy versions")
        version = next(iter(values))
        require(0 <= rollout - version <= version_limit, "Sample exceeds policy-lag budget")
        require(group not in versions or versions[group] == version, "Prompt group mixes policy versions")
        versions[group] = version
        length = sample["response_length"]
        tokens = sample["tokens"]
        require(type(length) is int and 0 < length <= response_cap and length < len(tokens), "Invalid response length")
        prompt_tokens = [int(value) for value in tokens[:-length]]
        token_hash = hashlib.sha256((json.dumps(prompt_tokens, sort_keys=True) + "\n").encode()).hexdigest()
        require(
            token_hash == row["metadata"]["run_prompt_token_ids_sha256"],
            "Prompt token identity differs from immutable preparation",
        )
        logprobs = sample.get("rollout_log_probs")
        require(
            logprobs is not None and len(logprobs) == length and all(finite(value) for value in logprobs),
            "Missing/nonfinite response logprobs",
        )
        require(
            sample.get("status") in ("completed", "truncated") and not sample.get("remove_sample", False),
            "Unsuccessful generation",
        )
        specs = row["metadata"]["verifiers"]
        require(all(spec["name"] == "gsm8k" for spec in specs), "This bounded audit only independently rescores GSM8K")
        score = sum(spec.get("weight", 1.0) * verifier([], sample["response"], spec["target"]).score for spec in specs)
        require(
            finite(sample.get("reward")) and score == sample["reward"],
            "Stored reward differs from direct GSM8K verification",
        )
        records.append(
            {
                "id": key,
                "group": group,
                "version": version,
                "reward": score,
                "response_tokens": length,
                "at_cap": length == response_cap,
                "response_sha256": hashlib.sha256(sample["response"].encode()).hexdigest(),
            }
        )
    require(
        len(counts) == group_count and set(counts.values()) == {multiplicity},
        "Incorrect prompt count or response multiplicity",
    )
    require(len(groups) == group_count, "Incorrect distinct prompt groups")
    if evaluation:
        require(set(groups.values()) == set(prepared), "Evaluation membership differs from heldout set")
    return {
        "samples": len(samples),
        "groups": sorted(groups),
        "ids": sorted(set(groups.values())),
        "versions": versions,
        "mean_reward": mean(row["reward"] for row in records),
        "response_tokens": sum(row["response_tokens"] for row in records),
        "at_cap": sum(row["at_cap"] for row in records),
        "records": records,
    }


def audit(root, *, counters_only=False):
    root = Path(root)
    spec = json.loads((root / "run-spec.json").read_text())
    plan = json.loads((root / "resolved-plan.json").read_text())
    state = json.loads((root / "workflow.json").read_text())
    require(state.get("status") == "complete", "Researcher workflow did not complete")
    require(
        state["spec_sha256"] == hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "Run-spec identity does not match workflow",
    )
    miles, core = plan["miles"], plan["core"]
    original_root = Path(spec["output"]["root"])

    def artifact(path):
        path = Path(path)
        return root / path.relative_to(original_root) if path.is_relative_to(original_root) else path

    updates = miles["num_rollout"]
    world = miles["actor_num_nodes"] * miles["actor_num_gpus_per_node"]
    batch = miles["global_batch_size"]
    require(
        updates > 1 and batch == miles["rollout_batch_size"] * miles["n_samples_per_prompt"],
        "Audit requires multiple collections with one optimizer step per collection",
    )
    require(
        not miles.get("load")
        and not miles.get("balance_data")
        and not miles.get("custom_convert_samples_to_train_data_path"),
        "Audit requires a fresh run with standard rank-strided distribution",
    )
    require(
        miles.get("fully_async") and miles.get("use_tis") and not miles.get("use_rollout_logprobs"),
        "Expected bounded async trainer-scoring/TIS objective",
    )
    manifest_path = root / "prepared/data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["contract"]["data"] == spec["data"], "Prepared data contract differs from submitted run")
    unavailable_sources = []
    for path, expected in manifest["inputs"].items():
        source = artifact(path)
        if counters_only and not source.is_file():
            unavailable_sources.append(path)
            continue
        require(digest(source) == expected, f"Prepared source changed: {path}")
    for filename, expected in manifest["outputs"].items():
        require(digest(manifest_path.parent / filename) == expected, f"Prepared output changed: {filename}")
    metrics = artifact(miles["save"])
    contracts = {str(rank): read_jsonl(metrics / f"training_contract_rank{rank}.jsonl") for rank in range(world)}
    report = validate_counters(
        contracts,
        read_jsonl(metrics / "publication.jsonl"),
        read_jsonl(metrics / "driver_timing.jsonl"),
        updates=updates,
        batch_size=batch,
        world=world,
        diagnostic_interval=core["diagnostic_interval"],
        equality=miles.get("check_weight_update_equal", False),
        max_lag=core["max_policy_lag"],
    )
    report.update(
        root=str(root),
        configured_updates=updates,
        samples_per_collection=batch,
        groups_per_collection=miles["rollout_batch_size"],
        samples_per_prompt=miles["n_samples_per_prompt"],
        max_policy_lag=core["max_policy_lag"],
        preparation_sha256=digest(manifest_path),
        run_spec_sha256=digest(root / "run-spec.json"),
        resolved_plan_sha256=digest(root / "resolved-plan.json"),
        full_sample_audit=not counters_only,
        limitations=[
            "One fresh run, one optimizer step per collection, standard rank-strided partitioning; no checkpoint/restart, fault recovery, or learning-quality qualification.",
            "Cycle timers exclude startup/evaluation; async generation_wait is consumer stall, not total inference work.",
        ],
    )
    if counters_only:
        report.update(passed=True, qualification="counters_only", unavailable_prepared_sources=unavailable_sources)
        report["limitations"].append("Samples, token identity, rewards and heldout membership were not audited.")
        return report
    torch = importlib.import_module("torch")
    verifier = importlib.import_module("open_instruct.ground_truth_utils").GSM8KVerifier()
    training = prepared_rows(artifact(miles["prompt_data"]))
    evaluation = {}
    for _name, path in zip(miles["eval_prompt_data"][::2], miles["eval_prompt_data"][1::2], strict=True):
        rows = prepared_rows(artifact(path))
        require(not set(evaluation) & set(rows), "Duplicate heldout IDs across eval sources")
        evaluation.update(rows)
    require(evaluation and not set(training) & set(evaluation), "Heldout IDs overlap training")
    require(
        not {row["input"] for row in training.values()} & {row["input"] for row in evaluation.values()},
        "Heldout prompt text overlaps training",
    )
    consumed, groups_seen, reports = set(), set(), []
    dump_paths = []
    for rollout in range(updates):
        path = artifact(miles["save_debug_rollout_data"].format(rollout_id=rollout))
        payload = torch.load(path, map_location="cpu", weights_only=True)
        require(payload.get("rollout_id") == rollout, "Training dump rollout ID mismatch")
        result = audit_samples(
            payload["samples"],
            training,
            rollout=rollout,
            version_limit=core["max_policy_lag"],
            multiplicity=miles["n_samples_per_prompt"],
            group_count=miles["rollout_batch_size"],
            consumed=consumed,
            groups_seen=groups_seen,
            response_cap=miles["rollout_max_response_len"],
            verifier=verifier,
        )
        for rank in range(world):
            versions = sorted(
                {result["versions"][sample["group_index"]] for sample in payload["samples"][rank::world]}
            )
            step = report["optimizer_by_rank"][str(rank)][rollout]
            require(
                versions == step["local_behavior_versions"],
                "Trainer/rank sample versions differ from retained rollout",
            )
            require(
                step["normalization"]["model_tokens"] == sum(len(sample["tokens"]) for sample in payload["samples"]),
                "Trainer model-token normalization differs from retained rollout",
            )
        consumed.update(result["ids"])
        groups_seen.update(result["groups"])
        result.update(rollout_id=rollout, sha256=digest(path))
        dump_paths.append(path)
        reports.append(result)
    eval_reports = []
    for dump_id, policy_version in ((0, 0), (updates - 1, updates)):
        path = artifact(miles["save_debug_rollout_data"].format(rollout_id=f"eval_{dump_id}"))
        payload = torch.load(path, map_location="cpu", weights_only=True)
        require(payload.get("rollout_id") == dump_id, "Evaluation dump rollout ID mismatch")
        result = audit_samples(
            payload["samples"],
            evaluation,
            rollout=policy_version,
            version_limit=0,
            multiplicity=miles.get("n_samples_per_eval_prompt", 1),
            group_count=len(evaluation),
            consumed=set(),
            groups_seen=set(),
            response_cap=miles.get("eval_max_response_len", miles["rollout_max_response_len"]),
            verifier=verifier,
            evaluation=True,
        )
        result.update(policy_version=policy_version, sha256=digest(path))
        dump_paths.append(path)
        eval_reports.append(result)
    report.update(
        passed=True,
        qualification="bounded_workflow_gsm8k",
        training=reports,
        evaluation=eval_reports,
        consumed_samples=sum(row["samples"] for row in reports),
        unique_consumed_prompts=len(consumed),
        consumed_response_tokens=sum(row["response_tokens"] for row in reports),
        dump_files=[str(path) for path in dump_paths],
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--counters-only", action="store_true")
    args = parser.parse_args()
    try:
        report = audit(args.root, counters_only=args.counters_only)
    except Exception as error:
        report = {"passed": False, "root": str(args.root), "error": f"{type(error).__name__}: {error}"}
    report["auditor_sha256"] = digest(__file__)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.report), "error": report.get("error")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
