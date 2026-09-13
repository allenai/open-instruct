"""Read-only source inventory and retained-sample checks for readiness qualification."""

import argparse
import asyncio
import collections
import copy
import hashlib
import importlib
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

from scripts.miles import audit_workflow, prepare_colleague_exercises

from open_instruct.miles import general_judge, rewards, run_data, workflow

require = audit_workflow.require


def verify_sample(sample, row, *, version, max_lag, response_cap):
    metadata = sample["metadata"]
    require(sample["prompt"] == row["input"] and sample["label"] == row["label"], "Prompt/label changed")
    require(metadata["verifiers"] == row["metadata"]["verifiers"], "Verifier targets changed")
    length = sample["response_length"]
    require(0 < length <= response_cap and length < len(sample["tokens"]), "Invalid response boundary")
    prompt_tokens = sample["tokens"][:-length]
    expected = row["metadata"]["run_prompt_token_ids_sha256"]
    actual = hashlib.sha256((json.dumps(prompt_tokens, sort_keys=True) + "\n").encode()).hexdigest()
    require(actual == expected, "Prompt token identity changed")
    versions = {int(v) for v in sample["weight_versions"]}
    require(len(versions) == 1 and 0 <= version - next(iter(versions)) <= max_lag, "Invalid behavior policy")
    components = metadata["reward_components"]
    targets = metadata["verifiers"]
    require(len(components) == len(targets), "Missing reward components")
    for component, target in zip(components, targets, strict=True):
        require(component["name"] == target["name"], "Reward domain changed")
        require(component["weight"] == target.get("weight", 1.0), "Reward weight changed")
        require(math.isfinite(component["score"]), "Nonfinite component reward")
        name = component["name"]
        if name.startswith("general-"):
            detail = metadata["verifier_diagnostics"][name]
            require(detail["context_checked"] and detail["binding"]["judge"] == "general", "Wrong judge binding")
            require(detail["model"] == "Qwen/Qwen3-32B", "Wrong judge model")
            require(
                general_judge.parse_judge_response(detail["raw_reply"])[1] == component["score"], "Judge grade changed"
            )
    total = sum(c["score"] * c["weight"] for c in components)
    require(math.isfinite(total) and total == sample["reward"], "Reward accounting mismatch")
    return next(iter(versions))


def audit(root):
    root = Path(root)
    state = json.loads((root / "workflow.json").read_text())
    require(state["status"] == "complete", "Workflow incomplete")
    plan = json.loads((root / "resolved-plan.json").read_text())
    miles, core = plan["miles"], plan["core"]
    require(not miles.get("load"), "This audit covers fresh runs only")
    updates = miles["num_rollout"]
    batch = miles["rollout_batch_size"] * miles["n_samples_per_prompt"]
    optimizer_steps = batch // miles["global_batch_size"]
    require(batch % miles["global_batch_size"] == 0, "Incomplete optimizer batch")
    partitions = {"train": [Path(miles["prompt_data"])], "eval": [Path(p) for p in miles["eval_prompt_data"][1::2]]}
    prepared = {
        key: {r["metadata"]["prepared_sample_id"]: r for p in paths for r in audit_workflow.read_jsonl(p)}
        for key, paths in partitions.items()
    }
    require(not set(prepared["train"]) & set(prepared["eval"]), "Train/eval identity overlap")
    require(
        not {r["input"] for r in prepared["train"].values()} & {r["input"] for r in prepared["eval"].values()},
        "Train/eval text overlap",
    )
    counts, scores, diagnostics, mixed = (
        collections.Counter(),
        collections.defaultdict(collections.Counter),
        collections.Counter(),
        collections.Counter(),
    )
    examples, dump_files, seen_groups = [], [], set()
    labels = [(i, i * optimizer_steps, "train") for i in range(updates)]
    # Initial eval and final eval filenames follow the driver rollout clock.
    labels += [("eval_0", 0, "eval"), (f"eval_{updates - 1}", updates * optimizer_steps, "eval")]
    for label, version, split in labels:
        path = Path(miles["save_debug_rollout_data"].format(rollout_id=label))
        samples = audit_workflow.load_rollout(path)["samples"]
        require(
            len(samples)
            == (batch if split == "train" else len(prepared[split]) * miles.get("n_samples_per_eval_prompt", 1)),
            "Wrong sample count",
        )
        groups = collections.defaultdict(list)
        for sample in samples:
            key = sample["metadata"]["prepared_sample_id"]
            require(key in prepared[split], "Sample outside prepared split")
            behavior = verify_sample(
                sample,
                prepared[split][key],
                version=version,
                max_lag=core["max_policy_lag"] if split == "train" else 0,
                response_cap=miles["rollout_max_response_len"],
            )
            group = sample.get("group_index")
            groups[group].append((key, behavior, sample["reward"]))
            for c in sample["metadata"]["reward_components"]:
                name = c["name"]
                counts[f"{split}:{name}"] += 1
                scores[f"{split}:{name}"][str(c["score"])] += 1
                detail = sample["metadata"].get("verifier_diagnostics", {}).get(name, {})
                if name in ("code", "code_stdio"):
                    require(detail, "Missing code service outcome")
                    diagnostics[f"{split}:{name}:{detail.get('status')}"] += 1
                if split == "train" and sum(x["domain"] == name for x in examples) < 3:
                    examples.append(
                        {
                            "domain": name,
                            "update": version,
                            "response": sample["response"],
                            "reward": c["score"],
                            "diagnostics": detail,
                        }
                    )
        if split == "train":
            require(len(groups) == miles["rollout_batch_size"], "Wrong group count")
            require(not seen_groups & set(groups), "Consumed group repeated")
            seen_groups.update(groups)
            for group in groups.values():
                require(
                    len(group) == miles["n_samples_per_prompt"] and len({(k, v) for k, v, _ in group}) == 1,
                    "Group membership/version mismatch",
                )
                name = prepared[split][group[0][0]]["metadata"]["verifiers"][0]["name"]
                mixed[name] += int(len({reward for _, _, reward in group}) > 1)
        dump_files.append({"path": str(path), "sha256": audit_workflow.digest(path), "samples": len(samples)})
    return {
        "passed": True,
        "root": str(root),
        "counts": dict(counts),
        "score_counts": dict(scores),
        "code_outcomes": dict(diagnostics),
        "mixed_reward_groups": dict(mixed),
        "examples": examples,
        "dumps": dump_files,
        "limitations": [
            "Retained rewards/accounting checked; deterministic math/IF/code rewards have not been independently re-executed.",
            "Judge grades reparsed from retained replies, without new stochastic model requests.",
            "No optimizer/checkpoint numerical-equivalence claim from this sample audit.",
        ],
    }


def inspect(model):
    model = Path(model)
    config = json.loads((model / "config.json").read_text())
    tokenizer = run_data._tokenizer(model)
    manifest, partitions, inputs = prepare_colleague_exercises.source_rows()
    keys = manifest["miles"]
    inventory, candidates = {}, []
    for split, rows in partitions.items():
        buckets, maxima = collections.Counter(), collections.Counter()
        for index, source in enumerate(rows):
            metadata = source[keys["metadata_key"]]
            domain = metadata["verifiers"][0]["name"]
            messages = run_data._messages({"messages": source[keys["input_key"]]}, strip_answer=False)
            prompt = run_data._render(messages, tokenizer, tokenizer.chat_template)
            length = len(tokenizer.encode(prompt, add_special_tokens=False))
            bucket = next((str(n) for n in (2048, 4096, 8192, 12288, 16384) if length <= n), "over16384")
            buckets[f"{domain}:le{bucket}"] += 1
            maxima[domain] = max(maxima[domain], length)
            if 4096 < length <= 12288 and domain in ("math", "ifeval", "code", "code_stdio"):
                candidates.append(
                    {
                        "split": split,
                        "index": index,
                        "domain": domain,
                        "tokens": length,
                        "identity": metadata.get("prepared_sample_id"),
                    }
                )
        inventory[split] = {"rows": len(rows), "length_buckets": dict(buckets), "max_prompt_tokens": dict(maxima)}
    return {
        "passed": True,
        "model": str(model),
        "config": config,
        "sources": inputs,
        "inventory": inventory,
        "long_candidates": candidates,
        "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    }


def check_optimizer_sequence(events, updates):
    optimizers = [event for event in events if event.get("event") == "optimizer"]
    require([event["step"] for event in optimizers] == list(range(1, updates + 1)), "Optimizer step discontinuity")
    require([event["rollout_id"] for event in optimizers] == list(range(updates)), "Rollout discontinuity")
    require(all(not event["optimizer_skipped"] for event in optimizers), "Skipped optimizer step")
    require(
        all(event["lr_used"] and all(lr > 0 for lr in event["lr_used"]) for event in optimizers),
        "Zero learning rate in resume exercise",
    )
    return optimizers


def lifecycle(root):
    """Audit our four-update, save-every-update, two-process readiness cases."""
    root = Path(root)
    state = json.loads((root / "workflow.json").read_text())
    require(state["status"] == "complete", "Resume workflow incomplete")
    require(state["result"]["start_rollout_id"] == 2, "No fresh-process resume from update two")
    require(state["result"]["completed_rollout_ids"] == [2, 3], "Wrong resumed horizon")
    plan = json.loads((root / "resolved-plan.json").read_text())
    require(plan["miles"].get("load"), "Resume plan did not load a checkpoint")
    directory = root / "checkpoints"
    latest = json.loads((directory / "core-latest.json").read_text())
    require(latest["rollout_id"] == 3, "Final native checkpoint not committed")
    boundaries = []
    for rollout in range(4):
        path = directory / "core" / f"rollout_{rollout:07d}"
        manifest = json.loads((path / "complete.json").read_text())
        clock = manifest["clock"]
        require(
            clock["completed_steps"] == rollout + 1 and clock["next_rollout_id"] == rollout + 1, "Wrong saved clock"
        )
        cursor = directory / "rollout" / f"global_dataset_state_dict_{rollout}.pt"
        require(audit_workflow.digest(cursor) == manifest["cursor_sha256"], "Saved cursor integrity failure")
        # Read descriptors, not many gigabytes of optimizer tensors. Native restore
        # itself is exercised by the resumed GPU process and must also pass.
        checkpoint = importlib.import_module("olmo_core.distributed.checkpoint")
        metadata = checkpoint.get_checkpoint_metadata(path / "model")
        names = list(metadata.state_dict_metadata)
        require(
            any("optim" in name or name.endswith(".exp_avg") for name in names), "Native save lacks optimizer state"
        )
        require(
            all((path / f"rank_{rank}.pt").is_file() for rank in range(manifest["world_size"])), "Missing rank state"
        )
        boundaries.append(
            {
                "rollout_id": rollout,
                "clock": clock,
                "cursor_sha256": manifest["cursor_sha256"],
                "native_entries": len(names),
            }
        )
    rank_events = {}
    for rank in range(manifest["world_size"]):
        events = audit_workflow.read_jsonl(directory / f"training_contract_rank{rank}.jsonl")
        optimizers = check_optimizer_sequence(events, 4)
        checks = [event for event in events if event.get("event") == "scoring_check"]
        require({0, 2} <= {event["rollout_id"] for event in checks}, "Missing first-forward check after restart")
        require(all(event["max_abs"] <= event["tolerance"] for event in checks), "Scoring check failed")
        rank_events[str(rank)] = {"steps": [event["step"] for event in optimizers], "scoring_checks": checks}
    publications = audit_workflow.read_jsonl(directory / "publication.jsonl")
    require([event["version"] for event in publications] == [0, 1, 2, 2, 3, 4], "Publication/republish discontinuity")
    return {
        "passed": True,
        "root": str(root),
        "boundaries": boundaries,
        "ranks": rank_events,
        "publication_versions": [event["version"] for event in publications],
        "limitations": [
            "Restore lifecycle and saved optimizer descriptors checked; not an uninterrupted-versus-resumed numerical-equivalence experiment."
        ],
    }


def rescore(root):
    """Re-execute low/high retained rewards for each deterministic domain."""
    root = Path(root)
    plan = json.loads((root / "resolved-plan.json").read_text())
    domains = {"math", "ifeval", "code", "code_stdio"}
    candidates = collections.defaultdict(list)
    for rollout in range(plan["miles"]["num_rollout"]):
        path = Path(plan["miles"]["save_debug_rollout_data"].format(rollout_id=rollout))
        for sample in audit_workflow.load_rollout(path)["samples"]:
            names = [entry["name"] for entry in sample["metadata"]["verifiers"]]
            if len(names) == 1 and names[0] in domains:
                candidates[names[0]].append((rollout, sample))
    require(set(candidates) == domains, "Missing deterministic verifier domain")
    registry = json.loads(Path(plan["core"]["reward_config"]).read_text())
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "verifiers.json"
        path.write_text(json.dumps({name: registry[name] for name in sorted(domains)}))
        args = SimpleNamespace(olmo_core=SimpleNamespace(reward_config=str(path)))

        async def execute():
            results = []
            for domain, items in sorted(candidates.items()):
                ordered = sorted(items, key=lambda item: item[1]["reward"])
                for rollout, sample in (ordered[0], ordered[-1]):
                    duplicate = SimpleNamespace(**copy.deepcopy(sample))
                    score = await asyncio.wait_for(rewards._score(args, duplicate), timeout=180)
                    results.append(
                        {
                            "domain": domain,
                            "rollout": rollout,
                            "id": sample["metadata"]["prepared_sample_id"],
                            "response_sha256": hashlib.sha256(sample["response"].encode()).hexdigest(),
                            "recorded": sample["reward"],
                            "rescored": score,
                            "matched": math.isclose(score, sample["reward"], rel_tol=0, abs_tol=1e-10),
                            "diagnostics": duplicate.metadata.get("verifier_diagnostics", {}),
                        }
                    )
            return results

        results = asyncio.run(execute())
    return {
        "passed": all(row["matched"] for row in results),
        "root": str(root),
        "results": results,
        "limits": "Two retained training responses per deterministic domain, selected at reward extremes. Fresh verifier/service execution; no new judge calls or optimizer-equivalence claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "audit", "lifecycle", "rescore"))
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failed = False
    for index, path in enumerate(args.paths):
        try:
            report = {"inspect": inspect, "audit": audit, "lifecycle": lifecycle, "rescore": rescore}[args.mode](path)
        except Exception as error:
            report = {"passed": False, "path": str(path), "error": f"{type(error).__name__}: {error}"}
            failed = True
        failed = failed or not report["passed"]
        workflow.write_json(args.output / f"{args.mode}-{index}.json", report)
        print(json.dumps({"path": str(path), "passed": report["passed"], "error": report.get("error")}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
