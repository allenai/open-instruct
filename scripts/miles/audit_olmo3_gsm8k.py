"""Read-only audit of the synchronous dense Olmo 3 GSM8K qualification on WEKA."""

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from scripts.miles.audit_workflow import audit_samples, digest, prepared_rows, read_jsonl, require

from open_instruct.ground_truth_utils import GSM8KVerifier
from open_instruct.miles import checkpoint, workflow


def audit(root):
    root = Path(root)
    state = json.loads((root / "workflow.json").read_text())
    require(state["status"] == "complete", "Workflow did not complete")
    spec = json.loads((root / "run-spec.json").read_text())
    require(workflow.fingerprint(spec) == state["spec_sha256"], "Run identity changed")
    plan = json.loads((root / "resolved-plan.json").read_text())
    miles, core = plan["miles"], plan["core"]
    require(not miles.get("fully_async") and not miles.get("load"), "Expected fresh synchronous run")
    count = miles["num_rollout"]
    world = miles["actor_num_nodes"] * miles["actor_num_gpus_per_node"]
    require(
        miles["global_batch_size"] == miles["rollout_batch_size"] * miles["n_samples_per_prompt"],
        "Unexpected batch geometry",
    )
    manifest = json.loads((root / "prepared/data/manifest.json").read_text())
    for path, expected in manifest["inputs"].items():
        require(digest(path) == expected, f"Input changed: {path}")
    for name, expected in manifest["outputs"].items():
        require(digest(root / "prepared/data" / name) == expected, f"Prepared output changed: {name}")
    training = prepared_rows(miles["prompt_data"])
    evaluation = {}
    for path in miles["eval_prompt_data"][1::2]:
        rows = prepared_rows(path)
        require(not set(rows) & set(evaluation), "Duplicate evaluation identities")
        evaluation.update(rows)
    require(not set(training) & set(evaluation), "Training/evaluation overlap")
    require(
        not {r["input"] for r in training.values()} & {r["input"] for r in evaluation.values()}, "Prompt text overlap"
    )
    verifier = GSM8KVerifier()
    sample_reports, examples = {}, {}
    for label, rollout, version, rows, is_eval in [(str(i), i, i, training, False) for i in range(count)] + [
        ("eval_0", 0, 0, evaluation, True),
        (f"eval_{count - 1}", count - 1, count, evaluation, True),
    ]:
        payload = torch.load(
            miles["save_debug_rollout_data"].format(rollout_id=label), map_location="cpu", weights_only=True
        )
        require(payload["rollout_id"] == rollout, "Unexpected dump index")
        sample_reports[label] = audit_samples(
            payload["samples"],
            rows,
            rollout=version,
            version_limit=0,
            multiplicity=1 if is_eval else miles["n_samples_per_prompt"],
            group_count=len(rows) if is_eval else miles["rollout_batch_size"],
            consumed=set(),
            groups_seen=set(),
            response_cap=miles["rollout_max_response_len"],
            verifier=verifier,
            evaluation=is_eval,
        )
        examples[label] = [
            {
                "id": s["metadata"]["prepared_sample_id"],
                "reward": s["reward"],
                "response_tokens": s["response_length"],
                "response": s["response"],
            }
            for s in payload["samples"][:2]
        ]
    metrics = Path(miles["save"])
    optimizer = {}
    for rank in range(world):
        records = read_jsonl(metrics / f"training_contract_rank{rank}.jsonl")
        require(not any(r["event"] == "optimizer_rejected" for r in records), "Optimizer rejected")
        steps = [r for r in records if r["event"] == "optimizer"]
        require([r["step"] for r in steps] == list(range(1, count + 1)), "Missing optimizer step")
        require(all(not r["optimizer_skipped"] for r in steps), "Optimizer skipped")
        for i, step in enumerate(steps):
            require(step["rollout_id"] == i and step["published_step"] == i, "Wrong optimizer/policy boundary")
            require(step["local_behavior_versions"] == [i], "Wrong behavior policy version")
            require(step["normalization"]["samples"] == miles["global_batch_size"], "Wrong sample normalization")
            require(step["normalization"]["world_size"] == world, "Wrong distributed normalization")
        # Uniform-reward groups legitimately have zero GRPO gradients; retain them explicitly.
        optimizer[str(rank)] = steps
    path, saved = checkpoint.resume_manifest(metrics)
    checkpoint.validate_topology(saved, world, core["expert_parallel_size"])
    require(saved["clock"]["completed_steps"] == count, "Checkpoint misses completed updates")
    require(all((path / f"rank_{rank}.pt").is_file() for rank in range(world)), "Missing rank state")
    require((path / "model/.metadata").is_file(), "Missing distributed checkpoint metadata")
    exported = Path(spec["output"]["hf_dir"])
    hf = json.loads((exported / "config.json").read_text())
    require(hf["model_type"] == "olmo3" and hf["rope_scaling"]["rope_type"] == "yarn", "Wrong exported architecture")
    require(bool(list(exported.glob("*.safetensors"))), "Missing HF exported weights")
    publications = read_jsonl(metrics / "publication.jsonl")
    require(
        [r["version"] for r in publications if not r["repeated_version"]] == list(range(count + 1)),
        "Missing publication",
    )
    timings = read_jsonl(metrics / "driver_timing.jsonl")
    require(all(r["passed"] for r in timings), "Failed driver stage")
    return dict(
        passed=True,
        root=str(root),
        qualification="synchronous_dense_olmo3_gsm8k",
        samples=sample_reports,
        examples=examples,
        optimizer_by_rank=optimizer,
        checkpoint={"path": str(path), "clock": saved["clock"], "world_size": world},
        publications=publications,
        driver_timings=timings,
        mean_publication_seconds=mean(r["total_seconds"] for r in publications if not r["repeated_version"]),
        limitations=[
            "Two updates are a machinery qualification, not a learning comparison.",
            "Checkpoint topology, cursor and rank-state presence checked; fresh-process full-model optimizer resume is not covered by this audit.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    workflow.write_json(args.report, result)
    print(json.dumps({"passed": True, "reward": {k: v["mean_reward"] for k, v in result["samples"].items()}}))


if __name__ == "__main__":
    main()
