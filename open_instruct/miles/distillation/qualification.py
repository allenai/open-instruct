"""Freeze a screened cohort and prepare a bounded cancellation learning check."""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

from tokenizers import Tokenizer

from open_instruct.miles.configuration.run_spec import RunSpec


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def select_cohort(rows, checkpoint, train_count, seed):
    if not 0 < train_count < len(rows):
        raise ValueError("Require nonempty training and disjoint held-out cohorts")
    identities, questions = set(), set()
    for row in rows:
        gate = row["gate"]
        if row["checkpoint"] != checkpoint or gate["weight_version"] != 0:
            raise ValueError("Screen must come from unchanged H045 at weight version zero")
        responses = gate["initial_responses"]
        if len(responses) != 8 or any(r["reward"] != 0 for r in responses):
            raise ValueError("Qualification input requires a recorded zero-of-eight screen")
        identity = row["sample_id"]
        question = row["row"]["metadata"]["query"]
        if identity in identities or question in questions:
            raise ValueError("Duplicate question or source identity in screened pool")
        identities.add(identity)
        questions.add(question)
    ordered = sorted(rows, key=lambda r: r["sample_id"])
    random.Random(seed).shuffle(ordered)
    return ordered[:train_count], ordered[train_count:]


def render_row(row, template, tokenizer, training):
    query = row["row"]["metadata"]["query"]
    original = template["metadata"]["query"]
    if template["input"].count(original) != 1:
        raise ValueError("Template must contain exactly one original question")
    bare = template["input"].replace(original, query)
    metadata = copy.deepcopy(row["row"]["metadata"])
    reference = "\n\n".join(metadata.pop("bread")["sections"])
    if not reference.strip():
        raise ValueError("A full prepared R1 reference is required")
    prompt_ids = tokenizer.encode(bare, add_special_tokens=False).ids
    if len(prompt_ids) > 2048:
        raise ValueError("Bare prompt exceeds the qualification prompt budget")
    metadata.update(
        prompt_token_ids=prompt_ids,
        opd_screen_failure_kind=row["gate"]["failure_kind"],
        opd_screen_sha256=digest(json.dumps(row["gate"], sort_keys=True).encode()),
    )
    if training:
        contexts = {
            "opd_teacher_prompt": reference,
            "opd_answer_prompt": f"The final answer is {row['row']['ground_truth']}.",
        }
        for key, text in contexts.items():
            notes = (
                query
                + "\n\nReference material:\n<reference_solution>\n"
                + text
                + "\n</reference_solution>\nUse it to help solve the problem. "
                "Continue the current reasoning coherently; another valid approach is allowed."
            )
            prompt = template["input"].replace(original, notes)
            context_tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            if context_tokens + 16384 > 65528:
                raise ValueError("Full reference plus response exceeds teacher context budget")
            metadata[key] = prompt
            metadata[key + "_tokens"] = context_tokens
        metadata.update(
            opd_reference_source="prepared_openr1_r1_sections",
            opd_reference_sha256=digest(reference.encode()),
            opd_reference_index=row["row"]["metadata"]["bread"]["reference_index"],
        )
    return {"input": bare, "label": row["row"]["ground_truth"], "metadata": metadata}


def prepare(pool, template_data, tokenizer_path, base_run, output, train_count=40, seed=120045):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; qualification inputs are immutable")
    output.mkdir(parents=True, exist_ok=True)
    document = json.loads(base_run.read_text())
    rows = [json.loads(line) for line in pool.read_text().split("\n") if line.strip()]
    train, heldout = select_cohort(rows, document["model"]["source"], train_count, seed)
    template = json.loads(next(template_data.open()))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    partitions = {
        "train": [render_row(r, template, tokenizer, True) for r in train],
        "train_diagnostic": [render_row(r, template, tokenizer, False) for r in train],
        "heldout": [render_row(r, template, tokenizer, False) for r in heldout],
    }
    data = output / "data"
    name = "h045-opd-cancellation-qualification"
    document["name"] = name
    document["output"]["root"] = "/weka/olmo-3p5-checkpoints/scratch/robertb/open-instruct/runs/" + name
    document["data"] = {
        "prompt_data": str(data / "train.jsonl"),
        "reward_config": str(data / "verifiers.json"),
        "eval_prompt_data": [
            "train_diagnostic",
            str(data / "train_diagnostic.jsonl"),
            "heldout",
            str(data / "heldout.jsonl"),
        ],
        "in_sample_eval": ["train_diagnostic"],
        "seed": seed,
        "shuffle": True,
    }
    document["training"].update(num_rollouts=16, eval_interval=16, save_interval=4, skip_eval_before_train=False)
    document["inference"].update(
        sglang_server_concurrency=8, sglang_max_running_requests=8, sglang_cuda_graph_max_bs_decode=8
    )
    document["launch"].update(timeout="2h", priority="high", min_runtime=0, auto_resume=False)
    document["optimizer"].update(learning_rate=1e-6, lr_warmup_iters=0, weight_decay=0.0)
    document["teacher"].update(
        objective="cancel_answer_positive", pressure_clip=None, kl_coef=1.0, calibrate_initial=True
    )
    document["core"].update(max_run_seconds=6600, checkpoint_keep_every=4)
    document["miles"].update(rollout_seed=seed, seed=seed, n_samples_per_eval_prompt=4)
    document["tracking"].update(wandb_group=name)
    # Validate the actual review recipe before recording it.
    RunSpec.from_dict(document, config_path=output / "run-local.json").plan()
    data.mkdir()
    for split, values in partitions.items():
        (data / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values))
    (data / "verifiers.json").write_text(
        json.dumps(
            {"openr1_math": {"factory": "open_instruct.miles.rewards.openr1_verifier.OpenR1MathVerifier"}}, indent=2
        )
        + "\n"
    )
    (output / "run-local.json").write_text(json.dumps(document, indent=2) + "\n")
    launch = copy.deepcopy(document)
    launch["data"]["prompt_data"] = "/opt/h045-qualification-data/train.jsonl"
    launch["data"]["reward_config"] = "/opt/h045-qualification-data/verifiers.json"
    launch["data"]["eval_prompt_data"] = [
        "train_diagnostic",
        "/opt/h045-qualification-data/train_diagnostic.jsonl",
        "heldout",
        "/opt/h045-qualification-data/heldout.jsonl",
    ]
    (output / "run.json").write_text(json.dumps(launch, indent=2) + "\n")
    manifest = {
        "seed": seed,
        "training_checkpoint": document["model"]["source"],
        "frozen_teacher_checkpoint": document["teacher"]["source"],
        "reference_source": "Full prepared OpenR1 R1 sections; not the hybrid BREAD success",
        "cohorts": {
            name: [r["metadata"]["prepared_sample_id"] for r in values] for name, values in partitions.items()
        },
        "screen_is_not_baseline": True,
        "inputs": {str(p): digest(p.read_bytes()) for p in (pool, template_data, tokenizer_path, base_run)},
        "outputs": {p.name: digest(p.read_bytes()) for p in data.iterdir()},
        "recipes": {p.name: digest(p.read_bytes()) for p in (output / "run.json", output / "run-local.json")},
        "training_context_tokens": [r["metadata"]["opd_teacher_prompt_tokens"] for r in partitions["train"]],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "template-data", "tokenizer", "base-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=120045)
    args = parser.parse_args()
    result = prepare(
        args.pool, args.template_data, args.tokenizer, args.base_run, args.output, args.train_count, args.seed
    )
    print(json.dumps({"cohorts": {k: len(v) for k, v in result["cohorts"].items()}}))


if __name__ == "__main__":
    main()
