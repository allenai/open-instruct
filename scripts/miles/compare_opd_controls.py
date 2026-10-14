"""Paired-question factorial contrasts for the fixed-cohort OPD experiment.

Intervals describe evaluation-question variation for these four trained models;
they do not establish reproducibility across training seeds or selection traces.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.miles import compare_opd_repeated

ARMS = ("admitted-p1", "admitted-p4", "selected-p1", "selected-p4")
CONTRASTS = {
    "selection_at_age_zero": (-1, 0, 1, 0),
    "selection_with_stale_policy": (0, -1, 0, 1),
    "staleness_on_admitted_prompts": (-1, 1, 0, 0),
    "staleness_on_selected_prompts": (0, 0, -1, 1),
    "selection_mean_over_age_settings": (-0.5, -0.5, 0.5, 0.5),
    "staleness_mean_over_prompt_settings": (-0.5, 0.5, -0.5, 0.5),
    "selection_by_staleness_interaction": (1, -1, -1, 1),
}


def summarize(arms):
    if set(arms) != set(ARMS):
        raise ValueError("All four prespecified control arms are required")
    reference = arms[ARMS[0]]
    if not reference or any(rows.keys() != reference.keys() for rows in arms.values()):
        raise ValueError("Exactly matched complete question/repetition panels are required")
    groups = defaultdict(lambda: defaultdict(list))
    for key, first in sorted(reference.items()):
        paired = [arms[name][key] for name in ARMS]
        for row in paired:
            if any(row[field] != first[field] for field in ("prompt", "label", "input_ids", "seed")):
                raise ValueError("Evaluation inputs or seeds differ between arms")
            if row["status"] not in ("completed", "truncated") or not math.isfinite(row["reward"]):
                raise ValueError("Invalid or unfinished evaluation response")
        groups[(key[0], key[2])][key[1]].append([int(row["reward"] > 0) for row in paired])
    result = {}
    for (dataset, mode), questions in groups.items():
        sizes = {len(values) for values in questions.values()}
        if len(sizes) != 1:
            raise ValueError("Unequal repetitions across questions")
        repetitions = sizes.pop()
        question_means = {
            key: [statistics.mean(row[i] for row in values) for i in range(len(ARMS))]
            for key, values in questions.items()
        }
        contrasts = {}
        for name, coefficients in CONTRASTS.items():
            differences = {
                key: sum(c * x for c, x in zip(coefficients, means, strict=True))
                for key, means in question_means.items()
            }
            rng = random.Random(42)
            values = list(differences.values())
            draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(5000))
            contrasts[name] = {
                "accuracy_delta": statistics.mean(values),
                "paired_question_bootstrap_95_interval": [draws[124], draws[4874]],
                "question_deltas": differences,
            }
        result[f"{dataset}/{mode}"] = {
            "questions": len(questions),
            "repetitions": repetitions,
            "responses_per_arm": len(questions) * repetitions,
            "pass_at_1": {
                arm: statistics.mean(row[i] for row in question_means.values()) for i, arm in enumerate(ARMS)
            },
            "contrasts": contrasts,
        }
    return result


def compare(paths):
    if set(paths) != set(ARMS):
        raise ValueError("All four control paths are required")
    arms = {}
    reference = None
    for name in ARMS:
        root = paths[name]
        provenance = json.loads((root / "provenance.json").read_text())
        complete = json.loads((root / "complete.json").read_text())
        settings = {
            key: provenance[key]
            for key in (
                "panel_sha256",
                "repeats",
                "head",
                "sampling_temperature",
                "response_cap",
                "request_order_sha256",
            )
        }
        serving = list(provenance["command"])
        serving[serving.index("--model-path") + 1] = "CHECKPOINT"
        settings["serving_command_except_checkpoint"] = serving
        if reference is not None and settings != reference:
            raise ValueError("Serving/evaluation settings differ between arms")
        reference = settings
        path = root / "responses.jsonl"
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != complete["responses_sha256"]:
                raise ValueError("Completed answer artifact changed")
        arms[name] = compare_opd_repeated.read(path)
        if len(arms[name]) != provenance["expected_responses"] or complete["responses"] != len(arms[name]):
            raise ValueError("Incomplete evaluation arm")
    return {
        "scope": "One training seed and one frozen selection trace. Question bootstrap excludes training-seed uncertainty. Contrasts are exploratory and intervals are not multiplicity-adjusted. Positive deltas favor selected prompts or stale policies respectively; interaction is the change in selection effect under stale policies.",
        "settings": reference,
        "datasets": summarize(arms),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="NAME=RESULT_DIRECTORY")
    options = parser.parse_args()
    paths = dict((name, Path(path)) for name, path in (item.split("=", 1) for item in options.arm))
    print(json.dumps(compare(paths), indent=2))
