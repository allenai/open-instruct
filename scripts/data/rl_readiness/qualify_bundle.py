"""Qualify complete prepared sources using training preparation and reward controls.

No generation, downloads, topic filtering or backend launches. A source remains
prepared unless every observed stratum has reviewed gold/wrong/near-miss controls.
"""

import argparse
import asyncio
import copy
import dataclasses
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from open_instruct.miles.datasets import run_data
from open_instruct.miles.datasets import source_preparation as sources
from open_instruct.miles.rewards import complete_code, judge_registry, rewards


def read_jsonl(path):
    # Unicode separators inside JSON strings are prompt content, not records.
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def bundled_path(root, descriptor):
    path = (root / descriptor["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Catalog artifact escapes its bundle")
    return sources.checked_file(descriptor, root)


async def check_control(row, cases, reward_path):
    by_kind = {case["kind"]: case for case in cases}
    if len(by_kind) != len(cases) or not {"gold", "wrong", "near_miss"} <= by_kind.keys():
        raise ValueError("Require distinct gold, wrong and near_miss controls")
    # ToolRL preserves its published reward scale instead of silently rescaling
    # training to fit the qualification tool's usual [0, 1] contract.
    low, high = 0, 1
    registry = json.loads(Path(reward_path).read_text())
    specs = row["metadata"]["verifiers"]
    if len(specs) == 1 and registry[specs[0]["name"]]["factory"] == sources.FACTORIES["toolrl"]:
        if specs[0].get("weight", 1) != 1:
            raise ValueError("ToolRL controls require the unweighted source reward")
        if specs[0]["target"]["calls"]:
            low, high = -3, 4
    if by_kind["gold"]["expected"] != high or any(
        not isinstance(by_kind[k]["expected"], (int, float)) or not low <= by_kind[k]["expected"] < high
        for k in ("wrong", "near_miss")
    ):
        raise ValueError(f"Gold must expect {high}; wrong and near_miss must be in [{low}, {high})")
    expanded = [
        *cases,
        {"kind": "empty", "response": "", "expected": low},
        {"kind": "truncated", "response": by_kind["gold"]["response"], "expected": 0},
    ]
    args = SimpleNamespace(
        reward_key=None,
        olmo_core=SimpleNamespace(
            reward_config=str(reward_path), reward_zero_truncated=True, reward_final_answer_only=False
        ),
    )
    results = []
    for case in expanded:
        sample = SimpleNamespace(
            prompt=row["input"],
            metadata=copy.deepcopy(row["metadata"]),
            response=case["response"],
            tokens=[],
            response_length=0,
            status="truncated" if case["kind"] == "truncated" else "completed",
        )
        actual = await rewards.registered_reward(args, sample)
        diagnostics = sample.metadata.get("verifier_diagnostics", {})
        if any(value.get("status") == "timeout" for value in diagnostics.values()):
            raise ValueError(f"{case['kind']} control timed out")
        results.append(
            {
                "kind": case["kind"],
                "expected": case["expected"],
                "actual": actual,
                "diagnostics": diagnostics,
                "response_sha256": sources.digest(case["response"]),
            }
        )
        if not math.isclose(actual, float(case["expected"]), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{case['kind']} expected {case['expected']}, got {actual}")
    return results


def validate_prompt_targets(rows, controls, registry):
    """Different hidden code tests may share a prompt if a common witness passes all.

    Every witness is subsequently executed by check_control. Other differing
    targets remain ambiguous and require source review before qualification.
    """
    groups = defaultdict(list)
    golds = defaultdict(set)
    for control in controls:
        for case in control["cases"]:
            if case["kind"] == "gold" and case["expected"] == 1:
                golds[control["sample_id"]].add(case["response"])
    for row in rows:
        groups[sources.digest(row["input"])].append(row)
    for group in groups.values():
        if len({sources.digest(row["metadata"]["verifiers"]) for row in group}) <= 1:
            continue
        witnesses = None
        for row in group:
            verifiers = row["metadata"]["verifiers"]
            if len(verifiers) != 1 or registry[verifiers[0]["name"]]["factory"] != sources.FACTORIES["complete_code"]:
                raise ValueError("Conflicting targets for the same prompt")
            candidates = golds[row["metadata"]["sample_id"]]
            witnesses = candidates if witnesses is None else witnesses & candidates
        if not witnesses:
            raise ValueError("Different code tests for one prompt require a shared gold control for every row")


async def qualify(catalog_path, controls_path, hf, output, max_prompt_length, quarantine_path=None):
    catalog_path, controls_path = Path(catalog_path).resolve(), Path(controls_path).resolve()
    output = Path(output).resolve()
    catalog = json.loads(catalog_path.read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("Unknown catalog version")
    quarantine = defaultdict(dict)
    if quarantine_path is not None:
        for record in read_jsonl(quarantine_path):
            name, identity = record["source"], record["id"]
            if name not in catalog["sources"] or identity in quarantine[name] or not record.get("reason"):
                raise ValueError("Quarantine requires known sources, distinct IDs and explicit reasons")
            quarantine[name][identity] = record["reason"]
    controls = defaultdict(list)
    for entry in read_jsonl(controls_path):
        if entry["source"] not in catalog["sources"]:
            raise ValueError("Control names an unknown source")
        controls[entry["source"]].append(entry)
    output.mkdir(parents=True, exist_ok=False)
    qualified = copy.deepcopy(catalog)
    reports = {}
    for name, entry in qualified["sources"].items():
        # Avoid trusting a pre-existing status; re-qualify against these controls.
        entry["status"] = "prepared"
        report = {"status": "deferred", "controls": []}
        reports[name] = report
        try:
            bundled_path(catalog_path.parent, entry["hygiene_receipt"])
            omitted = entry.get("counts", {}).get("omitted_rows", 0)
            if omitted or quarantine[name]:
                if "dispositions" not in entry:
                    raise ValueError("Quarantine requires a hashed dispositions report")
                dispositions_path = bundled_path(catalog_path.parent, entry["dispositions"])
                dispositions = read_jsonl(dispositions_path)
                omitted_ids = {row["id"] for row in dispositions if row["status"] == "omitted"}
                if len(omitted_ids) != omitted or omitted_ids != set(quarantine[name]):
                    raise ValueError("Every omission must match an explicitly reviewed quarantine row")
                report["quarantined_rows"] = omitted
            paths = {key: bundled_path(catalog_path.parent, entry[key]) for key in ("prompt_data", "reward_config")}
            # Bind the evidence to the inputs checked here, before running controls
            # or copying artifacts into the published bundle. Paths may relocate.
            report["artifact_sha256"] = {key: entry[key]["sha256"] for key in paths}
            registry = json.loads(paths["reward_config"].read_text())
            for key, spec in registry.items():
                if spec.get("factory") not in set(sources.FACTORIES.values()):
                    raise ValueError("Unreviewed verifier factory")
                expected = {"factory": spec["factory"]}
                if spec["factory"] == sources.FACTORIES["reference_judge"]:
                    expected["config"] = {"name": key}
                    if not judge_registry.bound(key):
                        raise ValueError("Missing resolved named judge binding: " + key)
                if spec["factory"] == sources.FACTORIES["complete_code"]:
                    config = complete_code.Config(**spec.get("config", {}))
                    complete_code.validate_config(config)
                    expected["config"] = dataclasses.asdict(config)
                if spec != expected:
                    raise ValueError("Unreviewed verifier options")
            rows = read_jsonl(paths["prompt_data"])
            by_id = {row["metadata"]["sample_id"]: row for row in rows}
            if len(by_id) != len(rows):
                raise ValueError("Repeated sample identity")
            validate_prompt_targets(rows, controls[name], registry)
            prepared = run_data.prepare_data(
                {"sources": [{"name": name, "fraction": 1.0, **{k: str(v) for k, v in paths.items()}}]},
                Path(hf),
                output / name / "loader",
                max_prompt_length=max_prompt_length,
                seed=17,
            )
            if any(spec["factory"] == sources.FACTORIES["reference_judge"] for spec in registry.values()):
                judge_registry.validate_data(prepared, judge_registry.registry())
            covered = set()
            for control in controls[name]:
                row = by_id[control["sample_id"]]
                result = await check_control(row, control["cases"], paths["reward_config"])
                covered.add(row["metadata"]["subtask"])
                report["controls"].append({"sample_id": control["sample_id"], "cases": result})
            required = {row["metadata"]["subtask"] for row in rows}
            missing = sorted(required - covered)
            report.update(
                required_strata=len(required),
                covered_strata=len(covered),
                missing_strata=missing,
                rows=len(rows),
                loader_manifest_sha256=sources.file_hash(prepared["manifest"]),
            )
            if missing:
                raise ValueError("Missing reviewed controls for one or more strata")
            if not rows:
                raise ValueError("Empty source")
            entry["status"] = report["status"] = "qualified"
        except Exception as error:
            report["reason"] = f"{type(error).__name__}: {error}"
        # Preserve all inputs and hashes even for deferred sources. Never shrink a source to passing controls.
        for key in ("prompt_data", "reward_config", "hygiene_receipt", "dispositions"):
            if key not in entry:
                continue
            path = bundled_path(catalog_path.parent, entry[key])
            target = output / name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            copied_hash = sources.file_hash(target)
            if copied_hash != entry[key]["sha256"]:
                raise ValueError("Catalog artifact changed while copying: " + key)
            entry[key] = {"path": str(target.relative_to(output)), "sha256": copied_hash}
    receipt = {
        "schema_version": 1,
        "catalog_sha256": sources.file_hash(catalog_path),
        "controls_sha256": sources.file_hash(controls_path),
        "max_prompt_length": max_prompt_length,
        "quarantine_sha256": sources.file_hash(quarantine_path) if quarantine_path else None,
        "judges_sha256": sources.digest(judge_registry.registry()),
        "scope": "Whole-pool loader validation; stratum controls via runtime bridge. No GPU training or policy screen.",
        "sources": reports,
    }
    if quarantine_path is not None:
        shutil.copyfile(quarantine_path, output / "quarantine.jsonl")
    sources.write(output / "qualification.json", receipt)
    for entry in qualified["sources"].values():
        entry["qualification"] = {
            "path": "qualification.json",
            "sha256": sources.file_hash(output / "qualification.json"),
        }
    sources.write(output / "catalog.json", qualified)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument(
        "--quarantine", type=Path, help="Reviewed JSONL records with source, id and reason for every omitted row"
    )
    args = parser.parse_args()
    receipt = asyncio.run(
        qualify(args.catalog, args.controls, args.tokenizer, args.output, args.max_prompt_length, args.quarantine)
    )
    print(json.dumps({name: result["status"] for name, result in receipt["sources"].items()}, indent=2))
    if any(result["status"] != "qualified" for result in receipt["sources"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
