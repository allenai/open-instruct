"""Qualify complete prepared sources using training preparation and reward controls.

No generation, downloads, topic filtering or backend launches. A source remains
prepared unless every observed stratum has reviewed gold/wrong/near-miss controls.
"""

import argparse
import asyncio
import copy
import json
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from open_instruct.miles.datasets import run_data
from open_instruct.miles.datasets import source_preparation as sources
from open_instruct.miles.rewards import judge_registry, rewards


def bundled_path(root, descriptor):
    path = (root / descriptor["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Catalog artifact escapes its bundle")
    return sources.checked_file(descriptor, root)


async def check_control(row, cases, reward_path):
    by_kind = {case["kind"]: case for case in cases}
    if len(by_kind) != len(cases) or not {"gold", "wrong", "near_miss"} <= by_kind.keys():
        raise ValueError("Require distinct gold, wrong and near_miss controls")
    if by_kind["gold"]["expected"] != 1 or any(by_kind[k]["expected"] != 0 for k in ("wrong", "near_miss")):
        raise ValueError("Gold must expect one; wrong and near_miss must expect zero")
    expanded = [
        *cases,
        {"kind": "empty", "response": "", "expected": 0},
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
        results.append(
            {
                "kind": case["kind"],
                "expected": case["expected"],
                "actual": actual,
                "response_sha256": sources.digest(case["response"]),
            }
        )
        if actual != case["expected"]:
            raise ValueError(f"{case['kind']} expected {case['expected']}, got {actual}")
    return results


async def qualify(catalog_path, controls_path, hf, output, max_prompt_length, accept_omissions=False):
    catalog_path, controls_path = Path(catalog_path).resolve(), Path(controls_path).resolve()
    output = Path(output).resolve()
    catalog = json.loads(catalog_path.read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("Unknown catalog version")
    controls = defaultdict(list)
    for line in controls_path.read_text().splitlines():
        entry = json.loads(line)
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
            if entry.get("counts", {}).get("omitted_rows", 0) and not accept_omissions:
                raise ValueError(
                    "Review omissions; use --accept-omissions only for an explicitly accepted source variant"
                )
            paths = {key: bundled_path(catalog_path.parent, entry[key]) for key in ("prompt_data", "reward_config")}
            registry = json.loads(paths["reward_config"].read_text())
            for key, spec in registry.items():
                if spec.get("factory") not in set(sources.FACTORIES.values()):
                    raise ValueError("Unreviewed verifier factory")
                expected = {"factory": spec["factory"]}
                if spec["factory"] == sources.FACTORIES["reference_judge"]:
                    expected["config"] = {"name": key}
                    if not judge_registry.bound(key):
                        raise ValueError("Missing resolved named judge binding: " + key)
                if spec != expected:
                    raise ValueError("Unreviewed verifier options")
            rows = [json.loads(line) for line in paths["prompt_data"].read_text().splitlines()]
            by_id = {row["metadata"]["sample_id"]: row for row in rows}
            if len(by_id) != len(rows):
                raise ValueError("Repeated sample identity")
            prompt_targets = {}
            for row in rows:
                key, target = sources.digest(row["input"]), sources.digest(row["metadata"]["verifiers"])
                if key in prompt_targets and prompt_targets[key] != target:
                    raise ValueError("Conflicting targets for the same prompt")
                prompt_targets[key] = target
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
        for key in ("prompt_data", "reward_config", "hygiene_receipt"):
            path = bundled_path(catalog_path.parent, entry[key])
            target = output / name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            entry[key] = {"path": str(target.relative_to(output)), "sha256": sources.file_hash(target)}
    receipt = {
        "schema_version": 1,
        "catalog_sha256": sources.file_hash(catalog_path),
        "controls_sha256": sources.file_hash(controls_path),
        "max_prompt_length": max_prompt_length,
        "accept_omissions": accept_omissions,
        "judges_sha256": sources.digest(judge_registry.registry()),
        "scope": "Whole-pool loader validation; stratum controls via runtime bridge. No GPU training or policy screen.",
        "sources": reports,
    }
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
    parser.add_argument("--accept-omissions", action="store_true")
    args = parser.parse_args()
    receipt = asyncio.run(
        qualify(
            args.catalog, args.controls, args.tokenizer, args.output, args.max_prompt_length, args.accept_omissions
        )
    )
    print(json.dumps({name: result["status"] for name, result in receipt["sources"].items()}, indent=2))
    if any(result["status"] != "qualified" for result in receipt["sources"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
