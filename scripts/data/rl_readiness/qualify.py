"""Qualify exported candidate controls through the real MILES reward and loader paths.

Run in the pinned MILES image plus requirements.txt. Exports are trusted inputs
produced by a reviewed adapter; arbitrary dataset rows cannot choose factories.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from miles.rollout.filter_hub import common_filters
from miles.utils.types import Sample
from rdkit import Chem
from rdkit.Chem import Descriptors
from transformers import AutoTokenizer

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.datasets import run_data
from open_instruct.miles.rewards import rewards
from open_instruct.miles.rollout import provider_source

FACTORIES = {
    "open_instruct.ground_truth_utils.MathVerifier",
    "open_instruct.miles.rewards.candidate_verifiers.CandidateVerifier",
    "open_instruct.miles.rewards.reasoning_gym.ReasoningGymVerifier",
    "open_instruct.miles.rewards.json_schema.JSONSchemaVerifier",
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def descriptor(path, root):
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def sample(row, text, index, status=Sample.Status.COMPLETED):
    return Sample(
        prompt=row["input"],
        label=copy.deepcopy(row["label"]),
        metadata=copy.deepcopy(row["metadata"]),
        response=text,
        tokens=[],
        response_length=0,
        status=status,
        index=index,
        group_index=0,
    )


def independent_chemistry(entry):
    if not entry.get("chemistry"):
        return
    target = entry["row"]["label"]["value"]
    prop, smiles = entry["chemistry"]["property"], entry["chemistry"]["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid molecule")
    elements = {"NumN": {7}, "NumO": {8}, "NumS": {16}, "NumHalogens": {9, 17, 35, 53}}
    if prop in elements:
        expected = sum(a.GetAtomicNum() in elements[prop] for a in mol.GetAtoms())
    elif prop == "FormalCharge":
        expected = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    else:
        expected = dict(Descriptors.descList)[prop](mol)
    if float(target) != expected:
        raise ValueError("RDKit reference mismatch")


async def controls(entry, args):
    independent_chemistry(entry)
    row = entry["row"]
    texts = [entry["gold"], entry["negative"], "", entry["gold"]]
    samples = [
        sample(row, text, i, Sample.Status.TRUNCATED if i == 3 else Sample.Status.COMPLETED)
        for i, text in enumerate(texts)
    ]
    values = await rewards.registered_reward(args, samples)
    if values != [1.0, 0.0, 0.0, 0.0]:
        raise ValueError(f"positive/wrong/empty/cap controls: {values}")
    for s, value in zip(samples, values, strict=True):
        s.reward = value
    if not common_filters.apply_reward_nonzero_std_filter(args, samples).keep:
        raise ValueError("Native filter rejected mixed group with valid capped zeros")
    for s in samples:
        s.reward = 0.0
    if common_filters.apply_reward_nonzero_std_filter(args, samples).keep:
        raise ValueError("Native filter accepted all-zero group")
    return {"values": values, "mixed_kept": True, "all_zero_filtered": True, "cap_is_valid_zero": True}


def loader_args(prepared, hf, root):
    return SimpleNamespace(
        rollout_global_dataset=True,
        hf_checkpoint=str(hf),
        chat_template_path=None,
        dump_details=None,
        prompt_data=prepared["prompt_data"],
        rollout_max_prompt_len=32768,
        input_key="input",
        multimodal_keys=None,
        label_key="label",
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs={},
        rollout_seed=17,
        rollout_shuffle=False,
        n_samples_per_prompt=4,
        buffer_filter_path=None,
        save=str(root),
        load=None,
        start_rollout_id=0,
        fully_async=True,
        use_miles_dashboard=False,
        olmo_core=SimpleNamespace(
            prompt_provider="mixture", prompt_mixture=prepared["prompt_mixture"], records_root=None
        ),
    )


async def main(args):
    root = args.output.resolve()
    if (root / "catalog.json").exists():
        raise FileExistsError("Choose a fresh output; catalogs are immutable")
    root.mkdir(parents=True, exist_ok=True)
    entries = [json.loads(line) for line in args.export.open()]
    registry = {}
    for entry in entries:
        for name, spec in entry["registry"].items():
            if spec["factory"] not in FACTORIES or set(spec) != {"factory"}:
                raise ValueError("Untrusted verifier factory/config")
            if name in registry and registry[name] != spec:
                raise ValueError("Registry collision")
            registry[name] = spec
    write(root / "controls-registry.json", registry)
    core = CoreConfig(reward_config=str(root / "controls-registry.json"), reward_zero_truncated=True)
    reward_args = SimpleNamespace(olmo_core=core, reward_key=None)
    passed, receipts = defaultdict(list), []
    started = time.monotonic()
    semaphore = asyncio.Semaphore(4)

    async def check(entry):
        async with semaphore:
            receipt = {
                "source": entry["source"],
                "id": entry["row"]["metadata"]["sample_id"],
                "stratum": entry["stratum"],
                "gold_type": entry["gold_type"],
            }
            try:
                receipt.update(await controls(entry, reward_args), status="passed")
                passed[entry["source"]].append(entry)
            except Exception as error:
                receipt.update(status="deferred", reason=f"{type(error).__name__}: {error}")
            receipts.append(receipt)
            if len(receipts) % 50 == 0:
                print(f"controls {len(receipts)}/{len(entries)}", flush=True)

    await asyncio.gather(*(check(entry) for entry in entries))
    write(root / "controls.json", receipts)
    tokenizer = AutoTokenizer.from_pretrained(args.hf)
    catalog = {
        "schema_version": 1,
        "sources": {},
        "scope": "Bounded prepared slices; only these rows qualified",
        "semantics": "Cap=valid zero. Toy all-zero groups bypass variance filtering; controls exercise it.",
    }
    coverage, all_registry = [], {}
    for name, values in sorted(passed.items()):
        values.sort(key=lambda e: e["row"]["metadata"]["sample_id"])
        folder = root / "sources" / name
        folder.mkdir(parents=True)
        rows = [entry["row"] for entry in values]
        prompts, reward_path = folder / "prompts.jsonl", folder / "verifiers.json"
        prompts.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        source_registry = {key: value for entry in values for key, value in entry["registry"].items()}
        write(reward_path, source_registry)
        data = {
            "sources": [
                {"name": name, "fraction": 1.0, "prompt_data": str(prompts), "reward_config": str(reward_path)}
            ],
            "seed": 17,
            "shuffle": False,
        }
        record = {
            "candidate": values[0]["candidate"],
            "rows": len(rows),
            "status": "deferred",
            "strata": sorted({v["stratum"] for v in values}),
            "prompt_data": descriptor(prompts, root),
            "reward_config": descriptor(reward_path, root),
            "controls_passed": True,
            "gpu_exercised": False,
        }
        try:
            prepared = run_data.prepare_data(data, args.hf, folder / "prepared", max_prompt_length=32768, seed=17)
            loader = provider_source.ProviderRolloutDataSource(loader_args(prepared, args.hf, root))
            groups = loader.get_samples(len(rows))
            actual_ids = [group[0].metadata["sample_id"] for group in groups]
            assert actual_ids == [row["metadata"]["sample_id"] for row in rows]
            assert all(len(group) == 4 for group in groups)
            assert all(s.metadata["mixture_source"] == name for group in groups for s in group)
            state = copy.deepcopy(loader.provider.state_dict())
            expected = loader.provider.next_samples(1)[0].metadata
            loader.provider.load_state_dict(state)
            assert loader.provider.next_samples(1)[0].metadata == expected
            record.update(status="qualified", native_loader_passed=True, resume_cursor_passed=True)
            # The GPU pass visits every source deterministically, using one of its actual rows.
            representative = min(rows, key=lambda row: len(tokenizer.encode(row["input"], add_special_tokens=False)))
            length = len(tokenizer.encode(representative["input"], add_special_tokens=False))
            record["gpu_prompt_tokens"] = length
            if length <= 8160:
                coverage.append(representative)
            else:
                record["gpu_deferral"] = "Shortest prompt exceeds the bounded 8192-token toy fixture"
            all_registry.update(source_registry)
        except Exception as error:
            record["reason"] = f"{type(error).__name__}: {error}"
        catalog["sources"][name] = record
        print(name, record["status"], record.get("reason", ""), flush=True)
    write(root / "catalog.json", catalog)
    (root / "gpu-prompts.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in coverage))
    write(root / "gpu-verifiers.json", all_registry)
    write(
        root / "qualification.json",
        {
            "seconds": time.monotonic() - started,
            "controls": len(receipts),
            "passed": sum(r["status"] == "passed" for r in receipts),
            "named_sources": len(catalog["sources"]),
            "gpu_coverage_rows": len(coverage),
        },
    )
    print(json.dumps(json.loads((root / "qualification.json").read_text())), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
