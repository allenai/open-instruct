"""Observe one production EP1 contract update without changing its arithmetic."""

import argparse
import dataclasses
import json
import time
from pathlib import Path
from unittest import mock

import ep_contract
import ep_stress_contract
import torch
from triton.runtime import cache as triton_cache

from open_instruct.miles import compiler_cache as cache
from open_instruct.miles.config import RunConfig


def model_state(worker):
    return {name: value.detach().cpu().clone() for name, value in worker.model.named_parameters()}


def run(root):
    expected = RunConfig.load(root / "expected.toml")
    report = {
        "configuration_matches": False,
        "completed_optimizer_steps": 0,
        "observation_scope": "Actual FileCacheManager calls during ep_contract.run; not all compiler families",
        "triton": {"put_calls": 0, "put_filenames": [], "group_hits": 0, "group_misses": 0},
        "input_inventory": {
            "hf": cache.inventory(root / "hf"),
            "prompts_sha256": cache.sha256(root / "prompts.jsonl"),
        },
    }
    tensors = {"scores": [], "initial_model": {}, "final_model": {}}
    original_config = ep_contract.RunConfig
    original_score = ep_contract.actor.OLMoCoreTrainRayActor._score
    original_state = ep_contract.full_optimizer_state
    original_put = triton_cache.FileCacheManager.put
    original_group = triton_cache.FileCacheManager.get_group

    def construct_config(*arguments, **kwargs):
        actual = original_config(*arguments, **kwargs)
        if dataclasses.asdict(actual) != dataclasses.asdict(expected):
            raise ValueError("Production EP contract config differs from fingerprinted expected.toml")
        report["configuration_matches"] = True
        report["actual_config"] = dataclasses.asdict(actual)
        return actual

    def score(worker, module, batches, *, use_replay):
        if not tensors["initial_model"]:
            tensors["initial_model"] = model_state(worker)
        started = time.monotonic()
        result = original_score(worker, module, batches, use_replay=use_replay)
        tensors["scores"].append(
            {"use_replay": use_replay, "scores": [value.detach().cpu().clone() for value in result]}
        )
        report.setdefault("scoring_seconds", []).append(time.monotonic() - started)
        return result

    def capture_state(worker, *, gradients=False):
        state = original_state(worker, gradients=gradients)
        if not gradients:
            tensors["final_model"] = model_state(worker)
            report["completed_optimizer_steps"] = worker.clock.completed_steps
        return state

    def put(manager, data, filename, binary=True):
        report["triton"]["put_calls"] += 1
        report["triton"]["put_filenames"].append(filename)
        return original_put(manager, data, filename, binary=binary)

    def get_group(manager, filename):
        result = original_group(manager, filename)
        report["triton"]["group_hits" if result else "group_misses"] += 1
        return result

    started = time.monotonic()
    try:
        if (root / "routes.pt").exists():
            raise ValueError("Both arms must capture routes from the identical initial fixture")
        with (
            mock.patch.object(ep_contract, "RunConfig", side_effect=construct_config),
            mock.patch.object(ep_contract.actor.OLMoCoreTrainRayActor, "_score", score),
            mock.patch.object(ep_contract, "full_optimizer_state", capture_state),
            mock.patch.object(triton_cache.FileCacheManager, "put", put),
            mock.patch.object(triton_cache.FileCacheManager, "get_group", get_group),
        ):
            ep_contract.run(root, "combined", False, capture_gradients=True)
        evidence = torch.load(root / "stress-ep1-rank0.pt", weights_only=True)
        report["optimizer_self_check"] = ep_stress_contract.inspect(evidence)
        records = [
            json.loads(line)
            for line in (root / "ep1-combined-ac0/training_contract_rank0.jsonl").read_text().splitlines()
        ]
        updates = [record for record in records if record["event"] == "optimizer"]
        if len(updates) != 1 or updates[0]["step"] != 1 or updates[0]["optimizer_skipped"]:
            raise ValueError("Expected exactly one accepted optimizer update")
        report["optimizer_seconds"] = updates[0]["elapsed_seconds"]
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["contract_seconds"] = time.monotonic() - started
        torch.save(tensors, root / "observation.pt")
        (root / "observation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
