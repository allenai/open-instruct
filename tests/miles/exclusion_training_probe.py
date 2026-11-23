"""Bounded local GPU qualification of exclusion through the production Core trainer.

Inside the pinned image, with tests/miles on PYTHONPATH:
  python tests/miles/exclusion_training_probe.py --output /output/exclusion

Uses the existing random tiny hybrid-MoE fixture and real GSM8K grading, with
controlled completions and injected verifier outages. Rollout transport is a
fixed batch; generation, Ray scheduling, and service HTTP are not exercised.
"""

import argparse
import asyncio
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import ep_contract
import torch
from miles.ray.rollout import train_data_conversion
from miles.rollout.filter_hub import common_filters
from miles.utils.types import Sample

from open_instruct.miles.configuration.config import EXCLUDE_TRUNCATED, RunConfig
from open_instruct.miles.rewards import exclusion, rewards
from open_instruct.miles.rewards.verifiers import GSM8KVerifier


class ProbeVerifier(GSM8KVerifier):
    async def async_call(self, tokens, prediction, target, **kwargs):
        if prediction == "outage":
            raise RuntimeError("injected verifier outage")
        return await super().async_call(tokens, prediction, target, **kwargs)


def run_case(root, estimator, scenario):
    report = {"estimator": estimator, "scenario": scenario, "dropped_groups": {}}
    native_rollout = ep_contract.rollout_for
    resolved = {}

    def configuration(core, miles):
        core = dataclasses.replace(
            core,
            filter_zero_std_groups=scenario != "zeros",
            reward_zero_truncated=scenario != "truncated",
            reward_config=str(root / "verifiers.json"),
        )
        miles = dict(
            miles,
            advantage_estimator=estimator,
            normalize_advantages=estimator.startswith("reinforce_plus_plus"),
            grpo_std_normalization=False,
            weight_decay=0.0,
        )
        if scenario == "truncated":
            miles["custom_reward_post_process_path"] = EXCLUDE_TRUNCATED
        config = RunConfig(core, miles)
        resolved.update(core=core, miles=config.resolved_miles())
        return config

    def rollout(rank, world, mode):
        assert world == 1
        original = native_rollout(rank, world, mode)
        args = argparse.Namespace(**resolved["miles"], olmo_core=resolved["core"])
        args.reward_key = None
        args.rewards_normalization = True
        args.rollout_top_logprobs_num = 0
        args.use_dynamic_global_batch_size = False
        # ep_contract captures real routing IDs after this callback and supplies
        # them to the trainer; this earlier grading/conversion phase has none yet.
        args.use_rollout_routing_replay = False
        postprocess = exclusion.exclude_truncated if scenario == "truncated" else exclusion.exclude_ungraded
        group_filter = (
            exclusion.filter_finished_reward_variance
            if scenario == "truncated"
            else exclusion.filter_graded_responses
            if scenario == "zeros"
            else exclusion.filter_graded_reward_variance
        )

        def candidates(responses):
            values = []
            for i, response in enumerate(responses):
                value = Sample(
                    prompt="What is 1 + 1?",
                    response=response,
                    tokens=original["tokens"][i].tolist(),
                    response_length=original["response_lengths"][i],
                    group_index=0,
                    metadata={"verifiers": [{"name": "math", "target": "2"}]},
                )
                value.index = i
                value.loss_mask = original["loss_masks"][i].tolist()
                value.status = (
                    Sample.Status.TRUNCATED if scenario == "truncated" and i == 2 else Sample.Status.COMPLETED
                )
                values.append(value)
            for value, reward in zip(values, asyncio.run(rewards.rollout_reward(args, values)), strict=True):
                value.reward = reward
            return values

        # Exercise rejection before admission; an empty group never reaches train().
        rejected = candidates(["outage"] * 4)
        result = common_filters.apply_preput_filters(args, group_filter, rejected)
        assert not result.keep and result.reason == "no_trainable_responses"
        report["dropped_groups"][result.reason] = 1
        if scenario != "zeros":
            result = common_filters.apply_preput_filters(args, group_filter, candidates(["0"] * 4))
            assert not result.keep
            report["dropped_groups"][result.reason] = 1
        samples = candidates(["0"] * 4 if scenario == "zeros" else ["2", "0", "0", "outage"])
        assert common_filters.apply_preput_filters(args, group_filter, samples).keep
        converted = train_data_conversion.convert_samples_to_train_data(args, samples, {}, None, postprocess)
        original["rewards"] = converted["rewards"]
        original["loss_masks"] = [torch.tensor(mask, device="cuda") for mask in converted["loss_masks"]]
        original["rollout_mask_sums"] = [
            torch.tensor(total, device="cuda") for total in converted["rollout_mask_sums"]
        ]
        report.update(
            raw_rewards=converted["raw_reward"],
            training_rewards=converted["rewards"],
            mask_sums=converted["rollout_mask_sums"],
            ungraded=rewards.ungraded_count(samples),
        )
        return original

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (
        mock.patch.object(ep_contract, "RunConfig", side_effect=configuration),
        mock.patch.object(ep_contract, "rollout_for", rollout),
    ):
        ep_contract.run(root, "policy", False, capture_gradients=True)
    state = torch.load(root / "stress-ep1-rank0.pt", weights_only=True)
    tensors = list(state["before"].values()) + list(state["after"].values()) + list(state["state"].values())
    assert all(torch.isfinite(tensor).all() for tensor in tensors)
    gradient_norm = sum(float(tensor.double().square().sum()) for tensor in state["before"].values()) ** 0.5
    assert (gradient_norm == 0.0) == (scenario == "zeros")
    records = [
        json.loads(line) for line in (root / "ep1-policy-ac0/training_contract_rank0.jsonl").read_text().splitlines()
    ]
    steps = [record for record in records if record["event"] == "optimizer"]
    assert len(steps) == 1 and steps[0]["step"] == 1 and not steps[0]["optimizer_skipped"]
    report.update(
        status="passed",
        optimizer_steps=1,
        gradient_norm=gradient_norm,
        seconds=time.monotonic() - started,
        optimizer_seconds=steps[0]["elapsed_seconds"],
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
    )
    (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--estimator")
    parser.add_argument("--scenario", choices=("mixed", "zeros", "truncated"))
    args = parser.parse_args()
    if args.estimator:
        run_case(args.output, args.estimator, args.scenario)
        return
    root = args.output
    ep_contract.bootstrap(root / "fixture")
    cases = [
        (e, s)
        for e in ("grpo", "gspo", "reinforce_plus_plus_baseline", "reinforce_plus_plus")
        for s in ("mixed", "zeros")
    ]
    cases.append(("grpo", "truncated"))
    results = []
    for estimator, scenario in cases:
        destination = root / f"{estimator}-{scenario}"
        shutil.copytree(root / "fixture", destination)
        (destination / "verifiers.json").write_text(
            json.dumps({"math": {"factory": "exclusion_training_probe.ProbeVerifier", "config": {}}})
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--master-addr=127.0.0.1",
                "--master-port=29500",
                "--nnodes=1",
                "--nproc-per-node=1",
                __file__,
                "--output",
                str(destination),
                "--estimator",
                estimator,
                "--scenario",
                scenario,
            ],
            check=True,
            timeout=180,
            env=dict(os.environ, OMP_NUM_THREADS="2"),
        )
        results.append(json.loads((destination / "result.json").read_text()))
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print("EXCLUSION_TRAINING_PROBE_PASSED", flush=True)


if __name__ == "__main__":
    main()
