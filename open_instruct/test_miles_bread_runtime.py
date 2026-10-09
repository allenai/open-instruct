"""Exercise the BREAD collector against the pinned runtime's callback contracts."""

import asyncio
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.test_miles_bread import samples


@pytest.fixture
def runtime(monkeypatch):
    # The CPU application environment intentionally has no serving stack.
    names = [
        "miles",
        "miles.backends",
        "miles.backends.core_utils",
        "miles.backends.core_utils.rollout",
        "miles.backends.core_utils.rollout.errors",
        "miles.rollout",
        "miles.rollout.base_types",
        "miles.rollout.filter_hub",
        "miles.rollout.filter_hub.base_types",
        "miles.rollout.inference_rollout",
        "miles.rollout.inference_rollout.inference_rollout_common",
        "miles.utils",
        "miles.utils.dumper_utils",
        "miles.utils.function_registry",
    ]
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], child, module)

    class Interrupted(RuntimeError):
        pass

    modules[names[4]].GenerationInterrupted = Interrupted
    modules["miles.rollout.base_types"].RolloutFnTrainOutput = SimpleNamespace
    common = modules["miles.rollout.inference_rollout.inference_rollout_common"]
    common.InferenceRolloutFn = object

    async def configure(args):
        pass

    modules["miles.utils.dumper_utils"].configure_sglang = configure
    modules["miles.utils.function_registry"].load_function = lambda _: None
    path = Path(__file__).parent / "miles/rollout/bread_runtime.py"
    spec = importlib.util.spec_from_file_location("bread_runtime_test_subject", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, common, Interrupted


def collector(runtime, tmp_path, score_sequence):
    module, common, _ = runtime
    fn = module.BreadRolloutFn()
    calls, retired = [], []

    async def generate(state, group, params, evaluation):
        calls.append(group)
        for i, sample in enumerate(group):
            sample.status = SimpleNamespace(value="completed")
            sample.reward = {"score": score_sequence[len(calls) - 1][i]}
            prompt_tokens = list(sample.prompt.encode())
            sample.tokens = prompt_tokens + [99, 100]
            sample.response = "student answer"
            sample.response_length = 2
            sample.rollout_log_probs = [-1.0, -2.0]
        return group

    common.generate_and_rm_group = generate

    def get_samples(n):
        assert n == 1
        group = samples()
        for i, sample in enumerate(group):
            sample.index, sample.group_index = i, 3
            sample.metadata["prepared_sample_id"] = "openr1:test"
        return [group]

    args = SimpleNamespace(
        olmo_core=CoreConfig(bread_max_groups_per_batch=2),
        rollout_batch_size=1,
        dynamic_sampling_filter_path=None,
        reward_key="score",
        rollout_max_context_len=1000,
        rollout_max_response_len=20,
        save=str(tmp_path),
    )
    fn.state = SimpleNamespace(
        args=args,
        sampling_params={},
        aborted=False,
        tokenizer=SimpleNamespace(encode=lambda text, **_: list(text.encode())),
        reset=lambda: None,
    )
    fn.data_source = SimpleNamespace(get_samples=get_samples, add_samples=retired.append)
    return fn, calls, retired


def test_collector_rescues_before_filter_and_keeps_prefix_outside_loss_span(runtime, tmp_path):
    fn, calls, retired = collector(runtime, tmp_path, [[0, 0], [0, 1], [1, 0]])
    output = asyncio.run(fn._call_train(SimpleNamespace(rollout_id=7, weight_version=2)))
    assert len(calls) == 3 and retired == [[]]
    assert output.samples == [calls[-1]]
    for sample in output.samples[0]:
        assert sample.prompt.endswith("first\n\n")
        assert sample.tokens[: -sample.response_length] == list(sample.prompt.encode())
        assert len(sample.rollout_log_probs) == sample.response_length == 2
    assert output.metrics["bread/generated_tokens"] == 12
    event = json.loads((tmp_path / "bread/groups.jsonl").read_text())
    assert event["accepted"] and event["outcome"] == "rescued"
    assert event["sample_id"] == "openr1:test" and event["weight_version"] == 2


def test_collection_budget_fails_instead_of_looping_and_does_not_acknowledge(runtime, tmp_path):
    fn, _, retired = collector(runtime, tmp_path, [[1, 1], [1, 1]])
    with pytest.raises(RuntimeError, match="budget exhausted"):
        asyncio.run(fn._call_train(SimpleNamespace(rollout_id=7, weight_version=2)))
    assert retired == []


def test_aborted_generation_propagates_to_recovery_without_consumption(runtime, tmp_path):
    fn, _, retired = collector(runtime, tmp_path, [[0, 0]])
    fn.state.aborted = True
    with pytest.raises(runtime[2], match="interrupted"):
        asyncio.run(fn._call_train(SimpleNamespace(rollout_id=7, weight_version=2)))
    assert retired == []


def test_concurrent_waves_refill_in_input_order_without_overfetch(runtime, tmp_path):
    fn, _, retired = collector(runtime, tmp_path, [])
    fn.state.args.olmo_core = dataclasses.replace(
        fn.state.args.olmo_core, bread_concurrent_groups=2, bread_max_groups_per_batch=4
    )
    fn.state.args.rollout_batch_size = 3
    requested, finished = [], []
    next_id = 0

    def get_samples(count):
        nonlocal next_id
        requested.append(count)
        groups = []
        for _ in range(count):
            group = samples()
            for i, sample in enumerate(group):
                sample.index, sample.group_index = next_id * 2 + i, next_id
            groups.append(group)
            next_id += 1
        return groups

    fn.data_source.get_samples = get_samples

    async def run():
        second_finished = asyncio.Event()

        async def generate(state, group, params, evaluation):
            identity = group[0].group_index
            if identity == 0:
                await second_finished.wait()
            for i, sample in enumerate(group):
                sample.status = SimpleNamespace(value="completed")
                sample.reward = {"score": 1 if identity == 1 else i}
                sample.response_length = 2
            finished.append(identity)
            if identity == 1:
                second_finished.set()
            return group

        runtime[1].generate_and_rm_group = generate
        return await fn._call_train(SimpleNamespace(rollout_id=7, weight_version=2))

    output = asyncio.run(run())
    assert finished == [1, 0, 2, 3]
    assert requested == [2, 2] and retired == [[]]
    assert [group[0].group_index for group in output.samples] == [0, 2, 3]
    assert output.metrics["bread/generated_tokens"] == 16
    assert output.metrics["bread/dropped_groups"] == 1


def test_failed_concurrent_wave_settles_siblings_without_acknowledging(runtime, tmp_path):
    fn, _, retired = collector(runtime, tmp_path, [])
    fn.state.args.olmo_core = dataclasses.replace(fn.state.args.olmo_core, bread_concurrent_groups=2)
    fn.state.args.rollout_batch_size = 2
    groups = [samples(), samples()]
    for index, group in enumerate(groups):
        for sample in group:
            sample.group_index = index
    fn.data_source.get_samples = lambda count: groups
    settled = []

    async def run():
        sibling_started = asyncio.Event()

        async def generate(state, group, params, evaluation):
            if group[0].group_index == 0:
                await sibling_started.wait()
                raise runtime[2]("interrupted")
            try:
                sibling_started.set()
                await asyncio.Event().wait()
            finally:
                settled.append(True)

        runtime[1].generate_and_rm_group = generate
        return await fn._call_train(SimpleNamespace(rollout_id=7, weight_version=2))

    with pytest.raises(runtime[2], match="interrupted"):
        asyncio.run(run())
    assert settled == [True] and retired == []
