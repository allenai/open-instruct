"""Resolve the cross-repository hooks in a qualified MILES runtime."""

from importlib import import_module
from types import SimpleNamespace

import pytest
from miles.ray.specs.train import _TRAINER_ACTOR_CLASSES
from miles.utils import arguments

from open_instruct.miles.configuration import run_spec


@pytest.mark.parametrize("mode", ["barrier", "refresh", "engine_drain"])
def test_runtime_hooks_resolve(mode, tmp_path):
    spec = run_spec.RunSpec.from_dict(
        {
            "schema_version": 1,
            "name": "import-contract",
            "model": {"source": "model", "format": "hf"},
            "output": {"root": str(tmp_path / "run")},
            "data": {"tasks": [{"task": "gsm8k", "train_count": 32, "eval_count": 8}]},
        },
        config_path=tmp_path / "run.toml",
    )
    compiled = spec.compile()
    args = SimpleNamespace(
        train_backend="olmo_core",
        fully_async=True,
        olmo_core=SimpleNamespace(publication_mode=mode),
        rollout_function_path=None,
        multi_lora=False,
        colocate=False,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        pause_generation_mode="retract",
        recompute_logprobs_via_prefill=False,
        rollout_all_samples_process_path=None,
        eval_num_gpus=0,
        eval_function_path=None,
    )
    arguments._resolve_rollout_functions(args)
    paths = [
        _TRAINER_ACTOR_CLASSES["olmo_core"],
        args.rollout_function_path,
        args.eval_function_path,
        compiled.miles["custom_rm_path"],
        compiled.miles["custom_rollout_log_function_path"],
        "open_instruct.miles.rollout.data_source.DashboardDrainingRolloutDataSource",
        "open_instruct.miles.rollout.async_buffer.RefreshPolicyDataBuffer",
        "open_instruct.miles.rollout.async_buffer.HomogeneousPolicyDataBuffer",
        "open_instruct.miles.rewards.code_rewards.CodeVerifier",
        "open_instruct.miles.rewards.judge_registry.NamedJudgeVerifier",
    ]
    for path in paths:
        module, _, name = path.rpartition(".")
        assert callable(getattr(import_module(module), name)), path
