"""Megatron verifier GRPO against the image's pinned MILES: converter guard, DPPO loss wiring,
async producer and the reward post-process on native samples."""

import ast
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from miles.backends.core_utils.rollout import async_rollout
from miles.backends.training_utils import cp_utils
from miles.backends.training_utils.loss_hub import logit_processors, losses
from miles.utils.types import Sample

from open_instruct.miles.megatron_grpo import conversion, dppo_loss, grpo_async, grpo_config, grpo_hooks


def native_converter():
    native = importlib.util.find_spec("miles")
    assert native is not None and native.origin is not None
    return (Path(native.origin).resolve().parents[1] / "tools/convert_hf_to_torch_dist.py").read_text()


def run_get_args(source, tp, keep_pp1):
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_args")
    environ = {"WORLD_SIZE": "2", **({"CONVERT_KEEP_PP1": "1"} if keep_pp1 else {})}
    namespace = {
        "parse_args": lambda _: SimpleNamespace(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            num_layers=24,
            decoder_last_pipeline_num_layers=None,
        ),
        "set_default_megatron_args": lambda args: args,
        "add_conversion_args": None,
        "os": SimpleNamespace(environ=environ, getenv=environ.get),
    }

    def validate(args):
        assert 2 % (args.tensor_model_parallel_size * args.pipeline_model_parallel_size) == 0

    namespace["validate_args"] = validate
    exec(compile(ast.Module(body=[function], type_ignores=[]), "native-converter", "exec"), namespace)
    return namespace["get_args"]()


def test_pinned_converter_keeps_the_trainer_mesh():
    source = native_converter()
    conversion.verify_keep_pp1(source, "native-converter")
    for keep_pp1 in (False, True):
        args = run_get_args(source, tp=2, keep_pp1=keep_pp1)
        assert args.pipeline_model_parallel_size == 1 and args.decoder_last_pipeline_num_layers is None
    assert run_get_args(source, tp=1, keep_pp1=False).pipeline_model_parallel_size == 2
    assert run_get_args(source, tp=1, keep_pp1=True).pipeline_model_parallel_size == 1


def test_dppo_loss_matches_the_pinned_custom_loss_contract():
    selected = SimpleNamespace(loss_type="custom_loss", custom_loss_function_path=grpo_config.DPPO_LOSS)
    assert losses.get_loss_function(selected) is dppo_loss.policy_loss
    assert list(inspect.signature(dppo_loss.policy_loss).parameters) == [
        "args",
        "batch",
        "logits",
        "sum_of_sample_mean",
    ]
    masks = inspect.signature(cp_utils.get_local_response_loss_masks).parameters
    assert list(masks)[:5] == ["total_lengths", "response_lengths", "loss_masks", "qkv_format", "max_seq_lens"]
    log_probs = inspect.signature(logit_processors.get_log_probs_and_entropy).parameters
    for name in ("args", "unconcat_tokens", "total_lengths", "response_lengths", "with_entropy", "max_seq_lens"):
        assert name in log_probs


def test_dppo_loss_reports_opd_reverse_kl_like_the_native_loss():
    assert 'batch.get("opd_reverse_kl")' in inspect.getsource(losses)
    assert 'metrics["opd_reverse_kl"]' in inspect.getsource(dppo_loss.policy_loss)


def test_async_producer_is_the_managed_native_producer():
    assert issubclass(grpo_async.VerifierAsyncRollout, async_rollout.ManagedFullyAsyncRolloutFn)
    assert grpo_async.VerifierAsyncRollout.eval_reward_path == grpo_config.REWARD_HOOK
    assert grpo_config.ASYNC_ROLLOUT == "open_instruct.miles.megatron_grpo.grpo_async.VerifierAsyncRollout"


def test_post_process_on_native_samples(tmp_path, monkeypatch):
    monkeypatch.setenv(grpo_hooks.OUTPUT_ENV, str(tmp_path))
    samples = []
    for index, reward in enumerate([1.0, 0.0, 0.0, 0.0]):
        sample = Sample(
            index=index,
            group_index=index // 2,
            tokens=[5, 6, 7 + index],
            response="answer",
            response_length=1,
            reward=reward,
            metadata={"verifiers": [{"name": "math", "target": "1"}]},
        )
        sample.status = Sample.Status.TRUNCATED if index == 3 else Sample.Status.COMPLETED
        samples.append(sample)
    args = SimpleNamespace(n_samples_per_prompt=2, grpo_std_normalization=False)
    raw, centered = grpo_hooks.post_process(args, samples)
    assert raw == [1.0, 0.0, 0.0, 0.0] and centered == [0.5, -0.5, 0.0, 0.0]
    records = [json.loads(line) for line in (tmp_path / grpo_hooks.EVIDENCE).read_text().splitlines()]
    assert [row["status"] for row in records] == ["completed", "completed", "completed", "truncated"]


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_post_process_rejects_nonfinite_native_rewards(tmp_path, monkeypatch, value):
    monkeypatch.setenv(grpo_hooks.OUTPUT_ENV, str(tmp_path))
    samples = [Sample(index=i, tokens=[1, 2], response_length=1, reward=r) for i, r in enumerate([value, 0.0])]
    with pytest.raises(ValueError, match="nonfinite"):
        grpo_hooks.post_process(SimpleNamespace(n_samples_per_prompt=2, grpo_std_normalization=False), samples)
