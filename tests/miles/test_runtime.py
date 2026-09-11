"""Integration tests run inside the pinned MILES + patched Core image."""

import contextlib
import copy
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from miles.backends.fsdp_utils import lr_scheduler
from miles.backends.training_utils import parallel
from miles.utils import arguments
from miles.utils.ft_utils.process_group_utils import GroupInfo
from olmo_core.nn.hf.config import _register_olmo3moe_auto_classes
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from torch import distributed as dist
from transformers import AutoModelForCausalLM, Olmo3Config, Qwen3Config

from open_instruct.miles import actor, checkpoint, data, models, scheduler
from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.state import PolicyClock


@pytest.fixture(params=["qwen3", "kda", "kda_latent", "olmo3_full", "olmo3_sliding"])
def parsed_args(tmp_path, monkeypatch, request):
    path = tmp_path / "hf"
    hf = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=128,
    )
    if request.param.startswith("olmo3_"):
        hf = Olmo3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=128,
            sliding_window=3,
            layer_types=["full_attention", "full_attention"]
            if request.param == "olmo3_full"
            else ["sliding_attention", "full_attention"],
        )
    elif request.param != "qwen3":
        _register_olmo3moe_auto_classes()
        hf = Olmo3MoeConfig(
            vocab_size=256,
            hidden_size=128,
            attention_hidden_size=128,
            head_dim=64,
            dense_mlp_intermediate_size=256,
            dense_mlp_uses_shared_experts=True,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            n_routed_experts=4,
            num_experts_per_tok=2,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            use_head_qk_norm=True,
            use_rope=False,
            attention_gate_type="elementwise",
            # Eight KDA heads keep DDP flat parameter offsets 16-byte aligned.
            linear_num_key_heads=8,
            linear_num_value_heads=8,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            latent_moe_dim=64 if request.param == "kda_latent" else None,
            layer_types=["linear_attention", "full_attention"],
            dense_layers_indices=[0],
            use_peri_ln=True,
            max_position_embeddings=128,
        )
    model = AutoModelForCausalLM.from_config(hf).to(torch.bfloat16)
    # HF conversion classes leave these direct parameters empty on construction.
    # Real exported checkpoints supply their initialized Core values.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                parameter.zero_()
    model.save_pretrained(path)
    config = RunConfig(
        CoreConfig(
            attention_backend="torch", max_sequence_length=128, activation_checkpointing=False, diagnostic_interval=1
        ),
        {
            "hf_checkpoint": str(path),
            "global_batch_size": 4,
            "rollout_batch_size": 1,
            "n_samples_per_prompt": 4,
            "num_rollout": 2,
            "debug_train_only": True,
            "save": str(tmp_path / "checkpoints"),
            "rollout_global_dataset": True,
            "prompt_data": str(tmp_path / "prompts.jsonl"),
        },
    )
    (tmp_path / "prompts.jsonl").write_text('{"input": "test", "label": "test"}\n')
    monkeypatch.setattr(sys, "argv", ["test", *config.arguments()])
    return arguments.parse_args()


def test_pinned_parser_selects_core(parsed_args):
    assert parsed_args.train_backend == "olmo_core"
    assert parsed_args.olmo_core.max_sequence_length == 128
    assert not parsed_args.offload_train


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_miles_loss_core_update_and_native_resume(parsed_args, tmp_path, monkeypatch):
    args = parsed_args
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        group = GroupInfo(rank=0, size=1, group=dist.group.WORLD, gloo_group=dist.group.WORLD)
        trivial = GroupInfo(rank=0, size=1, group=None)
        parallel.set_parallel_state(
            parallel.ParallelState(
                intra_dp=group,
                intra_dp_cp=group,
                cp=trivial,
                tp=trivial,
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
            )
        )
        monkeypatch.setattr(actor.distributed_utils, "get_gloo_group", lambda: dist.group.WORLD)
        worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
        worker.args = args
        dense_olmo = json.loads((Path(args.hf_checkpoint) / "config.json").read_text())["model_type"] == "olmo3"
        backend_kinds = []
        original_backend = models._backend

        def checked_backend(kind):
            backend_kinds.append(kind)
            if dense_olmo:
                assert kind == "standard", "Dense Olmo 3 entered the MoE trainer backend"
            return original_backend(kind)

        monkeypatch.setattr(models, "_backend", checked_backend)
        worker.train_module, worker.hf_config, worker.model_config = models.build_train_module(args)
        if worker.hf_config.model_type == "olmo3":
            assert worker.train_module._miles_model_backend == "standard"
            assert all(kind == "standard" for kind in backend_kinds)
            assert not any("routed_experts" in name for name, _ in worker.train_module.model.named_modules())
        # Exercise publication from the actual wrapped train module, not just
        # conversion of an unwrapped standalone model.
        exported = models.export_state(worker.train_module, worker.hf_config)
        reference = AutoModelForCausalLM.from_pretrained(
            args.hf_checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        assert set(exported) == set(reference.state_dict())
        for name, value in reference.state_dict().items():
            # HF reload promotes KDA decay/bias parameters to FP32. The fixture
            # checkpoint stored BF16 values; require exact values across that promotion.
            torch.testing.assert_close(exported[name], value, rtol=0, atol=0, check_dtype=False)
        del reference, exported
        worker.model = worker.train_module.model
        worker.optimizer = worker.train_module.optim
        worker.lr_scheduler = scheduler.CoreLRScheduler(args, worker.optimizer)
        worker.clock = PolicyClock()
        worker.clock.published()
        worker.ref_module = None
        worker._heartbeat = SimpleNamespace(bump=lambda: None)
        before = {name: tensor.detach().clone() for name, tensor in worker.model.state_dict().items()}
        rollout = {
            "tokens": [torch.tensor([1, 2, 3, 4 + index, 8], device="cuda") for index in range(4)],
            "total_lengths": [5] * 4,
            "response_lengths": [3] * 4,
            "loss_masks": [torch.tensor([1, 0, 1], device="cuda") for _ in range(4)],
            "rewards": [1.0, -1.0, 1.0, -1.0],
            "weight_versions": [["0"] for _ in range(4)],
            "rollout_log_probs": [torch.full((3,), -5.0, device="cuda") for _ in range(4)],
        }
        monkeypatch.setattr(actor.miles_data, "get_rollout_data", lambda *a, **kw: (rollout, contextlib.nullcontext()))
        original_core_config = args.olmo_core
        args.olmo_core = dataclasses.replace(original_core_config, max_train_rollout_logprob_abs_diff=0.0)
        with pytest.raises(RuntimeError, match="logprob difference"):
            worker.train(0, None)
        assert worker.clock.completed_steps == 0
        for name, value in worker.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        args.olmo_core = original_core_config
        worker.train(0, None)
        assert worker.clock.completed_steps == 1 and worker.clock.next_rollout_id == 1
        after = worker.model.state_dict()
        assert any(not torch.equal(before[name], value) for name, value in after.items())
        worker.save_model(0, force_sync=True)
        cursor = Path(args.save) / "rollout" / "global_dataset_state_dict_0.pt"
        cursor.parent.mkdir(parents=True)
        torch.save({"sample_offset": 4}, cursor)
        worker.finalize_checkpoint(0)
        _, manifest = checkpoint.resume_manifest(args.save)
        assert manifest["clock"]["completed_steps"] == 1
        assert manifest["clock"]["next_rollout_id"] == 1
        native = tmp_path / "native"
        models.save_native(worker.train_module, native)
        expected = {name: tensor.detach().clone() for name, tensor in after.items()}
        with torch.no_grad():
            next(worker.model.parameters()).add_(1)
        models.load_native(worker.train_module, native)
        for name, tensor in worker.model.state_dict().items():
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)

        # Same next batch after native model+optimizer+scheduler+clock restore.
        worker.clock.published()
        rollout["weight_versions"] = [["1"] for _ in range(4)]
        next_batch = copy.deepcopy(rollout)
        worker.train(1, None)
        uninterrupted = {name: value.detach().clone() for name, value in worker.model.state_dict().items()}
        uninterrupted_lr = worker.lr_scheduler.state_dict()
        uninterrupted_optimizer = copy.deepcopy(worker.optimizer.state_dict())
        args.load = args.save
        checkpoint.restore(worker)
        worker.train_module._trainer.global_step = worker.clock.completed_steps
        worker.clock.published()
        rollout.clear()
        rollout.update(copy.deepcopy(next_batch))
        worker.train(1, None)
        for name, tensor in worker.model.state_dict().items():
            torch.testing.assert_close(tensor, uninterrupted[name], rtol=0, atol=0)
        assert worker.lr_scheduler.state_dict() == uninterrupted_lr
        torch.testing.assert_close(worker.optimizer.state_dict(), uninterrupted_optimizer, rtol=0, atol=0)
        assert worker.clock.completed_steps == 2 and worker.clock.next_rollout_id == 2
        records = [
            json.loads(line) for line in (Path(args.save) / "training_contract_rank0.jsonl").read_text().splitlines()
        ]
        steps = [record for record in records if record["event"] == "optimizer"]
        assert [record["step"] for record in steps] == [1, 2, 2]
        assert all(record["normalization"]["active_tokens"] == 8 for record in steps)
        assert all(record["local_pre_optimizer_gradients"] for record in steps)
        assert all(record["sampled_model_updates"] for record in steps)
        print(
            "CORE_RESUME_CONTRACT",
            json.dumps({"model": worker.hf_config.model_type, "next_update_exact": True, "steps": steps}),
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("style", ["constant", "linear", "cosine", "inverse-square-root", "WSD"])
def test_scheduler_matches_miles_and_restores(style):
    args = SimpleNamespace(
        num_rollout=12,
        rollout_batch_size=1,
        n_samples_per_prompt=4,
        global_batch_size=4,
        lr_decay_iters=None,
        lr_warmup_init=0.0,
        lr=0.01,
        min_lr=0.001,
        lr_warmup_fraction=None,
        lr_warmup_iters=2,
        lr_decay_style=style,
        lr_wsd_decay_iters=3,
        lr_wsd_decay_style="cosine",
        override_lr_scheduler=False,
        use_checkpoint_lr_scheduler=True,
    )
    reference_optim = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=args.lr)
    reference = lr_scheduler.get_lr_scheduler(args, reference_optim)
    native = SimpleNamespace(param_groups=[{"lr": args.lr}])
    actual = scheduler.CoreLRScheduler(args, native)
    for index in range(15):
        assert actual.get_last_lr() == reference.get_last_lr()
        if index == 5:
            state = actual.state_dict()
            actual = scheduler.CoreLRScheduler(args, native)
            actual.load_state_dict(state)
            assert actual.get_last_lr() == reference.get_last_lr()
        reference_optim.step()
        reference.step()
        actual.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("recompute", [False, True])
def test_row_specialization_training_parity(parsed_args, tmp_path, monkeypatch, recompute):
    if json.loads((Path(parsed_args.hf_checkpoint) / "config.json").read_text())["model_type"] != "olmo3moe":
        pytest.skip("row specialization applies to routed MoE experts")
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/paired-rendezvous", rank=0, world_size=1)
    try:
        group = GroupInfo(rank=0, size=1, group=dist.group.WORLD, gloo_group=dist.group.WORLD)
        trivial = GroupInfo(rank=0, size=1, group=None)
        parallel.set_parallel_state(
            parallel.ParallelState(
                intra_dp=group,
                intra_dp_cp=group,
                cp=trivial,
                tp=trivial,
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
            )
        )
        monkeypatch.setattr(actor.distributed_utils, "get_gloo_group", lambda: dist.group.WORLD)
        results = []
        for mode in ("static", "dynamic"):
            args = copy.deepcopy(parsed_args)
            args.save = str(tmp_path / mode)
            args.olmo_core = dataclasses.replace(
                args.olmo_core, row_specialization=mode, activation_checkpointing=recompute
            )
            worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
            worker.args = args
            worker.train_module, worker.hf_config, worker.model_config = models.build_train_module(args)
            worker.model = worker.train_module.model
            experts = [m for m in worker.model.modules() if hasattr(m, "row_specialization")]
            assert experts and all(m.row_specialization == mode for m in experts)
            worker.optimizer = worker.train_module.optim
            captured_gradients = {}
            original_clip = worker.optimizer._clip_grad

            def capture_clip(optimizer=worker.optimizer, captured=captured_gradients, clip=original_clip):
                captured.update({k: v.detach().cpu().clone() for k, v in optimizer.main_grad.items()})
                return clip()

            monkeypatch.setattr(worker.optimizer, "_clip_grad", capture_clip)
            worker.lr_scheduler = scheduler.CoreLRScheduler(args, worker.optimizer)
            worker.clock = PolicyClock()
            worker.clock.published()
            worker.ref_module = None
            worker._heartbeat = SimpleNamespace(bump=lambda: None)
            lengths = [5, 13, 19, 25]
            rollout = {
                "tokens": [torch.arange(1, length + 1, device="cuda") for length in lengths],
                "total_lengths": lengths,
                "response_lengths": [3] * 4,
                "loss_masks": [torch.tensor([1, 0, 1], device="cuda") for _ in lengths],
                "rewards": [1.0, -1.0, 1.0, -1.0],
                "weight_versions": [["0"] for _ in lengths],
                "rollout_log_probs": [torch.full((3,), -5.0, device="cuda") for _ in lengths],
            }
            before = worker._score(worker.train_module, data.sample_batches(rollout, 128), use_replay=False)
            monkeypatch.setattr(
                actor.miles_data,
                "get_rollout_data",
                lambda *a, payload=rollout, **kw: (payload, contextlib.nullcontext()),
            )
            worker.train(0, None)
            after = worker._score(worker.train_module, data.sample_batches(rollout, 128), use_replay=False)
            assert worker.clock.completed_steps == 1
            assert any(not torch.equal(a, b) for a, b in zip(before, after))
            results.append(
                {
                    "before": [v.cpu() for v in before],
                    "after": [v.cpu() for v in after],
                    "weights": {k: v.detach().cpu().clone() for k, v in worker.model.state_dict().items()},
                    "optimizer": copy.deepcopy(worker.optimizer.state_dict()),
                    "gradients": captured_gradients,
                }
            )
            assert results[-1]["gradients"]
            del worker
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _single_rank_worker(args, tmp_path, monkeypatch, name):
    args = copy.deepcopy(args)
    args.save = str(tmp_path / name)
    monkeypatch.setattr(actor.distributed_utils, "get_gloo_group", lambda: dist.group.WORLD)
    worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = args
    worker.train_module, worker.hf_config, worker.model_config = models.build_train_module(args)
    worker.model = worker.train_module.model
    worker.optimizer = worker.train_module.optim
    worker.lr_scheduler = scheduler.CoreLRScheduler(args, worker.optimizer)
    worker.clock = PolicyClock()
    worker.clock.published()
    worker.ref_module = None
    worker._heartbeat = SimpleNamespace(bump=lambda: None)
    return worker


def _fixed_rollout():
    lengths = [5, 13, 19, 25]
    return {
        "tokens": [torch.arange(1, length + 1, device="cuda") for length in lengths],
        "total_lengths": lengths,
        "response_lengths": [3] * 4,
        "loss_masks": [torch.tensor([1, 0, 1], device="cuda") for _ in lengths],
        "rewards": [1.0, -1.0, 1.0, -1.0],
        "weight_versions": [["0"] for _ in lengths],
        "rollout_log_probs": [torch.full((3,), -5.0, device="cuda") for _ in lengths],
    }


def _contract_records(args):
    path = Path(args.save) / "training_contract_rank0.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_skipped_scoring_pass_matches_required_pass(parsed_args, tmp_path, monkeypatch):
    """One optimizer step per collection: the training forward is the old policy."""
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/scoring-rendezvous", rank=0, world_size=1)
    try:
        group = GroupInfo(rank=0, size=1, group=dist.group.WORLD, gloo_group=dist.group.WORLD)
        trivial = GroupInfo(rank=0, size=1, group=None)
        parallel.set_parallel_state(
            parallel.ParallelState(
                intra_dp=group,
                intra_dp_cp=group,
                cp=trivial,
                tp=trivial,
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
            )
        )
        results = []
        for required in (True, False):
            args = copy.deepcopy(parsed_args)
            args.olmo_core = dataclasses.replace(
                args.olmo_core, scoring_pass_required=required, scoring_check_interval=1000
            )
            worker = _single_rank_worker(args, tmp_path, monkeypatch, "required" if required else "auto")
            rollout = _fixed_rollout()
            monkeypatch.setattr(
                actor.miles_data,
                "get_rollout_data",
                lambda *a, payload=rollout, **kw: (payload, contextlib.nullcontext()),
            )
            scored = []
            original_score = worker._score

            def counting_score(module, batches, *, use_replay, original=original_score, scored=scored):
                scored.append(use_replay)
                return original(module, batches, use_replay=use_replay)

            monkeypatch.setattr(worker, "_score", counting_score)
            worker.train(0, None)
            worker.clock.published()
            rollout["weight_versions"] = [["1"] for _ in rollout["tokens"]]
            worker.train(1, None)
            assert worker.clock.completed_steps == 2
            records = _contract_records(worker.args)
            modes = [record["scoring_pass"] for record in records if record["event"] == "optimizer"]
            sources = [record["source"] for record in records if record["event"] == "scores"]
            checks = [record for record in records if record["event"] == "scoring_check"]
            if required:
                assert modes == ["standalone", "standalone"] and sources == ["standalone", "standalone"]
                assert scored == [True, True] and not checks
            else:
                assert modes == ["checked", "skipped"] and sources == ["standalone", "training_forward"]
                assert scored == [True], "only the checked first update ran the standalone pass"
                assert len(checks) == 1 and checks[0]["step"] == 0
                assert checks[0]["active_tokens"] == 8 and checks[0]["mean_abs"] == 0.0
            assert all(not record.get("skip_actor_forward_only") for record in records)
            assert worker.args.skip_actor_forward_only is False
            results.append(
                {
                    "weights": {k: v.detach().cpu().clone() for k, v in worker.model.state_dict().items()},
                    "optimizer": copy.deepcopy(worker.optimizer.state_dict()),
                }
            )
            del worker
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_scoring_check_failure_stops_before_the_optimizer_step(parsed_args, tmp_path, monkeypatch):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/check-rendezvous", rank=0, world_size=1)
    try:
        group = GroupInfo(rank=0, size=1, group=dist.group.WORLD, gloo_group=dist.group.WORLD)
        trivial = GroupInfo(rank=0, size=1, group=None)
        parallel.set_parallel_state(
            parallel.ParallelState(
                intra_dp=group,
                intra_dp_cp=group,
                cp=trivial,
                tp=trivial,
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
            )
        )
        worker = _single_rank_worker(parsed_args, tmp_path, monkeypatch, "drift")
        rollout = _fixed_rollout()
        monkeypatch.setattr(actor.miles_data, "get_rollout_data", lambda *a, **kw: (rollout, contextlib.nullcontext()))
        original_score = worker._score

        def drifted_score(module, batches, *, use_replay):
            # Emulate a no-grad kernel path that rounds differently from the gradient path.
            return [value + 0.01 for value in original_score(module, batches, use_replay=use_replay)]

        monkeypatch.setattr(worker, "_score", drifted_score)
        before = {name: tensor.detach().clone() for name, tensor in worker.model.state_dict().items()}
        with pytest.raises(RuntimeError, match="Standalone scoring differs"):
            worker.train(0, None)
        assert worker.clock.completed_steps == 0
        for name, value in worker.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert worker.args.skip_actor_forward_only is False
        # A wider tolerance admits the same drift and the update proceeds.
        worker.args.olmo_core = dataclasses.replace(worker.args.olmo_core, scoring_check_tolerance=0.05)
        worker.train(0, None)
        assert worker.clock.completed_steps == 1
        checks = [record for record in _contract_records(worker.args) if record["event"] == "scoring_check"]
        assert len(checks) == 1 and checks[0]["mean_abs"] == pytest.approx(0.01, abs=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_multiple_steps_per_collection_keep_the_standalone_pass(parsed_args, tmp_path, monkeypatch):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/steps-rendezvous", rank=0, world_size=1)
    try:
        group = GroupInfo(rank=0, size=1, group=dist.group.WORLD, gloo_group=dist.group.WORLD)
        trivial = GroupInfo(rank=0, size=1, group=None)
        parallel.set_parallel_state(
            parallel.ParallelState(
                intra_dp=group,
                intra_dp_cp=group,
                cp=trivial,
                tp=trivial,
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
            )
        )
        args = copy.deepcopy(parsed_args)
        # Four collected samples, two optimizer steps; the config validator requires a
        # lag budget covering the second step, so mirror that here.
        args.global_batch_size = 2
        args.olmo_core = dataclasses.replace(args.olmo_core, max_policy_lag=1)
        worker = _single_rank_worker(args, tmp_path, monkeypatch, "two-steps")
        assert worker._scoring_pass().standalone
        assert worker._scoring_pass().optimizer_steps_per_collection == 2
        rollout = _fixed_rollout()
        monkeypatch.setattr(actor.miles_data, "get_rollout_data", lambda *a, **kw: (rollout, contextlib.nullcontext()))
        worker.train(0, None)
        assert worker.clock.completed_steps == 2
        records = _contract_records(worker.args)
        assert [r["scoring_pass"] for r in records if r["event"] == "optimizer"] == ["standalone", "standalone"]
        assert not [r for r in records if r["event"] == "scoring_check"]
    finally:
        dist.destroy_process_group()
