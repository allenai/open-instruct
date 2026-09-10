"""Integration tests run inside the pinned MILES + patched Core image."""

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from miles.backends.fsdp_utils import lr_scheduler
from miles.backends.training_utils import parallel
from miles.utils import arguments
from miles.utils.ft_utils.process_group_utils import GroupInfo
from torch import distributed as dist
from transformers import AutoModelForCausalLM, Qwen3Config

from open_instruct.miles import actor, checkpoint, models
from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.state import PolicyClock


@pytest.fixture
def parsed_args(tmp_path, monkeypatch):
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
    model = AutoModelForCausalLM.from_config(hf).to(torch.bfloat16)
    model.save_pretrained(path)
    config = RunConfig(
        CoreConfig(attention_backend="torch", max_sequence_length=128, activation_checkpointing=False),
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
        worker.train_module, worker.hf_config, worker.model_config = models.build_train_module(args)
        worker.model = worker.train_module.model
        worker.optimizer = worker.train_module.optim
        worker.lr_scheduler = lr_scheduler.get_lr_scheduler(args, worker.optimizer)
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
    finally:
        dist.destroy_process_group()
