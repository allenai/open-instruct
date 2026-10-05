"""Compare accumulated gradients/updates with a full-batch token-mean reference."""

import contextlib
import copy
import os
import tempfile
import types
import unittest
from datetime import timedelta
from unittest import mock

import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState
from accelerate.utils import DistributedType, GradientAccumulationPlugin
from accelerate.utils.deepspeed import DeepSpeedEngineWrapper
from deepspeed.runtime.comm import coalesced_collectives
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM

from open_instruct import sft_loss_utils


class _TokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(20, dtype=torch.float64).reshape(5, 4) / 20)

    def forward(self, input_ids):
        return self.weight[input_ids]


def _batches(rank=0, sp_size=1):
    # Five microbatches: a full GAS=3 window, followed by a short window. The
    # counts differ across microbatches AND ranks, with an empty shard on rank 0.
    masks = ([0, 1, 4, 2, 3], [4, 2, 1, 4, 1], [1, 4, 2, 1, 4], [3, 1, 4, 3, 2])[rank]
    batches = []
    for index, count in enumerate(masks):
        inputs = (torch.arange(5) + index + rank).remainder(5).unsqueeze(0)
        labels = (inputs + 1).remainder(4)
        labels[..., 0] = 3  # non-ignored first label must NOT be counted
        labels[..., 1 + count :] = -100
        if sp_size > 1:
            # Ulysses has already shifted globally, before sharding.
            labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
            batches.append({"input_ids": inputs, "shift_labels": labels})
        else:
            batches.append({"input_ids": inputs, "labels": labels})
    return batches


def _loss(model, batch):
    logits = model(batch["input_ids"])
    if "shift_labels" in batch:
        labels = batch["shift_labels"]
    else:
        logits = logits[..., :-1, :]
        labels = batch["labels"][..., 1:]
    return F.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1), reduction="mean"), logits


def _reference(world_size, sp_size=1, aux=False):
    model = _TokenModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    gradients = []
    for start in (0, 3):
        logits, labels, aux_losses = [], [], []
        for rank in range(world_size):
            for batch in _batches(rank, sp_size)[start : start + 3]:
                pred = model(batch["input_ids"])
                if "shift_labels" in batch:
                    targets = batch["shift_labels"]
                else:
                    pred = pred[..., :-1, :]
                    targets = batch["labels"][..., 1:]
                logits.append(pred.reshape(-1, 4))
                labels.append(targets.reshape(-1))
                aux_losses.append(model.weight.square().mean() * (rank + 1))
        loss = F.cross_entropy(torch.cat(logits), torch.cat(labels))
        if aux:
            loss = loss + 0.2 * torch.stack(aux_losses).mean()
        loss.backward()
        gradients.append(model.weight.grad.clone())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.15)
        optimizer.step()
        optimizer.zero_grad()
    return model.weight.detach(), gradients


class _DeepSpeedTestEngine:
    """CPU stand-in for ZeRO's SP-sum/DP-average reduction and in-backward step.

    Runs through Accelerate's actual DeepSpeedEngineWrapper. This does not replace
    a real ZeRO/Ulysses smoke run, but catches late normalization, GAS double
    division, incorrect boundaries, and the wrong SP divisor.
    """

    def __init__(self, model, world_size, sp_size, zero_stage=2):
        self.model = model
        self.dp_size = world_size // sp_size
        self.zero_stage = zero_stage
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        self.boundary = False
        self.gradients = []
        self.backward_count = 0
        self.boundaries = []

    def set_gradient_accumulation_boundary(self, is_boundary):
        self.boundary = is_boundary

    def backward(self, loss, scale_wrt_gas=True):
        if scale_wrt_gas:
            loss = loss / 3
        loss.backward()
        self.backward_count += 1

    def step(self):
        assert self.boundary
        if self.zero_stage == 3 and dist.is_initialized():
            # Execute the installed DeepSpeed coalesced reduction itself; replace
            # only its transport shim with CPU gloo. Its world-size divisor is
            # not reimplemented by this test engine.
            transport = types.SimpleNamespace(
                get_rank=dist.get_rank,
                get_world_size=dist.get_world_size,
                reduce_scatter_fn=dist.reduce_scatter_tensor,
            )
            with mock.patch.object(coalesced_collectives, "dist", transport):
                partition = coalesced_collectives.reduce_scatter_coalesced([self.model.weight.grad])[0]
            partitions = [torch.empty_like(partition) for _ in range(dist.get_world_size())]
            dist.all_gather(partitions, partition)
            self.model.weight.grad.copy_(torch.cat(partitions).view_as(self.model.weight))
        else:
            if dist.is_initialized():
                dist.all_reduce(self.model.weight.grad)
            self.model.weight.grad.div_(self.dp_size)
        self.gradients.append(self.model.weight.grad.clone())
        self.boundaries.append(self.backward_count)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.15)
        self.optimizer.step()
        self.optimizer.zero_grad()


class _DeepSpeedTestAccelerator:
    def __init__(self, engine, world_size):
        self.device = torch.device("cpu")
        self.num_processes = world_size
        self.distributed_type = DistributedType.DEEPSPEED
        self.gradient_accumulation_steps = 3
        self.sync_gradients = False
        self.wrapper = DeepSpeedEngineWrapper(engine)
        self.state = types.SimpleNamespace(
            deepspeed_plugin=types.SimpleNamespace(
                deepspeed_config={"zero_optimization": {"stage": engine.zero_stage, "reduce_scatter": True}}
            )
        )

    def reduce(self, tensor, reduction):
        assert reduction == "sum"
        if dist.is_initialized():
            dist.all_reduce(tensor)
        return tensor

    def no_sync(self, model):
        return contextlib.nullcontext()

    def backward(self, loss, **kwargs):
        self.wrapper.backward(loss, sync_gradients=self.sync_gradients, **kwargs)


def _run_accumulation(model, accelerator, batches, optimizer=None, sp_size=1, aux=False, sync_each_batch=False):
    gradients, boundaries = [], []
    for index, (batch, count, total, last, size) in enumerate(
        sft_loss_utils.iter_token_normalized_batches(batches, accelerator, 3)
    ):
        with sft_loss_utils.accumulate_token_window(accelerator, model, last, sync_each_batch):
            batch = sft_loss_utils.prepare_empty_causal_lm_batch(batch, count)
            model_loss, _ = _loss(model, batch)
            aux_loss = None
            if aux:
                # Different router regularizers per rank deliberately avoid a
                # reference that would hide token-weighting of the auxiliary term.
                rank = dist.get_rank() if dist.is_initialized() else 0
                base = model.module if hasattr(model, "module") else model
                aux_loss = base.weight.square().mean() * (rank + 1)
                model_loss = model_loss + 0.2 * aux_loss
            mean_loss = sft_loss_utils.mean_causal_lm_loss(model_loss, count, aux_loss, 0.2)
            loss = sft_loss_utils.token_normalized_loss(
                mean_loss, count, total, accelerator, size, sp_size, aux_loss, 0.2
            )
            sft_loss_utils.backward_token_normalized_loss(accelerator, loss)
            if optimizer is not None:
                if accelerator.sync_gradients:
                    base = model.module if hasattr(model, "module") else model
                    gradients.append(base.weight.grad.clone())
                    boundaries.append(index + 1)
                    accelerator.clip_grad_norm_(model.parameters(), 0.15)
                optimizer.step()
                optimizer.zero_grad()
    return gradients, boundaries


def _collate_one_batch(items):
    return items[0]


def _distributed_worker(rank, world_size, init_file, output_dir, mode):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size, timeout=timedelta(seconds=90)
    )
    try:
        model = _TokenModel()
        sp_size = 2 if mode.startswith("sp") else 1
        aux = mode.startswith("sp")
        if mode.startswith("ddp"):
            # Initialize Accelerate via the normal launcher environment; use a
            # real DDP model, AcceleratedOptimizer, and Accelerator.backward.
            os.environ.update(
                RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world_size)
            )
            accelerator = Accelerator(
                cpu=True,
                gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=3, sync_with_dataloader=False),
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
            global_batches = [_batches(source_rank)[index] for index in range(5) for source_rank in range(world_size)]
            dataloader = DataLoader(global_batches, batch_size=1, collate_fn=_collate_one_batch)
            model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
            dataloader.device = None
            assert len(dataloader) == 5, "Every rank must keep its full shard after preparation"
            gradients, boundaries = _run_accumulation(
                model, accelerator, dataloader, optimizer, sync_each_batch=mode == "ddp_sync"
            )
            weight = accelerator.unwrap_model(model).weight.detach()
        else:
            engine = _DeepSpeedTestEngine(model, world_size, sp_size, zero_stage=3 if mode == "sp_z3" else 2)
            accelerator = _DeepSpeedTestAccelerator(engine, world_size)
            _run_accumulation(model, accelerator, _batches(rank, sp_size), sp_size=sp_size, aux=aux)
            weight, gradients, boundaries = model.weight.detach(), engine.gradients, engine.boundaries
        torch.save((weight, gradients, boundaries), os.path.join(output_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


class TestSftLossNormalization(unittest.TestCase):
    def tearDown(self):
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()

    def test_accelerate_updates_match_full_batch_with_partial_final_window(self):
        accelerator = Accelerator(
            cpu=True, gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=3, sync_with_dataloader=False)
        )
        model = _TokenModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        # A prepared dataloader marks end_of_dataloader while the last window is
        # being prefetched. There must still be exactly two optimizer updates.
        dataloader = DataLoader(_batches(), batch_size=None)
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        dataloader.device = None
        gradients, boundaries = _run_accumulation(model, accelerator, dataloader, optimizer)
        expected_weight, expected_gradients = _reference(1)
        self.assertEqual(boundaries, [3, 5])
        torch.testing.assert_close(model.weight, expected_weight)
        for actual, expected in zip(gradients, expected_gradients):
            torch.testing.assert_close(actual, expected)

    def test_single_microbatch_retains_local_token_mean(self):
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
        mean_loss = torch.tensor(2.0, requires_grad=True)
        loss = sft_loss_utils.token_normalized_loss(mean_loss, 7, torch.tensor(7), accelerator, 1)
        accelerator.backward(loss)
        torch.testing.assert_close(loss, mean_loss)
        torch.testing.assert_close(mean_loss.grad, torch.tensor(1.0))

    def test_deepspeed_wrapper_normalizes_before_clipping_and_step(self):
        model = _TokenModel()
        engine = _DeepSpeedTestEngine(model, 1, 1)
        accelerator = _DeepSpeedTestAccelerator(engine, 1)
        _run_accumulation(model, accelerator, _batches())
        expected_weight, expected_gradients = _reference(1)
        self.assertEqual(engine.boundaries, [3, 5])
        torch.testing.assert_close(model.weight, expected_weight)
        for actual, expected in zip(engine.gradients, expected_gradients):
            torch.testing.assert_close(actual, expected)

    def test_hf_causal_shift_and_gradient_match(self):
        torch.manual_seed(17)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=13, hidden_size=16, intermediate_size=24, num_hidden_layers=1, num_attention_heads=2
            )
        ).double()
        reference = copy.deepcopy(model)
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=3)
        batches = [
            {"input_ids": torch.tensor([[1, 2, 3, 4]]), "labels": torch.tensor([[1, -100, 3, 4]])},
            {"input_ids": torch.tensor([[2, 1, 5, 4]]), "labels": torch.tensor([[2, -100, -100, 4]])},
        ]
        for batch, count, total, last, size in sft_loss_utils.iter_token_normalized_batches(batches, accelerator, 3):
            with sft_loss_utils.accumulate_token_window(accelerator, model, last):
                outputs = model(**batch)
                loss = sft_loss_utils.token_normalized_loss(outputs.loss, count, total, accelerator, size)
                sft_loss_utils.backward_token_normalized_loss(accelerator, loss)
        reference(**{key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}).loss.backward()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
        self.assertEqual(sum(sft_loss_utils.count_causal_lm_tokens(batch) for batch in batches), 3)

    def test_empty_microbatch_keeps_finite_hf_gradients_and_original_labels(self):
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=13, hidden_size=16, intermediate_size=24, num_hidden_layers=1, num_attention_heads=2
            )
        )
        batch = {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.full((1, 3), -100)}
        prepared = sft_loss_utils.prepare_empty_causal_lm_batch(batch, 0)
        outputs = model(**prepared)
        loss = sft_loss_utils.mean_causal_lm_loss(outputs.loss, 0)
        loss.backward()
        self.assertTrue(batch["labels"].eq(-100).all())
        self.assertEqual(loss.item(), 0)
        for parameter in model.parameters():
            self.assertTrue(parameter.grad.eq(0).all())

    def test_shift_labels_count_includes_first_shard_position(self):
        batch = {"labels": torch.tensor([[1, -100, -100]]), "shift_labels": torch.tensor([[2, -100, 3]])}
        self.assertEqual(sft_loss_utils.count_causal_lm_tokens(batch), 2)

    def test_empty_window_rejected_before_backward(self):
        accelerator = Accelerator(cpu=True)
        batches = [{"labels": torch.tensor([[1, -100, -100]])}]
        with self.assertRaisesRegex(ValueError, "no supervised causal LM tokens"):
            list(sft_loss_utils.iter_token_normalized_batches(batches, accelerator, 3))

    def test_ddp_unequal_rank_tokens_match_full_batch(self):
        self._assert_distributed_matches_reference("ddp", 2, 1)

    def test_ddp_sync_each_batch_matches_full_batch(self):
        self._assert_distributed_matches_reference("ddp_sync", 2, 1)

    def test_sp_and_dp_with_empty_shard_and_auxiliary_loss_match_reference(self):
        self._assert_distributed_matches_reference("sp", 4, 2)

    def test_zero3_coalesced_reduction_with_sp_matches_reference(self):
        self._assert_distributed_matches_reference("sp_z3", 4, 2)

    def test_inconsistent_zero3_sp_config_is_rejected(self):
        accelerator = _DeepSpeedTestAccelerator(_DeepSpeedTestEngine(_TokenModel(), 4, 2, zero_stage=3), 4)
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["reduce_scatter"] = False
        with self.assertRaisesRegex(ValueError, "requires reduce_scatter=true"):
            sft_loss_utils.gradient_reduction_divisor(accelerator, 2)
        with self.assertRaisesRegex(ValueError, "divide the process count"):
            sft_loss_utils.gradient_reduction_divisor(accelerator, 3)

    def _assert_distributed_matches_reference(self, mode, world_size, sp_size):
        expected_weight, expected_gradients = _reference(world_size, sp_size, aux=mode.startswith("sp"))
        with tempfile.TemporaryDirectory() as directory:
            mp.start_processes(
                _distributed_worker,
                args=(world_size, os.path.join(directory, "init"), directory, mode),
                nprocs=world_size,
                start_method="spawn",
            )
            for rank in range(world_size):
                weight, gradients, boundaries = torch.load(
                    os.path.join(directory, f"rank{rank}.pt"), weights_only=True
                )
                self.assertEqual(boundaries, [3, 5])
                torch.testing.assert_close(weight, expected_weight)
                for actual, expected in zip(gradients, expected_gradients):
                    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
