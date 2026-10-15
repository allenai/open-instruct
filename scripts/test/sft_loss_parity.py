"""GPU integration check for the legacy SFT token-normalization loop.

Run through launch_sft_loss_parity.sh. Each torchrun invocation owns a fresh
Accelerate/DeepSpeed state, including fresh engines for checkpoint reloads.
"""

import argparse
import collections
import copy
import json
import math
import os
import types
from pathlib import Path

import torch
from accelerate import Accelerator, ParallelismConfig
from accelerate.utils import DeepSpeedPlugin, DeepSpeedSequenceParallelConfig, GradientAccumulationPlugin
from deepspeed import utils as ds_utils
from torch import distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import modeling_utils as hf_modeling_utils

from open_instruct import sft_loss_utils

try:
    from liger_kernel import transformers as liger_transformers
except ImportError:
    liger_transformers = None


GAS = 3
LR = 0.1
CLIP = 0.05
SEQUENCE_LENGTH = 16


def examples():
    rows = []
    for index in range(20):
        tokens = (torch.arange(SEQUENCE_LENGTH) * 7 + index * 3).remainder(63) + 1
        labels = torch.full_like(tokens, -100)
        labels[0] = 7  # This unshifted first target must never enter the denominator.
        count = (0, 1, 3, 9, 15)[index % 5]
        if count:
            labels[-count:] = tokens[-count:]
        rows.append(
            dict(
                input_ids=tokens,
                position_ids=torch.arange(SEQUENCE_LENGTH),
                labels=labels,
                sample_ids=torch.full_like(tokens, index),
            )
        )
    return rows


def flatten(tensors):
    return torch.cat([tensor.detach().float().reshape(-1) for tensor in tensors])


def check_close(actual, expected, name, bf16):
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    error = (actual - expected).norm().item() / max(expected.norm().item(), 1e-12)
    max_error = (actual - expected).abs().max().item()
    relative_limit, absolute_limit = (0.03, 0.002) if bf16 else (0.0001, 0.00001)
    if not torch.isfinite(actual).all() or error > relative_limit or max_error > absolute_limit:
        index = (actual - expected).abs().argmax().item()
        value = expected[index].item()
        diagnostics = dict(
            index=index,
            actual=actual[index].item(),
            expected=value,
            actual_norm=actual.norm().item(),
            expected_norm=expected.norm().item(),
            bf16_spacing=2 ** (math.frexp(abs(value))[1] - 8) if value else 0,
        )
        raise AssertionError(
            f"PARITY_MISMATCH {name}: relative_l2={error}, max_abs={max_error}, "
            f"limits=({relative_limit}, {absolute_limit}), diagnostics={json.dumps(diagnostics)}"
        )
    return dict(relative_l2=error, max_abs=max_error)


class Reference:
    def __init__(self, model):
        self.model = model
        self.original_sdpa = hf_modeling_utils.ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.parameters = list(model.parameters())
        self.masters = [torch.nn.Parameter(parameter.detach().float().clone()) for parameter in self.parameters]
        self.initial = flatten(self.masters).clone()
        self.optimizer = torch.optim.SGD(self.masters, lr=LR, momentum=0.5)
        self.seen = collections.Counter()
        self.steps = 0

    def backward(self, local_ids, sp_size):
        rank_ids = [None] * dist.get_world_size()
        dist.all_gather_object(rank_ids, local_ids)
        occurrences = collections.Counter(index for ids in rank_ids for index in ids)
        assert all(count % sp_size == 0 for count in occurrences.values()), occurrences
        indices = [index for index, count in sorted(occurrences.items()) for _ in range(count // sp_size)]
        self.seen.update(indices)
        rows = examples()
        batch = {
            key: torch.stack([rows[index][key] for index in indices]).to(self.parameters[0].device)
            for key in ("input_ids", "position_ids", "labels")
        }
        # Ulysses overwrites the process-wide attention registry during prepare.
        # The unsharded reference must keep the original noncollective attention.
        ulysses_sdpa = hf_modeling_utils.ALL_ATTENTION_FUNCTIONS["sdpa"]
        try:
            hf_modeling_utils.ALL_ATTENTION_FUNCTIONS["sdpa"] = self.original_sdpa
            logits = self.model(
                input_ids=batch["input_ids"], position_ids=batch["position_ids"], use_cache=False
            ).logits
        finally:
            hf_modeling_utils.ALL_ATTENTION_FUNCTIONS["sdpa"] = ulysses_sdpa
        targets = batch["labels"][..., 1:].reshape(-1)
        loss = F.cross_entropy(logits[..., :-1, :].float().reshape(-1, 64), targets)
        loss.backward()
        for master, parameter in zip(self.masters, self.parameters):
            master.grad = parameter.grad.detach().float().clone()
        return flatten([master.grad for master in self.masters]), int(targets.ne(-100).sum())

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.masters, CLIP)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.zero_grad()
        with torch.no_grad():
            for parameter, master in zip(self.parameters, self.masters):
                parameter.copy_(master)
        self.steps += 1

    def save(self, path):
        torch.save(
            dict(
                masters=[parameter.detach().cpu() for parameter in self.masters],
                optimizer=self.optimizer.state_dict(),
                seen=dict(self.seen),
                steps=self.steps,
            ),
            path,
        )

    def load(self, path):
        state = torch.load(path, map_location=self.parameters[0].device, weights_only=True)
        with torch.no_grad():
            for parameter, master, saved in zip(self.parameters, self.masters, state["masters"]):
                master.copy_(saved)
                parameter.copy_(saved)
        self.optimizer.load_state_dict(state["optimizer"])
        self.seen = collections.Counter(state["seen"])
        self.steps = state["steps"]


def main(args):
    assert torch.cuda.is_available(), "Real CUDA is required"
    assert int(os.environ["WORLD_SIZE"]) == 4, "Use four GPUs to test SP=2 and DP=2 together"
    torch.set_num_threads(2)
    torch.manual_seed(1728)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bf16 = args.precision == "bf16"
    parallelism = None
    if args.sp_size > 1:
        parallelism = ParallelismConfig(
            sp_backend="deepspeed",
            sp_size=args.sp_size,
            dp_shard_size=4 // args.sp_size,
            sp_handler=DeepSpeedSequenceParallelConfig(sp_seq_length_is_variable=True, sp_attn_implementation="sdpa"),
        )
    plugin = None
    if args.zero_stage:
        plugin = DeepSpeedPlugin(
            hf_ds_config=dict(
                train_micro_batch_size_per_gpu=1,
                gradient_accumulation_steps=GAS,
                gradient_clipping=CLIP,
                bf16=dict(enabled=bf16),
                fp16=dict(enabled=False),
                zero_optimization=dict(
                    stage=args.zero_stage,
                    reduce_scatter=True,
                    overlap_comm=False,
                    contiguous_gradients=True,
                    reduce_bucket_size=100000,
                    stage3_prefetch_bucket_size=100000,
                    stage3_param_persistence_threshold=0,
                ),
                steps_per_print=100000,
            ),
            zero3_init_flag=False,
        )
    accelerator = Accelerator(
        deepspeed_plugin=plugin,
        parallelism_config=parallelism,
        mixed_precision="bf16" if bf16 else "no",
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=GAS, sync_with_dataloader=False, sync_each_batch=args.sync_each_batch
        ),
    )
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=False,
        attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config).to(accelerator.device, dtype=torch.bfloat16 if bf16 else torch.float32)
    reference_model = copy.deepcopy(model)
    # Liger patches the HF class. Keep the reference's original unfused forward.
    reference_model.forward = types.MethodType(LlamaForCausalLM.forward, reference_model)
    reference = Reference(reference_model)
    if args.liger:
        assert liger_transformers is not None, "Pinned Liger dependency is required"
        liger_transformers.apply_liger_kernel_to_llama(
            rope=False, rms_norm=False, cross_entropy=False, fused_linear_cross_entropy=True, swiglu=False
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.5)
    dataloader = DataLoader(examples(), batch_size=1, shuffle=False)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    getattr(dataloader, "dl", dataloader).device = None
    divisor = sft_loss_utils.gradient_reduction_divisor(accelerator, args.sp_size)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint"
    if args.phase == "resume":
        accelerator.load_state(str(checkpoint))
        reference.load(output / "reference.pt")
        # Exercise the same resume iterator as finetune.py, including an odd
        # skipped SP microbatch count that ends inside the adapter's refill.
        active_dataloader = sft_loss_utils.skip_first_batches(dataloader, accelerator, GAS, args.sp_size)
    else:
        active_dataloader = dataloader
    parameters = list(accelerator.unwrap_model(model).parameters())
    results = []
    expected_gradient = None
    current_tokens = None
    current_microbatch = 0

    def full_parameters():
        if args.zero_stage:
            return flatten([ds_utils.safe_get_full_fp32_param(parameter) for parameter in parameters])
        return flatten(parameters)

    def inspect_gradients():
        if args.zero_stage:
            gradient = flatten([ds_utils.safe_get_full_grad(parameter) for parameter in parameters])
        else:
            gradient = flatten([parameter.grad for parameter in parameters])
        try:
            errors = check_close(gradient, expected_gradient, "gradient-before-clipping", bf16)
        except AssertionError:
            if accelerator.is_main_process:
                torch.save(
                    dict(actual=gradient.cpu(), expected=expected_gradient.cpu()), output / "gradient-mismatch.pt"
                )
            raise
        errors["unclipped_norm"] = gradient.norm().item()
        assert expected_gradient.norm().item() > CLIP, "The update check must exercise active gradient clipping"
        return errors

    def inspect_update(gradient_errors):
        reference.step()
        assert reference.steps <= math.ceil(len(dataloader) / GAS), "Iterator exceeded its epoch boundary"
        update_errors = check_close(
            full_parameters() - reference.initial,
            flatten(reference.masters) - reference.initial,
            "clipped-SGD-update",
            bf16,
        )
        if args.zero_stage:
            assert model.global_steps == reference.steps, (model.global_steps, reference.steps)
        result = dict(
            step=reference.steps,
            boundary=current_microbatch,
            tokens=current_tokens,
            gradient=gradient_errors,
            update=update_errors,
        )
        results.append(result)
        if accelerator.is_main_process:
            print("SFT_PARITY_STEP " + json.dumps(result), flush=True)

    if args.zero_stage:
        original_step = model.step

        def checked_step(*positional, **keyword):
            gradient_errors = inspect_gradients()
            result = original_step(*positional, **keyword)
            inspect_update(gradient_errors)
            return result

        model.step = checked_step

    local_ids = []
    empty_shards = 0
    emitted_no_logits = False
    for index, (batch, count, total, last, window_size) in enumerate(
        sft_loss_utils.iter_token_normalized_batches(active_dataloader, accelerator, GAS)
    ):
        current_microbatch = index + 1 + (GAS if args.phase == "resume" else 0)
        local_ids.append(int(batch.pop("sample_ids")[0, 0]))
        empty_shards += int(count == 0)
        if last:
            expected_gradient, current_tokens = reference.backward(local_ids, args.sp_size)
            assert current_tokens == total.item(), (current_tokens, total.item())
            local_ids = []
        batch = {key: value.to(accelerator.device) for key, value in batch.items()}
        if "shift_labels" in batch:
            batch["labels"] = batch["shift_labels"]
        batch = sft_loss_utils.prepare_empty_causal_lm_batch(batch, count)
        with sft_loss_utils.accumulate_token_window(accelerator, model, last, args.sync_each_batch):
            outputs = model(**batch, use_cache=False)
            emitted_no_logits |= getattr(outputs, "logits", None) is None
            mean_loss = sft_loss_utils.mean_causal_lm_loss(outputs.loss, count)
            loss = (
                mean_loss
                if args.legacy_control
                else sft_loss_utils.token_normalized_loss(
                    mean_loss, count, total, accelerator, window_size, args.sp_size, gradient_divisor=divisor
                )
            )
            loss = loss * args.loss_scale
            sft_loss_utils.backward_token_normalized_loss(accelerator, loss)
            if not args.zero_stage and last:
                gradient_errors = inspect_gradients()
                accelerator.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            optimizer.zero_grad()
            if not args.zero_stage and last:
                inspect_update(gradient_errors)
        if last and args.phase == "save":
            accelerator.save_state(str(checkpoint), safe_serialization=False)
            if accelerator.is_main_process:
                reference.save(output / "reference.pt")
            accelerator.wait_for_everyone()
            break
    if args.phase != "save":
        assert reference.steps == math.ceil(len(dataloader) / GAS), reference.steps
        assert reference.seen == collections.Counter(range(20)), reference.seen
        assert results[-1]["boundary"] == len(dataloader), results
        assert len(dataloader) % GAS, "A partial final accumulation window is mandatory"
    totals = torch.tensor([empty_shards, int(emitted_no_logits)], device=accelerator.device)
    totals = accelerator.reduce(totals, reduction="sum").tolist()
    assert totals[0] > 0, "The check must exercise empty local batches/shards"
    if args.liger:
        assert totals[1] > 0, "The fused path must actually omit logits"
    if accelerator.is_main_process:
        report = dict(args=vars(args), steps=results, empty_local_shards=totals[0], fused_without_logits=totals[1])
        (output / f"{args.phase}.json").write_text(json.dumps(report, indent=2))
        print("SFT_LOSS_GPU_CASE_PASS " + json.dumps(vars(args)), flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-stage", type=int, choices=(0, 2, 3), default=0)
    parser.add_argument("--sp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--liger", action="store_true")
    parser.add_argument("--sync-each-batch", action="store_true")
    parser.add_argument("--legacy-control", action="store_true")
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--phase", choices=("full", "save", "resume"), default="full")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
