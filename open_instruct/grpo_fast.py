# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------
# Part of the code is adapted from https://github.com/OpenRLHF/OpenRLHF
# which has the following license:
# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# isort: off
import contextlib
import os
import pathlib
from concurrent import futures
from datetime import timedelta

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
with contextlib.suppress(Exception):
    import deepspeed
    from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
    from deepspeed.utils import groups

from open_instruct import data_loader as data_loader_lib
from open_instruct import data_types, grpo_utils, utils
from open_instruct.data_loader import DataPreparationActor, accumulate_inference_batches, add_prompt_to_generator

# isort: on
import asyncio
import dataclasses
import logging
import math
import random
import shutil
import socket
import threading
import time
from dataclasses import asdict
from queue import Empty, Full, Queue
from typing import Any

import backoff
import datasets
import numpy as np
import pandas as pd
import ray
import torch
import torch.distributed as dist
import torch.utils
import torch.utils.data
import wandb
from datasets import Dataset
from huggingface_hub import HfApi
from peft import PeftModel, get_peft_model_state_dict
from ray.util import queue as ray_queue
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rich.pretty import pprint
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, get_scheduler
from transformers.integrations import HfDeepSpeedConfig

from open_instruct import logger_utils, vllm_utils
from open_instruct.actor_manager import ActorManager
from open_instruct.data_types import ShutdownSentinel
from open_instruct.dataset_transformation import (
    INPUT_IDS_PROMPT_KEY,
    TOOLS_COLUMN_KEY,
    TokenizerConfig,
    get_cached_dataset_tulu,
    validate_dataset_tools,
    visualize_token,
)
from open_instruct.ground_truth_utils import RewardConfig, build_all_verifiers, cleanup_all_llm_judge_clients
from open_instruct.model_utils import (
    ModelConfig,
    disable_dropout_in_model,
    entropy_from_logits,
    estimate_kl,
    get_olmo3_generation_config,
    load_ref_policy,
    log_softmax_and_gather,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
)
from open_instruct.rl_utils import Timer, masked_mean
from open_instruct.tools.parsers import create_tool_parser
from open_instruct.tools.tools import TOOL_REGISTRY, GenericMCPToolConfig
from open_instruct.tools.utils import BaseToolConfig, ParsedToolConfig, ToolsConfig
from open_instruct.utils import (
    INVALID_LOGPROB,
    ArgumentParserPlus,
    BeakerRuntimeConfig,
    RayProcess,
    UlyssesSPSplitter,
    _z3_params_to_fetch,
    clean_last_n_checkpoints_deepspeed,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
    get_wandb_tags,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_update_beaker_description,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    ray_get_with_progress,
    sync_gs_bucket,
)

logger = logger_utils.setup_logger(__name__)

CHECKPOINT_COMPLETE_MARKER = ".checkpoint_complete"


def to_device_inplace(tensors_list: list[torch.Tensor], device: torch.device):
    for i in range(len(tensors_list)):
        tensors_list[i] = tensors_list[i].to(device, non_blocking=True)


@ray.remote(num_gpus=1)
class PolicyTrainerRayProcess(RayProcess):
    def __init__(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        master_addr: str | None,
        master_port: int | None,
        args: grpo_utils.ExperimentConfig,
        streaming_config: data_loader_lib.StreamingDataLoaderConfig,
        vllm_config: data_loader_lib.VLLMConfig,
        data_prep_actor_name: str,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__(world_size, rank, local_rank, master_addr, master_port)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.num_mini_batches = args.num_mini_batches
        self._args = args
        self.streaming_config = streaming_config
        self.vllm_config = vllm_config
        self.world_size = world_size
        self.local_rank = local_rank
        self.dp_world_size = world_size // args.sequence_parallel_size
        self._data_prep_actor_name = data_prep_actor_name

    def get_dataloader_state(self) -> dict[str, Any]:
        return self._streaming_dataloader.state_dict()

    def load_dataloader_state(self, state_dict: dict[str, Any]) -> None:
        self._streaming_dataloader.load_state_dict(state_dict)

    def from_pretrained(
        self,
        args: grpo_utils.ExperimentConfig,
        model_config: ModelConfig,
        beaker_config: BeakerRuntimeConfig,
        wandb_url: str,
        tokenizer: PreTrainedTokenizer,
    ) -> int:
        # ------------------------------------------------------------
        # Monkey patch to load checkpoints with `weights_only=False`
        # otherwise it errors out with:
        # `_pickle.UnpicklingError: Weights only load failed. ` with pytorch 2.6.0
        from deepspeed.runtime.checkpoint_engine import torch_checkpoint_engine  # noqa: PLC0415
        from deepspeed.utils import logger  # noqa: PLC0415

        def load(self, path: str, map_location=None):
            logger.info(f"[Torch] Loading checkpoint from {path}...")
            partition = torch.load(path, map_location=map_location, weights_only=False)
            logger.info(f"[Torch] Loaded checkpoint from {path}.")
            return partition

        torch_checkpoint_engine.TorchCheckpointEngine.load = load

        # ------------------------------------------------------------
        self.args = args
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.beaker_config = beaker_config
        self.wandb_url = wandb_url
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(self.local_rank)

        # Set seeds for this worker (different per rank to avoid correlation)
        worker_seed = args.seed + self.local_rank
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

        # Pre-initialize torch.distributed WITHOUT device_id to avoid NCCL hangs.
        # DeepSpeed 0.17.3 and up sets device_id in init_process_group which can cause hangs
        # when multiple process groups exist (e.g., for weight sync to vLLM).
        # By initializing first, DeepSpeed will detect it and wrap it instead of re-initializing.
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", timeout=timedelta(minutes=args.backend_timeout))
        deepspeed.init_distributed(timeout=timedelta(minutes=args.backend_timeout))

        ds_config = get_train_ds_config(
            offload=args.deepspeed_offload_param,
            adam_offload=args.deepspeed_offload_optimizer,
            stage=args.deepspeed_stage,
            bf16=True,
            zpg=args.deepspeed_zpg,
            sequence_parallel_size=args.sequence_parallel_size,
        )
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["gradient_accumulation_steps"] = 1
        # @vwxyzjn: MAGIC: it's actually needed to initialize this `dschf`, so
        # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
        # next line instructs transformers to partition the model directly over multiple gpus using
        # deepspeed.zero.Init when model's `from_pretrained` method is called.
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        logger.info(f"Deepspeed config: {dschf=}")

        # set sequence parallel
        # note this returns None if sequence_parallel_size == 1
        self.mpu = UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=model_config.model_name_or_path,
            core_attn_implementation=model_config.attn_implementation,
            sequence_parallel_size=args.sequence_parallel_size,
            micro_batch_size=args.per_device_train_batch_size,
            seq_length_is_variable=True,
        )
        self.policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            dtype=torch.bfloat16,
            attn_implementation=model_config.attn_implementation,
            use_cache=False,
            **({"device_map": {"": self.local_rank}} if args.deepspeed_stage != 3 else {}),
        )
        disable_dropout_in_model(self.policy)
        self.policy.gradient_checkpointing_enable()
        if args.set_weight_decay_on_bias_and_norm:
            optim_params = get_optimizer_grouped_parameters(self.policy, args.weight_decay)
        else:
            optim_params = self.policy.parameters()
        self.optimizer = torch.optim.AdamW(optim_params, lr=args.learning_rate, fused=args.fused_optimizer)
        num_scheduler_steps = args.num_training_steps * args.num_epochs * args.num_mini_batches
        warm_up_steps = args.warm_up_steps
        if args.warmup_ratio > 0.0:
            warm_up_steps = int(num_scheduler_steps * args.warmup_ratio)
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=num_scheduler_steps,
        )
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.policy,
            optimizer=self.optimizer,
            config=ds_config,
            lr_scheduler=scheduler,
            dist_init_required=False,
            mpu=self.mpu,
        )
        optimization_steps_done = 0
        if args.checkpoint_state_dir:
            # check if the dir exists
            if not os.path.exists(args.checkpoint_state_dir):
                logger.warning(
                    f"Skipping loading checkpoint state from {args.checkpoint_state_dir} because it does not exist!"
                )
            else:
                # remove mpu for loading checkpoints, add it back after loading
                old_mpu = self.mpu
                self.model.mpu = None
                path, states = self.model.load_checkpoint(
                    args.checkpoint_state_dir,
                    load_module_strict=True,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True,
                    load_module_only=False,
                )
                self.model.mpu = old_mpu
                if path is None:
                    raise ValueError(f"Failed to load checkpoint from {args.checkpoint_state_dir}")
                optimization_steps_done = states["training_step"]

                rng_states = states["rng_states"]
                torch.set_rng_state(rng_states["torch_cpu_rng_state"])
                np.random.set_state(rng_states["numpy_rng_state"])
                random.setstate(rng_states["python_rng_state"])

                if torch.cuda.is_available() and "torch_cuda_rng_states" in rng_states:
                    # device_str, e.g. "cuda:0"
                    for device_str, rng_state in rng_states["torch_cuda_rng_states"].items():
                        device_id = int(device_str.split(":")[1])
                        torch.cuda.set_rng_state(rng_state, device_id)
                    if "torch_cuda_rng_state_all" in rng_states:
                        torch.cuda.set_rng_state_all(rng_states["torch_cuda_rng_state_all"])

                logger.info(f"{self.rank=}: Restored RNG states from checkpoint")

                # Save reference policy path to load later (after ref_policy is initialized)
                self.ref_policy_checkpoint_path = None
                if args.load_ref_policy and states.get("ref_policy_saved", False):
                    ref_policy_dir = os.path.join(args.checkpoint_state_dir, "ref_policy")
                    model_path = os.path.join(ref_policy_dir, "pytorch_model.bin")
                    if os.path.exists(model_path):
                        self.ref_policy_checkpoint_path = model_path
                        logger.info(f"{self.rank=}: Will load reference policy from {model_path}")

                logger.info(
                    f"{self.rank=}: Loaded checkpoint from {args.checkpoint_state_dir} with {optimization_steps_done=}"
                )
        self.model.train()

        # reference model
        if args.load_ref_policy:
            ds_config, self.ref_policy_hf_ds_config = get_eval_ds_config(
                offload=False,
                # inference model only has stage 3 (sharding) or stage 0 (no sharding)
                # stage 2 is optimizer sharding which doesn't apply to inference
                stage=args.deepspeed_stage if args.deepspeed_stage == 3 else 0,
                bf16=True,
                per_device_train_batch_size=args.per_device_train_batch_size,
            )

            self.ref_policy: PreTrainedModel = load_ref_policy(
                model_config=model_config,
                ds_config=ds_config,
                deepspeed_stage=args.deepspeed_stage,
                local_rank=self.local_rank,
                device=self.device,
                rank=self.rank,
                checkpoint_path=self.ref_policy_checkpoint_path
                if hasattr(self, "ref_policy_checkpoint_path")
                else None,
                mpu=self.mpu,
                ref_policy_update_freq=args.ref_policy_update_freq,
                alpha=args.alpha,
            )
        self.local_metrics = utils.MetricsTracker(max_metrics=512, device=self.device)

        if self.mpu is not None:
            self.splitter = UlyssesSPSplitter(
                sp_rank=groups._get_sequence_parallel_rank(),
                sp_group=groups._get_sequence_parallel_group(),
                sp_world_size=groups._get_sequence_parallel_world_size(),
                device=self.device,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        else:
            self.splitter = None

        # dp_rank = which data-parallel group this worker belongs to
        # With SP, workers in the same SP group share the same dp_rank
        dp_rank = self.rank // args.sequence_parallel_size
        assert dp_rank < self.dp_world_size

        # Verify SP groups are consecutive as we assume for above logic (e.g., [0,1,2,3], [4,5,6,7], ...)
        # getting the dp_rank directly does not work right now with the mpus :/
        if self.mpu is not None:
            sp_group = groups._get_sequence_parallel_group()
            sp_ranks = sorted(torch.distributed.get_process_group_ranks(sp_group))
            expected = list(range(dp_rank * args.sequence_parallel_size, (dp_rank + 1) * args.sequence_parallel_size))
            assert sp_ranks == expected, f"SP group {sp_ranks} != expected {expected}"

        self._streaming_dataloader = streaming_config.build_dataloader(
            data_prep_actor_name=self._data_prep_actor_name,
            tokenizer=tokenizer,
            dp_rank=dp_rank,
            fs_local_rank=self.local_rank,
            num_training_steps=args.num_training_steps,
            work_dir=args.output_dir,
            dp_world_size=self.dp_world_size,
        )
        self.dataloader = iter(self._streaming_dataloader)

        return optimization_steps_done

    def forward(
        self,
        model: PreTrainedModel,
        query_response: torch.LongTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        pad_token_id: int,
        temperature: float,
        return_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Replace pad tokens with 0s so that we don't run into index out of bounds errors
        padding_mask = query_response != pad_token_id
        input_ids = torch.masked_fill(query_response, ~padding_mask, 0)
        # NOTE: the [:-1] and [1:] are because the logits and generated tokens are off by 1 in index
        output = model(
            input_ids=input_ids[:, :-1],
            # @vwxyzjn: without clamp, we get index out of bounds errors; TODO: investigate
            attention_mask=attention_mask[:, :-1].clamp(0, 1),
            position_ids=position_ids[:, :-1],
            return_dict=True,
        )
        logits = output.logits
        logits /= temperature + 1e-7
        logprob = log_softmax_and_gather(logits, input_ids[:, 1:])

        # For now, entropy is just for monitoring, and we don't pass gradients through it.
        entropy = None
        if return_entropy:
            with torch.no_grad():
                entropy = entropy_from_logits(logits)

        return logprob, entropy

    def setup_model_update_group(self, vllm_engines):
        self.vllm_engines = vllm_engines
        self.model_update_group = None
        if self.rank == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            vllm_num_engines, vllm_tensor_parallel_size = (
                self.vllm_config.vllm_num_engines,
                self.vllm_config.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
            backend = self.vllm_config.vllm_sync_backend
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    "openrlhf",
                    backend=backend,
                    timeout_minutes=self.args.backend_timeout,
                )
                for i, engine in enumerate(vllm_engines)
            ]
            torch.cuda.set_device(self.local_rank)
            self.model_update_group = vllm_utils.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="openrlhf",
                timeout=timedelta(minutes=self.args.backend_timeout),
            )
            ray_get_with_progress(refs, desc="Initializing vLLM process groups", timeout=600)
        torch.distributed.barrier()

    def broadcast_to_vllm(self):
        # avoid OOM
        torch.cuda.empty_cache()
        # Ensure CUDA device is set before broadcast operations.
        # DeepSpeed 0.17.3+ sets device_id in init_process_group which affects NCCL device binding.
        torch.cuda.set_device(self.local_rank)
        return vllm_utils.broadcast_weights_to_vllm(
            model=self.model.module,
            vllm_engines=self.vllm_engines,
            model_update_group=self.model_update_group,
            deepspeed_stage=self.args.deepspeed_stage,
            gather_whole_model=self.args.gather_whole_model,
        )

    def update_ref_policy(self):
        if not self.args.load_ref_policy:
            return
        for ref_param, param in zip(self.ref_policy.parameters(), self.model.parameters()):
            if self.args.deepspeed_stage == 3:
                with deepspeed.zero.GatheredParameters([param, ref_param], modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)
            else:
                ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)

    def compute_logprobs(
        self, model: PreTrainedModel, data_BT: data_types.CollatedBatchData, pad_token_id: int, use_grad: bool = False
    ) -> list[torch.Tensor]:
        logprobs_BT: list[torch.Tensor] = []

        context = contextlib.nullcontext() if use_grad else torch.no_grad()
        with context:
            for i in range(len(data_BT.query_responses)):
                logprob_BT, _ = self.forward(
                    model,
                    data_BT.query_responses[i],
                    data_BT.attention_masks[i],
                    data_BT.position_ids[i],
                    pad_token_id,
                    self.streaming_config.temperature,
                    return_entropy=False,
                )

                response_mask_BT = data_BT.response_masks[i]
                logprob_BT = torch.masked_fill(logprob_BT, ~response_mask_BT[:, 1:], INVALID_LOGPROB)
                logprobs_BT.append(logprob_BT)

                torch.cuda.empty_cache()

        return logprobs_BT

    def calculate_token_counts(
        self, accumulation_steps: int, data_BT: data_types.CollatedBatchData
    ) -> dict[int, float]:
        accumulation_counts: dict[int, float] = {}
        local_counts = [mask[:, 1:].sum().float() for mask in data_BT.response_masks]
        if not local_counts:
            return accumulation_counts

        counts_tensor = torch.stack(local_counts)
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)

        for i, count in enumerate(counts_tensor):
            group_idx = i // accumulation_steps
            key = int(group_idx * accumulation_steps)
            accumulation_counts[key] = accumulation_counts.get(key, 0.0) + count.item()

        return accumulation_counts

    def _compute_loss_metrics(
        self, loss_stats_B: dict[str, torch.Tensor], total_valid_tokens: int
    ) -> dict[str, float]:
        """Compute weighted average metrics from per-batch loss statistics."""
        token_counts = loss_stats_B["token_count"]
        total_tokens = token_counts.sum()
        # Zero weights when no tokens - all weighted sums become 0
        weights = token_counts / total_tokens if total_tokens > 0 else torch.zeros_like(token_counts)

        if self.args.load_ref_policy:
            for j in range(4):
                self.local_metrics[f"objective/kl{j}_avg"] = (loss_stats_B["kl"][j] * weights).sum()
            self.local_metrics["loss/kl_avg"] = (loss_stats_B["kl_loss"] * weights).sum()
        self.local_metrics["loss/policy_avg"] = (loss_stats_B["pg_loss"] * weights).sum()
        self.local_metrics["loss/total_avg"] = (loss_stats_B["loss"] * weights).sum()
        self.local_metrics["policy/clipfrac_avg"] = (loss_stats_B["pg_clipfrac"] * weights).sum()
        self.local_metrics["val/ratio"] = (loss_stats_B["ratio"] * weights).sum()
        weighted_mean_ratio = self.local_metrics["val/ratio"]
        self.local_metrics["val/ratio_var"] = (weights * (loss_stats_B["ratio"] - weighted_mean_ratio) ** 2).sum()
        if self.args.record_entropy:
            self.local_metrics["policy/entropy_avg"] = (loss_stats_B["entropy"] * weights).sum()

        self.local_metrics["lr"] = self.scheduler.get_last_lr()[0]
        self.local_metrics["_token_count"] = total_valid_tokens

    def step(self):
        """Execute one training step: fetch data from the dataloader and train on it.

        Returns:
            Tuple of (metrics_list, array_metrics) from training.
        """
        batch_data = next(self.dataloader)
        data_BT = batch_data["batch"]
        if len(data_BT) == 0:
            logger.warning("[Training] Empty batch received, skipping training step")
            return [], {}

        # split batch for sequence parallelism. Do before moving data to GPU.
        if self.splitter is not None:
            with Timer("✂️ Splitting batch for SP", noop=self.rank != 0):
                data_BT = self.splitter.split_collated_batch(data_BT)

        for f in dataclasses.fields(data_BT):
            to_device_inplace(getattr(data_BT, f.name), self.device)
        data_BT.response_masks = [mask.bool() for mask in data_BT.response_masks]
        num_samples = len(data_BT)
        accumulation_steps = max(math.ceil(num_samples / self.num_mini_batches - 0.5), 1)
        leftover = num_samples % accumulation_steps
        if leftover > 0:
            data_BT = data_BT[:-leftover]
            logger.warning(f"{leftover} samples are dropped due to batch size {self.num_mini_batches}")

        num_mini_batches = len(data_BT.query_responses) // accumulation_steps

        ref_logprobs_BT: list[torch.Tensor] = []
        if self.args.load_ref_policy:
            with Timer("Inference Calculation", noop=self.rank != 0):
                ref_logprobs_BT = self.compute_logprobs(self.ref_policy, data_BT, self.pad_token_id, use_grad=False)

        # if we have multiple minibatches, we need to calculate the old logprobs for each minibatch
        # following gtrl scripts in just doing this on the current active policy, rather than use the logprobs
        # from the generator (note that async mode means these are a bit diff!)
        old_logprobs_BT: list[torch.Tensor | None] = [None for _ in range(len(data_BT.query_responses))]
        if num_mini_batches > 1:
            with Timer("Old logprobs Calculation", noop=self.rank != 0):
                local_old_logprobs_BT = None
                if not self.args.use_vllm_logprobs:
                    local_old_logprobs_BT = self.compute_logprobs(
                        self.model, data_BT, self.pad_token_id, use_grad=False
                    )

                with torch.no_grad():
                    for i in range(len(data_BT.query_responses)):
                        vllm_old_logprob_BT = data_BT.vllm_logprobs[i][:, 1:]
                        vllm_old_logprob_BT = torch.masked_fill(
                            vllm_old_logprob_BT, ~data_BT.response_masks[i][:, 1:], INVALID_LOGPROB
                        )
                        vllm_old_logprob_BT = torch.nan_to_num(vllm_old_logprob_BT, nan=INVALID_LOGPROB)

                        if self.args.use_vllm_logprobs:
                            old_logprobs_BT[i] = vllm_old_logprob_BT
                        else:
                            old_logprobs_BT[i] = local_old_logprobs_BT[i]

                        torch.cuda.empty_cache()

        local_step = 0
        num_samples = len(data_BT.query_responses)
        # Pre-compute token counts per sample (for weighted averaging across SP ranks)
        # This only needs to be done once since response_masks don't change across epochs
        token_counts_per_sample = torch.stack([mask[:, 1:].sum().float() for mask in data_BT.response_masks])
        total_valid_tokens = token_counts_per_sample.sum().item()
        device = token_counts_per_sample.device
        # Do multiple epochs of training on on-policy data (PPO-style), with a fresh random shuffle in each epoch
        with Timer("[Training Processes] Loss calculation", noop=self.rank != 0):
            loss_stats_B: dict[str, torch.Tensor] = {
                "kl": torch.zeros(4, num_samples, device=device),
                "kl_loss": torch.zeros(num_samples, device=device),
                "pg_clipfrac": torch.zeros(num_samples, device=device),
                "pg_loss": torch.zeros(num_samples, device=device),
                "loss": torch.zeros(num_samples, device=device),
                "ratio": torch.zeros(num_samples, device=device),
                "entropy": torch.zeros(num_samples, device=device),
                "token_count": token_counts_per_sample,
            }
            for epoch_idx in range(self.args.num_epochs):
                # Pre-compute total tokens for each accumulation group if using "token" normalization
                # This ensures all minibatches in an accumulation group are normalized by the same total
                if self.args.loss_denominator == "token":
                    accumulation_token_counts = self.calculate_token_counts(accumulation_steps, data_BT)
                else:
                    accumulation_token_counts = {
                        int(group_idx * accumulation_steps): float(self.args.loss_denominator)
                        for group_idx in range((len(data_BT.query_responses) // accumulation_steps) + 1)
                    }

                for i in range(num_samples):
                    response_mask_BT = data_BT.response_masks[i][:, 1:]
                    # retrieve the loss denominator for the current batch
                    batch_start = (i // accumulation_steps) * accumulation_steps
                    loss_denominator = accumulation_token_counts[batch_start]
                    local_logprobs_BT, entropy_BT = self.forward(
                        self.model,
                        data_BT.query_responses[i],
                        data_BT.attention_masks[i],
                        data_BT.position_ids[i],
                        self.pad_token_id,
                        self.streaming_config.temperature,
                        return_entropy=self.args.record_entropy,
                    )
                    local_logprobs_BT = torch.masked_fill(local_logprobs_BT, ~response_mask_BT, INVALID_LOGPROB)
                    vllm_logprobs_BT = data_BT.vllm_logprobs[i][:, 1:]
                    vllm_logprobs_BT = torch.masked_fill(vllm_logprobs_BT, ~response_mask_BT, INVALID_LOGPROB)
                    vllm_logprobs_BT = torch.nan_to_num(vllm_logprobs_BT, nan=INVALID_LOGPROB)

                    # Compare vLLM logprobs with local logprobs
                    with torch.no_grad():
                        valid_mask_BT = response_mask_BT & ~torch.isnan(vllm_logprobs_BT)
                        logprob_diff_BT = (local_logprobs_BT - vllm_logprobs_BT).abs()
                        masked_diff_BT = torch.masked_fill(logprob_diff_BT, ~valid_mask_BT, 0.0)
                        mean_diff = masked_diff_BT.sum() / valid_mask_BT.sum() if valid_mask_BT.sum() > 0 else 0.0
                        max_diff = masked_diff_BT.max()
                        std_diff = masked_diff_BT[valid_mask_BT].std() if valid_mask_BT.sum() > 1 else 0.0

                        self.local_metrics["debug/vllm_vs_local_logprob_diff_mean"] = float(mean_diff)
                        self.local_metrics["debug/vllm_vs_local_logprob_diff_max"] = float(max_diff)
                        self.local_metrics["debug/vllm_vs_local_logprob_diff_std"] = float(std_diff)

                        reverse_kl_BT = torch.exp(vllm_logprobs_BT) * (vllm_logprobs_BT - local_logprobs_BT)
                        masked_reverse_kl_BT = torch.masked_fill(reverse_kl_BT, ~valid_mask_BT, 0.0)
                        mean_reverse_kl = (
                            masked_reverse_kl_BT.sum() / valid_mask_BT.sum() if valid_mask_BT.sum() > 0 else 0.0
                        )
                        self.local_metrics["debug/vllm_local_reverse_kl"] = float(mean_reverse_kl)

                    new_logprobs_BT = local_logprobs_BT

                    # Cache the old logprobs
                    if num_mini_batches > 1:
                        old_logprob_BT = old_logprobs_BT[i]
                    else:
                        with torch.no_grad():
                            if epoch_idx == 0:
                                if self.args.use_vllm_logprobs:
                                    old_logprobs_BT[i] = vllm_logprobs_BT
                                else:
                                    old_logprobs_BT[i] = local_logprobs_BT.detach()
                            old_logprob_BT = old_logprobs_BT[i]

                    old_logprobs_mask_BT = old_logprob_BT != INVALID_LOGPROB
                    assert torch.all(old_logprobs_mask_BT == response_mask_BT), (
                        f"Old logprobs mask should match response mask. "
                        f"old_mask sum={old_logprobs_mask_BT.sum()}, "
                        f"response_mask sum={response_mask_BT.sum()}"
                    )

                    # Calculate the policy's loss
                    logprobs_diff_BT = new_logprobs_BT - old_logprob_BT
                    ratio_BT = torch.exp(logprobs_diff_BT)
                    if self.args.loss_fn == "dapo":
                        pg_losses_BT = -data_BT.advantages[i][:, 1:] * ratio_BT
                        pg_losses2_BT = -data_BT.advantages[i][:, 1:] * torch.clamp(
                            ratio_BT, 1.0 - self.args.clip_lower, 1.0 + self.args.clip_higher
                        )
                    elif self.args.loss_fn == "cispo":
                        # cispo: directly clip ratio, no lower bound.
                        # reinforce loss, so multiply by new logprobs
                        pg_losses_BT = (
                            -data_BT.advantages[i][:, 1:]
                            * torch.clamp(ratio_BT.detach(), max=1.0 + self.args.clip_higher)
                            * new_logprobs_BT
                        )
                        pg_losses2_BT = pg_losses_BT
                    else:
                        raise ValueError(f"Invalid loss function: {self.args.loss_fn}")

                    # Apply truncated importance sampling if enabled
                    if self.args.truncated_importance_sampling_ratio_cap > 0 and vllm_logprobs_BT is not None:
                        old_logprobs_mask_BT = old_logprob_BT != INVALID_LOGPROB
                        vllm_logprobs_mask_BT = vllm_logprobs_BT != INVALID_LOGPROB

                        assert torch.all(old_logprobs_mask_BT == response_mask_BT), (
                            f"Old logprobs mask should match response mask. "
                            f"old_mask sum={old_logprobs_mask_BT.sum()}, "
                            f"response_mask sum={response_mask_BT.sum()}"
                        )
                        assert torch.all(vllm_logprobs_mask_BT == response_mask_BT), (
                            f"vLLM logprobs mask should match response mask. "
                            f"vllm_mask sum={vllm_logprobs_mask_BT.sum()}, "
                            f"response_mask sum={response_mask_BT.sum()}"
                        )

                        valid_mask_BT = response_mask_BT

                        # Initialize importance ratio to 1.0 (no effect) for all positions
                        tis_imp_ratio_BT = torch.ones_like(old_logprob_BT)

                        if valid_mask_BT.any():
                            # Calculate logprob difference only for valid positions
                            logprob_diff_is_BT = old_logprob_BT - vllm_logprobs_BT
                            # Clamp to prevent numerical overflow in exp
                            logprob_diff_is_BT = torch.where(
                                valid_mask_BT,
                                logprob_diff_is_BT.clamp(-10.0, 10.0),
                                torch.zeros_like(logprob_diff_is_BT),
                            )
                            # Compute importance ratio only for valid positions
                            tis_imp_ratio_BT = torch.where(
                                valid_mask_BT, torch.exp(logprob_diff_is_BT), tis_imp_ratio_BT
                            )
                            # Apply cap
                            tis_imp_ratio_BT = torch.clamp(
                                tis_imp_ratio_BT, max=self.args.truncated_importance_sampling_ratio_cap
                            )

                        # Apply importance sampling to losses
                        pg_losses_BT = pg_losses_BT * tis_imp_ratio_BT
                        pg_losses2_BT = pg_losses2_BT * tis_imp_ratio_BT

                    pg_loss_max_BT = torch.max(pg_losses_BT, pg_losses2_BT)

                    if self.args.load_ref_policy:
                        ref_logprob_BT = ref_logprobs_BT[i]
                        # Here we recalculate kl: we want the KL loss to backpropagate through the model
                        # We also clamp the KL loss to avoid numerical instability
                        # https://chatgpt.com/share/679d0ed9-8f48-8011-926e-e274b15ae8ae
                        ref_logprobs_diff_BT = (new_logprobs_BT - ref_logprob_BT).clamp(-40.0, 40.0)
                        kl_4BT = estimate_kl(ref_logprobs_diff_BT, ratio_BT)
                        # grpo change: directly subtract KL in loss (add)
                        loss = masked_mean(
                            pg_loss_max_BT + self.args.beta * kl_4BT[self.args.kl_estimator],
                            response_mask_BT,
                            None,
                            loss_denominator,
                        )
                    else:
                        loss = masked_mean(pg_loss_max_BT, response_mask_BT, None, loss_denominator)

                    # we already took world size into account via the tokens
                    # but deepspeed will try to average over ranks, so multiply back
                    # up, adjusting for the sequence parallel size (adjust by dp world size).
                    loss *= self.args.world_size // self.args.sequence_parallel_size

                    # Clear CUDA cache before backward pass to free memory for reduce_scatter operations
                    torch.cuda.empty_cache()
                    self.model.backward(loss)
                    if (local_step + 1) % accumulation_steps == 0:
                        self.model.step()
                    local_step += 1
                    with torch.no_grad():
                        if self.args.load_ref_policy:
                            # NOTE: in packed implementation, kl calculation are averages over response tokens
                            loss_stats_B["kl"][:, i] = masked_mean(kl_4BT, response_mask_BT).float()
                            loss_stats_B["kl_loss"][i] = loss_stats_B["kl"][self.args.kl_estimator, i] * self.args.beta
                        loss_stats_B["pg_clipfrac"][i] = masked_mean(
                            (pg_losses2_BT > pg_losses_BT).float(), response_mask_BT
                        )
                        loss_stats_B["pg_loss"][i] = masked_mean(pg_loss_max_BT, response_mask_BT)
                        loss_stats_B["loss"][i] = loss
                        loss_stats_B["ratio"][i] = masked_mean(ratio_BT, response_mask_BT)
                        if self.args.record_entropy:
                            loss_stats_B["entropy"][i] = masked_mean(entropy_BT, response_mask_BT).float()

            batch_metrics = batch_data["metrics"]
            with torch.no_grad():
                self._compute_loss_metrics(loss_stats_B, total_valid_tokens)
                array_metrics = {}
                for key, value in batch_metrics.items():
                    if value is None:
                        continue
                    if isinstance(value, (int, float, np.floating, np.integer)):
                        self.local_metrics[key] = value
                    else:
                        array_metrics[key] = value
                return self.local_metrics.get_metrics_list(), array_metrics

    def save_checkpoint_state(self, checkpoint_state_dir: str, client_state: dict[str, Any]) -> None:
        args = self.args

        # Save comprehensive RNG states for each rank
        rng_states = {
            "torch_cpu_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        # Save CUDA RNG states for all devices
        if torch.cuda.is_available():
            rng_states["torch_cuda_rng_states"] = {
                f"cuda:{i}": torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
            }
            rng_states["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        # Add RNG states to client_state
        client_state["rng_states"] = rng_states
        client_state["rank"] = self.rank

        # Save reference policy checkpoint (model only, no optimizer)
        if self.args.load_ref_policy:
            ref_policy_dir = os.path.join(checkpoint_state_dir, "ref_policy")
            os.makedirs(ref_policy_dir, exist_ok=True)

            # For reference policy, we save just the model weights
            # We can't use save_checkpoint because it would try to save DummyOptim
            # which doesn't have state_dict
            if self.rank == 0:
                # Only rank 0 saves the model state
                model_to_save = self.ref_policy.module if hasattr(self.ref_policy, "module") else self.ref_policy
                # Save the state dict
                torch.save(model_to_save.state_dict(), os.path.join(ref_policy_dir, "pytorch_model.bin"))
                logger.info(f"Saved reference policy model to {ref_policy_dir}")

            client_state["ref_policy_saved"] = True

        # Save the main model checkpoint with enhanced client state
        # mpu is just used for sequence parallel, so we remove it for saving, and then re-add it after.
        old_mpu = None
        if self.model.mpu is not None:
            old_mpu = self.mpu
            self.model.mpu = None
        self.model.save_checkpoint(checkpoint_state_dir, client_state=client_state)

        # `save_checkpoint` needs to be called on all ranks, only rank 0 will have all the states
        if self.rank == 0:
            if args.keep_last_n_checkpoints >= 0:
                clean_last_n_checkpoints_deepspeed(checkpoint_state_dir, args.keep_last_n_checkpoints)

            # Sync to GCS if configured (check the actual target, not just gs_bucket_path)
            if args.gs_checkpoint_state_dir is not None:
                ray.remote(sync_gs_bucket).options(num_cpus=1).remote(
                    checkpoint_state_dir, args.gs_checkpoint_state_dir
                )
        # add back the mpu
        if old_mpu is not None:
            self.model.mpu = old_mpu

    def save_model(self, output_dir: str, chat_template_name: str, tokenizer: PreTrainedTokenizer) -> None:
        output_path = pathlib.Path(output_dir)
        marker_path = output_path / CHECKPOINT_COMPLETE_MARKER
        if marker_path.exists():
            logger.info(f"Checkpoint already complete at {output_dir}, skipping save")
            return

        model_to_save = self.model

        if self.rank == 0:
            output_path.mkdir(parents=True, exist_ok=True)

        # save model weights for ZeRO2/3
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        # Set generation config after unwrapping to ensure it's on the actual model being saved
        # Check both chat_template_name and model name for OLMo 3 detection
        model_name = getattr(model_to_save.config, "_name_or_path", "") or ""
        is_olmo3 = (
            chat_template_name is not None and "olmo" in chat_template_name.lower()
        ) or "olmo-3" in model_name.lower()
        if is_olmo3:
            model_to_save.generation_config = get_olmo3_generation_config(tokenizer)

        # gather parameters
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # only gather z3 params
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if self.rank == 0:
                    output_state_dict[k] = vv

        if self.rank == 0:
            state_dict = model_to_save.state_dict()

            # copy named_buffers with `persistent=True`
            for k, v in model_to_save.named_buffers():
                if k not in state_dict:
                    continue
                vv = v.data.cpu()
                output_state_dict[k] = vv

            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())

            # corner case for tie_word_embeddings, such as Qwen2-0.5B
            if getattr(model_to_save.config, "tie_word_embeddings", False) and "lm_head.weight" in state_dict_keys:
                state_dict_keys.remove("lm_head.weight")

            assert state_dict_keys.issubset(output_state_dict_keys), (
                f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"
            )

            # only save peft weights https://github.com/microsoft/DeepSpeed/issues/4295
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir)
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict), output_path / "adapter_model.bin"
                    )
            else:
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict)

            self.tokenizer.save_pretrained(output_dir)
            marker_path.touch()

    # we need this because we don't know which node is rank 0 is on
    def launch_ai2_evals_on_weka_wrapper(self, step_dir, leaderboard_name, wandb_url, training_step):
        args = self.args
        if self.rank == 0:
            ray.remote(launch_ai2_evals_on_weka).options(num_cpus=1).remote(
                path=step_dir,
                leaderboard_name=leaderboard_name,
                oe_eval_max_length=args.oe_eval_max_length,
                wandb_url=wandb_url,
                training_step=training_step,
                oe_eval_tasks=args.oe_eval_tasks,
                stop_strings=streaming_config.stop_strings,
                gs_bucket_path=args.gs_bucket_path,
                eval_priority=args.eval_priority,
                eval_workspace=args.eval_workspace,
                beaker_image=args.oe_eval_beaker_image,
                oe_eval_gpu_multiplier=args.oe_eval_gpu_multiplier,
            )


class ModelGroup:
    def __init__(
        self,
        pg: PlacementGroup,
        ray_process_cls: RayProcess,
        num_gpus_per_node: list[int],
        single_gpu_mode: bool,
        args: grpo_utils.ExperimentConfig,
        streaming_config: data_loader_lib.StreamingDataLoaderConfig,
        vllm_config: data_loader_lib.VLLMConfig,
        data_prep_actor_name: str,
        tokenizer: PreTrainedTokenizer,
    ):
        self.pg = pg
        self.ray_process_cls = ray_process_cls
        self.num_gpus_per_node = num_gpus_per_node
        self.num_gpus_per_actor = 0.48 if single_gpu_mode else 1
        self.num_cpus_per_actor = 4
        self.models = []
        world_size = sum(self.num_gpus_per_node)
        master_policy = ray_process_cls.options(
            num_cpus=self.num_cpus_per_actor,
            num_gpus=self.num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=0
            ),
        ).remote(world_size, 0, 0, None, None, args, streaming_config, vllm_config, data_prep_actor_name, tokenizer)

        self.models.append(master_policy)
        results, _ = ray_get_with_progress(
            [master_policy.get_master_addr_port.remote()], desc="Getting master address"
        )
        (master_addr, master_port) = results[0]

        def get_bundle_index(rank, num_gpus_per_node):
            """given a rank and a list of num_gpus_per_node, return the index of the bundle that the rank belongs to"""
            bundle_idx = 0
            while rank >= num_gpus_per_node[bundle_idx]:
                rank -= num_gpus_per_node[bundle_idx]
                bundle_idx += 1
            return bundle_idx

        assert get_bundle_index(0, [7, 8, 4]) == 0
        assert get_bundle_index(1, [7, 8, 4]) == 0
        assert get_bundle_index(7, [7, 8, 4]) == 1
        assert get_bundle_index(8, [7, 8, 4]) == 1
        assert get_bundle_index(9, [7, 8, 4]) == 1
        assert get_bundle_index(16, [7, 8, 4]) == 2

        # Setup worker models
        for rank in range(1, world_size):
            logger.debug(f"{rank=}, {world_size=}, {rank=}, {master_addr=}, {master_port=}")
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=get_bundle_index(rank, self.num_gpus_per_node)
            )
            worker_policy = ray_process_cls.options(
                num_cpus=self.num_cpus_per_actor,
                num_gpus=self.num_gpus_per_actor,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                world_size,
                rank,
                0,
                master_addr,
                master_port,
                args,
                streaming_config,
                vllm_config,
                data_prep_actor_name,
                tokenizer,
            )
            self.models.append(worker_policy)


def compute_token_weights(metrics_list: list[dict[str, float]]) -> list[float]:
    """Compute token-weighted weights for averaging metrics across ranks.

    Important for sequence parallel where different ranks may have different token counts.
    """
    token_counts = []
    total_tokens = 0.0
    for m in metrics_list:
        tc = m.get("_token_count", 1.0)
        token_counts.append(tc)
        total_tokens += tc
    if total_tokens > 0:
        return [tc / total_tokens for tc in token_counts]
    return [1.0 / len(metrics_list)] * len(metrics_list)


def validate_configs(
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    num_learners_per_node: tuple[int, ...],
    sequence_parallel_size: int,
) -> None:
    """Validate cross-cutting config constraints."""
    if streaming_config.num_unique_prompts_rollout < vllm_config.vllm_num_engines:
        logger.warning(
            f"With num_unique_prompts_rollout={streaming_config.num_unique_prompts_rollout} < "
            f"vllm_num_engines={vllm_config.vllm_num_engines}, vllm will be generating data for multiple "
            "batches simultaneously. This is fine but might be unexpected behaviour."
        )
    assert (
        streaming_config.num_samples_per_prompt_rollout * streaming_config.num_unique_prompts_rollout
        >= sum(num_learners_per_node) // sequence_parallel_size
    ), (
        "num_samples_per_prompt_rollout * num_unique_prompts_rollout must be greater than or equal to world_size // sequence_parallel_size to ensure we have a batch for each rank in distributed training."
    )


def setup_runtime_variables(
    args: grpo_utils.ExperimentConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    tools_config: ToolsConfig,
) -> grpo_utils.ExperimentConfig:
    """Set up runtime variables for the experiment."""
    if tools_config.enabled and (args.use_vllm_logprobs or args.truncated_importance_sampling_ratio_cap > 0.0):
        assert streaming_config.mask_tool_use, (
            "Must mask tool use when using vLLM logprobs or truncated importance sampling."
        )
    args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    streaming_config.dataset_local_cache_dir = os.path.abspath(streaming_config.dataset_local_cache_dir)
    if is_beaker_job():
        streaming_config.dataset_local_cache_dir = (
            "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        )
    args.world_size = sum(args.num_learners_per_node)
    args.num_training_steps = args.total_episodes // (
        streaming_config.num_unique_prompts_rollout * streaming_config.num_samples_per_prompt_rollout
    )
    args.try_launch_beaker_eval_jobs_on_weka = args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job()
    if args.push_to_hub:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    if args.with_tracking and args.wandb_entity is None:
        args.wandb_entity = maybe_use_ai2_wandb_entity()
    return args


def setup_experiment_tracking(args: grpo_utils.ExperimentConfig, tc: TokenizerConfig, model_config: ModelConfig):
    """Setup experiment tracking and seeds."""
    all_configs = {}
    beaker_config = None
    if is_beaker_job():
        beaker_config = maybe_get_beaker_config()
        all_configs.update(vars(beaker_config))
    all_configs.update(**asdict(args), **asdict(tc), **asdict(model_config))

    wandb_url = None
    if args.with_tracking:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=all_configs,
            name=args.run_name,
            save_code=True,
            tags=[args.exp_name] + get_wandb_tags(),
        )
        wandb_url = wandb.run.get_url()
        maybe_update_beaker_description(wandb_url=wandb_url)

    return beaker_config, wandb_url


def _validate_and_log_dataset_tools(dataset, configured_tool_names: list[str] | None, dataset_name: str) -> None:
    """Validate and log per-sample tool configuration for a dataset."""
    if dataset and TOOLS_COLUMN_KEY in dataset.column_names and configured_tool_names:
        logger.info(
            f"{dataset_name} has '{TOOLS_COLUMN_KEY}' column - validating configured tools against dataset tools"
        )
        validate_dataset_tools(dataset, configured_tool_names, dataset_name)
        logger.info(f"{dataset_name} has '{TOOLS_COLUMN_KEY}' column - per-sample tool activation enabled")


def setup_datasets(
    args: grpo_utils.ExperimentConfig,
    tc: TokenizerConfig,
    tokenizer: PreTrainedTokenizer,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    tool_definitions: list[dict[str, Any]],
    pass_tools_to_chat_template: bool,
    configured_tool_call_names: list[str] | None = None,
):
    """Set up training and evaluation datasets.

    Args:
        args: Training arguments.
        tc: Tokenizer configuration.
        tokenizer: The tokenizer.
        streaming_config: Data loading configuration.
        tool_definitions: Global tool definitions in OpenAI format.
        pass_tools_to_chat_template: Whether to pass tools to chat template.
        configured_tool_call_names: List of tool call names configured in the launch job.
            Used to validate against per-sample tools in datasets.
    """
    system_prompt_override = None
    if streaming_config.system_prompt_override_file is not None:
        logger.info(f"Loading system prompt override from {streaming_config.system_prompt_override_file}")
        with open(streaming_config.system_prompt_override_file) as f:
            system_prompt_override = f.read().strip()
        logger.info(f"System prompt overriden to:\n#####\n{system_prompt_override}\n#####\n")

    transform_fn_args = [
        {
            "system_prompt_override": system_prompt_override,
            "tool_definitions": tool_definitions,
            "pass_tools_to_chat_template": pass_tools_to_chat_template,
        },
        {"max_prompt_token_length": streaming_config.max_prompt_token_length},
    ]
    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=streaming_config.dataset_mixer_list,
        dataset_mixer_list_splits=streaming_config.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=streaming_config.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_cache_mode=streaming_config.dataset_cache_mode,
        dataset_config_hash=streaming_config.dataset_config_hash,
        hf_entity=args.hf_entity,
        dataset_local_cache_dir=streaming_config.dataset_local_cache_dir,
        dataset_skip_cache=streaming_config.dataset_skip_cache,
        system_prompt_override=system_prompt_override,
    )

    _validate_and_log_dataset_tools(train_dataset, configured_tool_call_names, "train_dataset")
    train_dataset = train_dataset.shuffle(seed=args.seed)

    if len(streaming_config.dataset_mixer_eval_list) > 0:
        eval_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=streaming_config.dataset_mixer_eval_list,
            dataset_mixer_list_splits=streaming_config.dataset_mixer_eval_list_splits,
            tc=tc,
            dataset_transform_fn=streaming_config.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            hf_entity=args.hf_entity,
            dataset_cache_mode=streaming_config.dataset_cache_mode,
            dataset_config_hash=streaming_config.dataset_config_eval_hash,
            dataset_local_cache_dir=streaming_config.dataset_local_cache_dir,
            dataset_skip_cache=streaming_config.dataset_skip_cache,
            system_prompt_override=system_prompt_override,
        )

        _validate_and_log_dataset_tools(eval_dataset, configured_tool_call_names, "eval_dataset")
        if streaming_config.shuffle_eval_dataset:
            eval_dataset = eval_dataset.shuffle(seed=args.seed)
    else:
        eval_dataset = None

    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)

    return train_dataset, eval_dataset


def create_tools(parsed_tools: list[ParsedToolConfig]) -> tuple[list[ray.actor.ActorHandle], list[str]]:
    """Create tool actors based on tool configuration using the TOOL_REGISTRY.

    Args:
        parsed_tools: List of ParsedTool instances containing name, call_name, and config.

    Returns:
        A tuple of (tool_actors, tool_call_names) where:
        - tool_actors: List of Ray actor handles for the requested tools.
        - tool_call_names: List of call names for each tool (may differ from input for MCP tools, which decide their own call names).

    Raises:
        ValueError: If an unknown tool is requested, configs are invalid, or required fields are missing.
    """
    tool_actors = []
    tool_call_names = []

    for parsed_tool in parsed_tools:
        if parsed_tool.name not in TOOL_REGISTRY:
            available_tools = ", ".join(TOOL_REGISTRY.keys())
            raise ValueError(f"Unknown tool: {parsed_tool.name}. Available tools: {available_tools}")

        tool_config_class = TOOL_REGISTRY[parsed_tool.name]
        # Build config from dictionary
        try:
            config = tool_config_class(**parsed_tool.config)
        except Exception as e:
            raise ValueError(f"Invalid config for tool '{parsed_tool.name}': {e}") from e

        # Collect (config, call_name, tool_class) tuples to process
        # special logic for MCP tools: we ask the mcp server what tools it has, and then create actors for each.
        configs_to_create: list[tuple[BaseToolConfig, str, type]] = []

        if isinstance(config, GenericMCPToolConfig) and config.tool_name is None:
            logger.info(f"Auto-discovering tools from MCP server for '{parsed_tool.name}'...")
            expanded_configs = asyncio.run(config.expand_tools())
            for expanded_config in expanded_configs:
                configs_to_create.append((expanded_config, expanded_config.tool_name, tool_config_class.tool_class))
            logger.info(
                f"Discovered {len(expanded_configs)} tools from MCP server: {[c.tool_name for c in expanded_configs]}"
            )
        else:
            configs_to_create.append((config, parsed_tool.call_name, tool_config_class.tool_class))

        for cfg, call_name, tool_class in configs_to_create:
            _kwarg_dict = asdict(cfg) | {"call_name": call_name}
            # max_concurrency is only needed for Ray actor options, not passed to the tool class
            tool_actors.append(
                ray.remote(tool_class)
                .options(max_concurrency=_kwarg_dict.pop("max_concurrency"))
                .remote(**_kwarg_dict)
            )
            tool_call_names.append(call_name)

    return tool_actors, tool_call_names


def create_model_and_optimizer(
    args: grpo_utils.ExperimentConfig,
    tc: TokenizerConfig,
    model_config: ModelConfig,
    beaker_config: BeakerRuntimeConfig,
    wandb_url: str,
    tokenizer: PreTrainedTokenizer,
    inference_results_Q: ray_queue.Queue,
    prompt_Q: ray_queue.Queue,
    evaluation_inference_results_Q: ray_queue.Queue,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    train_dataset: Dataset,
    eval_dataset,
    reward_config: RewardConfig,
    generation_config,
    data_prep_actor_state: dict | None = None,
    tool_actors: list[ray.actor.ActorHandle] | None = None,
    tools_config: ToolsConfig | None = None,
) -> tuple[
    ModelGroup, list[vllm_utils.LLMRayActor], int, int, ray.actor.ActorHandle, utils.ModelDims, ray.actor.ActorHandle
]:
    """Create the model, optimizer, and vLLM engines."""
    # Create placement group
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.num_learners_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray_get_with_progress([pg.ready()], desc="Waiting for placement group")

    queues_to_monitor = {
        "Inference Results Queue": inference_results_Q,
        "Prompt Queue": prompt_Q,
        "Evaluation Queue": evaluation_inference_results_Q,
    }
    actor_manager = ray.remote(ActorManager).remote(queues_to_monitor, args, streaming_config, vllm_config)

    # Get model_dims early from HuggingFace config (doesn't require vLLM)
    model_dims = utils.ModelDims.from_hf_config(model_config.model_name_or_path)

    # Create DataPreparationActor FIRST so StreamingDataLoader can find it
    data_prep_actor_name = "data_prep_singleton"
    _data_prep_actor = DataPreparationActor.options(name=data_prep_actor_name, num_cpus=2).remote(
        dataset=train_dataset,
        inference_results_Q=inference_results_Q,
        param_prompt_Q=prompt_Q,
        tokenizer=tokenizer,
        config=streaming_config,
        generation_config=generation_config,
        num_training_steps=args.num_training_steps,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        global_batch_size=streaming_config.num_unique_prompts_rollout,
        dp_world_size=args.world_size // args.sequence_parallel_size,
        max_possible_score=streaming_config.max_possible_score,
        actor_manager=actor_manager,
        model_dims=model_dims,
        verbose=args.verbose,
        work_dir=args.output_dir,
        tool_names=tools_config.tool_call_names if tools_config else [],
        run_name=args.run_name,
        model_name=model_config.model_name_or_path,
        initial_state=data_prep_actor_state,
    )

    # Create policy group and start model loading BEFORE vLLM engines (matches main branch order).
    # This ensures policy trainer actors are scheduled first, which affects how Ray schedules
    # the vLLM placement group and prevents port collisions during vLLM initialization.
    wandb_url = wandb.run.get_url() if args.with_tracking else None
    policy_group = ModelGroup(
        pg,
        PolicyTrainerRayProcess,
        args.num_learners_per_node,
        args.single_gpu_mode,
        args=args,
        streaming_config=streaming_config,
        vllm_config=vllm_config,
        data_prep_actor_name=data_prep_actor_name,
        tokenizer=tokenizer,
    )
    inits = [
        model.from_pretrained.remote(args, model_config, beaker_config, wandb_url, tokenizer)
        for model in policy_group.models
    ]

    # Create vLLM engines with queues
    vllm_engines = vllm_utils.create_vllm_engines(
        vllm_config.vllm_num_engines,
        vllm_config.vllm_tensor_parallel_size,
        vllm_config.vllm_enforce_eager,
        tc.tokenizer_name_or_path,
        model_config.model_name_or_path,
        model_config.model_revision,
        args.seed,
        vllm_config.vllm_enable_prefix_caching,
        streaming_config.max_prompt_token_length + streaming_config.response_length,  # max_model_len
        vllm_config.vllm_gpu_memory_utilization,
        args.single_gpu_mode,
        pg=pg if args.single_gpu_mode else None,
        tool_actors=tool_actors,
        tool_parser_type=tools_config.tool_parser_type if tools_config else "legacy",
        max_tool_calls=tools_config.max_tool_calls if tools_config else 5,
        mask_tool_use=streaming_config.mask_tool_use,
        prompt_queue=prompt_Q,
        results_queue=inference_results_Q,
        eval_results_queue=evaluation_inference_results_Q,
        actor_manager=actor_manager,
        inflight_updates=streaming_config.inflight_updates,
        reward_config=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    logger.info("======== ✅ vLLM engines and actor_manager initialized =========")

    if vllm_engines:
        kv_cache_max_concurrency = ray.get(vllm_engines[0].get_kv_cache_info.remote())
        ray.get(actor_manager.set_kv_cache_max_concurrency.remote(kv_cache_max_concurrency))
        expected_batch_size = (
            streaming_config.num_unique_prompts_rollout
            * streaming_config.num_samples_per_prompt_rollout
            // vllm_config.vllm_num_engines
        )
        if kv_cache_max_concurrency < expected_batch_size:
            nodes_needed = (
                streaming_config.num_unique_prompts_rollout
                * streaming_config.num_samples_per_prompt_rollout
                // kv_cache_max_concurrency
            )
            logger.warning(
                f"kv_cache_max_concurrency ({kv_cache_max_concurrency}) is lower than "
                f"num_unique_prompts_rollout * num_samples_per_prompt_rollout // vllm_num_engines ({expected_batch_size}). "
                f"This means actors will have to run multiple sequential batches, hurting performance. "
                f"You might want to use more inference nodes ({nodes_needed} nodes to generate the entire batch simultaneously)."
            )
    else:
        ray.get(actor_manager.set_kv_cache_max_concurrency.remote(-1))

    # Wait for policy models to finish loading
    results, _ = ray_get_with_progress(inits, desc="Initializing models")
    resume_training_step = results[0] + 1
    episode = (
        (resume_training_step - 1)
        * streaming_config.num_unique_prompts_rollout
        * streaming_config.num_samples_per_prompt_rollout
    )
    logger.info("======== ✅ all models initialized =========")

    ray_get_with_progress(
        [m.setup_model_update_group.remote(vllm_engines=vllm_engines) for m in policy_group.models],
        desc="Setting up model update group",
    )
    logger.info("======== ✅ model update group setup successfully =========")

    return (policy_group, vllm_engines, resume_training_step, episode, actor_manager, model_dims, _data_prep_actor)


def create_generation_configs(
    args: grpo_utils.ExperimentConfig, streaming_config: data_loader_lib.StreamingDataLoaderConfig
):
    """Create generation configs for training and evaluation."""
    generation_config = vllm_utils.SamplingConfig(
        temperature=streaming_config.temperature,
        top_p=vllm_config.vllm_top_p,
        max_tokens=streaming_config.response_length,
        n=streaming_config.num_samples_per_prompt_rollout,
        stop=streaming_config.stop_strings,
        seed=args.seed,
        logprobs=1,
    )
    eval_generation_config = dataclasses.replace(generation_config, n=1)
    return {"train": generation_config, "eval": eval_generation_config}


def weight_sync_thread(
    args: grpo_utils.ExperimentConfig,
    stop_event: threading.Event,
    weight_sync_trigger_event: threading.Event,
    policy_group: ModelGroup,
    actor_manager: ActorManager,
    weight_sync_metrics_Q: Queue,
    resume_training_step: int = 1,
):
    """Thread function that handles weight sync operations and actor manager coordination."""
    logger.info("[Weight Sync Thread] 🚀 Starting weight sync thread")
    if resume_training_step > 1:
        weight_sync_trigger_event.set()

    while not stop_event.is_set():
        # Wait for weight sync trigger from main thread
        if not weight_sync_trigger_event.wait(timeout=1.0):
            continue

        # Clear the event for next iteration
        weight_sync_trigger_event.clear()

        with Timer("[Weight Sync]") as timer:
            logger.debug("[Weight Sync Thread] Starting weight sync")

            # Set actors to stop
            ray.get(actor_manager.set_should_stop.remote(True))
            logger.debug("[Weight Sync Thread] Set should_stop to True for weight sync")

            # Broadcast weights to vLLM engines
            # First get the futures
            weight_broadcast_futures: list[ray.ObjectRef] = [m.broadcast_to_vllm.remote() for m in policy_group.models]

            # Wait for all weight updates to complete and collect individual timings
            _, actor_sync_times = ray_get_with_progress(
                weight_broadcast_futures,
                desc="[Weight Sync Thread] Waiting for weight updates to complete",
                enable=args.verbose,
            )

            # Allow actors to resume
            ray.get(actor_manager.set_should_stop.remote(False))
            logger.debug("[Weight Sync Thread] Set should_stop to False after weight sync")

        # Calculate distribution statistics
        sync_time_stats = {
            "time/weight_sync": timer.duration,
            "time/weight_sync_mean": np.mean(actor_sync_times),
            "time/weight_sync_min": np.min(actor_sync_times),
            "time/weight_sync_max": np.max(actor_sync_times),
            "time/weight_sync_median": np.median(actor_sync_times),
        }

        try:
            weight_sync_metrics_Q.put_nowait(sync_time_stats)
        except Full:
            logger.warning("[Weight Sync Thread] weight sync metrics queue full, skipping metric")

    logger.info("[Weight Sync Thread] 🛑 Stopping weight sync thread")


def one_training_step(
    args: grpo_utils.ExperimentConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    policy_group: ModelGroup,
    tokenizer: PreTrainedTokenizer,
    data_thread_metrics: dict[str, Any],
    episode: int,
    training_step: int,
    num_total_tokens: int,
    start_time: float,
    train_dataset: datasets.Dataset,
    training_start_time: float,
    wandb_url: str,
    chat_template_name: str,
    model_dims: utils.ModelDims,
    actor_manager: ActorManager | None = None,
) -> int:
    """Train the model for one step. Returns the number of tokens processed."""
    update_ref_policy_future = []
    with Timer("[Main Thread] 🗡️ Training") as train_timer:
        results, _ = ray_get_with_progress(
            [policy_group.models[i].step.remote() for i in range(args.world_size)],
            desc=f"Running training step {training_step}",
        )
        metrics, array_metrics = zip(*results)
        if all(len(m) == 0 for m in metrics):
            logger.warning("[Main Thread] 🤡 After packing, there is not enough data to train")
            return 0
        if (
            args.load_ref_policy
            and args.ref_policy_update_freq is not None
            and training_step % args.ref_policy_update_freq == 0
            and args.alpha > 0
        ):
            update_ref_policy_future.extend(
                [policy_group.models[i].update_ref_policy.remote() for i in range(args.world_size)]
            )
            ray_get_with_progress(update_ref_policy_future, desc=f"Updating reference policy at step {training_step}")

    save_time = maybe_save_checkpoint(args, training_step, policy_group, chat_template_name, tokenizer, wandb_url)

    ray.get(actor_manager.report_training_step_time.remote(train_timer.duration))

    # Note: metrics contains scalar metrics from each worker, array_metrics contains list/array metrics
    weights = compute_token_weights(metrics)

    # Metrics that should be token-weighted (averages over tokens)
    token_weighted_metrics = {
        "objective/kl0_avg",
        "objective/kl1_avg",
        "objective/kl2_avg",
        "objective/kl3_avg",
        "loss/kl_avg",
        "loss/policy_avg",
        "loss/total_avg",
        "policy/clipfrac_avg",
        "policy/entropy_avg",
        "val/ratio",
        "val/ratio_var",
    }
    average_metrics = {}
    # Average scalar metrics from each worker
    for k in metrics[0]:
        if k == "_token_count":
            # Don't include internal token count in final metrics
            continue
        if k in token_weighted_metrics:
            # Token-weighted average
            average_metrics[k] = sum(m[k] * w for m, w in zip(metrics, weights))
        else:
            # Simple average for other metrics
            average_metrics[k] = sum(m[k] for m in metrics) / len(metrics)
    # Pass through array metrics from the first worker (these are the same across workers)
    for k, v in array_metrics[0].items():
        average_metrics[k] = v
    step_time = time.perf_counter() - start_time
    total_training_time = time.perf_counter() - training_start_time

    total_generation_time = average_metrics["time/getting_response"]
    prompt_lengths = array_metrics[0]["batch/prompt_lengths"]
    response_lengths = array_metrics[0]["batch/response_lengths"]
    num_step_tokens = sum(prompt_lengths) + sum(response_lengths)

    utilization_metrics = utils.calculate_utilization_metrics(
        model_dims=model_dims,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        total_generation_time=total_generation_time,
        samples_per_prompt=streaming_config.num_samples_per_prompt_rollout,
        num_engines=vllm_config.vllm_num_engines,
        num_gpus_per_engine=vllm_config.vllm_tensor_parallel_size,
        training_time=train_timer.duration,
        num_training_gpus=args.world_size,
    )

    metrics = {
        "episode": episode,
        "global_step": episode,
        "training_step": training_step,
        "val/num_total_tokens": num_total_tokens,
        "val/num_step_tokens": num_step_tokens,
        "epoch": episode / streaming_config.num_samples_per_prompt_rollout / len(train_dataset),
        "learner_tokens_per_second_overall": num_total_tokens / total_training_time,
        "learner_tokens_per_second_step": num_step_tokens / step_time,
        "time/total": step_time,
        "time/training": train_timer.duration,
        "time/saving": save_time,
        **data_thread_metrics,
        **average_metrics,
        **utilization_metrics,
    }
    # Print only scalar metrics
    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, float | int)}
    print_rich_single_line_metrics(scalar_metrics)

    if args.with_tracking:
        # Convert array/list metrics to wandb histograms for logging
        for key, value in metrics.items():
            if (isinstance(value, np.ndarray | list)) and len(value) > 0:
                metrics[key] = wandb.Histogram(value)
        wandb.log(metrics, step=episode)

    return num_step_tokens


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def maybe_save_checkpoint(
    args: grpo_utils.ExperimentConfig,
    training_step: int,
    policy_group: ModelGroup,
    chat_template_name: str,
    tokenizer: PreTrainedTokenizer,
    wandb_url: str,
) -> float:
    save_time = 0
    if args.save_freq > 0 and training_step % args.save_freq == 0 and (args.eval_on_step_0 or training_step > 1):
        with Timer("[Main Thread] 🗡️ Saving model") as timer:
            checkpoint_dir = f"{args.output_dir}_checkpoints"
            step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
            logger.info(f"Saving model at step {training_step} to {step_dir}")
            ray_get_with_progress(
                [
                    policy_group.models[i].save_model.remote(step_dir, chat_template_name, tokenizer)
                    for i in range(args.world_size)
                ],
                desc=f"Saving model at step {training_step}",
            )
            if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
                leaderboard_name = f"{args.hf_repo_revision}_step_{training_step}"
                for i in range(args.world_size):
                    policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                        step_dir, leaderboard_name, wandb_url, training_step
                    )
        save_time = timer.duration

    return save_time


def maybe_evaluate(
    args: grpo_utils.ExperimentConfig,
    training_step: int,
    evaluation_inference_results_Q: ray_queue.Queue,
    tokenizer,
    episode,
    eval_dataset: Dataset,
    eval_generation_config,
    model_dims: utils.ModelDims,
    actor_manager=None,
):
    """Optionally evaluate the model."""
    if eval_dataset is None:
        return

    try:
        # timeout 0.01 if this is not the last training step
        # otherwise, wait to get the last evaluation generations (long timeout just in case)
        timeout = 0.01 if training_step < args.num_training_steps else 100

        # Accumulate evaluation results from all vLLM engines
        eval_result, eval_batch, eval_reward_metrics, _ = accumulate_inference_batches(
            evaluation_inference_results_Q,
            eval_generation_config,
            num_prompts=len(eval_dataset),
            model_dims=model_dims,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            actor_manager=actor_manager,
            timeout=timeout,
            active_sampling=False,
            filter_zero_std_samples=False,
            replenish_prompts=False,
        )

        logger.info("[Main Thread] 📊 Evaluation responses received")

        eval_sequence_lengths = np.array([len(response) for response in eval_result.responses])
        eval_stop_rate = sum(int(finish_reason == "stop") for finish_reason in eval_result.finish_reasons) / len(
            eval_result.finish_reasons
        )
        eval_reward_metrics = {f"eval/{key}": val for key, val in eval_reward_metrics.items()}
        eval_metrics = {
            "eval/scores": np.array(eval_batch.scores).mean(),
            "eval/sequence_lengths": eval_sequence_lengths.mean(),
            "eval/sequence_lengths_min": eval_sequence_lengths.min(),
            "eval/sequence_lengths_max": eval_sequence_lengths.max(),
            "eval/stop_rate": eval_stop_rate,
            **eval_reward_metrics,
        }

        total_tokens = (
            eval_result.token_statistics.num_prompt_tokens + eval_result.token_statistics.num_response_tokens
        )
        eval_metrics["eval/actor_tokens_per_second"] = total_tokens / eval_result.token_statistics.generation_time

        print_rich_single_line_metrics(eval_metrics)

        table = {}
        table["prompt"] = tokenizer.batch_decode(eval_batch.queries if eval_batch else [])
        table["response"] = eval_batch.decoded_responses
        table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
        table["scores"] = eval_batch.scores
        table["ground_truth"] = eval_batch.ground_truths if eval_batch else []
        if eval_batch.active_tools is not None:
            table["active_tools"] = [str(tools) if tools is not None else "all" for tools in eval_batch.active_tools]
        df = pd.DataFrame(table)

        if args.with_tracking:
            eval_metrics["sample_completions"] = wandb.Table(dataframe=df)
            wandb.log(eval_metrics, step=episode)
        else:
            print_rich_table(df.iloc[:1])
        del table
    except Empty:
        logger.warning("[Main Thread] 🙈 Evaluation responses not received")


def save_final_model(
    args: grpo_utils.ExperimentConfig,
    policy_group: ModelGroup,
    tokenizer: PreTrainedTokenizer,
    training_step: int,
    wandb_url: str,
    chat_template_name: str,
):
    """Save the final model and launch evaluation jobs if configured."""
    logger.info(f"Saving final model at step {training_step} to {args.output_dir}")
    with Timer("[Main Thread] 🗡️ Saving model"):
        ray_get_with_progress(
            [
                policy_group.models[i].save_model.remote(args.output_dir, chat_template_name, tokenizer)
                for i in range(args.world_size)
            ],
            desc="Saving final model",
        )
        if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
            leaderboard_name = args.hf_repo_revision
            for i in range(args.world_size):
                policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                    args.output_dir, leaderboard_name, wandb_url, training_step
                )


def make_tokenizer(tc: TokenizerConfig, model_config: ModelConfig):
    """Setup tokenizer with appropriate configuration."""
    tc.tokenizer_revision = model_config.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        model_config.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if (
        tc.tokenizer_revision != model_config.model_revision
        and tc.tokenizer_name_or_path != model_config.model_name_or_path
    ):
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{model_config.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{model_config.model_name_or_path=}`."""
        logger.warning(warning)
    return tc.tokenizer


def cleanup_judge_clients():
    """Cleans up all LLM judge clients."""
    asyncio.run(cleanup_all_llm_judge_clients())
    logger.info("✅ LLM judge clients cleaned up")


def cleanup_training_resources(
    stop_event: threading.Event,
    executor: futures.ThreadPoolExecutor,
    queues: list[ray_queue.Queue],
    actor_manager: ActorManager,
) -> None:
    """Clean up all training resources including threads and Ray queues."""
    stop_event.set()

    logger.info("Signaling all actors to stop...")
    ray.get(actor_manager.set_should_stop.remote(True))
    logger.info("✅ Signaled all actors to stop")

    # Clean up ActorManager resources
    logger.info("Cleaning up ActorManager resources...")
    ray.get(actor_manager.cleanup.remote())
    logger.info("✅ ActorManager resources cleaned up")

    logger.info("Pushing shutdown sentinel to queues...")
    # Push sentinel to the first queue (inference_results_Q)
    if queues and len(queues) > 0:
        queues[0].put(ShutdownSentinel(), timeout=1)

    logger.info("Shutting down Ray queues...")
    if queues and len(queues) > 0:
        [queue.shutdown() for queue in queues]
    logger.info("Shutting down thread pool executor...")
    executor.shutdown(wait=True)

    # Clean up judge clients
    cleanup_judge_clients()

    # Shutdown Ray only from the main process (rank 0) or when DDP isn't initialized
    try:
        is_ddp = dist.is_available() and dist.is_initialized()
        is_rank0 = (not is_ddp) or (dist.get_rank() == 0)
        if is_rank0 and ray.is_initialized():
            logger.info("Shutting down Ray...")
            ray.shutdown()
            logger.info("✅ Ray shut down")
    except Exception as e:
        logger.warning(f"Ray shutdown failed: {e}")

    # Clean up distributed process group if it was initialized
    if dist.is_initialized():
        logger.info("Destroying process group...")
        dist.destroy_process_group()
        logger.info("✅ Process group destroyed")


def run_training(
    args,
    streaming_config,
    tokenizer,
    train_dataset,
    eval_dataset,
    policy_group,
    vllm_engines,
    generation_configs,
    resume_training_step,
    episode,
    wandb_url,
    tc,
    stop_event,
    executor,
    inference_results_Q,
    prompt_Q,
    evaluation_inference_results_Q,
    weight_sync_metrics_Q,
    actor_manager: ActorManager,
    model_dims: utils.ModelDims,
    checkpoint_state=None,
):
    if resume_training_step > 1:
        logger.info(f"[Main Thread] Resuming training from step {resume_training_step}")

    # Restore dataloader state if available in checkpoint
    if checkpoint_state and "dataloader_state" in checkpoint_state:
        ray_get_with_progress(
            [
                policy_group.models[i].load_dataloader_state.remote(checkpoint_state["dataloader_state"])
                for i in range(args.world_size)
            ],
            desc="Restoring dataloader state",
        )
        logger.info("Restored dataloader state from checkpoint")

    logger.info("======== ✅ weight sync thread starts =========")
    weight_sync_trigger_event = threading.Event()
    weight_sync_thread_future = executor.submit(
        weight_sync_thread,
        args,
        stop_event,
        weight_sync_trigger_event,
        policy_group,
        actor_manager,
        weight_sync_metrics_Q,
        resume_training_step,
    )

    """Run the main training loop with worker threads."""
    ray_get_with_progress(
        [engine.ready.remote() for engine in vllm_engines], "Checking engines are ready to work", timeout=300
    )

    logger.info("======== ✅ Dataloaders already initialized in actors =========")

    def health_check_fn():
        [f.result() for f in [weight_sync_thread_future] if f.done()]
        ray_get_with_progress(
            [engine.check_background_threads.remote() for engine in vllm_engines],
            desc="Checking vLLM engine health",
            enable=False,
        )

    if checkpoint_state and "num_total_tokens" in checkpoint_state:
        num_total_tokens = checkpoint_state["num_total_tokens"]
        logger.info(f"Restored num_total_tokens: {num_total_tokens}")
    else:
        num_total_tokens = 0

    if eval_dataset is not None:
        eval_data_loader = data_loader_lib.HFDataLoader(
            dataset=eval_dataset,
            batch_size=1,
            seed=args.seed,
            dp_rank=0,
            dp_world_size=1,
            work_dir=args.output_dir,
            automatic_reshuffle=False,
            collator=lambda x: x[0],
        )
    else:
        eval_data_loader = None
    training_start_time = time.perf_counter()  # Track overall training start time
    maybe_update_beaker_description(
        current_step=resume_training_step - 1,
        total_steps=args.num_training_steps,
        start_time=training_start_time,
        wandb_url=wandb_url,
    )
    for training_step in range(resume_training_step, args.num_training_steps + 1):
        start_time = time.perf_counter()

        # Check if any of the threads have raised an exception.
        health_check_start = time.perf_counter()
        health_check_fn()
        health_check_time = time.perf_counter() - health_check_start

        if (
            training_step % args.local_eval_every == 0
            and eval_data_loader is not None
            and (args.eval_on_step_0 or training_step > 1)
        ):
            for eval_example in iter(eval_data_loader):
                add_prompt_to_generator(eval_example, 0, prompt_Q, generation_configs["eval"], is_eval=True)

        episode += streaming_config.num_unique_prompts_rollout * streaming_config.num_samples_per_prompt_rollout

        data_thread_metrics = {}
        try:
            data_thread_metrics |= weight_sync_metrics_Q.get_nowait()
        except Empty:
            logger.info("[Main Thread] didn't get train generation metrics")

        data_thread_metrics["time/health_check"] = health_check_time

        num_step_tokens = one_training_step(
            args,
            streaming_config,
            policy_group,
            tokenizer,
            data_thread_metrics,
            episode,
            training_step,
            num_total_tokens,
            start_time,
            train_dataset,
            training_start_time,
            wandb_url,
            tc.chat_template_name,
            model_dims,
            actor_manager,
        )
        num_total_tokens += num_step_tokens

        # Checkpoint after one_training_step (or even if it was skipped)
        # This ensures we checkpoint progress even if the exact checkpoint step has no data
        if (
            args.checkpoint_state_freq > 0
            and training_step % args.checkpoint_state_freq == 0
            and args.checkpoint_state_dir is not None
        ):
            utils.warn_if_low_disk_space(args.checkpoint_state_dir, send_slack_alerts=args.send_slack_alerts)
            with Timer("[Main Thread] 🗡️ Saving checkpoint state"):
                # Save comprehensive client state including dataloader state
                client_state = {
                    "training_step": training_step,
                    "episode": episode,
                    "num_total_tokens": num_total_tokens,
                }

                # Save dataloader state from Ray actor
                client_state["dataloader_state"] = ray.get(policy_group.models[0].get_dataloader_state.remote())

                # Save DataPreparationActor state
                data_prep_actor = ray.get_actor("data_prep_singleton")
                client_state["data_prep_actor_state"] = ray.get(data_prep_actor.get_state.remote())

                ray_get_with_progress(
                    [
                        policy_group.models[i].save_checkpoint_state.remote(args.checkpoint_state_dir, client_state)
                        for i in range(args.world_size)
                    ],
                    desc=f"Saving checkpoint state at step {training_step}",
                )
                logger.info(f"Saved checkpoint state at step {training_step} to {args.checkpoint_state_dir}")

        logger.debug(f"[Main Thread] Triggered weight sync for step {training_step}")
        weight_sync_trigger_event.set()

        maybe_evaluate(
            args,
            training_step,
            evaluation_inference_results_Q,
            tokenizer,
            episode,
            eval_dataset,
            generation_configs["eval"],
            model_dims,
            actor_manager,
        )

        maybe_update_beaker_description(
            current_step=training_step,
            total_steps=args.num_training_steps,
            start_time=training_start_time,
            wandb_url=wandb_url,
        )

    if resume_training_step > args.num_training_steps:
        raise ValueError(f"Training didn't run since {resume_training_step=} > {args.num_training_steps=}")

    save_final_model(args, policy_group, tokenizer, training_step, wandb_url, tc.chat_template_name)


def initialize_tools(tools_config: ToolsConfig, tokenizer) -> tuple[list, list, list[str], list[str]]:
    """Initialize tool actors and get tool definitions and stop sequences.

    Args:
        tools_config: Configuration for tools.
        tokenizer: Tokenizer for the model.

    Returns:
        Tuple of (tool_actors, tool_definitions, stop_sequences, tool_call_names).
        Note: tool_call_names may differ from tools_config.tool_call_names if MCP
        tools were auto-expanded.
    """
    tool_actors, tool_call_names = create_tools(tools_config._parsed_tools)
    tool_definitions = (
        ray.get([actor.get_openai_tool_definitions.remote() for actor in tool_actors]) if tool_actors else []
    )

    # Create parser temporarily to get stop sequences for generation config
    # The actual parser used during generation will be created inside vLLM actors
    stop_sequences = []
    if tool_actors:
        stop_sequences = create_tool_parser(
            parser_type=tools_config.tool_parser_type, tool_actors=tool_actors, tokenizer=tokenizer
        ).stop_sequences

    return tool_actors, tool_definitions, stop_sequences, tool_call_names


def main(
    args: grpo_utils.ExperimentConfig,
    tc: TokenizerConfig,
    model_config: ModelConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    tools_config: ToolsConfig,
):
    tokenizer = make_tokenizer(tc, model_config)
    args = setup_runtime_variables(args, streaming_config, tools_config)
    validate_configs(streaming_config, vllm_config, tuple(args.num_learners_per_node), args.sequence_parallel_size)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)

    beaker_config, wandb_url = setup_experiment_tracking(args, tc, model_config)

    # We have to initialize ray earlier for constructing Tools (they are implemented as ray actors).
    ray.init(dashboard_host="0.0.0.0", runtime_env={"excludes": [".git/"], "env_vars": dict(os.environ)})

    tool_actors, tool_definitions, tool_stop_sequences, tool_call_names = initialize_tools(tools_config, tokenizer)
    logger.info(
        f"Initialized {len(tool_actors)} tool actors with definitions: {[d['function']['name'] for d in tool_definitions]}"
    )
    # Update tools_config with expanded tool call names (for MCP auto-expansion)
    tools_config.tool_call_names = tool_call_names
    if tool_stop_sequences:
        logger.info(f"Adding tool stop sequences to config: {tool_stop_sequences}")
        streaming_config.stop_strings.extend(tool_stop_sequences)

    train_dataset, eval_dataset = setup_datasets(
        args,
        tc,
        tokenizer,
        streaming_config,
        tool_definitions,
        pass_tools_to_chat_template=tools_config.pass_tools_to_chat_template,
        configured_tool_call_names=tools_config.tool_call_names if tools_config.enabled else None,
    )

    if len(train_dataset) < (
        needed := max(streaming_config.async_steps, 1) * streaming_config.num_unique_prompts_rollout
    ):
        raise ValueError(
            f"Train dataset is too small! Is {len(train_dataset)} prompts, but {needed} are needed to have enough prompts for bsz and prefill. Try reducing async_steps or num_unique_prompts_rollout, or increasing the dataset size."
        )

    if args.cache_dataset_only:
        return

    pprint([args, model_config, streaming_config, vllm_config, tools_config])

    # Create Ray queues.
    # Since we now send/receive individual prompts, queue size should accommodate
    # - all prompts from async_steps + 1 training steps
    # - all eval prompts
    num_eval_prompts = len(eval_dataset) if eval_dataset is not None else 0
    queue_size = (streaming_config.async_steps + 1) * streaming_config.num_unique_prompts_rollout + num_eval_prompts
    inference_results_Q = ray_queue.Queue(maxsize=queue_size)
    prompt_Q = ray_queue.Queue(maxsize=queue_size)
    # We don't care if we ever hit the max, so we let the queue be unbounded.
    evaluation_inference_results_Q = ray_queue.Queue()

    reward_config = RewardConfig(
        apply_r1_style_format_reward=streaming_config.apply_r1_style_format_reward,
        r1_style_format_reward=streaming_config.r1_style_format_reward,
        apply_verifiable_reward=streaming_config.apply_verifiable_reward,
        verification_reward=streaming_config.verification_reward,
        non_stop_penalty=streaming_config.non_stop_penalty,
        non_stop_penalty_value=streaming_config.non_stop_penalty_value,
        only_reward_good_outputs=tools_config.only_reward_good_outputs,
        additive_format_reward=streaming_config.additive_format_reward,
        verifier_functions=build_all_verifiers(args, streaming_config),
    )

    # AFTER potentially adding tool stop sequences, create generation configs
    generation_configs = create_generation_configs(args, streaming_config)

    checkpoint_state = None
    data_prep_actor_state = None
    if args.checkpoint_state_dir and os.path.exists(args.checkpoint_state_dir):
        checkpoint_path = os.path.join(args.checkpoint_state_dir, "global_0", "state.pt")
        if os.path.exists(checkpoint_path):
            checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            logger.info(f"Loaded checkpoint state from {checkpoint_path}")
            data_prep_actor_state = checkpoint_state.get("data_prep_actor_state")
            if data_prep_actor_state:
                # Use trainer's authoritative training_step for DataPreparationActor.
                # iter_dataloader state may be ahead but that's ok (prompts are shuffled, training is stochastic)
                data_prep_actor_state["training_step"] = checkpoint_state.get("training_step", 0)

    (policy_group, vllm_engines, resume_training_step, episode, actor_manager, model_dims, _data_prep_actor) = (
        create_model_and_optimizer(
            args,
            tc,
            model_config,
            beaker_config,
            wandb_url,
            tokenizer,
            inference_results_Q,
            prompt_Q,
            evaluation_inference_results_Q,
            streaming_config,
            vllm_config,
            train_dataset,
            eval_dataset,
            reward_config,
            generation_configs["train"],
            data_prep_actor_state,
            tool_actors,
            tools_config,
        )
    )

    if checkpoint_state:
        episode = checkpoint_state["episode"]
        logger.info(f"Restored episode count: {episode}")

    # Create additional queues (main queues already created above)
    weight_sync_metrics_Q = Queue(maxsize=streaming_config.async_steps)

    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="grpo")

    try:
        episode = run_training(
            args,
            streaming_config,
            tokenizer,
            train_dataset,
            eval_dataset,
            policy_group,
            vllm_engines,
            generation_configs,
            resume_training_step,
            episode,
            wandb_url,
            tc,
            stop_event,
            executor,
            inference_results_Q,
            prompt_Q,
            evaluation_inference_results_Q,
            weight_sync_metrics_Q,
            actor_manager,
            model_dims,
            checkpoint_state,
        )

        if args.push_to_hub and (not dist.is_initialized() or dist.get_rank() == 0):
            push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)
    except Exception as e:
        if args.send_slack_alerts:
            utils.send_slack_message(f"<!here> A RL job has died. Error message: {e}.")
        raise
    finally:
        cleanup_training_resources(
            stop_event, executor, [inference_results_Q, prompt_Q, evaluation_inference_results_Q], actor_manager
        )

    # Ai2 logic: we use /output to store the artifacts of the job, so we
    # make a copy of the model to `/output` in the end.
    if (
        args.try_auto_save_to_beaker
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
        and os.path.isdir(args.output_dir)
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)
    logger.info("finished training")

    # Check for runtime leaks before exiting
    logger.info("Checking for runtime leaks...")

    utils.check_runtime_leaks()


if __name__ == "__main__":
    utils.check_oe_eval_internal()

    parser = ArgumentParserPlus(
        (
            grpo_utils.ExperimentConfig,
            TokenizerConfig,
            ModelConfig,
            data_loader_lib.StreamingDataLoaderConfig,
            data_loader_lib.VLLMConfig,
            ToolsConfig,
        )
    )
    args, tokenizer_config, model_config, streaming_config, vllm_config, tools_config = (
        parser.parse_args_into_dataclasses()
    )
    assert isinstance(args, grpo_utils.ExperimentConfig)
    assert isinstance(tokenizer_config, TokenizerConfig)
    assert isinstance(model_config, ModelConfig)
    assert isinstance(streaming_config, data_loader_lib.StreamingDataLoaderConfig)
    assert isinstance(vllm_config, data_loader_lib.VLLMConfig)
    assert isinstance(tools_config, ToolsConfig)

    main(args, tokenizer_config, model_config, streaming_config, vllm_config, tools_config)
