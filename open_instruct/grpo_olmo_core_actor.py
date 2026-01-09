"""
OLMo-core based GRPO training actor for Ray-distributed training.

This module provides a Ray actor that wraps OLMo-core's training infrastructure,
allowing distributed GRPO training across multiple GPUs and nodes.
"""

import logging
import os
import socket
import time
from datetime import timedelta
from typing import Any

import ray
import torch
from olmo_core import train
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, LinearWithWarmup
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)
from transformers import PreTrainedTokenizer

from open_instruct import data_loader as data_loader_lib
from open_instruct import vllm_utils
from open_instruct.grpo_callbacks import olmo_core_to_hf_name
from open_instruct.grpo_train_module import GRPOConfig, GRPOTrainModule
from open_instruct.utils import RayProcess

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class PolicyTrainerOLMoCoreProcess(RayProcess):
    """Ray actor for OLMo-core based GRPO training.

    Each actor represents one GPU in the distributed training setup.
    Actors coordinate via torch.distributed for FSDP gradient synchronization.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        master_addr: str | None,
        master_port: int | None,
        num_nodes: int,
        local_world_size: int,
        model_name_or_path: str,
        grpo_config: GRPOConfig,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warm_up_steps: int,
        warmup_ratio: float,
        num_training_steps: int,
        num_epochs: int,
        num_mini_batches: int,
        per_device_train_batch_size: int,
        max_sequence_length: int,
        single_gpu_mode: bool,
        load_ref_policy: bool,
        beta: float,
        seed: int,
        output_dir: str,
        streaming_config: data_loader_lib.StreamingDataLoaderConfig,
        vllm_config: data_loader_lib.VLLMConfig,
        data_prep_actor_name: str,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__(world_size, rank, local_rank, master_addr, master_port)
        self.num_nodes = num_nodes
        self.local_world_size = local_world_size
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.model_name_or_path = model_name_or_path
        self.grpo_config = grpo_config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.lr_scheduler_type = lr_scheduler_type
        self.warm_up_steps = warm_up_steps
        self.warmup_ratio = warmup_ratio
        self.num_training_steps = num_training_steps
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.per_device_train_batch_size = per_device_train_batch_size
        self.max_sequence_length = max_sequence_length
        self.single_gpu_mode = single_gpu_mode
        self.load_ref_policy = load_ref_policy
        self.beta = beta
        self.seed = seed
        self.output_dir = output_dir
        self.streaming_config = streaming_config
        self.vllm_config = vllm_config
        self.data_prep_actor_name = data_prep_actor_name

        self.model = None
        self.ref_policy = None
        self.train_module = None
        self.dataloader = None
        self.dataloader_iter = None
        self.vllm_engines = None
        self.model_update_group = None
        self.local_metrics = {}

    def setup_model(self) -> int:
        """Initialize the OLMo-core model and training infrastructure.

        Returns:
            The training step to resume from (1 if starting fresh).
        """
        os.environ["NUM_NODES"] = str(self.num_nodes)
        os.environ["LOCAL_WORLD_SIZE"] = str(self.local_world_size)

        backend = "cpu:gloo,cuda:nccl"
        train.prepare_training_environment(seed=self.seed, backend=backend)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_basename = self.model_name_or_path.split("/")[-1]
        config_name = model_basename.replace("-", "_").replace(".", "_")
        config_name = config_name[:-1].lower() + "B" if config_name.endswith("B") else config_name.lower()

        if not hasattr(TransformerConfig, config_name):
            available = [
                m for m in dir(TransformerConfig) if not m.startswith("_") and callable(getattr(TransformerConfig, m))
            ]
            raise ValueError(f"No TransformerConfig.{config_name}() found. Available: {available}")

        logger.info(f"[Rank {self.rank}] Building OLMo-core model with TransformerConfig.{config_name}()")
        model_config_olmo = getattr(TransformerConfig, config_name)()
        self.model = model_config_olmo.build(init_device="cpu")

        logger.info(f"[Rank {self.rank}] Loading HuggingFace weights from {self.model_name_or_path}")
        load_hf_model(self.model_name_or_path, self.model.state_dict(), work_dir=self.output_dir)

        if self.load_ref_policy and self.beta > 0:
            logger.info(f"[Rank {self.rank}] Building reference policy...")
            self.ref_policy = model_config_olmo.build(init_device="cpu")
            load_hf_model(self.model_name_or_path, self.ref_policy.state_dict(), work_dir=self.output_dir)
            self.ref_policy = self.ref_policy.to(device=device, dtype=torch.bfloat16).eval()

        self.dataloader = self.streaming_config.build_dataloader(
            data_prep_actor_name=self.data_prep_actor_name,
            tokenizer=self.tokenizer,
            dp_rank=self.rank,
            fs_local_rank=self.rank,
            num_training_steps=self.num_training_steps,
            work_dir=self.output_dir,
            dp_world_size=self.world_size,
        )
        self.dataloader_iter = iter(self.dataloader)

        num_scheduler_steps = self.num_training_steps * self.num_epochs * self.num_mini_batches
        warmup_steps = self.warm_up_steps
        if self.warmup_ratio > 0.0:
            warmup_steps = int(num_scheduler_steps * self.warmup_ratio)

        if self.lr_scheduler_type == "cosine":
            scheduler = CosWithWarmup(warmup_steps=warmup_steps)
        else:
            scheduler = LinearWithWarmup(warmup_steps=warmup_steps, alpha_f=0.0)

        optim_config = AdamWConfig(lr=self.learning_rate, weight_decay=self.weight_decay)

        dp_config = None
        if not self.single_gpu_mode and self.world_size > 1:
            dp_config = TransformerDataParallelConfig(
                name=DataParallelType.hsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
                wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            )

        self.grpo_config.temperature = self.streaming_config.temperature

        self.train_module = GRPOTrainModule(
            model=self.model,
            optim=optim_config,
            rank_microbatch_size=self.per_device_train_batch_size,
            max_sequence_length=self.max_sequence_length,
            grpo_config=self.grpo_config,
            tokenizer=self.tokenizer,
            ref_policy=self.ref_policy,
            dp_config=dp_config,
            max_grad_norm=self.max_grad_norm,
            scheduler=scheduler,
            device=device,
        )

        self.trainer = train.TrainerConfig(
            save_folder=self.output_dir,
            max_duration=train.Duration.steps(self.num_training_steps),
            metrics_collect_interval=10,
        ).build(self.train_module, self.dataloader)

        logger.info(f"[Rank {self.rank}] OLMo-core model setup complete")
        return 1

    def setup_model_update_group(self, vllm_engines: list) -> None:
        """Set up the process group for weight synchronization with vLLM engines."""
        self.vllm_engines = vllm_engines

        if not vllm_engines or self.rank != 0:
            return

        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        vllm_world_size = self.vllm_config.vllm_num_engines * self.vllm_config.vllm_tensor_parallel_size + 1
        backend = self.vllm_config.vllm_sync_backend

        refs = [
            engine.init_process_group.remote(
                master_address,
                master_port,
                i * self.vllm_config.vllm_tensor_parallel_size + 1,
                vllm_world_size,
                "openrlhf",
                backend=backend,
                timeout_minutes=120,
            )
            for i, engine in enumerate(vllm_engines)
        ]

        torch.cuda.set_device(self.local_rank)
        self.model_update_group = vllm_utils.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=vllm_world_size,
            rank=0,
            group_name="openrlhf",
            timeout=timedelta(minutes=120),
        )

        ray.get(refs)
        logger.info(f"[Rank {self.rank}] vLLM model update group initialized")

    def step(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute one training step.

        Returns:
            Tuple of (scalar_metrics, array_metrics).
        """
        try:
            batch_data = next(self.dataloader_iter)
        except StopIteration:
            return {}, {}

        batch = {"batch": batch_data["batch"]}
        self.train_module.train_batch(batch, dry_run=False)

        metrics = batch_data.get("metrics", {})
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        array_metrics = {k: v for k, v in metrics.items() if not isinstance(v, (int, float))}

        return scalar_metrics, array_metrics

    def broadcast_to_vllm(self) -> list:
        """Broadcast model weights to vLLM engines.

        Follows the same pattern as grpo_fast.py - returns Ray ObjectRefs
        for the caller to wait on, rather than blocking internally.

        Returns:
            List of Ray ObjectRefs for the weight update calls.
        """
        import sys

        print(f"[DEBUG] broadcast_to_vllm starting on rank {self.rank}", file=sys.stderr, flush=True)
        torch.cuda.empty_cache()
        print(f"[DEBUG] after cuda.empty_cache on rank {self.rank}", file=sys.stderr, flush=True)
        torch.cuda.set_device(self.local_rank)
        print(f"[DEBUG] after cuda.set_device on rank {self.rank}", file=sys.stderr, flush=True)

        print(f"[DEBUG] getting train_module.model on rank {self.rank}", file=sys.stderr, flush=True)
        model = self.train_module.model
        print(f"[DEBUG] got model on rank {self.rank}", file=sys.stderr, flush=True)
        params_list = list(model.named_parameters())
        print(
            f"[DEBUG] got params_list with {len(params_list)} params on rank {self.rank}", file=sys.stderr, flush=True
        )
        num_params = len(params_list)
        refss = []

        print(
            f"[DEBUG] starting broadcast loop for {num_params} params on rank {self.rank}", file=sys.stderr, flush=True
        )

        if torch.distributed.get_rank() == 0 and self.vllm_engines:
            print("[DEBUG] checking vllm engines are ready", file=sys.stderr, flush=True)
            ready_refs = [engine.ready.remote() for engine in self.vllm_engines]
            ray.get(ready_refs)
            print("[DEBUG] vllm engines are ready", file=sys.stderr, flush=True)

        for count, (name, param) in enumerate(params_list, start=1):
            hf_name = olmo_core_to_hf_name(name)
            if count == 1:
                print(
                    f"[DEBUG] first param: {hf_name}, rank={torch.distributed.get_rank()}", file=sys.stderr, flush=True
                )
            if torch.distributed.get_rank() == 0:
                if count <= 3:
                    print(
                        f"[DEBUG] Param {count}/{num_params}: calling update_weight.remote()",
                        file=sys.stderr,
                        flush=True,
                    )
                refs = [
                    engine.update_weight.remote(
                        hf_name, dtype=str(param.dtype), shape=tuple(param.shape), empty_cache=count == num_params
                    )
                    for engine in self.vllm_engines
                ]
                refss.extend(refs)
                if count == 1:
                    time.sleep(0.5)
                if count <= 3:
                    print(f"[DEBUG] Param {count}/{num_params}: calling broadcast()", file=sys.stderr, flush=True)
            if torch.distributed.get_rank() == 0:
                torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
                if count <= 3:
                    print(f"[DEBUG] Param {count}/{num_params}: broadcast complete", file=sys.stderr, flush=True)

        logger.info(f"[Rank {self.rank}] All broadcasts complete, returning {len(refss)} refs")
        all_refs = []
        if torch.distributed.get_rank() == 0:
            all_refs.extend(refss)
        return all_refs

    def update_ref_policy(self) -> None:
        """Update reference policy using Polyak averaging."""
        if self.ref_policy is None:
            return

        alpha = self.grpo_config.alpha
        with torch.no_grad():
            for param, ref_param in zip(self.train_module.model.parameters(), self.ref_policy.parameters()):
                ref_param.data.mul_(1 - alpha).add_(param.data, alpha=alpha)

    def save_model(self, output_dir: str, chat_template_name: str, tokenizer: PreTrainedTokenizer) -> None:
        """Save model checkpoint."""
        if self.rank != 0:
            return

        os.makedirs(output_dir, exist_ok=True)

        state_dict = {}
        model = self.train_module.model
        for name, param in model.named_parameters():
            hf_name = olmo_core_to_hf_name(name)
            state_dict[hf_name] = param.data.cpu()

        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(output_dir)
        logger.info(f"[Rank {self.rank}] Model saved to {output_dir}")

    def get_dataloader_state(self) -> dict[str, Any]:
        """Get dataloader state for checkpointing."""
        return self.dataloader.state_dict() if hasattr(self.dataloader, "state_dict") else {}

    def load_dataloader_state(self, state_dict: dict[str, Any]) -> None:
        """Load dataloader state from checkpoint."""
        if hasattr(self.dataloader, "load_state_dict"):
            self.dataloader.load_state_dict(state_dict)


class OLMoCoreModelGroup:
    """Manager class for OLMo-core training actors.

    Similar to ModelGroup in grpo_fast.py but uses OLMo-core based actors.
    """

    def __init__(
        self,
        pg,
        num_gpus_per_node: list[int],
        single_gpu_mode: bool,
        model_name_or_path: str,
        grpo_config: GRPOConfig,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warm_up_steps: int,
        warmup_ratio: float,
        num_training_steps: int,
        num_epochs: int,
        num_mini_batches: int,
        per_device_train_batch_size: int,
        max_sequence_length: int,
        load_ref_policy: bool,
        beta: float,
        seed: int,
        output_dir: str,
        streaming_config: data_loader_lib.StreamingDataLoaderConfig,
        vllm_config: data_loader_lib.VLLMConfig,
        data_prep_actor_name: str,
        tokenizer: PreTrainedTokenizer,
    ):
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from open_instruct.utils import ray_get_with_progress

        self.pg = pg
        self.num_gpus_per_node = num_gpus_per_node
        self.num_gpus_per_actor = 0.48 if single_gpu_mode else 1
        self.num_cpus_per_actor = 4
        self.models = []
        world_size = sum(num_gpus_per_node)
        num_nodes = len(num_gpus_per_node)

        def get_node_info(rank, num_gpus_per_node):
            """Returns (node_index, local_rank, local_world_size) for a given global rank."""
            node_idx = 0
            remaining_rank = rank
            while remaining_rank >= num_gpus_per_node[node_idx]:
                remaining_rank -= num_gpus_per_node[node_idx]
                node_idx += 1
            return node_idx, remaining_rank, num_gpus_per_node[node_idx]

        node_idx, local_rank, local_world_size = get_node_info(0, num_gpus_per_node)

        master_policy = PolicyTrainerOLMoCoreProcess.options(
            num_cpus=self.num_cpus_per_actor,
            num_gpus=self.num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0),
        ).remote(
            world_size=world_size,
            rank=0,
            local_rank=local_rank,
            master_addr=None,
            master_port=None,
            num_nodes=num_nodes,
            local_world_size=local_world_size,
            model_name_or_path=model_name_or_path,
            grpo_config=grpo_config,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warm_up_steps=warm_up_steps,
            warmup_ratio=warmup_ratio,
            num_training_steps=num_training_steps,
            num_epochs=num_epochs,
            num_mini_batches=num_mini_batches,
            per_device_train_batch_size=per_device_train_batch_size,
            max_sequence_length=max_sequence_length,
            single_gpu_mode=single_gpu_mode,
            load_ref_policy=load_ref_policy,
            beta=beta,
            seed=seed,
            output_dir=output_dir,
            streaming_config=streaming_config,
            vllm_config=vllm_config,
            data_prep_actor_name=data_prep_actor_name,
            tokenizer=tokenizer,
        )

        self.models.append(master_policy)
        results, _ = ray_get_with_progress(
            [master_policy.get_master_addr_port.remote()], desc="Getting master address"
        )
        (master_addr, master_port) = results[0]

        for rank in range(1, world_size):
            node_idx, local_rank, local_world_size = get_node_info(rank, num_gpus_per_node)
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=node_idx
            )
            worker_policy = PolicyTrainerOLMoCoreProcess.options(
                num_cpus=self.num_cpus_per_actor,
                num_gpus=self.num_gpus_per_actor,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                world_size=world_size,
                rank=rank,
                local_rank=local_rank,
                master_addr=master_addr,
                master_port=master_port,
                num_nodes=num_nodes,
                local_world_size=local_world_size,
                model_name_or_path=model_name_or_path,
                grpo_config=grpo_config,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                lr_scheduler_type=lr_scheduler_type,
                warm_up_steps=warm_up_steps,
                warmup_ratio=warmup_ratio,
                num_training_steps=num_training_steps,
                num_epochs=num_epochs,
                num_mini_batches=num_mini_batches,
                per_device_train_batch_size=per_device_train_batch_size,
                max_sequence_length=max_sequence_length,
                single_gpu_mode=single_gpu_mode,
                load_ref_policy=load_ref_policy,
                beta=beta,
                seed=seed,
                output_dir=output_dir,
                streaming_config=streaming_config,
                vllm_config=vllm_config,
                data_prep_actor_name=data_prep_actor_name,
                tokenizer=tokenizer,
            )
            self.models.append(worker_policy)
