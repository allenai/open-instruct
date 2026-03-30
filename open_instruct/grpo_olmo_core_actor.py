"""
OLMo-core based GRPO training actor for Ray-distributed training.

This module provides a Ray actor that wraps OLMo-core's training infrastructure,
allowing distributed GRPO training across multiple GPUs and nodes.
"""

import os
from datetime import timedelta
from typing import Any

import ray
import torch
import transformers
from olmo_core import train
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.optim import AdamWConfig, CosWithWarmup, LinearWithWarmup
from olmo_core.train import callbacks
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_utils, logger_utils, olmo_core_utils, vllm_utils
from open_instruct.grpo_callbacks import RefPolicyUpdateCallback, VLLMWeightSyncCallback, olmo_core_to_hf_name
from open_instruct.olmo_core_callbacks import BeakerCallbackV2
from open_instruct.olmo_core_train_modules import GRPOTrainModule
from open_instruct.utils import RayProcess, is_beaker_job, ray_get_with_progress

logger = logger_utils.setup_logger(__name__)


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
        local_world_size: int,
        model_name_or_path: str,
        grpo_config: grpo_utils.ExperimentConfig,
        max_sequence_length: int,
        streaming_config: data_loader_lib.StreamingDataLoaderConfig,
        vllm_config: data_loader_lib.VLLMConfig,
        data_prep_actor_name: str,
        tokenizer: transformers.PreTrainedTokenizer,
        attn_implementation: str = "flash_3",
    ):
        super().__init__(world_size, rank, local_rank, master_addr, master_port)
        self.local_world_size = local_world_size
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path
        self.grpo_config = grpo_config
        self.max_sequence_length = max_sequence_length
        self.streaming_config = streaming_config
        self.vllm_config = vllm_config
        self.data_prep_actor_name = data_prep_actor_name
        self.attn_implementation = attn_implementation

        self.ref_policy = None
        self.vllm_engines = None
        self.model_update_group = None
        self.actor_manager = None
        self.with_tracking = False
        self.wandb_project = None
        self.wandb_entity = None
        self.run_name = None
        self.json_config = None
        self.ref_policy_update_freq = None

    def setup_model(self) -> int:
        """Initialize the OLMo-core model and training infrastructure.

        Returns:
            The training step to resume from (1 if starting fresh).
        """
        os.environ["NUM_NODES"] = str(self.grpo_config.num_nodes)
        os.environ["LOCAL_WORLD_SIZE"] = str(self.local_world_size)
        os.environ["FS_LOCAL_RANK"] = str(self.rank)

        # Ray sets CUDA_VISIBLE_DEVICES per actor, so device 0 is always correct
        torch.cuda.set_device(0)
        logger.info(
            f"[Rank {self.rank}] Set CUDA device to 0, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
        )

        backend = "cpu:gloo,cuda:nccl"
        logger.info(f"[Rank {self.rank}] Calling train.prepare_training_environment...")
        train.prepare_training_environment(seed=self.grpo_config.seed, backend=backend)
        logger.info(f"[Rank {self.rank}] train.prepare_training_environment completed")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hf_config = transformers.AutoConfig.from_pretrained(self.model_name_or_path)
        vocab_size = hf_config.vocab_size

        torch_dtype = grpo_utils.TORCH_DTYPES[self.grpo_config.model_dtype]
        olmo_core_dtype = {"bfloat16": DType.bfloat16, "float32": DType.float32}[self.grpo_config.model_dtype]

        self.model_config = olmo_core_utils.get_transformer_config(
            self.model_name_or_path, vocab_size, attn_backend=self.attn_implementation
        )
        logger.info(f"[Rank {self.rank}] Building OLMo-core model from {self.model_name_or_path}")
        self.model = self.model_config.build(init_device="cpu")

        if self.grpo_config.load_ref_policy and self.grpo_config.beta > 0:
            logger.info(f"[Rank {self.rank}] Building reference policy...")
            self.ref_policy = self.model_config.build(init_device="cpu")
            load_hf_model(self.model_name_or_path, self.ref_policy.state_dict(), work_dir=self.grpo_config.output_dir)
            self.ref_policy = self.ref_policy.to(device=device, dtype=torch_dtype).eval()

        assert self.grpo_config.num_training_steps is not None, "num_training_steps must be set"
        self.dataloader = self.streaming_config.build_dataloader(
            data_prep_actor_name=self.data_prep_actor_name,
            tokenizer=self.tokenizer,
            dp_rank=self.rank,
            fs_local_rank=self.rank,
            num_training_steps=self.grpo_config.num_training_steps,
            work_dir=self.grpo_config.output_dir,
            dp_world_size=self.world_size,
        )

        num_scheduler_steps = (
            self.grpo_config.num_training_steps * self.grpo_config.num_epochs * self.grpo_config.num_mini_batches
        )
        warmup_steps = self.grpo_config.warm_up_steps
        if self.grpo_config.warmup_ratio > 0.0:
            warmup_steps = int(num_scheduler_steps * self.grpo_config.warmup_ratio)

        if self.grpo_config.lr_scheduler_type == "cosine":
            scheduler = CosWithWarmup(warmup_steps=warmup_steps)
        else:
            scheduler = LinearWithWarmup(warmup_steps=warmup_steps, alpha_f=0.0)

        optim_config = AdamWConfig(lr=self.grpo_config.learning_rate, weight_decay=self.grpo_config.weight_decay)

        dp_config = None
        if not self.grpo_config.single_gpu_mode and self.world_size > 1:
            dp_config = TransformerDataParallelConfig(
                name=DataParallelType.hsdp,
                param_dtype=olmo_core_dtype,
                reduce_dtype=DType.float32,
                wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            )

        self.train_module = GRPOTrainModule(
            model=self.model,
            optim=optim_config,
            sample_microbatch_size=self.grpo_config.per_device_train_batch_size,
            max_sequence_length=self.max_sequence_length,
            grpo_config=self.grpo_config,
            temperature=self.streaming_config.temperature,
            tokenizer=self.tokenizer,
            ref_policy=self.ref_policy,
            dp_config=dp_config,
            max_grad_norm=self.grpo_config.max_grad_norm,
            scheduler=scheduler,
            device=device,
        )

        # GRPOTrainModule.__init__ calls parallelize_model which reinitializes weights.
        # We must reload HF weights after parallelization (FSDP-first loading pattern).
        logger.info(f"[Rank {self.rank}] Reloading HuggingFace weights after parallelization...")
        sd = self.train_module.model.state_dict()
        load_hf_model(self.model_name_or_path, sd, work_dir=self.grpo_config.output_dir)
        self.train_module.model.load_state_dict(sd)

        if self.grpo_config.single_gpu_mode:
            logger.info(f"[Rank {self.rank}] Converting model to {self.grpo_config.model_dtype} for single_gpu_mode")
            self.train_module.model = self.train_module.model.to(dtype=torch_dtype)

        logger.info(f"[Rank {self.rank}] OLMo-core model setup complete")
        return 1

    def setup_model_update_group(self, vllm_engines: list) -> None:
        """Set up the process group for weight synchronization with vLLM engines."""
        self.vllm_engines = vllm_engines

        if not vllm_engines or self.rank != 0:
            return

        master_address = self.get_current_node_ip()
        master_port = self.get_free_port()

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
                timeout_minutes=self.grpo_config.backend_timeout,
            )
            for i, engine in enumerate(vllm_engines)
        ]

        # Ray sets CUDA_VISIBLE_DEVICES per actor, so device 0 is always correct
        torch.cuda.set_device(0)
        self.model_update_group = vllm_utils.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=vllm_world_size,
            rank=0,
            group_name="openrlhf",
            timeout=timedelta(minutes=self.grpo_config.backend_timeout),
        )

        ray.get(refs)
        logger.info(f"[Rank {self.rank}] vLLM model update group initialized")

    def setup_callbacks(
        self,
        actor_manager: Any,
        with_tracking: bool,
        wandb_project: str | None,
        wandb_entity: str | None,
        run_name: str | None,
        json_config: dict,
        ref_policy_update_freq: int | None = None,
    ) -> None:
        """Store callback configuration for use in fit()."""
        self.actor_manager = actor_manager
        self.with_tracking = with_tracking
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.run_name = run_name
        self.json_config = json_config
        self.ref_policy_update_freq = ref_policy_update_freq

    def fit(self) -> dict:
        """Run training using OLMo-core Trainer with callbacks.

        This method sets up callbacks for weight sync, ref policy updates,
        Beaker progress tracking, and wandb logging, then calls trainer.fit().
        """
        trainer_callbacks: dict[str, callbacks.Callback] = {}

        if self.vllm_engines:
            trainer_callbacks["vllm_sync"] = VLLMWeightSyncCallback(
                vllm_engines=self.vllm_engines,
                model_update_group=self.model_update_group,
                actor_manager=self.actor_manager,
                name_mapper=olmo_core_to_hf_name,
            )

        if self.ref_policy is not None and self.grpo_config.beta > 0 and self.ref_policy_update_freq is not None:
            trainer_callbacks["ref_policy"] = RefPolicyUpdateCallback(
                ref_policy=self.ref_policy, alpha=self.grpo_config.alpha, update_interval=self.ref_policy_update_freq
            )

        if is_beaker_job() and self.json_config is not None:
            trainer_callbacks["beaker"] = BeakerCallbackV2(config=self.json_config)

        if self.with_tracking:
            trainer_callbacks["wandb"] = callbacks.WandBCallback(
                name=self.run_name, project=self.wandb_project, entity=self.wandb_entity, config=self.json_config
            )

        assert self.grpo_config.num_training_steps is not None
        self.trainer = train.TrainerConfig(
            save_folder=self.grpo_config.output_dir,
            max_duration=train.Duration.steps(self.grpo_config.num_training_steps),
            metrics_collect_interval=10,
            callbacks=trainer_callbacks,
        ).build(self.train_module, self.dataloader)

        logger.info(f"[Rank {self.rank}] Starting trainer.fit() with callbacks: {list(trainer_callbacks.keys())}")
        self.trainer.fit()
        logger.info(f"[Rank {self.rank}] Training complete")

        return {}

    def save_model(
        self, output_dir: str, chat_template_name: str, tokenizer: transformers.PreTrainedTokenizer
    ) -> None:
        """Save model checkpoint.

        All ranks must call this method because state_dict() and full_tensor()
        are collective operations when FSDP is enabled.
        """
        state_dict = self.train_module.model.state_dict()
        state_dict = {
            k: v.full_tensor().cpu() if hasattr(v, "full_tensor") else v.cpu() for k, v in state_dict.items()
        }

        if self.rank != 0:
            return

        os.makedirs(output_dir, exist_ok=True)
        olmo_core_utils.save_state_dict_as_hf(
            self.model_config, state_dict, output_dir, self.model_name_or_path, tokenizer
        )
        logger.info(f"[Rank {self.rank}] Model saved to {output_dir}")


class OLMoCoreModelGroup:
    """Manager class for OLMo-core training actors.

    Similar to ModelGroup in grpo_fast.py but uses OLMo-core based actors.
    """

    def __init__(
        self,
        pg,
        num_gpus_per_node: list[int],
        model_name_or_path: str,
        grpo_config: grpo_utils.ExperimentConfig,
        max_sequence_length: int,
        streaming_config: data_loader_lib.StreamingDataLoaderConfig,
        vllm_config: data_loader_lib.VLLMConfig,
        data_prep_actor_name: str,
        tokenizer: transformers.PreTrainedTokenizer,
        attn_implementation: str = "flash_3",
    ):
        self.pg = pg
        self.num_gpus_per_node = num_gpus_per_node
        self.num_gpus_per_actor = 0.5 if grpo_config.single_gpu_mode else 1
        self.num_cpus_per_actor = 4
        self.models = []
        world_size = sum(num_gpus_per_node)

        def get_node_info(rank, num_gpus_per_node):
            """Returns (node_index, local_rank, local_world_size) for a given global rank."""
            node_idx = 0
            remaining_rank = rank
            while remaining_rank >= num_gpus_per_node[node_idx]:
                remaining_rank -= num_gpus_per_node[node_idx]
                node_idx += 1
            return node_idx, remaining_rank, num_gpus_per_node[node_idx]

        common_kwargs = {
            "world_size": world_size,
            "model_name_or_path": model_name_or_path,
            "grpo_config": grpo_config,
            "max_sequence_length": max_sequence_length,
            "streaming_config": streaming_config,
            "vllm_config": vllm_config,
            "data_prep_actor_name": data_prep_actor_name,
            "tokenizer": tokenizer,
            "attn_implementation": attn_implementation,
        }

        node_idx, local_rank, local_world_size = get_node_info(0, num_gpus_per_node)

        master_policy = PolicyTrainerOLMoCoreProcess.options(  # ty: ignore[unresolved-attribute]
            num_cpus=self.num_cpus_per_actor,
            num_gpus=self.num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0),
        ).remote(
            rank=0,
            local_rank=local_rank,
            master_addr=None,
            master_port=None,
            local_world_size=local_world_size,
            **common_kwargs,
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
            worker_policy = PolicyTrainerOLMoCoreProcess.options(  # ty: ignore[unresolved-attribute]
                num_cpus=self.num_cpus_per_actor,
                num_gpus=self.num_gpus_per_actor,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                rank=rank,
                local_rank=local_rank,
                master_addr=master_addr,
                master_port=master_port,
                local_world_size=local_world_size,
                **common_kwargs,
            )
            self.models.append(worker_policy)
