"""
Native OLMo-core MoE DPO entrypoint for the OLMoE3-dev-260429 checkpoint.

This uses Open Instruct's preference-data/tokenization pipeline and DPO loss, but
keeps the model on the native OLMo-core MoE V2 checkpoint path.
"""

import math
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any

# Keep this before olmo_core imports. Several modules import nvtx at import time.
os.environ.setdefault("NVTX_DISABLE", "1")

import torch
import torch.distributed as dist
from olmo_core import train
from olmo_core.config import DType
from olmo_core.distributed import utils as distributed_utils
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType, SlidingWindowAttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.moe import MoELoadBalancingLossGranularity, MoERouterGatingFunction
from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlockConfig
from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import (
    MoEFusedV2TransformerConfig,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.optim import OptimGroupOverride
from olmo_core.optim.moe_optimizer import MoEFusedV2OptimizerConfig
from olmo_core.train import callbacks
from olmo_core.train.callbacks import ConsoleLoggerCallback, ProfilerCallback
from olmo_core.train.train_module.transformer import config as transformer_config

from open_instruct import data_loader as data_loader_lib
from open_instruct import dataset_transformation, dpo_utils, logger_utils, olmo_core_utils, utils
from open_instruct.olmo_core_train_modules import MoEDPOTrainModule
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO

logger = logger_utils.setup_logger(__name__)

DEFAULT_LOAD_PATH = (
    "/workspace/checkpoint/"
    "OLMoE3-dev-260429-t001_2048d2560a_16L2048M1536S_40E4K1S_p1/step112000"
)
DEFAULT_TOKENIZER = "allenai/olmo-3.2-tokenizer-think-dev"
DEFAULT_DPO_MIX = "allenai/Dolci-Think-DPO-32B"
DEFAULT_DATASET_CACHE = "/workspace/checkpoint/open-instruct-dataset-cache"
DEFAULT_OUTPUT_ROOT = "/workspace/checkpoint"

NUM_EXPERTS = 40
TOP_K = 4
D_MODEL = 2048
D_ATTN = 2560
HEAD_DIM = 128
NUM_HEADS = D_ATTN // HEAD_DIM
NUM_KV_HEADS = NUM_HEADS // 2
MOE_HIDDEN_SIZE = 2048
NUM_SHARED_EXPERTS = 1
SHARED_MLP_HIDDEN_SIZE = 1536
DENSE_LAYER_HIDDEN_SIZE = TOP_K * MOE_HIDDEN_SIZE + SHARED_MLP_HIDDEN_SIZE
NUM_LAYERS = 16

USE_PERI_NORM = True
USE_NO_SYNC_EP = True
USE_ROWWISE_A2A = True
USE_FP8 = False
ROWWISE_A2A_NBLOCKS = 256
MODEL_INIT_SEED = 2026


@dataclass
class MoEDPOExperimentConfig(dpo_utils.DPOExperimentConfig):
    exp_name: str = "OLMoE3-dev-260429-t001-dpo-pilot"
    model_name_or_path: str = DEFAULT_LOAD_PATH
    mixer_list: list[str] = field(default_factory=lambda: [DEFAULT_DPO_MIX, "1.0"])
    max_train_samples: int | None = 1024
    max_train_steps: int | None = 100
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 7e-8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    compile_model: bool = False
    packing: bool = True
    push_to_hub: bool = False
    try_launch_beaker_eval_jobs: bool = False
    try_auto_save_to_beaker: bool = False
    output_dir: str = DEFAULT_OUTPUT_ROOT
    local_cache_dir: str = DEFAULT_DATASET_CACHE
    cache_mode: str = "local"
    ep_degree: int = 8
    load_thread_count: int = 8
    save_checkpoints: bool = False
    reference_cache_dir: str | None = None


def _attention_config(
    dtype: DType,
    layer_norm: LayerNormConfig,
    attention_backend: AttentionBackendName,
) -> AttentionConfig:
    return AttentionConfig(
        name=AttentionType.default,
        n_heads=NUM_HEADS,
        n_kv_heads=NUM_KV_HEADS,
        bias=False,
        rope=RoPEConfig(
            name=RoPEType.default,
            theta=500_000,
            scaling=None,
            full_precision=True,
        ),
        qk_norm=layer_norm,
        backend=attention_backend,
        use_head_qk_norm=True,
        dtype=dtype,
        d_attn=D_ATTN,
        use_recompute_qkv_prep=False,
    )


def build_model_config(vocab_size: int, attention_backend: AttentionBackendName) -> TransformerConfig:
    dtype = DType.float32
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)

    block = MoEFusedV2TransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        use_peri_norm=USE_PERI_NORM,
        ep_no_sync=USE_NO_SYNC_EP,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_attn=False,
        checkpoint_second_unpermute=False,
        ep_no_sync_share_combine_out=False,
        ep_no_sync_share_dispatch_out=False,
        ep_no_sync_shared_slots=1,
        ep_no_sync_use_rowwise_all_to_all=USE_ROWWISE_A2A,
        ep_no_sync_rowwise_nblocks=ROWWISE_A2A_NBLOCKS,
        ep_no_sync_capacity_factor=1.25,
        rowwise_fp8=MoERowwiseFP8Config(enabled=USE_FP8) if USE_ROWWISE_A2A else None,
        attention=_attention_config(dtype, layer_norm, attention_backend),
        attention_norm=layer_norm,
        routed_experts=RoutedExpertsConfig(
            d_model=D_MODEL,
            hidden_size=MOE_HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            bias=False,
            dtype=dtype,
        ),
        routed_experts_router=MoERouterConfigV2(
            d_model=D_MODEL,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            gating_function=MoERouterGatingFunction.softmax,
            uniform_expert_assignment=False,
            random_expert_assignment=False,
            lb_loss_weight=0.01,
            z_loss_weight=1e-3,
            lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
            dtype=dtype,
            normalize_expert_weights=1.0,
            restore_weight_scale=True,
            use_recompute_fp32_cast=False,
        ),
        shared_experts=SharedExpertsConfig(
            d_model=D_MODEL,
            hidden_size=SHARED_MLP_HIDDEN_SIZE,
            num_experts=NUM_SHARED_EXPERTS,
            bias=False,
            dtype=dtype,
        ),
        shared_experts_router=None,
        feed_forward_norm=layer_norm,
    )

    config = MoEFusedV2TransformerConfig(
        name=TransformerType.moe_fused_v2,
        init_seed=MODEL_INIT_SEED,
        d_model=D_MODEL,
        two_batch_overlap=False,
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
        vocab_size=vocab_size,
        n_layers=NUM_LAYERS,
        embed_scale=math.sqrt(D_MODEL),
        embedding_norm=layer_norm,
        block=block,
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        init_std=0.01,
        dtype=dtype,
    )
    config.lm_head.loss_implementation = LMLossImplementation.default
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[2048, -1],
    )
    config.block_overrides = {
        0: TransformerBlockConfig(
            name=TransformerBlockType.peri_norm if USE_PERI_NORM else TransformerBlockType.default,
            attention=_attention_config(dtype, layer_norm, attention_backend),
            attention_norm=layer_norm,
            feed_forward=FeedForwardConfig(
                hidden_size=DENSE_LAYER_HIDDEN_SIZE,
                bias=False,
                dtype=dtype,
            ),
            feed_forward_moe=None,
            feed_forward_norm=layer_norm,
        )
    }
    return config


def build_optimizer_config(args: MoEDPOExperimentConfig) -> MoEFusedV2OptimizerConfig:
    return MoEFusedV2OptimizerConfig(
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(
                params=[
                    "*embeddings.weight",
                    "*embedding_norm.weight",
                    "*q_norm.weight",
                    "*k_norm.weight",
                    "*input_norm.weight",
                    "*lm_head.norm.weight",
                    "*attention_norm.weight",
                    "*feed_forward_norm.weight",
                ],
                opts={"weight_decay": 0.0, "use_muon": False},
            )
        ],
        compile=False,
        dtype=DType.float32,
        sigma_factor=12,
        use_distributed=True,
    )


def _resolve_model_and_optim_dir(path: str) -> str:
    candidate = os.path.join(path, "model_and_optim")
    if os.path.exists(os.path.join(candidate, ".metadata")):
        return candidate
    return path


def _coerce_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def _setup_callbacks(args: MoEDPOExperimentConfig) -> dict[str, callbacks.Callback]:
    json_config = dpo_utils.config_to_json_serializable(vars(args))
    run_name = args.run_name or args.exp_name
    trainer_callbacks = olmo_core_utils.build_base_callbacks(
        config_dict=json_config,
        run_name=run_name,
        checkpointing_steps=args.checkpointing_steps,
        ephemeral_save_interval=args.ephemeral_save_interval,
        with_tracking=args.with_tracking,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        save_async=False,
    )
    if not args.save_checkpoints:
        trainer_callbacks.pop("checkpointer", None)
    trainer_callbacks["console_logger"] = ConsoleLoggerCallback(
        metrics_log_interval=args.logging_steps,
        metrics=[
            "train/loss",
            "train/logps_*",
            "train/rewards_*",
            "train/token_count",
            "train/aux_loss",
            "train/load balancing loss",
            "train/router Z loss",
            "train/block */load imbalance",
            "train/block */token drop rate",
            "train/block */symm buffer util",
            "gpu_memory/*",
            "optim/total grad norm",
            "optim/step skipped",
            "optim/LR*",
            "throughput/*",
            "checkpoint/*",
        ],
    )
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if args.send_slack_alerts and slack_webhook_url:
        trainer_callbacks["slack"] = callbacks.SlackNotifierCallback(name=run_name, webhook_url=slack_webhook_url)
    if args.profiling:
        trainer_callbacks["profiler"] = ProfilerCallback(
            skip_first=5, wait=1, warmup=2, active=3, repeat=1, profile_memory=True
        )
    return trainer_callbacks


def _build_model_dims(vocab_size: int, model_config: TransformerConfig) -> utils.ModelDims:
    return utils.ModelDims(
        num_layers=NUM_LAYERS,
        hidden_size=D_ATTN,
        intermediate_size=DENSE_LAYER_HIDDEN_SIZE,
        vocab_size=vocab_size,
        num_attn_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        num_params=int(getattr(model_config, "num_active_params", model_config.num_params)),
        sliding_window=2048,
        num_sliding_window_layers=NUM_LAYERS - 1,
    )


def _truncate_dataset(dataset, max_train_samples: int | None):
    if max_train_samples is None or len(dataset) <= max_train_samples:
        return dataset
    return dataset.select(range(max_train_samples))


def main(args: MoEDPOExperimentConfig, tc: dataset_transformation.TokenizerConfig) -> None:
    args.save_checkpoints = _coerce_bool(args.save_checkpoints)
    args.packing = _coerce_bool(args.packing)
    args.compile_model = _coerce_bool(args.compile_model)
    args.push_to_hub = _coerce_bool(args.push_to_hub)
    args.try_launch_beaker_eval_jobs = _coerce_bool(args.try_launch_beaker_eval_jobs)
    args.try_auto_save_to_beaker = _coerce_bool(args.try_auto_save_to_beaker)

    if args.use_lora:
        raise ValueError("LoRA is not supported for native MoE DPO.")
    if args.tensor_parallel_degree > 1 or args.context_parallel_degree > 1:
        raise NotImplementedError("Native MoE DPO currently supports DDP + EP only.")
    if args.dpo_use_paged_optimizer or args.use_8bit_optimizer:
        raise ValueError("Paged and 8-bit optimizers are not supported for native MoE DPO.")
    if not args.packing:
        logger.warning("Running native MoE DPO without packing; flash doc-lens masking will be disabled.")
    if tc.tokenizer_name_or_path is None:
        tc.tokenizer_name_or_path = DEFAULT_TOKENIZER

    tokenizer = olmo_core_utils.setup_tokenizer_and_cache(args, args, tc)
    transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]

    if args.cache_dataset_only:
        olmo_core_utils.load_dataset_distributed(args, tc, transform_fn_args, is_main_process=True)
        logger.info("Dataset cached successfully. Exiting because --cache_dataset_only was set.")
        return

    global_rank, world_size, is_main_process = olmo_core_utils.setup_distributed_env(seed=args.seed)
    dp_world_size = world_size
    dp_rank = global_rank

    utils.setup_experiment_paths(args, is_main_process)
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    if distributed_utils.is_distributed():
        dist.barrier()

    dataset = olmo_core_utils.load_dataset_distributed(args, tc, transform_fn_args, is_main_process)
    dataset = _truncate_dataset(dataset, args.max_train_samples)
    dataset = dataset.shuffle(seed=args.seed)
    dataset.set_format(type="pt")
    if is_main_process:
        logger.info("Loaded DPO dataset with %d examples", len(dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = olmo_core_utils.to_oc_tokenizer_config(tc).padded_vocab_size()
    model_config = build_model_config(vocab_size=vocab_size, attention_backend=args.attn_implementation)
    model = model_config.build(init_device="meta")

    if args.packing:
        logger.info("Using packing/padding-free DPO collation")
        collator = TensorDataCollatorWithFlatteningDPO(
            return_position_ids=True,
            return_flash_attn_kwargs=True,
            max_seq_length=args.max_seq_length,
        )
    else:
        collator = dpo_utils.DataCollatorForSeq2SeqDPO(tokenizer=tokenizer, model=None, padding="longest")

    rank_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    global_batch_size = rank_batch_size * dp_world_size
    data_loader = data_loader_lib.HFDataLoader(
        dataset=dataset,
        batch_size=global_batch_size,
        seed=args.seed,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        work_dir=args.output_dir,
        collator=collator,
        device=device,
        drop_last=True,
        fs_local_rank=global_rank,
        max_seq_length=args.max_seq_length * 2,
    )
    cache_data_loader = data_loader_lib.HFDataLoader(
        dataset=dataset,
        batch_size=int(args.per_device_train_batch_size * 4 * dp_world_size),
        seed=args.seed,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        work_dir=args.output_dir,
        collator=collator,
        device=device,
        drop_last=False,
        fs_local_rank=global_rank,
        max_seq_length=args.max_seq_length * 2,
    )

    data_loader.reshuffle(epoch=0)
    num_training_steps = len(data_loader) * args.num_epochs
    effective_steps = args.max_train_steps if args.max_train_steps is not None else num_training_steps

    scheduler = olmo_core_utils.build_scheduler(args.lr_scheduler_type, args.warmup_ratio, effective_steps)
    dp_config = transformer_config.TransformerDataParallelConfig(
        name=DataParallelType.ddp,
        reduce_grads_in_fp32=True,
        accumulate_grads_in_fp32=True,
    )
    ep_config = transformer_config.TransformerExpertParallelConfig(degree=args.ep_degree) if args.ep_degree > 1 else None
    ac_config = olmo_core_utils.build_ac_config(args.activation_memory_budget, args.compile_model)

    train_module = MoEDPOTrainModule(
        model=model,
        optim=build_optimizer_config(args),
        sample_microbatch_size=args.per_device_train_batch_size,
        max_sequence_length=args.max_seq_length,
        dpo_config=args,
        attn_implementation=args.attn_implementation,
        dp_config=dp_config,
        ep_config=ep_config,
        ac_config=ac_config,
        compile_model=args.compile_model,
        max_grad_norm=args.max_grad_norm,
        scheduler=scheduler,
        device=device,
        reset_optimizer_states_on_load=True,
    )

    load_dir = _resolve_model_and_optim_dir(args.model_name_or_path)
    logger.info("Loading native checkpoint from %s", load_dir)
    train_module.load_state_dict_direct(load_dir, work_dir=args.output_dir, thread_count=args.load_thread_count)
    if dist.is_initialized():
        dist.barrier()

    ref_cache_hash = dpo_utils.compute_reference_cache_hash(args, tc)
    reference_cache_dir = pathlib.Path(args.reference_cache_dir or os.path.join(args.output_dir, "reference_logprobs_cache"))
    reference_cache_path = reference_cache_dir / f"{ref_cache_hash}.pt"
    logger.info("Reference logprobs cache path: %s", reference_cache_path)

    forward_fn = dpo_utils.concatenated_forward_olmo if args.concatenated_forward else dpo_utils.separate_forward_olmo
    forward_kwargs: dict[str, Any] = {}
    if args.packing:
        forward_kwargs["packing"] = True
    average_log_prob = args.loss_type.is_average_loss

    if args.loss_type.needs_reference_model:
        logger.info("Caching reference logprobs...")
        train_module.reference_cache = dpo_utils.build_reference_logprobs_cache(
            model=train_module.model,
            dataloader=cache_data_loader,
            average_log_prob=average_log_prob,
            forward_fn=forward_fn,
            forward_kwargs=forward_kwargs,
            full_dataset_size=len(dataset),
            device=device,
            cache_path=reference_cache_path,
            is_main_process=is_main_process,
            model_dims=_build_model_dims(vocab_size, model_config),
            use_lora=False,
            disable_adapter_context=None,
        )

    if args.cache_logprobs_only:
        logger.info("--cache_logprobs_only set, exiting after cache build.")
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    trainer_callbacks = _setup_callbacks(args)
    max_duration = train.Duration.steps(args.max_train_steps if args.max_train_steps is not None else num_training_steps)
    trainer = train.TrainerConfig(
        save_folder=args.output_dir,
        max_duration=max_duration,
        metrics_collect_interval=args.logging_steps,
        callbacks=trainer_callbacks,
        save_overwrite=True,
        no_checkpoints=not args.save_checkpoints,
    ).build(train_module, data_loader)
    trainer.epoch = 0

    logger.info("Starting native MoE DPO training...")
    trainer.fit()
    logger.info("Native MoE DPO training complete.")

    train.teardown_training_environment()


if __name__ == "__main__":
    from open_instruct.utils import ArgumentParserPlus

    parser = ArgumentParserPlus((MoEDPOExperimentConfig, dataset_transformation.TokenizerConfig))
    args, tc = parser.parse()
    main(args, tc)
