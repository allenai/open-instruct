"""Single-GPU Qwen3-0.6B SFT runner using olmo-core APIs directly.

Mirrors open_instruct/olmo_core_finetune.py's HF->olmo-core weight-loading path so
both runs start from identical weights. Compare step-0/1/2 CE loss for byte-level parity.
"""

import argparse
import logging

import transformers

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyPackedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import GarbageCollectorCallback, GPUMemoryMonitorCallback
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--save_folder", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--config_name", default="qwen3_0_6B")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--rank_microbatch_size_tokens", type=int, default=1024)
    parser.add_argument("--global_batch_size_tokens", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_fraction", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--init_seed", type=int, default=33333)
    parser.add_argument("--data_loader_seed", type=int, default=34521)
    parser.add_argument("--compile_model", action="store_true", default=True)
    args = parser.parse_args()

    prepare_training_environment()
    seed_all(args.init_seed)

    tokenizer_config = TokenizerConfig.dolma2()

    dataset_config = NumpyPackedFSLDatasetConfig(
        tokenizer=tokenizer_config,
        work_dir=args.save_folder,
        paths=[f"{args.dataset_path.rstrip('/')}/token_ids_part_*.npy"],
        expand_glob=True,
        label_mask_paths=[f"{args.dataset_path.rstrip('/')}/labels_mask_*.npy"],
        generate_doc_lengths=True,
        long_doc_strategy=LongDocStrategy.truncate,
        sequence_length=args.seq_len,
    )

    ac_config = TransformerActivationCheckpointingConfig(
        mode=TransformerActivationCheckpointingMode.selected_modules,
        modules=["blocks.*.feed_forward"],
    )

    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.ddp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
    )

    hf_config = transformers.AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    vocab_size = hf_config.vocab_size
    log.info(f"Building olmo-core model with vocab_size={vocab_size} from HF config")

    model_config = getattr(TransformerConfig, args.config_name)(
        vocab_size=vocab_size,
        attn_backend=AttentionBackendName.flash_2,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=args.rank_microbatch_size_tokens,
        max_sequence_length=args.seq_len,
        z_loss_multiplier=None,
        compile_model=args.compile_model,
        optim=SkipStepAdamWConfig(
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            compile=False,
        ),
        dp_config=dp_config,
        cp_config=None,
        ac_config=ac_config,
        scheduler=LinearWithWarmup(warmup_fraction=args.warmup_fraction, alpha_f=0.0),
        max_grad_norm=args.max_grad_norm,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=args.global_batch_size_tokens,
        seed=args.data_loader_seed,
        num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=args.save_folder,
            load_strategy=LoadStrategy.never,
            checkpointer=CheckpointerConfig(save_thread_count=1, load_thread_count=1, throttle_uploads=True),
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=10,
            max_duration=Duration.steps(args.max_steps),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
    )

    model = model_config.build(init_device="cpu")
    train_module = train_module_config.build(model)
    dataset = dataset_config.build()
    data_loader = data_loader_config.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = trainer_config.build(train_module, data_loader)

    log.info(f"Loading HF weights from {args.model_name_or_path} into olmo-core model")
    sd = train_module.model.state_dict()
    load_hf_model(args.model_name_or_path, sd, work_dir=args.save_folder)
    train_module.model.load_state_dict(sd)

    data_loader.reshuffle(epoch=1)

    try:
        trainer.fit()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
