import contextlib
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"
with contextlib.suppress(Exception):
    pass

import logging

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from open_instruct import utils
from open_instruct.actor_manager import ActorManager
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.ground_truth_utils import cleanup_all_llm_judge_clients
from open_instruct.grpo_fast import (
    Args,
    ModelGroup,
    PolicyTrainerRayProcess,
    make_tokenizer,
    setup_datasets,
    setup_experiment_tracking,
    setup_runtime_variables,
)
from open_instruct.model_utils import ModelConfig

logger = logging.getLogger(__name__)


def create_model_and_optimizer(args: Args, tokenizer_config: TokenizerConfig, model_config: ModelConfig):
    logger.info("Creating model and optimizer (mock version)")

    num_learner_gpus = args.world_size

    pg = placement_group([{"GPU": 1, "CPU": 1}] * num_learner_gpus, strategy="STRICT_PACK")
    ray.get(pg.ready())

    policy_group = ModelGroup.from_pretrained(
        args=args,
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        ds_config=args.deepspeed,
        sched_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_capture_child_tasks=True),
        rank_offset=0,
        world_size=num_learner_gpus,
        actor_cls=PolicyTrainerRayProcess,
    )

    tool_objects = {}
    actor_manager = ActorManager([], tool_objects)

    return policy_group, [], tool_objects, 0, 0, actor_manager


def main(args: Args, tokenizer_config: TokenizerConfig, model_config: ModelConfig):
    logger.info("Starting mock_data_grpo_fast main")

    tokenizer = make_tokenizer(tokenizer_config, model_config)
    args = setup_runtime_variables(args)

    beaker_config, wandb_url = setup_experiment_tracking(args, tokenizer_config, model_config)
    train_dataset, eval_dataset = setup_datasets(args, tokenizer_config, tokenizer)

    ray.init(dashboard_host="0.0.0.0", runtime_env={"excludes": [".git/"], "env_vars": dict(os.environ)})

    logger.info("Creating model and optimizer")
    policy_group, vllm_engines, tool_objects, resume_training_step, episode, actor_manager = (
        create_model_and_optimizer(args, tokenizer_config, model_config)
    )

    model_dims = ray.get(policy_group.models[0].get_model_dims.remote())
    logger.info(f"Model dims: {model_dims}")
    logger.info(f"Vocab size: {model_dims.vocab_size}")

    logger.info("Generating mock training data for OOM testing...")
    import numpy as np

    pack_len = args.pack_length
    num_seqs = args.per_device_train_batch_size * args.world_size

    query_responses = np.random.randint(0, model_dims.vocab_size, (num_seqs, pack_len), dtype=np.int64)
    attention_masks = np.ones((num_seqs, pack_len, pack_len), dtype=np.float32)
    response_masks = np.ones((num_seqs, pack_len), dtype=np.float32)
    original_responses = query_responses.copy()
    advantages = np.random.randn(num_seqs, pack_len).astype(np.float32)
    position_ids = np.tile(np.arange(pack_len), (num_seqs, 1))

    logger.info(f"Running training step with {num_seqs} sequences of length {pack_len}...")
    logger.info(f"Total tokens per batch: {num_seqs * pack_len:,}")

    collated_data = {
        "query_responses": query_responses,
        "attention_masks": attention_masks,
        "response_masks": response_masks,
        "original_responses": original_responses,
        "advantages": advantages,
        "position_ids": position_ids,
    }

    try:
        for step in range(min(args.total_episodes // args.num_unique_prompts_rollout, 5)):
            logger.info(f"\n===== Training Step {step + 1} =====")
            results = ray.get(
                [policy_group.models[i].train.remote(collated_data) for i in range(len(policy_group.models))]
            )
            logger.info(f"Step {step + 1} completed successfully!")
            if results and len(results) > 0 and results[0]:
                loss = results[0].get("loss", "N/A")
                logger.info(f"Loss: {loss}")

        logger.info("\n✅ Mock OOM test completed successfully!")
        logger.info("All training steps passed without OOM. You can increase sequence length or batch size.")

    except Exception as e:
        logger.error(f"\n❌ OOM or error occurred: {e}")
        raise

    cleanup_all_llm_judge_clients()


if __name__ == "__main__":
    parser = utils.ArgumentParserPlus((Args, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse()
    main(args, tokenizer_config, model_config)
