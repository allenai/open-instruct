import contextlib
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"
with contextlib.suppress(Exception):
    pass

import logging
import random
import threading
import time
from queue import Queue

import ray
import torch
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import PreTrainedTokenizer

from open_instruct import utils
from open_instruct.actor_manager import ActorManager
from open_instruct.dataset_transformation import TokenizerConfig, get_cached_dataset_tulu
from open_instruct.ground_truth_utils import (
    build_all_verifiers,
    cleanup_all_llm_judge_clients,
    soft_format_reward_func,
)
from open_instruct.grpo_fast import (
    Args,
    BatchStatistics,
    ModelGroup,
    PendingQueriesMap,
    PolicyTrainerRayProcess,
    ShufflingIterator,
    data_preparation_thread,
    make_tokenizer,
    next_batch,
    run_training,
)
from open_instruct.model_utils import Batch, ModelConfig
from open_instruct.queue_types import GenerationResult, RequestInfo, TokenStatistics

logger = logging.getLogger(__name__)


def mock_accumulate_inference_batches(
    inference_results_Q: ray_queue.Queue,
    pending_queries_map: PendingQueriesMap,
    args: Args,
    generation_config,
    num_prompts: int,
    model_dims: utils.ModelDims,
    tokenizer: PreTrainedTokenizer,
    reward_fn,
    actor_manager=None,
    timeout: float | None = None,
    filter_zero_std_samples: bool = False,
    no_resampling_pass_rate: float | None = None,
    iter_dataloader: ShufflingIterator | None = None,
):
    generation_results = []
    queries = []
    ground_truths = []
    datasets_list = []
    raw_queries = []

    for _ in range(num_prompts):
        try:
            result = inference_results_Q.get(timeout=timeout) if timeout is not None else inference_results_Q.get()

            query, ground_truth, dataset, raw_query = pending_queries_map.pop(result.dataset_index)

            generation_results.append(result)
            queries.append(query)
            ground_truths.append(ground_truth)
            datasets_list.append(dataset)
            raw_queries.append(raw_query)

        except Exception as e:
            logger.warning(f"Error getting from queue: {e}")
            break

    if len(generation_results) == 0:
        return (
            None,
            None,
            None,
            BatchStatistics(
                prompt_lengths=[], response_lengths=[], filtered_prompts=0, zero_std_samples=0, no_resampling_samples=0
            ),
        )

    decoded_responses = []
    for result in generation_results:
        responses = []
        for response_tokens in result.responses:
            decoded = tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(decoded)
        decoded_responses.append(responses)

    batch = Batch(
        queries=queries,
        ground_truths=ground_truths,
        datasets=datasets_list,
        raw_queries=raw_queries,
        decoded_responses=decoded_responses,
        indices=None,
        scores=None,
    )

    all_responses = []
    for result in generation_results:
        all_responses.extend(result.responses)

    rewards, reward_metrics = reward_fn(batch=batch, responses=all_responses, num_candidates=args.num_generations)

    combined_result = GenerationResult(
        responses=[resp for result in generation_results for resp in result.responses],
        finish_reasons=[fr for result in generation_results for fr in result.finish_reasons],
        masks=[mask for result in generation_results for mask in result.masks],
        request_info=generation_results[0].request_info if generation_results else RequestInfo(),
        dataset_index=None,
        epoch_number=generation_results[0].epoch_number if generation_results else None,
        token_statistics=None,
        start_time=generation_results[0].start_time if generation_results else None,
        logprobs=None,
    )

    prompt_lengths = [len(q) for q in queries]
    response_lengths = [len(r) for r in combined_result.responses]

    stats = BatchStatistics(
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        filtered_prompts=0,
        zero_std_samples=0,
        no_resampling_samples=0,
    )

    return combined_result, batch, reward_metrics, stats


def mock_data_generator_thread(
    inference_results_Q: ray_queue.Queue,
    pending_queries_map: PendingQueriesMap,
    args: Args,
    model_dims: utils.ModelDims,
    iter_dataloader: ShufflingIterator,
    datasets: dict,
    stop_event: threading.Event,
):
    logger.info("Starting mock data generator thread")

    while not stop_event.is_set():
        try:
            indices = next(iter_dataloader)

            for idx in indices:
                batch = next_batch(datasets, [idx], args)

                query = batch.queries[0]
                ground_truth = batch.ground_truths[0]
                dataset = batch.datasets[0]
                raw_query = batch.raw_queries[0] if batch.raw_queries else ""

                pending_queries_map.put(idx, (query, ground_truth, dataset, raw_query))

                mock_responses = []
                finish_reasons = []
                masks = []

                for _ in range(args.num_generations):
                    response_len = args.max_completion_len
                    mock_response = torch.randint(0, model_dims.vocab_size, (response_len,)).tolist()
                    mock_responses.append(mock_response)
                    finish_reasons.append("stop" if random.random() > 0.1 else "length")
                    masks.append([1] * response_len)

                result = GenerationResult(
                    responses=mock_responses,
                    finish_reasons=finish_reasons,
                    masks=masks,
                    request_info=RequestInfo(),
                    dataset_index=idx,
                    epoch_number=iter_dataloader.epoch,
                    token_statistics=TokenStatistics(
                        num_prompt_tokens=len(query), num_generated_tokens=sum(len(r) for r in mock_responses)
                    ),
                    start_time=time.time(),
                    logprobs=None,
                )

                inference_results_Q.put(result)

            time.sleep(0.01)

        except StopIteration:
            logger.info("Mock data generator finished epoch")
            break
        except Exception as e:
            logger.error(f"Error in mock data generator: {e}")
            break

    logger.info("Mock data generator thread stopped")


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

    if args.with_tracking:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=args.__dict__,
            name=args.run_name,
            save_code=True,
        )

    logger.info("Loading datasets")

    system_prompt_override = None
    if args.system_prompt_override_file is not None:
        logger.info(f"Loading system prompt override from {args.system_prompt_override_file}")
        with open(args.system_prompt_override_file) as f:
            system_prompt_override = f.read().strip()
        logger.info(f"System prompt overriden to:\n#####\n{system_prompt_override}\n#####\n")

    transform_fn_args = [
        {"system_prompt_override": system_prompt_override},
        {"max_prompt_token_length": args.max_prompt_token_length},
    ]

    datasets = get_cached_dataset_tulu(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tokenizer_config,
        dataset_transform_fn=args.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_config_hash=args.dataset_config_hash,
        hf_entity=args.hf_entity,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
        system_prompt_override=system_prompt_override,
    )

    ray.init(dashboard_host="0.0.0.0", runtime_env={"excludes": [".git/"], "env_vars": dict(os.environ)})

    inference_results_Q = ray_queue.Queue(maxsize=args.inference_results_queue_size)
    param_prompt_Q = ray_queue.Queue(maxsize=args.param_prompt_queue_size)
    evaluation_inference_results_Q = ray_queue.Queue(maxsize=10)
    packed_sequences_Q = Queue(maxsize=args.packed_sequences_queue_size)

    logger.info("Creating model and optimizer")
    (policy_group, vllm_engines, tool_objects, resume_training_step, episode, actor_manager) = (
        create_model_and_optimizer(args, tokenizer_config, model_config)
    )

    model_dims = ray.get(policy_group.models[0].get_model_dims.remote())
    logger.info(f"Model dims: {model_dims}")
    logger.info(f"Vocab size: {model_dims.vocab_size}")

    logger.info("Building verifiers")
    verifier_fns = build_all_verifiers(
        args=args,
        tokenizer=tokenizer,
        model_config=model_config,
        beam_size=args.best_of_n_beam_size,
        max_search_expansion=args.best_of_n_max_expansion,
    )
    reward_fn = soft_format_reward_func(
        verifier_fns=verifier_fns,
        use_vllm_server_for_llm_judge=args.use_vllm_server_for_llm_judge,
        tool_objects=tool_objects,
        num_workers=1,
    )

    iter_dataloader = ShufflingIterator(
        dataset_sizes={name: len(ds) for name, ds in datasets.items()},
        dataset_mixer_fracs=args.dataset_mixer_fracs,
        seed=args.seed,
        start_epoch=0,
        start_step=0,
    )

    pending_queries_map = PendingQueriesMap()

    stop_event = threading.Event()
    mock_gen_thread = threading.Thread(
        target=mock_data_generator_thread,
        args=(inference_results_Q, pending_queries_map, args, model_dims, iter_dataloader, datasets, stop_event),
        daemon=True,
    )
    mock_gen_thread.start()

    data_prep_thread = threading.Thread(
        target=data_preparation_thread,
        args=(
            inference_results_Q,
            packed_sequences_Q,
            pending_queries_map,
            args,
            None,
            model_dims,
            tokenizer,
            reward_fn,
            actor_manager,
            mock_accumulate_inference_batches,
            None,
        ),
        daemon=True,
    )
    data_prep_thread.start()

    logger.info("Starting training")
    run_training(
        args=args,
        policy_group=policy_group,
        vllm_engines=vllm_engines,
        packed_sequences_Q=packed_sequences_Q,
        param_prompt_Q=param_prompt_Q,
        evaluation_inference_results_Q=evaluation_inference_results_Q,
        datasets=datasets,
        tokenizer=tokenizer,
        model_dims=model_dims,
        reward_fn=reward_fn,
        resume_training_step=resume_training_step,
        episode=episode,
        iter_dataloader=iter_dataloader,
        pending_queries_map=pending_queries_map,
        actor_manager=actor_manager,
    )

    stop_event.set()
    mock_gen_thread.join(timeout=5)

    cleanup_all_llm_judge_clients()

    if args.with_tracking:
        wandb.finish()


if __name__ == "__main__":
    parser = utils.ArgumentParserPlus((Args, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse()
    main(args, tokenizer_config, model_config)
