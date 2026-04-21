# Copyright 2026 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sibling training script for the GENERATIVE value model.

This script is a thin wrapper around ``open_instruct.grpo_fast`` that adds a *second* vLLM pool
hosting a generative value model, runs it at SAE (or fixed-chunk) segment boundaries, and trains
the generative value model in-place via REINFORCE using the rollout outcome as reward.

The generative value model has its own weights and vLLM pool.  During each policy training step
``grpo_fast.PolicyTrainerRayProcess.step()`` scores the actual rollout tokens at fixed-chunk
boundaries via the gen-value vLLM pool and uses the returned piecewise-constant scores as the
value function for GAE (replacing the scalar value head when ``use_generative_value_model=True``).

A background REINFORCE thread collects ``(prompt, generated_score, outcome)`` training pairs from
the policy actors via a shared Ray queue.  A ``GenValueTrainerActor`` (one per cluster) holds a
PyTorch copy of the gen-value model, computes REINFORCE gradients, and optionally syncs the
updated weights back to the gen-value vLLM pool every ``gen_value_sync_freq`` policy steps by
saving the model to a temp directory and restarting the gen-value vLLM engine.  Set
``gen_value_sync_freq=0`` (default) to train the PyTorch gen-value model without syncing back to
vLLM (useful for debugging the gradient path before wiring full weight-transfer infrastructure).

Usage::

    python open_instruct/grpo_fast_genvalue.py \\
        --model_name_or_path ... \\
        --use_generative_value_model \\
        --gen_value_model_name_or_path ... \\
        --gen_value_vllm_num_engines 1 \\
        --gen_value_segmentation sae \\
        --sae_threshold 0.2 \\
        --gen_value_score_min 0 --gen_value_score_max 10 \\
        --gen_value_conditioning gt \\
        --gen_value_reinforce_coef 0.1 \\
        ...
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import random
import tempfile
import threading
from concurrent import futures
from dataclasses import dataclass
from queue import Queue
from typing import Any

import ray
import torch
import torch.distributed as dist
import wandb
from ray.util import queue as ray_queue
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_fast_resource_plan, grpo_utils, utils, value_model_utils, vllm_utils

# grpo_fast is heavy (pulls in vLLM) so it is imported lazily inside main().
from open_instruct.dataset_transformation import INPUT_IDS_PROMPT_KEY, TokenizerConfig
from open_instruct.environments.tools.utils import EnvsConfig
from open_instruct.ground_truth_utils import RewardConfig, build_all_verifiers
from open_instruct.model_utils import ModelConfig
from open_instruct.utils import ArgumentParserPlus

logger = logging.getLogger(__name__)

_GEN_VALUE_SAMPLE_SIZE = 4  # prompts sampled per step for background scoring


@ray.remote(num_gpus=1)
class GenValueTrainerActor:
    """Ray actor that holds the gen-value model (PyTorch) and performs REINFORCE updates.

    Lives on the same GPU as the gen-value vLLM engine so that IPC-based weight sync
    (trainer → vLLM) can be used when needed.  The actor receives training pairs via
    ``reinforce_step`` and maintains a running EMA baseline for variance reduction.
    """

    def __init__(
        self,
        model_path: str,
        learning_rate: float,
        score_min: float,
        score_max: float,
        allow_cot: bool,
    ) -> None:
        self._lr = learning_rate
        self._score_min = score_min
        self._score_max = score_max
        self._allow_cot = allow_cot
        self._step_count = 0
        self._baseline = 0.5

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, use_cache=False
        ).cuda()
        self._model.train()
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)

    def reinforce_step(self, training_pairs: list[dict]) -> dict:
        """Apply one REINFORCE gradient step from a batch of (prompt, generated, outcome) pairs.

        Returns a metrics dict with ``gen_value/reinforce_loss`` and ``gen_value/outcome_mean``.
        """
        if not training_pairs:
            return {}

        total_loss = 0.0
        outcomes = [p["outcome"] for p in training_pairs if p["outcome"] is not None]
        if not outcomes:
            return {}

        self._optimizer.zero_grad()
        for pair in training_pairs:
            if pair["outcome"] is None:
                continue
            prompt = pair["prompt"]
            generated = pair["generated"]
            outcome = float(pair["outcome"])

            combined = prompt + generated
            enc = self._tokenizer(combined, return_tensors="pt", truncation=True, max_length=2048)
            prompt_len = len(self._tokenizer(prompt, add_special_tokens=False).input_ids)
            input_ids = enc.input_ids.cuda()
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100

            outputs = self._model(input_ids=input_ids, labels=labels)
            log_prob = -outputs.loss  # mean log-prob of generated tokens

            advantage = outcome - self._baseline
            loss = -log_prob * advantage
            loss.backward()
            total_loss += float(loss.detach())

            # EMA baseline update
            self._baseline = 0.95 * self._baseline + 0.05 * outcome

        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()
        self._step_count += 1

        return {
            "gen_value/reinforce_loss": total_loss / len(outcomes),
            "gen_value/outcome_mean": sum(outcomes) / len(outcomes),
            "gen_value/reinforce_steps": self._step_count,
        }

    def save_weights(self, output_dir: str) -> str:
        """Save the current model weights to ``output_dir`` and return the path."""
        os.makedirs(output_dir, exist_ok=True)
        self._model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)
        return output_dir

    def get_step_count(self) -> int:
        return self._step_count

    def ready(self) -> bool:
        return True


@dataclass
class GenValueExperimentConfig(grpo_utils.GRPOExperimentConfig):
    """Extended experiment config for the generative-value training script."""

    # Whether to enable the generative value model path (required for this script).
    use_generative_value_model: bool = False
    # Generative value model: its own weights + its own vLLM pool.
    gen_value_model_name_or_path: str | None = None
    gen_value_vllm_num_engines: int = 1
    gen_value_vllm_tensor_parallel_size: int = 1
    # Segmentation: 'sae' uses the policy-logprob-based SAE boundaries (requires --use_sae);
    # 'fixed' queries the gen value model every `gen_value_chunk_size` response tokens.
    gen_value_segmentation: str = "sae"
    gen_value_chunk_size: int = 512
    # Generation params for the gen value model's vLLM engine.
    gen_value_max_new_tokens: int = 8
    gen_value_temperature: float = 1.0
    # Score schema.
    gen_value_score_min: float = 0.0
    gen_value_score_max: float = 10.0
    gen_value_allow_cot: bool = False
    # Training coefficients.
    gen_value_learning_rate: float | None = None
    gen_value_reinforce_coef: float = 0.1
    # Conditioning for the gen-value prompt: one of none, gt, correct_demo, rollout_context.
    gen_value_conditioning: str = "none"
    # How often (in policy steps) to sync REINFORCE-trained gen-value weights back to vLLM.
    # 0 disables periodic sync (weights stay frozen in vLLM; REINFORCE gradient is still computed).
    gen_value_sync_freq: int = 0

    def __post_init__(self):
        super().__post_init__()
        if not self.use_generative_value_model:
            raise ValueError(
                "grpo_fast_genvalue.py requires --use_generative_value_model. "
                "Use grpo_fast.py for runs without a generative value model."
            )
        if self.gen_value_segmentation not in {"sae", "fixed"}:
            raise ValueError(
                f"--gen_value_segmentation must be 'sae' or 'fixed', got {self.gen_value_segmentation!r}."
            )
        if self.gen_value_segmentation == "sae" and not self.use_sae:
            raise ValueError(
                "--gen_value_segmentation=sae requires --use_sae (SAE boundaries come from the policy's vLLM logprobs)."
            )
        if self.gen_value_chunk_size <= 0:
            raise ValueError(f"--gen_value_chunk_size must be > 0, got {self.gen_value_chunk_size}.")
        if self.gen_value_score_max <= self.gen_value_score_min:
            raise ValueError("--gen_value_score_max must be greater than --gen_value_score_min.")
        if self.gen_value_conditioning not in value_model_utils.GEN_VALUE_CONDITIONING_TYPES:
            raise ValueError(
                f"--gen_value_conditioning must be one of "
                f"{sorted(value_model_utils.GEN_VALUE_CONDITIONING_TYPES)}, "
                f"got {self.gen_value_conditioning!r}."
            )


def segment_rollout(
    response_tokens: list[int],
    response_logprobs: list[float] | None,
    *,
    mode: str,
    sae_threshold: float = 0.2,
    fixed_chunk_size: int = 512,
) -> list[int]:
    """Return the list of boundary positions (response-token indices) at which the generative value
    model should be queried.

    In ``sae`` mode, a boundary is any response token whose probability (exp(logprob)) is below
    ``sae_threshold``. In ``fixed`` mode, boundaries are emitted every ``fixed_chunk_size`` tokens.
    Boundary indices are returned in ascending order and always include a final boundary at the
    end of the rollout so the terminal outcome is scored.
    """
    length = len(response_tokens)
    if length == 0:
        return []
    boundaries: list[int] = []
    if mode == "sae":
        if response_logprobs is None:
            raise ValueError("SAE segmentation requires response_logprobs.")
        log_threshold = math.log(max(sae_threshold, 1e-12))
        for t, lp in enumerate(response_logprobs):
            if lp < log_threshold:
                boundaries.append(t)
    else:  # fixed
        t = fixed_chunk_size
        while t < length:
            boundaries.append(t)
            t += fixed_chunk_size
    if not boundaries or boundaries[-1] != length - 1:
        boundaries.append(length - 1)
    return boundaries


def score_partial_rollout_batch(
    vllm_engines,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    score_min: float,
    score_max: float,
    allow_cot: bool,
) -> tuple[list[float | None], list[str]]:
    """Send a batch of partial-rollout scoring prompts to the gen-value vLLM pool.

    Returns (parsed_scores_in_0_1, raw_generations). ``None`` scores indicate parse failures; the
    caller is free to substitute a default (e.g. 0.5) and track ``parse_rate`` as a metric.
    """
    sampling_params = SamplingParams(n=1, temperature=temperature, max_tokens=max_new_tokens, top_p=1.0, logprobs=1)

    async def _score_one(engine_idx: int, prompt: str) -> tuple[str, float | None]:
        engine = vllm_engines[engine_idx % len(vllm_engines)]
        # The LLMRayActor's add_request interface varies; here we go through its OpenAI server when
        # available. Callers should wire this to match their vllm_utils API version.
        ref = engine.add_request.remote(prompt, sampling_params)
        output = await asyncio.to_thread(__import__("ray").get, ref)
        text = output.outputs[0].text if hasattr(output, "outputs") else str(output)
        parsed = value_model_utils.parse_generative_value_score(
            text, score_min=score_min, score_max=score_max, allow_cot=allow_cot
        )
        return text, parsed

    async def _runner():
        tasks = [_score_one(i, p) for i, p in enumerate(prompts)]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_runner())
    raw = [r[0] for r in results]
    scores = []
    for r in results:
        parsed = r[1]
        if parsed is None:
            scores.append(None)
        else:
            scores.append(value_model_utils.rescale_gen_value_score(parsed, score_min, score_max))
    return scores, raw


def _build_sample_scoring_prompts(
    args: GenValueExperimentConfig, tokenizer: Any, train_dataset: Any, n: int, ground_truths_key: str = "ground_truth"
) -> list[str]:
    """Sample n prompts from the dataset and build gen-value scoring prompts from them."""
    indices = random.sample(range(len(train_dataset)), min(n, len(train_dataset)))
    prompts = []
    for idx in indices:
        prompt_ids = train_dataset[idx][INPUT_IDS_PROMPT_KEY]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        ground_truth = train_dataset[idx].get(ground_truths_key, "")
        scoring_prompt = value_model_utils.build_generative_value_prompt(
            partial_response=prompt_text,
            conditioning=args.gen_value_conditioning,
            ground_truth=ground_truth,
            score_min=args.gen_value_score_min,
            score_max=args.gen_value_score_max,
            allow_cot=args.gen_value_allow_cot,
        )
        prompts.append(scoring_prompt)
    return prompts


def _gen_value_scoring_loop(
    args: GenValueExperimentConfig,
    tokenizer: Any,
    train_dataset: Any,
    gen_value_vllm_engines: list,
    step_trigger: threading.Event,
    stop_event: threading.Event,
    with_tracking: bool,
    ground_truths_key: str = "ground_truth",
) -> None:
    """Background thread: after each policy training step, score sample prompts with the gen-value
    pool and log ``gen_value/score_mean`` and ``gen_value/parse_rate`` to W&B."""
    logger.info("[GenValue] Scoring thread started.")
    while not stop_event.is_set():
        triggered = step_trigger.wait(timeout=1.0)
        if not triggered:
            continue
        step_trigger.clear()
        if stop_event.is_set():
            break
        try:
            prompts = _build_sample_scoring_prompts(
                args, tokenizer, train_dataset, _GEN_VALUE_SAMPLE_SIZE, ground_truths_key
            )
            scores, _ = score_partial_rollout_batch(
                gen_value_vllm_engines,
                tokenizer,
                prompts,
                max_new_tokens=args.gen_value_max_new_tokens,
                temperature=args.gen_value_temperature,
                score_min=args.gen_value_score_min,
                score_max=args.gen_value_score_max,
                allow_cot=args.gen_value_allow_cot,
            )
            valid = [s for s in scores if s is not None]
            if with_tracking:
                wandb.log(
                    {
                        "gen_value/score_mean": sum(valid) / len(valid) if valid else float("nan"),
                        "gen_value/parse_rate": len(valid) / len(scores) if scores else 0.0,
                    }
                )
            logger.debug(
                "[GenValue] scored %d prompts: mean=%.3f parse_rate=%.2f",
                len(scores),
                sum(valid) / len(valid) if valid else float("nan"),
                len(valid) / len(scores) if scores else 0.0,
            )
        except Exception:
            logger.exception("[GenValue] scoring failed for this step, continuing")
    logger.info("[GenValue] Scoring thread stopped.")


def _gen_value_reinforce_loop(
    trainer_actor: Any,
    training_queue: Any,
    stop_event: threading.Event,
    with_tracking: bool,
) -> None:
    """Background thread: drain the training-pairs queue and call ``GenValueTrainerActor.reinforce_step``.

    Training pairs are produced by ``PolicyTrainerRayProcess.step()`` during injection; each pair
    contains a gen-value scoring prompt, the text generated by the gen-value vLLM, and the actual
    rollout outcome.  The REINFORCE gradient update runs on a dedicated GPU inside
    ``GenValueTrainerActor``.
    """
    logger.info("[GenValue] REINFORCE thread started.")
    while not stop_event.is_set():
        try:
            pairs = ray.get(training_queue.get.remote(timeout=1.0))
        except Exception:
            continue
        if pairs is None or len(pairs) == 0:
            continue
        try:
            metrics = ray.get(trainer_actor.reinforce_step.remote(pairs))
            if with_tracking and metrics:
                wandb.log(metrics)
            logger.debug("[GenValue] REINFORCE step: %s", metrics)
        except Exception:
            logger.exception("[GenValue] REINFORCE step failed, continuing")
    logger.info("[GenValue] REINFORCE thread stopped.")


def _sync_gen_value_weights(
    gen_value_trainer: Any,
    gen_value_vllm_engines: list,
    policy_group: Any,
    gen_value_model_path: str,
    args: GenValueExperimentConfig,
    vllm_config: Any,
    streaming_config: Any,
    tc: Any,
    reward_config: Any,
) -> None:
    """Save gen-value trainer weights, kill old vLLM engines, recreate from checkpoint, re-wire policy actors."""
    logger.info("[GenValue] Syncing gen-value weights to vLLM engines.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_path = ray.get(gen_value_trainer.save_weights.remote(tmp_dir))
        logger.info("[GenValue] Saved weights to %s", saved_path)

        for engine in gen_value_vllm_engines:
            with contextlib.suppress(Exception):
                ray.kill(engine)

        gen_value_max_model_len = streaming_config.max_prompt_token_length + args.gen_value_max_new_tokens
        new_engines = vllm_utils.create_vllm_engines(
            len(gen_value_vllm_engines),
            args.gen_value_vllm_tensor_parallel_size,
            vllm_config.vllm_enforce_eager,
            tc.tokenizer_name_or_path or gen_value_model_path,
            saved_path,
            None,
            args.seed,
            False,
            gen_value_max_model_len,
            vllm_config.vllm_gpu_memory_utilization,
            False,
            pg=None,
            tool_parser_type="legacy",
            tool_definitions=None,
            tool_stop_sequences=[],
            max_steps=1,
            per_turn_max_tokens=None,
            mask_tool_use=False,
            pools={},
            prompt_queue=ray_queue.Queue(),
            results_queue=ray_queue.Queue(),
            eval_results_queue=ray_queue.Queue(),
            actor_manager=None,
            inflight_updates=False,
            reward_config=reward_config,
            train_dataset=None,
            eval_dataset=None,
            vllm_attention_backend=vllm_config.vllm_attention_backend,
        )

        gen_value_vllm_engines.clear()
        gen_value_vllm_engines.extend(new_engines)

    ray.get([a.set_gen_value_engines.remote(gen_value_vllm_engines) for a in policy_group.models])
    logger.info("[GenValue] Weight sync complete: %d engine(s) re-wired.", len(gen_value_vllm_engines))


def main():
    """Entry point: parse GenValueExperimentConfig, bring up the second vLLM pool, then train.

    Mirrors grpo_fast.main() setup step-by-step.  After policy model init, creates the gen-value
    vLLM pool and starts a background scoring thread that fires after each policy training step.
    """
    import grpo_fast as _grpo_fast  # noqa: PLC0415

    utils.check_oe_eval_internal()

    parser = ArgumentParserPlus(
        (
            GenValueExperimentConfig,
            TokenizerConfig,
            ModelConfig,
            data_loader_lib.StreamingDataLoaderConfig,
            data_loader_lib.VLLMConfig,
            EnvsConfig,
        )
    )
    parser.set_defaults(exp_name="grpo_genvalue", warmup_ratio=0.0, max_grad_norm=1.0, per_device_train_batch_size=1)
    args, tc, model_config, streaming_config, vllm_config, tools_config = parser.parse_args_into_dataclasses()
    assert isinstance(args, GenValueExperimentConfig)

    # Log combined resource requirements (policy pool + gen-value pool).
    base_reqs = grpo_fast_resource_plan.build_grpo_fast_startup_requirements(
        num_learners_per_node=args.num_learners_per_node,
        single_gpu_mode=args.single_gpu_mode,
        vllm_num_engines=vllm_config.vllm_num_engines,
        vllm_tensor_parallel_size=vllm_config.vllm_tensor_parallel_size,
    )
    gen_value_extra_gpus = args.gen_value_vllm_num_engines * args.gen_value_vllm_tensor_parallel_size
    logger.info(
        "Gen-value adds %d GPU(s) for second vLLM pool (%d engine(s) × TP%d). "
        "Policy needs ≥%d GPU(s); combined total ≥%d GPU(s).",
        gen_value_extra_gpus,
        args.gen_value_vllm_num_engines,
        args.gen_value_vllm_tensor_parallel_size,
        base_reqs["min_total_cluster_gpus"],
        base_reqs["min_total_cluster_gpus"] + gen_value_extra_gpus,
    )

    # ── Step 1: mirror grpo_fast.main() pre-ray setup ─────────────────────────
    tokenizer = _grpo_fast.make_tokenizer(tc, model_config)
    args = _grpo_fast.setup_runtime_variables(args, streaming_config, tools_config)
    _grpo_fast.validate_configs(
        streaming_config, vllm_config, tuple(args.num_learners_per_node), args.sequence_parallel_size
    )

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)

    beaker_config, wandb_url = _grpo_fast.setup_experiment_tracking(
        args, tc, model_config, streaming_config, vllm_config
    )

    # ── Step 2: ray.init ──────────────────────────────────────────────────────
    ray.init(
        runtime_env={
            "excludes": [".git/"],
            "env_vars": {k: v for k, v in os.environ.items() if k not in _grpo_fast.EXCLUDED_ENV_VARS},
        }
    )

    pool_size = tools_config.pool_size
    if pool_size is None:
        pool_size = streaming_config.num_unique_prompts_rollout * streaming_config.num_samples_per_prompt_rollout

    pools, tool_definitions, tool_stop_sequences = _grpo_fast.initialize_tools_and_envs(
        tools_config,
        tokenizer,
        pool_size=pool_size,
        dataset_mixer_list=streaming_config.dataset_mixer_list,
        dataset_mixer_list_splits=streaming_config.dataset_mixer_list_splits,
    )
    if tool_stop_sequences:
        streaming_config.stop_strings.extend(tool_stop_sequences)

    train_dataset, eval_dataset = _grpo_fast.setup_datasets(
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
            f"Train dataset is too small ({len(train_dataset)} prompts); need {needed}. "
            "Reduce async_steps / num_unique_prompts_rollout or increase the dataset."
        )

    if args.cache_dataset_only:
        return

    utils.ensure_hf_repo_cached(model_config.model_name_or_path, revision=model_config.model_revision)
    if tc.tokenizer_name_or_path and tc.tokenizer_name_or_path != model_config.model_name_or_path:
        utils.ensure_hf_repo_cached(tc.tokenizer_name_or_path, revision=tc.tokenizer_revision)
    if args.gen_value_model_name_or_path:
        utils.ensure_hf_repo_cached(args.gen_value_model_name_or_path)

    # ── Step 3: create policy model, optimizer, and policy vLLM pool ──────────
    num_eval_prompts = len(eval_dataset) if eval_dataset is not None else 0
    queue_size = (streaming_config.async_steps + 1) * streaming_config.num_unique_prompts_rollout + num_eval_prompts
    inference_results_Q = ray_queue.Queue(maxsize=queue_size)
    prompt_Q = ray_queue.Queue(maxsize=queue_size)
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
        reward_aggregator=streaming_config.reward_aggregator,
    )

    generation_configs = _grpo_fast.create_generation_configs(args, streaming_config, vllm_config)
    base_env_config = _grpo_fast.build_base_env_config(tools_config, pools)

    (
        policy_group,
        vllm_engines,
        resume_training_step,
        episode,
        actor_manager,
        model_dims,
        _data_prep_actor,
        checkpoint_state,
    ) = _grpo_fast.create_model_and_optimizer(
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
        base_env_config,
        tool_definitions,
        tools_config,
        pools,
        tool_stop_sequences,
    )

    if checkpoint_state:
        episode = checkpoint_state["episode"]
        logger.info("Restored episode count: %d", episode)

    # Several functions in grpo_fast.py reference module-level globals that are set by its
    # __main__ block in normal execution. Since we import grpo_fast as a module rather than
    # running it via __main__, we inject all required globals into its namespace here.
    _grpo_fast.vllm_config = vllm_config
    _grpo_fast.streaming_config = streaming_config
    _grpo_fast.args = args

    # ── Step 4: create gen-value vLLM pool ────────────────────────────────────
    gen_value_vllm_engines: list = []
    gen_value_model_path = args.gen_value_model_name_or_path or model_config.model_name_or_path

    if args.gen_value_vllm_num_engines > 0:
        # The gen-value engines are queried directly via score_partial_rollout_batch(); they do
        # not participate in the queue-driven rollout loop.  We pass sentinel Ray queues so the
        # LLMRayActor's internal prefetch thread doesn't crash (it blocks on an empty queue).
        gen_value_max_model_len = streaming_config.max_prompt_token_length + args.gen_value_max_new_tokens
        gen_value_prompt_Q: ray_queue.Queue = ray_queue.Queue()
        gen_value_results_Q: ray_queue.Queue = ray_queue.Queue()
        gen_value_eval_Q: ray_queue.Queue = ray_queue.Queue()

        gen_value_vllm_engines = vllm_utils.create_vllm_engines(
            args.gen_value_vllm_num_engines,
            args.gen_value_vllm_tensor_parallel_size,
            vllm_config.vllm_enforce_eager,
            tc.tokenizer_name_or_path or gen_value_model_path,
            gen_value_model_path,
            # Reuse policy model revision only when gen_value_model_name_or_path is unset.
            model_config.model_revision if not args.gen_value_model_name_or_path else None,
            args.seed,
            False,  # no prefix caching for value scoring
            gen_value_max_model_len,
            vllm_config.vllm_gpu_memory_utilization,
            False,  # gen-value pool never shares GPU with learners
            pg=None,
            tool_parser_type="legacy",
            tool_definitions=None,
            tool_stop_sequences=[],
            max_steps=1,
            per_turn_max_tokens=None,
            mask_tool_use=False,
            pools={},
            prompt_queue=gen_value_prompt_Q,
            results_queue=gen_value_results_Q,
            eval_results_queue=gen_value_eval_Q,
            actor_manager=None,
            inflight_updates=False,
            reward_config=reward_config,
            train_dataset=None,
            eval_dataset=None,
            vllm_attention_backend=vllm_config.vllm_attention_backend,
        )
        logger.info(
            "======== ✅ Gen-value vLLM pool ready (%d engine(s), model=%s) =========",
            len(gen_value_vllm_engines),
            gen_value_model_path,
        )
    else:
        logger.warning(
            "gen_value_vllm_num_engines=0: gen-value pool skipped. "
            "Set --gen_value_vllm_num_engines 1 (requires one additional GPU) to enable scoring."
        )

    # ── Step 4a: gen-value trainer actor + injection wiring ───────────────────
    # When gen-value vLLM engines are available we also spin up a GenValueTrainerActor that
    # holds a PyTorch copy of the gen-value model for REINFORCE gradient updates.  The training
    # pairs are produced by PolicyTrainerRayProcess.step() (injection path) and pushed into
    # gen_value_training_queue; a background REINFORCE thread drains that queue and calls
    # GenValueTrainerActor.reinforce_step().
    gen_value_trainer: Any = None
    gen_value_training_queue: Any = None
    gen_value_reinforce_thread: threading.Thread | None = None

    if gen_value_vllm_engines:
        gv_lr = args.gen_value_learning_rate or 1e-6
        gen_value_trainer = GenValueTrainerActor.remote(
            gen_value_model_path,
            gv_lr,
            args.gen_value_score_min,
            args.gen_value_score_max,
            args.gen_value_allow_cot,
        )
        ray.get(gen_value_trainer.ready.remote())
        logger.info("======== ✅ Gen-value trainer actor ready (lr=%.2e) =========", gv_lr)

        gen_value_training_queue = ray_queue.Queue(maxsize=200)

        # Wire gen-value engines and the training queue into each policy actor.
        ray.get([a.set_gen_value_engines.remote(gen_value_vllm_engines) for a in policy_group.models])
        ray.get([a.set_gen_value_training_queue.remote(gen_value_training_queue) for a in policy_group.models])
        logger.info(
            "Gen-value injection wired: %d engine(s) → %d policy actor(s).",
            len(gen_value_vllm_engines),
            len(policy_group.models),
        )

    # ── Step 5: background threads (scoring + REINFORCE) ──────────────────────
    gen_value_step_trigger = threading.Event()
    gen_value_stop_event = threading.Event()
    gen_value_thread: threading.Thread | None = None

    if gen_value_vllm_engines:
        gen_value_thread = threading.Thread(
            target=_gen_value_scoring_loop,
            args=(
                args,
                tokenizer,
                train_dataset,
                gen_value_vllm_engines,
                gen_value_step_trigger,
                gen_value_stop_event,
                args.with_tracking,
                tc.ground_truths_key,
            ),
            daemon=True,
            name="genvalue-scoring",
        )
        gen_value_thread.start()

        if gen_value_trainer is not None:
            gen_value_reinforce_thread = threading.Thread(
                target=_gen_value_reinforce_loop,
                args=(gen_value_trainer, gen_value_training_queue, gen_value_stop_event, args.with_tracking),
                daemon=True,
                name="genvalue-reinforce",
            )
            gen_value_reinforce_thread.start()

    # Wrap one_training_step to fire the scoring trigger and (periodically) sync gen-value weights.
    _original_one_training_step = _grpo_fast.one_training_step
    _gv_policy_step_count = [0]  # mutable counter shared with closure

    def _one_training_step_with_genvalue(*step_args, **step_kwargs):
        result = _original_one_training_step(*step_args, **step_kwargs)
        if gen_value_vllm_engines:
            gen_value_step_trigger.set()
        _gv_policy_step_count[0] += 1
        # Periodic weight sync: save PyTorch gen-value model → recreate vLLM engines.
        sync_freq = args.gen_value_sync_freq
        if (
            sync_freq > 0
            and gen_value_trainer is not None
            and _gv_policy_step_count[0] % sync_freq == 0
        ):
            _sync_gen_value_weights(
                gen_value_trainer,
                gen_value_vllm_engines,
                policy_group,
                gen_value_model_path,
                args,
                vllm_config,
                streaming_config,
                tc,
                reward_config,
            )
        return result

    _grpo_fast.one_training_step = _one_training_step_with_genvalue

    # ── Step 6: run policy training loop ─────────────────────────────────────
    weight_sync_metrics_Q: Queue = Queue(maxsize=streaming_config.async_steps)
    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="grpo_genvalue")

    try:
        _grpo_fast.run_training(
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
            base_env_config,
        )

        if args.push_to_hub and (not dist.is_initialized() or dist.get_rank() == 0):
            _grpo_fast.push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)
    except Exception as e:
        if args.send_slack_alerts:
            utils.send_slack_message(f"<!here> A gen-value RL job has died. Error message: {e}.")
        raise
    finally:
        _grpo_fast.one_training_step = _original_one_training_step
        gen_value_stop_event.set()
        if gen_value_thread is not None:
            gen_value_step_trigger.set()
            gen_value_thread.join(timeout=30)
        if gen_value_reinforce_thread is not None:
            gen_value_reinforce_thread.join(timeout=30)
        _grpo_fast.cleanup_training_resources(
            stop_event, executor, [inference_results_Q, prompt_Q, evaluation_inference_results_Q], actor_manager
        )

    logger.info("finished gen-value training")
    utils.check_runtime_leaks()


if __name__ == "__main__":
    main()
