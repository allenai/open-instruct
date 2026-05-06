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
PyTorch copy of the gen-value model and computes REINFORCE gradients using the paper's
MSE-shaped critic reward ``R_v = 1 - (outcome - v_hat)**2``, following §5.2 of "Bringing Value
Models Back" (arXiv:2604.10701).  On parse failure we treat the critic as having predicted
``v_hat = 0`` (instead of the paper's reward-zero convention) so the gradient signal stays
consistent with the piecewise-constant values we feed into GAE.

Weight sync from the trainer actor back to the gen-value vLLM pool happens in-place over NCCL,
mirroring how the policy syncs to its own vLLM pool in
``grpo_fast.PolicyTrainerRayProcess.broadcast_to_vllm``.  A NCCL group is established once at
startup via ``GenValueTrainerActor.setup_model_update_group``; every ``gen_value_sync_freq``
policy steps (default: 1 = every step) we then broadcast the latest PyTorch weights into the
running vLLM engines.  Set ``gen_value_sync_freq=0`` to skip sync entirely and keep the vLLM
critic frozen while REINFORCE gradients are still computed (useful for debugging).

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

import logging
import os
import queue as queue_lib
import random
import threading
from concurrent import futures
from dataclasses import dataclass
from queue import Full, Queue
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray.util import queue as ray_queue
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.distributed.weight_transfer.base import WeightTransferInitRequest
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

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

    The actor receives training pairs (prompt, generated, outcome) via ``reinforce_step``
    and optimises the paper's accuracy-shaped REINFORCE reward::

        R_v = 1 - (outcome - v_hat)**2    if v_hat was parsed from the generation
            = 0                           otherwise

    where ``outcome`` is clipped to [0, 1] and ``v_hat`` is the parsed/rescaled score
    from the critic's own generation. This matches §5.2 of
    "Bringing Value Models Back" (GenAC, arXiv:2604.10701).

    Weight sync to the gen-value vLLM pool is done in-place over NCCL via
    ``setup_model_update_group`` + ``broadcast_to_vllm``, mirroring how the policy
    pushes weights to its vLLM pool in ``grpo_fast.PolicyTrainerRayProcess``.
    """

    def __init__(
        self,
        model_path: str,
        learning_rate: float,
        score_min: float,
        score_max: float,
        tensor_parallel_size: int = 1,
        max_prompt_tokens: int = 8192,
    ) -> None:
        self._lr = learning_rate
        self._score_min = score_min
        self._score_max = score_max
        self._tp_size = tensor_parallel_size
        self._max_prompt_tokens = max_prompt_tokens
        self._step_count = 0

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, use_cache=False
        ).cuda()
        if hasattr(self._model, "gradient_checkpointing_enable"):
            self._model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self._model.train()
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)

        self._vllm_engines: list | None = None
        self._model_update_group: Any | None = None

    def _score_from_text(self, text: str) -> float | None:
        raw = value_model_utils.parse_generative_value_score(text, self._score_min, self._score_max)
        if raw is None:
            return None
        return value_model_utils.rescale_gen_value_score(raw, self._score_min, self._score_max)

    def reinforce_step(self, training_pairs: list[dict]) -> dict:
        """Apply one REINFORCE gradient step with the MSE-shaped critic reward.

        For each pair we compute ``R_v = 1 - (r - v_hat)^2``, with both ``r`` and
        ``v_hat`` clipped to [0, 1]. Parse failures are treated as the critic
        predicting ``v_hat = 0`` (rather than assigning a reward of 0), so the
        critic still receives a consistent gradient signal on malformed outputs.
        """
        if not training_pairs:
            return {}

        self._optimizer.zero_grad()
        total_loss = 0.0
        rewards: list[float] = []
        outcomes: list[float] = []
        mses: list[float] = []  # (r - v_hat)^2 with parse-failure → v_hat=0.
        parsed_v_hats: list[float] = []  # only for pairs where parsing succeeded.
        parsed_count = 0
        skipped_empty_generation = 0

        for pair in training_pairs:
            if pair["outcome"] is None:
                continue
            outcome = max(0.0, min(1.0, float(pair["outcome"])))
            prompt = pair["prompt"]
            generated = pair["generated"]

            v_hat = self._score_from_text(generated)
            if v_hat is None:
                # Parse failure: treat the critic's prediction as 0.
                effective_v_hat = 0.0
            else:
                effective_v_hat = v_hat
                parsed_count += 1
                parsed_v_hats.append(v_hat)
            squared_error = (outcome - effective_v_hat) ** 2
            reward = 1.0 - squared_error

            prompt_ids = self._tokenizer(prompt, add_special_tokens=False).input_ids
            generated_ids = self._tokenizer(generated, add_special_tokens=False).input_ids
            if not generated_ids:
                skipped_empty_generation += 1
                continue
            if len(prompt_ids) > self._max_prompt_tokens:
                prompt_ids = prompt_ids[-self._max_prompt_tokens :]

            input_ids = torch.tensor([prompt_ids + generated_ids], dtype=torch.long, device="cuda")
            attention_mask = torch.ones_like(input_ids)
            target_ids = input_ids[:, -len(generated_ids) :]

            # Only materialize logits needed to score the generated answer tokens.
            # Full-sequence logits for 8k-token prompts can allocate several extra GiB.
            outputs = self._model(
                input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=len(generated_ids) + 1
            )
            logits = outputs.logits[:, :-1, :]
            lm_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), target_ids.reshape(-1), reduction="mean"
            )
            log_prob = -lm_loss  # mean log-prob of generated tokens

            loss = -log_prob * reward
            loss.backward()
            total_loss += float(loss.detach())
            outcomes.append(outcome)
            rewards.append(reward)
            mses.append(squared_error)

        if not rewards:
            return {}

        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()
        self._step_count += 1

        metrics = {
            "gen_value/reinforce_loss": total_loss / len(rewards),
            "gen_value/reward_mean": sum(rewards) / len(rewards),
            "gen_value/outcome_mean": sum(outcomes) / len(outcomes),
            "gen_value/mse": sum(mses) / len(mses),
            "gen_value/parse_rate": parsed_count / len(rewards),
            "gen_value/reinforce_steps": self._step_count,
        }
        if skipped_empty_generation:
            metrics["gen_value/skipped_empty_generation"] = skipped_empty_generation
        if parsed_v_hats:
            # Mean of parsed predictions -- tells us whether the critic is biased high/low
            # vs. ``outcome_mean`` and whether it's moving over training. Undefined when
            # no pair parsed this step, so we only emit the key when we have signal.
            metrics["gen_value/v_hat_mean"] = sum(parsed_v_hats) / len(parsed_v_hats)
        return metrics

    def setup_model_update_group(self, vllm_engines: list) -> None:
        """One-time NCCL handshake between this trainer and the gen-value vLLM engines.

        World layout matches the policy pool: trainer is rank 0, then each vLLM
        engine owns ``tensor_parallel_size`` consecutive ranks starting from 1.
        """
        self._vllm_engines = vllm_engines
        if not vllm_engines:
            self._model_update_group = None
            return

        master_address = ray._private.services.get_node_ip_address().strip("[]")
        master_port = utils.find_free_port()
        world_size = len(vllm_engines) * self._tp_size + 1
        master_info = {"master_address": master_address, "master_port": master_port, "world_size": world_size}
        init_infos = [master_info | {"rank_offset": i * self._tp_size + 1} for i, _ in enumerate(vllm_engines)]

        # Submit the vLLM-side init RPCs first (async) so the NCCL handshake can
        # proceed on both sides in parallel, then wait.
        refs = [
            engine.init_weight_transfer_engine.remote(WeightTransferInitRequest(init_info=info))
            for engine, info in zip(vllm_engines, init_infos)
        ]
        torch.cuda.set_device(0)
        self._model_update_group = NCCLWeightTransferEngine.trainer_init(master_info)
        utils.ray_get_with_progress(refs, desc="Initializing gen-value vLLM weight transfer engines", timeout=600)

    def broadcast_to_vllm(self, model_step: int) -> list:
        """Push current PyTorch weights to the gen-value vLLM pool over NCCL.

        Returns the list of engine-side ``update_weights`` ObjectRefs so the
        caller can wait on them (mirrors ``PolicyTrainerRayProcess.broadcast_to_vllm``).
        """
        if not self._vllm_engines or self._model_update_group is None:
            return []
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)
        return vllm_utils.broadcast_weights_to_vllm(
            model=self._model,
            vllm_engines=self._vllm_engines,
            model_update_group=self._model_update_group,
            model_step=model_step,
            gather_whole_model=True,
        )

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
    # Cap on the number of gen-value queries per rollout (evenly downsampled when exceeded).
    gen_value_max_segments: int = 16
    # Generation params for the gen value model's vLLM engine.
    # Default matches GenAC's "Maximum Critic Response Length" (Table 5): the critic
    # needs enough budget to actually do CoT reasoning before emitting the score.
    gen_value_max_new_tokens: int = 1024
    gen_value_temperature: float = 1.0
    # Score schema.
    gen_value_score_min: float = 0.0
    gen_value_score_max: float = 10.0
    # Training coefficients.
    gen_value_learning_rate: float | None = None
    gen_value_reinforce_coef: float = 0.1
    gen_value_reinforce_max_prompt_tokens: int = 8192
    """Maximum prompt tokens kept for gen-value REINFORCE; generated score tokens are always kept."""
    # Conditioning for the gen-value prompt: one of none, gt, correct_demo, rollout_context.
    gen_value_conditioning: str = "none"
    # How often (in policy steps) to sync REINFORCE-trained gen-value weights back to vLLM.
    # Default=1 keeps the critic tracking the evolving actor every step (paper's joint-
    # training regime). Set to 0 to disable sync entirely -- weights stay frozen in vLLM
    # while REINFORCE gradients are still computed (useful for debugging).
    gen_value_sync_freq: int = 1

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
        if self.gen_value_reinforce_max_prompt_tokens <= 0:
            raise ValueError(
                "--gen_value_reinforce_max_prompt_tokens must be > 0, "
                f"got {self.gen_value_reinforce_max_prompt_tokens}."
            )
        if self.gen_value_conditioning not in value_model_utils.GEN_VALUE_CONDITIONING_TYPES:
            raise ValueError(
                f"--gen_value_conditioning must be one of "
                f"{sorted(value_model_utils.GEN_VALUE_CONDITIONING_TYPES)}, "
                f"got {self.gen_value_conditioning!r}."
            )


def score_partial_rollout_batch(
    vllm_engines,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    score_min: float,
    score_max: float,
) -> tuple[list[float | None], list[str]]:
    """Send a batch of partial-rollout scoring prompts to the gen-value vLLM pool.

    Returns (parsed_scores_in_0_1, raw_generations). Parse failures are reported as ``None`` in
    the returned list so callers can track ``parse_rate`` as a metric; downstream consumers that
    need a numeric value should substitute ``0.0`` to match the in-graph gen-value scorer and the
    REINFORCE trainer's handling of parse failures.
    """
    n_eng = len(vllm_engines)
    buckets: list[list[tuple[int, str]]] = [[] for _ in range(n_eng)]
    for k, prompt in enumerate(prompts):
        buckets[k % n_eng].append((k, prompt))
    non_empty = [(e, b) for e, b in enumerate(buckets) if b]
    refs = [
        vllm_engines[e].generate_completions.remote(
            [p for _, p in bucket],
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=1.0,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        for e, bucket in non_empty
    ]
    engine_results = ray.get(refs)
    raw: list[str] = [""] * len(prompts)
    for (_, bucket), bucket_texts in zip(non_empty, engine_results):
        for (k, _), text in zip(bucket, bucket_texts):
            raw[k] = text

    scores: list[float | None] = []
    for text in raw:
        parsed = value_model_utils.parse_generative_value_score(text, score_min=score_min, score_max=score_max)
        if parsed is None:
            scores.append(None)
        else:
            scores.append(value_model_utils.rescale_gen_value_score(parsed, score_min, score_max))
    return scores, raw


def _get_gen_value_max_model_len(streaming_config: Any, args: Any) -> int:
    """Reserve context for gen-value scoring prompts plus the critic's answer."""
    return streaming_config.pack_length * 2 + args.gen_value_max_new_tokens


def _build_sample_scoring_prompts(
    args: GenValueExperimentConfig, tokenizer: Any, train_dataset: Any, n: int, ground_truths_key: str = "ground_truth"
) -> list[str]:
    """Sample n prompts from the dataset and build gen-value scoring prompts from them.

    Used only by the diagnostic scoring thread; we probe the critic at the
    start-of-rollout state (i.e. ``partial_response=""``) so the logged score
    represents the prior value estimate for each sampled problem.
    """
    indices = random.sample(range(len(train_dataset)), min(n, len(train_dataset)))
    prompts = []
    for idx in indices:
        prompt_ids = train_dataset[idx][INPUT_IDS_PROMPT_KEY]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        ground_truth = train_dataset[idx].get(ground_truths_key, "")
        scoring_prompt = value_model_utils.build_generative_value_prompt(
            partial_response="",
            conditioning=args.gen_value_conditioning,
            ground_truth=ground_truth,
            score_min=args.gen_value_score_min,
            score_max=args.gen_value_score_max,
            problem=prompt_text,
        )
        prompts.append(scoring_prompt)
    return prompts


def _put_gen_value_metrics(metrics_Q: Queue, metrics: dict[str, Any], source: str) -> None:
    """Send background-thread metrics through the main training step for aligned W&B logging."""
    try:
        metrics_Q.put_nowait(metrics)
    except Full:
        logger.warning("[GenValue] metrics queue full, dropping %s metrics", source)


def _gen_value_scoring_loop(
    args: GenValueExperimentConfig,
    tokenizer: Any,
    train_dataset: Any,
    gen_value_vllm_engines: list,
    step_trigger: threading.Event,
    stop_event: threading.Event,
    engines_lock: threading.Lock,
    metrics_Q: Queue,
    ground_truths_key: str = "ground_truth",
) -> None:
    """Background thread: after each policy training step, score sample prompts with the gen-value
    pool and log ``gen_value/score_mean`` and ``gen_value/parse_rate`` to W&B.

    ``engines_lock`` serialises engine use with ``_sync_gen_value_weights`` so a weight sync
    can't put an engine to sleep while this thread has a ``generate_completions`` in flight
    (the resulting deadlock is visible as a hang in ``engine.wake_up``).
    """
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
            with engines_lock:
                scores, _ = score_partial_rollout_batch(
                    gen_value_vllm_engines,
                    tokenizer,
                    prompts,
                    max_new_tokens=args.gen_value_max_new_tokens,
                    temperature=args.gen_value_temperature,
                    score_min=args.gen_value_score_min,
                    score_max=args.gen_value_score_max,
                )
            valid = [s for s in scores if s is not None]
            score_metrics = {
                "gen_value/score_mean": sum(valid) / len(valid) if valid else float("nan"),
                "gen_value/score_parse_rate": len(valid) / len(scores) if scores else 0.0,
            }
            _put_gen_value_metrics(metrics_Q, score_metrics, "scoring")
            logger.debug(
                "[GenValue] scored %d prompts: mean=%.3f parse_rate=%.2f",
                len(scores),
                score_metrics["gen_value/score_mean"],
                score_metrics["gen_value/score_parse_rate"],
            )
        except Exception:
            logger.exception("[GenValue] scoring failed for this step, continuing")
    logger.info("[GenValue] Scoring thread stopped.")


def _gen_value_reinforce_loop(
    trainer_actor: Any, training_queue: Any, stop_event: threading.Event, metrics_Q: Queue
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
            # ray.util.queue.Queue.get is a regular sync method that blocks up
            # to `timeout` seconds and raises queue.Empty on timeout. The
            # previous `training_queue.get.remote(timeout=1.0)` raised
            # AttributeError (no .remote on a bound method) and was swallowed
            # by the bare `except Exception`, so this thread was spinning
            # without ever consuming a pair.
            pairs = training_queue.get(timeout=1.0)
        except queue_lib.Empty:
            continue
        if pairs is None or len(pairs) == 0:
            continue
        metrics = ray.get(trainer_actor.reinforce_step.remote(pairs))
        if metrics:
            _put_gen_value_metrics(metrics_Q, metrics, "REINFORCE")
        logger.debug("[GenValue] REINFORCE step: %s", metrics)
    logger.info("[GenValue] REINFORCE thread stopped.")


def _sync_gen_value_weights(
    gen_value_trainer: Any, gen_value_vllm_engines: list, model_step: int, engines_lock: threading.Lock
) -> None:
    """Push updated gen-value weights to the gen-value vLLM pool over NCCL.

    Mirrors the policy-side weight sync in ``grpo_fast.weight_sync_thread``:
    the engines are put to sleep inside ``broadcast_weights_to_vllm``, the
    trainer streams parameters over the NCCL group established at startup,
    and the engines are woken back up here.

    ``engines_lock`` is held for the duration of the sync so the diagnostic
    scoring thread cannot have a ``generate_completions`` in flight against
    a sleeping engine -- otherwise ``engine.wake_up()`` deadlocks behind the
    in-flight request on vLLM's single asyncio loop.
    """
    if not gen_value_vllm_engines:
        return
    logger.debug("[GenValue] Syncing weights at model_step=%d.", model_step)

    with engines_lock:
        engine_refs = ray.get(gen_value_trainer.broadcast_to_vllm.remote(model_step))
        if engine_refs:
            ray.get(engine_refs)
        ray.get([engine.wake_up.remote() for engine in gen_value_vllm_engines])

    logger.debug("[GenValue] Weight sync complete (%d engine(s)).", len(gen_value_vllm_engines))


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
        gen_value_max_model_len = _get_gen_value_max_model_len(streaming_config, args)
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

    if gen_value_vllm_engines:
        gv_lr = args.gen_value_learning_rate or 1e-6
        gen_value_trainer = GenValueTrainerActor.remote(
            gen_value_model_path,
            gv_lr,
            args.gen_value_score_min,
            args.gen_value_score_max,
            tensor_parallel_size=args.gen_value_vllm_tensor_parallel_size,
            max_prompt_tokens=args.gen_value_reinforce_max_prompt_tokens,
        )
        ray.get(gen_value_trainer.ready.remote())
        logger.info("======== ✅ Gen-value trainer actor ready (lr=%.2e) =========", gv_lr)

        # Establish the NCCL weight-transfer group between the trainer actor and the
        # gen-value vLLM engines. Mirrors `setup_model_update_group` on the policy
        # side (see grpo_fast.PolicyTrainerRayProcess) so we can push weights
        # in-place instead of killing and recreating engines.
        if args.gen_value_sync_freq > 0:
            ray.get(gen_value_trainer.setup_model_update_group.remote(gen_value_vllm_engines))
            logger.info("======== ✅ Gen-value NCCL weight-transfer group initialised =========")

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
    # Serialises gen-value vLLM engine use between the diagnostic scoring thread
    # and weight sync (see _sync_gen_value_weights docstring).
    gen_value_engines_lock = threading.Lock()
    # Cross-thread metrics shuttle: background threads put() per-step metric
    # dicts here; _one_training_step_with_genvalue drains them and merges into
    # data_thread_metrics so they land in the main pretty-print + wandb log.
    gen_value_metrics_Q: Queue = Queue(maxsize=64)
    gen_value_thread: threading.Thread | None = None
    gen_value_reinforce_future: futures.Future | None = None

    # Shared executor for training support threads; the REINFORCE future must be
    # observable from the main loop so trainer failures abort the whole run.
    weight_sync_metrics_Q: Queue = Queue(maxsize=streaming_config.async_steps)
    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="grpo_genvalue")

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
                gen_value_engines_lock,
                gen_value_metrics_Q,
                tc.ground_truths_key,
            ),
            daemon=True,
            name="genvalue-scoring",
        )
        gen_value_thread.start()

        assert gen_value_trainer is not None
        gen_value_reinforce_future = executor.submit(
            _gen_value_reinforce_loop,
            gen_value_trainer,
            gen_value_training_queue,
            gen_value_stop_event,
            gen_value_metrics_Q,
        )

    # Wrap one_training_step to fire the scoring trigger and (periodically) sync gen-value weights.
    _original_one_training_step = _grpo_fast.one_training_step
    _gv_policy_step_count = [0]  # mutable counter shared with closure

    def _raise_if_gen_value_reinforce_failed() -> None:
        if gen_value_reinforce_future is not None and gen_value_reinforce_future.done():
            gen_value_reinforce_future.result()

    def _one_training_step_with_genvalue(*step_args, **step_kwargs):
        _raise_if_gen_value_reinforce_failed()

        # Drain any gen-value metrics emitted by the background threads since
        # the previous step and merge them into data_thread_metrics (positional
        # arg 4 of grpo_fast.one_training_step, a mutable dict). Main thread
        # then includes them in the pretty-print and wandb.log(step=training_step).
        data_thread_metrics = step_args[4] if len(step_args) > 4 else step_kwargs.get("data_thread_metrics")
        if isinstance(data_thread_metrics, dict):
            while True:
                try:
                    data_thread_metrics.update(gen_value_metrics_Q.get_nowait())
                except queue_lib.Empty:
                    break

        result = _original_one_training_step(*step_args, **step_kwargs)
        _gv_policy_step_count[0] += 1
        # Sync FIRST, then fire the scoring trigger. Scoring needs to run on the
        # latest weights, and firing before sync risks a generate_completions
        # in flight when broadcast_weights_to_vllm calls engine.sleep() -- that
        # deadlocks engine.wake_up() on vLLM's asyncio loop.
        sync_freq = args.gen_value_sync_freq
        if sync_freq > 0 and gen_value_trainer is not None and _gv_policy_step_count[0] % sync_freq == 0:
            _sync_gen_value_weights(
                gen_value_trainer, gen_value_vllm_engines, _gv_policy_step_count[0], gen_value_engines_lock
            )
        if gen_value_vllm_engines:
            gen_value_step_trigger.set()
        _raise_if_gen_value_reinforce_failed()
        return result

    _grpo_fast.one_training_step = _one_training_step_with_genvalue

    # ── Step 6: run policy training loop ─────────────────────────────────────
    primary_exception: BaseException | None = None
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
        primary_exception = e
        if args.send_slack_alerts:
            utils.send_slack_message(f"<!here> A gen-value RL job has died. Error message: {e}.")
        raise
    finally:
        _grpo_fast.one_training_step = _original_one_training_step
        gen_value_stop_event.set()
        if gen_value_thread is not None:
            gen_value_step_trigger.set()
            gen_value_thread.join(timeout=30)
        _grpo_fast.cleanup_training_resources(
            stop_event, executor, [inference_results_Q, prompt_Q, evaluation_inference_results_Q], actor_manager
        )
        if primary_exception is None:
            _raise_if_gen_value_reinforce_failed()

    logger.info("finished gen-value training")
    utils.check_runtime_leaks()


if __name__ == "__main__":
    main()
