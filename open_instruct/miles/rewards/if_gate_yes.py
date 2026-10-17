"""H087's binary prompt-level loose constraint pass times the H080 yes verdict."""

import asyncio
import dataclasses
import hashlib
import importlib
import json
import time

from open_instruct import logger_utils
from open_instruct.miles.rewards import general_judge, h080, judge_metrics, judge_registry, task_verifiers

logger = logger_utils.setup_logger(__name__)
_ADMISSION = None


def _admission(limit):
    """Share admission across clients on the worker's event loop, never in Ray."""
    global _ADMISSION
    loop = asyncio.get_running_loop()
    if _ADMISSION is None or _ADMISSION[0] is not loop:
        _ADMISSION = (loop, {})
    return _ADMISSION[1].setdefault(limit, asyncio.Semaphore(limit))


def client_config(service, endpoint):
    return general_judge.GeneralJudgeConfig(
        api_url=endpoint.rstrip("/") + "/chat/completions",
        api_key="EMPTY",
        model=service["model"],
        max_tokens=16,
        max_context_length=service["max_context_length"],
        temperature=0.0,
        timeout=600.0,
        seed=1,
        max_concurrent_calls=32,
        check_context=True,
        scale_timeout=False,
    )


class VerdictClient:
    """Shared reward/qualification client: one logical request, at most three attempts."""

    def __init__(self, config, metrics=None):
        self.config = config
        self.metrics = metrics

    async def grade(self, conversation, response, *, instruction_ids=()):
        prompt = h080.render_prompt(conversation, response)
        metrics = self.metrics if self.metrics is not None else judge_metrics.tracker()
        started = time.monotonic()
        semaphore = _admission(self.config.max_concurrent_calls)
        metrics.arrive()
        admitted = None
        attempts, label, terminal_error = [], None, None
        terminal_failure = None
        try:
            await semaphore.acquire()
            admitted = time.monotonic()
            metrics.admit()
            for attempt in range(3):
                evidence = {"attempt": attempt + 1}
                try:
                    reply = await general_judge.async_request(self.config, prompt)
                    evidence["reply"] = reply
                    label = h080.parse_verdict(reply)
                    if label is None:
                        evidence["error"] = "parse"
                except RuntimeError as error:
                    evidence.update(error=str(error), diagnostics=getattr(error, "diagnostics", {}))
                    # The training caller selects the exhausted-request policy;
                    # overflow and configuration/programming failures are always fatal.
                    terminal_error = error
                attempts.append(evidence)
                if "error" in evidence:
                    logger.warning(
                        "H087 judge failed attempt: %s",
                        json.dumps(
                            {"rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), **evidence}
                        ),
                    )
                if label is not None:
                    break
            record = {
                "kind": "general_judge",
                "status": "ok" if label is not None else "judge_error",
                "label": label,
                "attempts": attempts,
                "elapsed_seconds": time.monotonic() - started,
                "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_sha256": h080.PROMPT_SHA256,
                "model": self.config.model,
                "seed": self.config.seed,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "context_checked": self.config.check_context,
                "instruction_ids": sorted(set(instruction_ids)),
            }
            if (
                label is None
                and terminal_error is not None
                and not isinstance(terminal_error, general_judge.JudgeResponseError)
            ):
                raise terminal_error
            terminal_failure = label is None
            return record
        finally:
            if admitted is not None:
                semaphore.release()
            finished = time.monotonic()
            metrics.finish(
                label is not None,
                len(attempts),
                finished - started,
                finished - admitted if admitted is not None else None,
                terminal_failure,
                instruction_ids,
            )


async def score_prompt(config, conversation, response, *, policy, metadata, name, instruction_ids=()):
    if policy not in ("zero", "zero_guarded", "raise"):
        raise ValueError("OI_MILES_JUDGE_FAILURE_POLICY must be 'zero', 'zero_guarded' or 'raise'")
    record = await VerdictClient(config).grade(conversation, response, instruction_ids=instruction_ids)
    record["failure_policy"] = policy
    metadata.setdefault("verifier_diagnostics", {})[name] = record
    if record["label"] is None:
        if policy == "raise":
            raise general_judge.JudgeResponseError("H080 judge exhausted three attempts", record)
        record["fallback_reward"] = 0.0
    return float(record["label"] == "yes")


@dataclasses.dataclass
class GateConfig:
    name: str = "ifeval"


def prompt_level_loose_pass(prediction, target):
    """H080/H086's binary all-constraints loose pass, using H070's IF checkers.

    H080 build_requests reads sample_metrics.ifeval.ifeval, not the fraction of
    constraints passed. Each constraint may pass a different loose variant.
    """
    bundle = task_verifiers.if_constraint_bundle(target)
    # As in ManifestIFVerifier, keep grading dependencies out of CPU planning.
    checker = importlib.import_module("open_instruct.ground_truth_utils")
    answer = checker.remove_thinking_section(prediction)
    lines = answer.split("\n")
    trimmed = ["\n".join(part).strip() for part in (lines[1:], lines[:-1], lines[1:-1])]
    variants = [answer, answer.replace("*", ""), *trimmed, *(part.replace("*", "") for part in trimmed)]
    followed = []
    for key, kwargs in zip(bundle["instruction_id"], bundle["kwargs"], strict=True):
        instruction = checker.instructions_registry.INSTRUCTION_DICT[key](key)
        instruction.build_description(**{k: v for k, v in (kwargs or {}).items() if v is not None})
        followed.append(any(variant.strip() and instruction.check_following(variant) for variant in variants))
    return float(bool(followed) and all(followed))


class IfGateYesVerifier(task_verifiers.ManifestIFVerifier):
    """Judge all eligible IF rows; retain H070's fractional C as a diagnostic."""

    get_config_class = staticmethod(lambda: GateConfig)

    def __init__(self, verifier_config=None):
        self.name = (verifier_config or GateConfig()).name

    async def score_sample(self, args, sample, tokens, response, target):
        result = await super().async_call(
            tokens,
            response,
            target,
            query=sample.metadata.get("query", sample.prompt),
            rollout_state=sample.metadata.get("rollout_state"),
        )
        constraint = result.score
        constraint_pass = prompt_level_loose_pass(response, target)
        conversation = h080.sample_conversation(sample.metadata)
        resolved = judge_registry.resolve_request(args, self.name, conversation, response, target)
        if resolved is None or resolved[3] != "h080-verdict":
            raise ValueError("IfGateYesVerifier requires an open-instruct/if-gate-yes binding")
        try:
            quality = await score_prompt(
                resolved[0],
                conversation,
                response,
                policy=general_judge.failure_policy(args, default="raise"),
                metadata=sample.metadata,
                name=self.name,
                instruction_ids=task_verifiers.if_constraint_bundle(target)["instruction_id"],
            )
        finally:
            record = sample.metadata.get("verifier_diagnostics", {}).get(self.name)
            if record is not None:
                record.update(constraint_score=constraint, constraint_pass=constraint_pass)
        return constraint_pass * quality
