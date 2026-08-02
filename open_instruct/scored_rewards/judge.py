"""A generative, multi-dimensional judge.

open-instruct's ``LMJudgeVerifier`` covers the common case: one score, one of a
fixed set of prompts, a remote model through LiteLLM. This is the same idea with
three differences that matter when the judge IS the reward rather than a filter.

RUBRIC AS DATA. Dimensions, their questions and their scale anchors are values
you pass in, from Python or a JSON file, so a rubric can be versioned next to
the run that used it and validated offline against human ratings before it is
ever optimised against.

MULTI-DIMENSIONAL. One scalar "goodness" is what lets style dominate a learned
reward: the policy finds the direction in text space that the judge likes and
goes there. Several separable questions are harder to satisfy with one trick,
and per-dimension normalisation (``guards.MultiDimensional``) then stops any one
of them owning the gradient.

REASON BEFORE SCORE. Each dimension is answered as ``{"why": ..., "score": N}``
with the justification first. A scalar head can satisfy "high quality" with a
style vector; a model that must justify a low score by pointing at what it saw
has a narrower way to be lazily right. The reasons are never part of the reward
- they exist so a suspicious score can be read instead of guessed at, which is
the entire argument for a generative judge over a probe.

A dimension the judge did not produce comes back as ``None``, not as a middling
default. A default is a silent vote for "average" on every parse failure, and
under group normalisation a systematic pull toward the mean is a bias.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence

GenerateFn = Callable[[Sequence[str]], Awaitable[list[str]]]

DEFAULT_SYSTEM = (
    "You are a demanding grader, of the kind who marks competent work 3 out of 5.\n"
    "Most of what you grade deserves a 2 or a 3. A 5 means you could not improve "
    "it if you tried; a 4 means it is genuinely good. If you find yourself giving "
    "5s to most of them you are grading the effort, not the work.\n"
    "Write the reason BEFORE the score, and make the reason quote or name the "
    "specific thing you are reacting to. A reason that would fit any answer means "
    "you have not read this one.\n"
    "Reply with JSON only - no preamble, no code fence."
)


@dataclasses.dataclass(frozen=True)
class Dimension:
    """One question the judge answers, with anchors for the scale.

    Anchors are what make a 1-5 scale mean the same thing twice. Without them
    the judge invents a private standard per call and the scores are not
    comparable across a group, which is the only comparison GRPO makes.
    """

    name: str
    question: str
    anchors: dict[int, str] = dataclasses.field(default_factory=dict)
    low: int = 1
    high: int = 5

    def render(self) -> str:
        anchors = "  ".join(f"{k}={v}" for k, v in sorted(self.anchors.items()))
        return f"- {self.name}: {self.question}" + (f"\n    {anchors}" if anchors else "")

    def normalize(self, raw: float) -> float | None:
        """Map the judge's integer onto [0, 1], or ``None`` if out of range."""
        if not (self.low <= raw <= self.high):
            return None
        span = self.high - self.low
        return (raw - self.low) / span if span else 1.0


@dataclasses.dataclass
class Rubric:
    """A set of dimensions and the instructions that frame them."""

    dimensions: tuple[Dimension, ...]
    system: str = DEFAULT_SYSTEM
    instructions: str = "Score the work below on each dimension. The high end of each scale is always good."

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.dimensions)

    def by_name(self, name: str) -> Dimension:
        for d in self.dimensions:
            if d.name == name:
                return d
        raise KeyError(name)

    @classmethod
    def from_dict(cls, blob: dict) -> Rubric:
        dims = tuple(
            Dimension(
                name=name,
                question=spec["question"],
                anchors={int(k): v for k, v in (spec.get("anchors") or {}).items()},
                low=int(spec.get("low", 1)),
                high=int(spec.get("high", 5)),
            )
            for name, spec in blob["dimensions"].items()
        )
        return cls(
            dimensions=dims,
            system=blob.get("system", DEFAULT_SYSTEM),
            instructions=blob.get("instructions", cls.instructions),
        )

    @classmethod
    def from_json_file(cls, path: str) -> Rubric:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def prompt(self, body: str, names: Iterable[str] | None = None) -> str:
        """Assemble the judge's user message around a caller-rendered ``body``.

        ``body`` is everything the judge needs to see - the task, the reference,
        the conversation, the thing being graded - and is the caller's job
        because only the caller knows what its task looks like.
        """
        chosen = tuple(names) if names is not None else self.names
        rubric = "\n".join(self.by_name(n).render() for n in chosen)
        schema = ", ".join(f'"{n}": {{"why": "...", "score": N}}' for n in chosen)
        example = self.by_name(chosen[0])
        return (
            f"{body.strip()}\n\n"
            f"{self.instructions}\n{rubric}\n\n"
            f'JSON only, exactly these keys. For each, give "why" FIRST - under 15 words, '
            f'naming something specific - then "score" as an integer '
            f"{example.low}-{example.high}:\n{{{schema}}}"
        )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_scores(text: str, rubric: Rubric, names: Iterable[str] | None = None) -> dict[str, float | None]:
    """Per-dimension scores from a judge reply, normalised to [0, 1]."""
    chosen = tuple(names) if names is not None else rubric.names
    out: dict[str, float | None] = {n: None for n in chosen}
    blob = _load(text)
    if blob is None:
        return out
    for name in chosen:
        value = blob.get(name)
        if isinstance(value, dict):
            value = value.get("score")
        try:
            raw = int(round(float(value)))
        except (TypeError, ValueError):
            continue
        out[name] = rubric.by_name(name).normalize(raw)
    return out


def parse_reasons(text: str, names: Iterable[str]) -> dict[str, str]:
    """The judge's justifications. Diagnostics only - never read by the reward."""
    blob = _load(text)
    if blob is None:
        return {}
    return {n: str(blob[n].get("why", "")) for n in names if isinstance(blob.get(n), dict)}


def _load(text: str) -> dict | None:
    match = _JSON_RE.search(text or "")
    if not match:
        return None
    try:
        blob = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return blob if isinstance(blob, dict) else None


class Judge:
    """Runs a rubric through any batched text generator.

    ``generate`` is ``list[str] -> list[str]`` and async, so the same judge
    object works against a served endpoint in training, a local model in a
    validation script, and a stub in tests, without knowing which. Keep it that
    way: the most expensive class of bug here is a judge validated in one
    harness and applied in another, because the mismatch is invisible until the
    policy has already moved.
    """

    def __init__(self, generate: GenerateFn, rubric: Rubric):
        self.generate = generate
        self.rubric = rubric
        self.parse_failures = 0
        self.calls = 0

    async def score(
        self, bodies: Sequence[str], names: Iterable[str] | None = None
    ) -> tuple[list[dict[str, float | None]], list[dict[str, str]]]:
        chosen = tuple(names) if names is not None else self.rubric.names
        if not bodies:
            return [], []
        prompts = [self.rubric.prompt(b, chosen) for b in bodies]
        self.calls += len(prompts)
        replies = await self.generate(prompts)
        scores, reasons = [], []
        for reply in replies:
            parsed = parse_scores(reply, self.rubric, chosen)
            if all(v is None for v in parsed.values()):
                self.parse_failures += 1
            scores.append(parsed)
            reasons.append(parse_reasons(reply, chosen))
        return scores, reasons


def mean_over_turns(per_turn: list[dict[str, float | None]], names: Iterable[str]) -> dict[str, float | None]:
    """Collapse several judged turns into one score per dimension.

    The reward is terminal and every turn earned it, so an episode is its mean
    turn. A dimension is ``None`` for the episode only if it is ``None`` for
    every turn - one unparsed turn should not discard the others.
    """
    out: dict[str, float | None] = {}
    for name in names:
        values = [t[name] for t in per_turn if t.get(name) is not None]
        out[name] = sum(values) / len(values) if values else None
    return out


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------


def openai_generator(
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    system: str = DEFAULT_SYSTEM,
    max_tokens: int = 320,
    temperature: float = 0.0,
    concurrency: int = 32,
    timeout: float = 120.0,
    retries: int = 2,
) -> GenerateFn:
    """Judge behind any OpenAI-compatible endpoint.

    The recommended shape for a real run: serve the judge once with
    ``vllm serve`` (or point at a hosted API) and let every rollout worker call
    it. It keeps the judge off the trainer's GPU, lets you swap judge models
    without touching the training job, and makes the judge independently
    benchmarkable - which you want, because a judge you cannot evaluate on its
    own is a reward you cannot debug.

    ``temperature`` defaults to 0. A judge is a measurement instrument; sampling
    it adds variance to every advantage in the group for no benefit.
    """
    import openai  # noqa: PLC0415

    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout)
    limiter = asyncio.Semaphore(concurrency)

    async def one(prompt: str) -> str:
        async with limiter:
            for attempt in range(retries + 1):
                try:
                    reply = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return reply.choices[0].message.content or ""
                except Exception:
                    if attempt == retries:
                        return ""
                    await asyncio.sleep(2**attempt)
            return ""

    async def generate(prompts: Sequence[str]) -> list[str]:
        return list(await asyncio.gather(*(one(p) for p in prompts)))

    return generate


def hf_generator(
    model,
    tokenizer,
    system: str = DEFAULT_SYSTEM,
    max_new_tokens: int = 320,
    batch_size: int = 8,
    temperature: float = 0.0,
) -> GenerateFn:
    """Judge as an in-process HuggingFace model.

    For smoke tests and offline validation. In training it competes with the
    policy for GPU memory - prefer ``openai_generator``.
    """
    import torch  # noqa: PLC0415

    @torch.no_grad()
    def run(prompts: Sequence[str]) -> list[str]:
        out: list[str] = []
        for start in range(0, len(prompts), batch_size):
            chats = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts[start : start + batch_size]
            ]
            enc = tokenizer(chats, return_tensors="pt", padding=True, padding_side="left").to(model.device)
            generated = model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=temperature or None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            out += [tokenizer.decode(x, skip_special_tokens=True) for x in generated[:, enc.input_ids.shape[1] :]]
        return out

    async def generate(prompts: Sequence[str]) -> list[str]:
        return await asyncio.get_event_loop().run_in_executor(None, run, list(prompts))

    return generate


def stub_generator(rubric: Rubric, seed: int = 0) -> GenerateFn:
    """Deterministic mid-range scores. Exercises parsing, not quality."""
    import random  # noqa: PLC0415

    async def generate(prompts: Sequence[str]) -> list[str]:
        out = []
        for prompt in prompts:
            rng = random.Random((hash(prompt) ^ seed) & 0xFFFF)
            body = ", ".join(f'"{d.name}": {{"why": "stub", "score": {rng.randint(2, 5)}}}' for d in rubric.dimensions)
            out.append("{" + body + "}")
        return out

    return generate
