"""The environment is another model.

open-instruct's text environments are games: Wordle knows the word, the counter
knows the target, and the observation is computed. A large class of tasks is not
like that. Tutoring, negotiation, interviewing, persuasion, elicitation,
customer support - in all of them the environment is a person, and the closest
you can get in a training loop is a frozen model standing in for one.

``PartnerModelEnv`` is that: each policy turn is sent to a frozen model, and its
reply comes back as the next observation. The partner is never trained. Only the
policy is.

FOUR THINGS THAT ARE EASY TO GET WRONG.

*The partner cannot open the conversation.* open-instruct discards the
observation returned by ``reset`` - the policy always generates first. If your
task needs the partner to speak first, bake its opening line into the dataset
prompt (``data.build_rows(opening=...)``). This is better than generating it at
runtime anyway: the opener is then identical across the G completions of a
group, so the group is a comparison between policies rather than between
partners, and it costs nothing at training time.

*The partner's tokens must not be trained on.* They are in the response stream
because the policy generated into it, and they are not the policy's. Leave
``--mask_tool_use true`` (the default) and open-instruct masks them out of the
loss for you.

*The partner's words must not be scored as the policy's.* Any reward rule that
punishes the policy for something it said has to read the policy's turns alone.
This env records those separately, under ``types.POLICY_TEXT_KEY``, and
``Sample.policy_text`` reads them back. Measured on one corpus of 864 dialogues,
17% of a leak rule's flags were the partner blurting the thing the policy was
being punished for naming.

*Control beats prompting.* A partner told "act confused" is a style, not an
environment. A ``Director`` picks what the partner will DO next from a closed
set before any text exists, which removes the failure mode instead of asking
politely for it not to happen. See ``Director`` below.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import random
from typing import Any, ClassVar, Protocol

from openenv.core.env_server.types import State

from open_instruct.environments.base import BaseEnvConfig, RLEnvironment, StepResult, TextRLEnvironment
from open_instruct.scored_rewards.types import POLICY_TEXT_KEY, TRANSCRIPT_KEY, parse_transcript  # noqa: F401


class Director(Protocol):
    """Decides what the partner does next, before the partner writes anything.

    The interesting part of a model environment. A frozen small model asked to
    "be a confused student" will drift into being a helpful assistant, because
    that is what it was trained to be, and an instruction not to is complied
    with some of the time - which is the worst case, since the failures are
    invisible in aggregate. Sampling the next behaviour from a closed set that
    contains no way to break character removes the option instead.

    It is also where within-group variance comes from. GRPO's advantage is
    computed from the spread inside a group; if every one of the G partners
    behaves identically, the only variation left is the policy's sampling
    temperature.
    """

    def system(self, item: dict, turn: int, rng: random.Random) -> str:
        """The partner's system prompt for this turn."""

    def user(self, item: dict, transcript: list[dict], turn: int, rng: random.Random) -> str:
        """What the partner is shown before it replies."""

    def observe(self, item: dict, policy_turn: str, partner_turn: str) -> None:
        """Update whatever state the director keeps. Called after each exchange."""

    def metrics(self) -> dict[str, float]:
        """Diagnostics for this episode."""


class StaticDirector:
    """A fixed system prompt. The baseline, and usually not enough - see ``Director``."""

    def __init__(self, system: str = "", user_template: str = "{transcript}\n\nYour reply:"):
        self._system = system
        self.user_template = user_template

    def system(self, item: dict, turn: int, rng: random.Random) -> str:
        return self._system.format(**item) if self._system else ""

    def user(self, item: dict, transcript: list[dict], turn: int, rng: random.Random) -> str:
        rendered = "\n".join(f"{t['who'].capitalize()}: {t['text']}" for t in transcript)
        return self.user_template.format(transcript=rendered, **item)

    def observe(self, item: dict, policy_turn: str, partner_turn: str) -> None:
        return

    def metrics(self) -> dict[str, float]:
        return {}


class PartnerModelEnv(TextRLEnvironment):
    """A frozen model as the environment, reached over an OpenAI-compatible API.

    Serve the partner separately (``vllm serve``) rather than loading it in
    process. The env runs in a Ray actor pool with one actor per concurrent
    rollout; loading a model in each would be absurd, and an endpoint batches
    across all of them for free.
    """

    config_name = "partner_model"
    response_role = "user"

    def __init__(
        self,
        model: str = "",
        base_url: str | None = None,
        api_key: str | None = None,
        max_turns: int = 3,
        temperature: float = 0.8,
        max_tokens: int = 96,
        director: str | None = None,
        director_kwargs: dict | None = None,
        system: str = "",
        timeout: float = 120.0,
        **_: Any,
    ):
        self.model = model
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._director_spec = director
        self._director_kwargs = director_kwargs or {}
        self._static_system = system

        self._client = None
        self._base_url, self._api_key = base_url, api_key

        self.item: dict = {}
        self.transcript: list[dict] = []
        self.policy_turns: list[str] = []
        self.turn = 0
        self.done = False
        self.director: Director = StaticDirector(system)
        self.rng = random.Random(0)

    # --- lifecycle --------------------------------------------------------

    def _ensure_client(self):
        if self._client is None:
            import openai  # noqa: PLC0415

            self._client = openai.AsyncOpenAI(
                base_url=self._base_url, api_key=self._api_key or "EMPTY", timeout=self.timeout
            )
        return self._client

    def _build_director(self) -> Director:
        if not self._director_spec:
            return StaticDirector(self._static_system)
        from open_instruct.scored_rewards import registry  # noqa: PLC0415

        module, _, attr = self._director_spec.rpartition(".")
        registry.load_plugins(module)
        import importlib  # noqa: PLC0415

        return getattr(importlib.import_module(module), attr)(**self._director_kwargs)

    async def _reset(self, **kwargs: Any) -> StepResult:
        """Per-row setup. ``kwargs`` is the row's ``env_config`` entry.

        The item travels as a JSON string because the dataset column is typed as
        strings; ``seed`` is derived from the item so every member of a group
        gets the SAME partner, which is what makes the group a comparison
        between policy samples.
        """
        raw = kwargs.get("item", "{}")
        self.item = json.loads(raw) if isinstance(raw, str) else dict(raw)
        self.max_turns = int(kwargs.get("max_turns", self.max_turns))
        self.transcript = []
        self.policy_turns = []
        self.turn = 0
        self.done = False
        self.director = self._build_director()

        opening = self.item.get("opening")
        if opening:
            self.transcript.append({"who": "partner", "text": str(opening)})

        seed = kwargs.get("seed")
        if seed is None:
            key = json.dumps(self.item, sort_keys=True).encode()
            seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        self.rng = random.Random(int(seed))
        return StepResult(result="")

    # --- the loop ---------------------------------------------------------

    async def text_step(self, text: str) -> StepResult:
        policy_turn = (text or "").strip()
        self.turn += 1
        self.transcript.append({"who": "policy", "text": policy_turn})
        self.policy_turns.append(policy_turn)

        if self.turn >= self.max_turns:
            self.done = True
            return StepResult(result="", reward=0.0, done=True)

        reply = await self._partner_reply()
        self.transcript.append({"who": "partner", "text": reply})
        self.director.observe(self.item, policy_turn, reply)
        return StepResult(result=reply, reward=0.0)

    async def _partner_reply(self) -> str:
        client = self._ensure_client()
        system = self.director.system(self.item, self.turn, self.rng)
        user = self.director.user(self.item, self.transcript, self.turn, self.rng)
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        try:
            response = await client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=self.max_tokens, temperature=self.temperature
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # a dead partner should end the episode, not the run
            self.done = True
            return f"(the other participant did not respond: {type(exc).__name__})"

    # --- what the scorer reads -------------------------------------------

    def get_metrics(self) -> dict[str, float]:
        """Episode diagnostics, plus the transcript the scorer needs.

        ``rollout.info`` is flattened from this dict, and the two string values
        below are how a scorer gets structured access to the episode. Everything
        else here is numeric and safe to average.
        """
        metrics: dict[str, Any] = {
            POLICY_TEXT_KEY: "\n".join(self.policy_turns),
            TRANSCRIPT_KEY: json.dumps(self.transcript),
            "partner_turns": float(self.turn),
        }
        metrics.update(self.director.metrics())
        return metrics

    def state(self) -> State:
        return State(step_count=self.turn)


@dataclasses.dataclass
class PartnerModelEnvConfig(BaseEnvConfig):
    """Construction-time config: one partner endpoint for the whole run."""

    tool_class: ClassVar[type[RLEnvironment]] = PartnerModelEnv
    model: str = ""
    base_url: str | None = None
    api_key: str | None = None
    max_turns: int = 3
    temperature: float = 0.8
    max_tokens: int = 96
    director: str | None = None
    director_kwargs: dict = dataclasses.field(default_factory=dict)
    system: str = ""
    timeout: float = 120.0


def register_env(name: str = PartnerModelEnv.config_name, config_class=PartnerModelEnvConfig) -> None:
    """Add an env to open-instruct's ``TOOL_REGISTRY``.

    Mutating the registry at plugin-import time is the whole registration
    mechanism - no upstream edit needed, because ``--reward_plugins`` is
    imported before ``initialize_tools_and_envs`` runs.
    """
    from open_instruct.environments.tools.tools import TOOL_REGISTRY  # noqa: PLC0415

    TOOL_REGISTRY[name] = config_class


async def gather_limited(coros, limit: int = 32):
    """Run coroutines with a concurrency cap. Handy inside a group scorer."""
    semaphore = asyncio.Semaphore(limit)

    async def run(c):
        async with semaphore:
            return await c

    return await asyncio.gather(*(run(c) for c in coros))
