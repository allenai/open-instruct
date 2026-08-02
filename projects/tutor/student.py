"""The student: the thing that talks, and the thing that answers.

They are deliberately two different objects, because they are two different
measurements and conflating them cost a run.

``StudentDirector``  drives the partner model's side of the dialogue. Controllable
                     - mastery, a five-trait profile, and an intent picked from a
                     closed set before any text exists.
``ChoiceStudent``    commits to an option by ranking them under length-normalised
                     log-probability. This is the ANCHOR's outcome channel and is
                     never part of the reward.

WHY A CONTROLLABLE SIMULATOR AND NOT A PROMPT. PEARL's ablation is the argument:
cold-start data from a simply-prompted simulator made the tutor WORSE than no
cold start at all (79.4 against 79.6), and the same pipeline with a controllable
simulator improved it (81.5). A student merely told "act like a kid" is not an
environment, it is a style. Concretely, rewriting the student's system prompt
into five explicit rules moved its median reply length by ZERO words (25 before,
25 after), and it complied with an instruction to emit a control token in 1
dialogue out of 1,000.

WHERE THIS DEPARTS FROM PEARL. Their planner and their mastery update are both
LLM calls; ours are formulas. Their simulator is a prompted 30B on 8 H200s and
can afford two extra generations per student turn. The paper does not specify
f_update's form, the intent taxonomy, or the mastery value type anyway.

NOTHING HERE CONSULTS GOLD. A mastery update that peeked at the answer would
make the student correct by construction and quietly convert the reward into a
measure of how many turns had elapsed. The update reads the dialogue only.
"""

from __future__ import annotations

import dataclasses
import math
import random
import re
from collections.abc import Sequence

LEVELS = ("low", "medium", "high")
LEVEL_SCORE = {"low": 1, "medium": 2, "high": 3}
TRAITS = ("activeness", "perseverance", "comprehension", "expressiveness", "attention")

MASTERY = ("none", "partial", "solid")
MASTERY_SCORE = {"none": 0, "partial": 1, "solid": 2}

# A closed set, deliberately missing an "explain the concept" intent. Every
# member is something a confused student does; none of them is teaching. This is
# the part that fixes role reversal - a 0.5B told "never explain" complies most
# of the time, and the failures are invisible in aggregate.
INTENTS = {
    "express_confusion": "say what you do not understand, without guessing",
    "ask_clarify": "ask the tutor one specific question about what they just said",
    "attempt_step": "try the next step out loud, even if you get it wrong",
    "state_misconception": "say what you currently believe and why, plainly",
    "answer": "commit to one option and say briefly why",
}

_CONFUSION = re.compile(
    r"\b(i (?:really )?(?:do ?n'?t|dont|don't) (?:know|get|understand)|no idea|"
    r"i'?m (?:so )?(?:lost|confused|stuck)|idk|what\?|huh)\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(r"\b(maybe|i guess|not sure|kind of|kinda|i think so)\b", re.IGNORECASE)
_QUESTION = re.compile(r"\?")


@dataclasses.dataclass
class StudentState:
    grade: int | None = None
    believes: str | None = None
    mastery: str = "none"
    profile: dict[str, str] = dataclasses.field(default_factory=dict)
    intent: str = "express_confusion"
    turns_seen: int = 0

    @property
    def ability(self) -> int:
        """PEARL's 5-15 ability score: the five traits, summed."""
        return sum(LEVEL_SCORE[self.profile.get(t, "medium")] for t in TRAITS)


def sample_profile(rng: random.Random, level: str | None = None) -> dict[str, str]:
    if level is not None:
        if level not in LEVELS:
            raise ValueError(f"level {level!r} not in {LEVELS}")
        return {t: level for t in TRAITS}
    return {t: rng.choice(LEVELS) for t in TRAITS}


def initial_state(item: dict, rng: random.Random, level: str | None = None, mastery: str = "none") -> StudentState:
    """Build a student for one item.

    ``believes`` is not invented. The screening pass recorded which option the
    student actually picks unaided, and every training item was KEPT because
    that pick is wrong - so it is a real misconception the tutor can work
    against rather than a plausible one we made up. Mastery starts at "none" for
    the same reason.
    """
    believes = item.get("believes")
    if believes is None and item.get("choose_pick") is not None:
        try:
            believes = item["choices"][int(item["choose_pick"])]
        except (IndexError, ValueError, TypeError):
            believes = None
    return StudentState(
        grade=item.get("grade"), believes=believes, mastery=mastery, profile=sample_profile(rng, level)
    )


def demonstrates_understanding(student_turn: str) -> bool:
    """Did the student's own words show they picked something up?

    Crude on purpose, and deliberately blind to gold. A turn counts when it is
    substantive, commits to something, and does not open by disclaiming - which
    is roughly the difference between "idk" and "oh so I add them first?".
    """
    text = (student_turn or "").strip()
    if len(text.split()) < 5:
        return False
    if _CONFUSION.search(text):
        return False
    return not _HEDGE.search(text)


def update_mastery(state: StudentState, tutor_turn: str, student_turn: str) -> StudentState:
    """PEARL's f_update (Eq. 3), as a formula rather than an LLM call.

    Rises one level when the student's own turn demonstrates understanding AND
    the tutor actually asked them to do something - a student that agrees with a
    lecture has not learned, it has nodded. Comprehension gates the rate, which
    is what makes the profile bite: a low-comprehension student needs two good
    exchanges where a high-comprehension one needs one.

    Never falls. Within a three-turn dialogue there is no mechanism for
    forgetting and adding one would be fiction.
    """
    state.turns_seen += 1
    if not (_QUESTION.search(tutor_turn or "") and demonstrates_understanding(student_turn)):
        return state
    if state.profile.get("comprehension", "medium") == "low" and state.turns_seen < 2:
        return state
    i = MASTERY.index(state.mastery)
    state.mastery = MASTERY[min(i + 1, len(MASTERY) - 1)]
    return state


def plan_intent(state: StudentState, rng: random.Random, tutor_turn: str | None = None) -> str:
    """Pick what the student does next, before any text exists.

    Weights, not rules, so a group of eight dialogues on one problem explores
    different student behaviours instead of eight copies of the same one -
    variance inside the group is what GRPO's advantage is computed from.

    The shape: mastery moves the student along confusion -> attempting ->
    answering, activeness moves it from waiting to volunteering, and a tutor
    question pulls hard toward responding to it rather than restating confusion.
    """
    m = MASTERY_SCORE[state.mastery]
    active = LEVEL_SCORE[state.profile.get("activeness", "medium")]
    persist = LEVEL_SCORE[state.profile.get("perseverance", "medium")]

    weights = {
        "express_confusion": max(0.1, 3.0 - 1.2 * m - 0.4 * (active - 2)),
        "ask_clarify": 0.8 + 0.5 * active - 0.3 * m,
        "attempt_step": 0.3 + 1.1 * m + 0.4 * persist,
        "state_misconception": 1.2 if (state.believes and m == 0) else 0.2,
        "answer": max(0.0, 1.4 * m - 0.6) if m >= 1 else 0.0,
    }
    if tutor_turn and _QUESTION.search(tutor_turn):
        weights["express_confusion"] *= 0.5
        weights["attempt_step"] *= 1.6
        weights["ask_clarify"] *= 1.2
    names = list(weights)
    return rng.choices(names, weights=[max(weights[n], 0.0) for n in names])[0]


def describe(state: StudentState) -> str:
    """The state as instructions the student model can act on.

    Traits are rendered as behaviour rather than as a rating - "you give up
    quickly" is actionable where "perseverance: low" is a label a small model
    will happily ignore.
    """
    bits = []
    if state.grade:
        bits.append(f"You are in grade {state.grade}.")
    bits.append(
        {
            "none": "You do not understand this question yet.",
            "partial": "You have started to get part of this, but not all of it.",
            "solid": "You think you understand it now.",
        }[state.mastery]
    )
    if state.believes and state.mastery == "none":
        bits.append(f'You currently think the answer is "{state.believes}", but you could not say why.')

    p = state.profile
    traits = []
    if p.get("activeness") == "low":
        traits.append("you answer only what you are asked")
    elif p.get("activeness") == "high":
        traits.append("you say what you are thinking without being asked")
    if p.get("perseverance") == "low":
        traits.append("you give up quickly")
    if p.get("comprehension") == "low":
        traits.append("explanations take a while to land")
    elif p.get("comprehension") == "high":
        traits.append("you follow explanations fast")
    if p.get("expressiveness") == "low":
        traits.append("you find it hard to put your thinking into words")
    if p.get("attention") == "low":
        traits.append("you drift off the question easily")
    if traits:
        bits.append("As a student, " + ", and ".join(traits) + ".")

    bits.append(f"RIGHT NOW, do exactly this: {INTENTS[state.intent]}.")
    return " ".join(bits)


class StudentDirector:
    """Drives the partner model. Plugs into ``PartnerModelEnv`` as its director."""

    def __init__(self, level: str | None = None, mastery: str = "none"):
        self.level = level
        self.mastery = mastery
        self.state: StudentState | None = None
        self._rng = random.Random(0)

    def _ensure(self, item: dict, rng: random.Random) -> StudentState:
        if self.state is None:
            self.state = initial_state(item, rng, level=self.level, mastery=self.mastery)
            self.state.intent = plan_intent(self.state, rng)
        return self.state

    def system(self, item: dict, turn: int, rng: random.Random) -> str:
        state = self._ensure(item, rng)
        self._rng = rng
        return (
            "You are a middle-school student talking to a tutor. You are the one being "
            "taught: never explain the topic, never tutor back, never state the correct "
            "answer as if you knew it. Reply in one or two short sentences, in your own "
            "words.\n" + describe(state)
        )

    def user(self, item: dict, transcript: list[dict], turn: int, rng: random.Random) -> str:
        """What the student is shown.

        Deliberately WITHOUT the answer options. Showing them here let the
        student read one aloud - it named gold in 34% of dialogues - and since
        the outcome is measured from the transcript, it then read its own words
        back and scored correct, paying the tutor for the student's knowledge.
        The answering channel still gets the options, because picking one
        requires them.
        """
        rendered = "\n".join(f"{'Tutor' if t['who'] == 'policy' else 'You'}: {t['text'].strip()}" for t in transcript)
        return f"Question you're stuck on:\n{item['question']}\n\nConversation so far:\n{rendered}\n\nYour reply:"

    def observe(self, item: dict, policy_turn: str, partner_turn: str) -> None:
        if self.state is None:
            return
        self.state = update_mastery(self.state, policy_turn, partner_turn)
        self.state.intent = plan_intent(self.state, self._rng, tutor_turn=policy_turn)

    def metrics(self) -> dict[str, float]:
        if self.state is None:
            return {}
        return {
            "student_mastery": float(MASTERY_SCORE[self.state.mastery]),
            "student_ability": float(self.state.ability),
        }


def opening_line(item: dict, seed: int = 0) -> str:
    """The student's first message, written offline at dataset-build time.

    Letting the student open changes what the tutor does. Cold-opening with
    nothing to respond to, the tutor structures the problem itself - which in
    real traces is when it starts walking the option list ("what about
    brimstone?"), one of the leak modes. Responding to a stated confusion gives
    it something specific to teach into.

    Templated rather than generated, so it is deterministic, free, and identical
    across the G completions of a group. The misconception in it is the real
    one the screening pass recorded.
    """
    rng = random.Random(seed)
    state = initial_state(item, rng)
    if state.believes:
        return f"Student: I think it's \"{state.believes}\" but I honestly don't know why - I'm stuck on this one."
    return rng.choice(
        [
            "Student: I don't really get what this question is asking.",
            "Student: I'm stuck - I don't know where to even start with this.",
            "Student: I read it a few times and I still don't understand it.",
        ]
    )


# --------------------------------------------------------------------------
# the answering channel
# --------------------------------------------------------------------------


class ChoiceStudent:
    """Commits to an option by length-normalised log-probability.

    This is the outcome the ANCHOR measures, and it is stricter than the
    student's actual ability: on 120 problems where this channel scored 1%, the
    same student answered 26% correctly in free text. That is fine - it is a
    fixed instrument, and every number in the run history was taken with it.
    Changing it changes what the results mean, which is why it is not
    configurable from the training script.

    Runs against an OpenAI-compatible completions endpoint using ``echo`` with
    logprobs, so no model is loaded in the training process.
    """

    PROMPT = "Fact: {hint}\nQuestion: {question}\nAnswer:"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, timeout: float = 120.0):
        import openai  # noqa: PLC0415

        self.model = model
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout)

    async def score_choices(self, question: str, choices: Sequence[str], hint: str = "") -> list[float]:
        prefix = self.PROMPT.format(hint=hint.strip(), question=question.strip())
        scores = []
        for choice in choices:
            response = await self.client.completions.create(
                model=self.model, prompt=f"{prefix} {choice}", max_tokens=0, echo=True, logprobs=0, temperature=0.0
            )
            logprobs = response.choices[0].logprobs
            tokens, values = logprobs.tokens, logprobs.token_logprobs
            # count back over the tokens the option contributed
            n = _suffix_token_count(tokens, f" {choice}")
            tail = [v for v in values[-n:] if v is not None]
            scores.append(sum(tail) / len(tail) if tail else -math.inf)
        return scores

    async def choose(self, question: str, choices: Sequence[str], hint: str = "") -> int:
        scores = await self.score_choices(question, choices, hint)
        return max(range(len(scores)), key=scores.__getitem__)


def _suffix_token_count(tokens: list[str], suffix: str) -> int:
    """How many trailing tokens make up ``suffix``. Tokenizer-agnostic."""
    total, count = "", 0
    for token in reversed(tokens):
        total = token + total
        count += 1
        if total.strip() == suffix.strip():
            return count
        if len(total) > len(suffix) + 8:
            break
    return max(count, 1)
