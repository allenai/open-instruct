"""Example environments for testing and demonstration."""

import random
import re
from dataclasses import dataclass
from typing import Any, ClassVar

import nltk
from nltk.corpus import words as nltk_words
from openenv.core.env_server.types import State

from .base import BaseEnvConfig, EnvCall, RLEnvironment, StepResult, TextRLEnvironment


class CounterEnv(RLEnvironment):
    """Simple counter environment. Increment to reach target, then submit."""

    config_name = "counter"
    max_steps = 20

    _tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "increment",
                "description": "Increment the counter by 1",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "decrement",
                "description": "Decrement the counter by 1",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "Submit current value as answer",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    def __init__(self, target: int = 5, **kwargs: Any):
        self._target = target
        self._current = 0
        self._step_count = 0
        self._done = False

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return cls._tool_definitions

    async def reset(self, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        self._current = 0
        self._step_count = 0
        self._done = False

        if "target" in kwargs:
            self._target = int(kwargs["target"])

        return (
            StepResult(result=f"Counter is at {self._current}. Reach {self._target} to win."),
            self._tool_definitions,
        )

    async def step(self, call: EnvCall) -> StepResult:
        self._step_count += 1

        if call.name == "increment":
            self._current += 1
            return StepResult(result=f"Counter is now {self._current}.", reward=-0.1)
        elif call.name == "decrement":
            self._current -= 1
            return StepResult(result=f"Counter is now {self._current}.", reward=-0.1)
        elif call.name == "submit":
            self._done = True
            if self._current == self._target:
                return StepResult(
                    result=f"Success! Counter is {self._current}, target was {self._target}.", reward=1.0, done=True
                )
            return StepResult(
                result=f"Wrong! Counter is {self._current}, target was {self._target}.", reward=-0.5, done=True
            )
        else:
            return StepResult(result=f"Unknown action: {call.name}. Available: increment, decrement, submit.")

    def get_metrics(self) -> dict[str, float]:
        return {
            "step_count": float(self._step_count),
            "final_value": float(self._current),
            "reached_target": 1.0 if self._current == self._target else 0.0,
        }

    def state(self) -> State:
        return State(step_count=self._step_count)


@dataclass
class CounterEnvConfig(BaseEnvConfig):
    """Configuration for CounterEnv."""

    tool_class: ClassVar[type[RLEnvironment]] = CounterEnv
    target: int = 5


class GuessNumberEnv(RLEnvironment):
    """Number guessing game. Guess a secret number between min and max."""

    config_name = "guess_number"
    max_steps = 10

    _tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "guess",
                "description": "Make a guess",
                "parameters": {
                    "type": "object",
                    "properties": {"number": {"type": "integer", "description": "Your guess"}},
                    "required": ["number"],
                },
            },
        }
    ]

    def __init__(self, min_val: int = 1, max_val: int = 100, **kwargs: Any):
        self._min_val = min_val
        self._max_val = max_val
        self._secret = 0
        self._guesses = 0
        self._done = False

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return cls._tool_definitions

    async def reset(self, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        if "number" in kwargs:
            self._secret = int(kwargs["number"])
        else:
            self._secret = random.randint(self._min_val, self._max_val)
        self._guesses = 0
        self._done = False

        return (
            StepResult(result=f"Guess a number between {self._min_val} and {self._max_val}."),
            self._tool_definitions,
        )

    async def step(self, call: EnvCall) -> StepResult:
        self._guesses += 1

        if call.name != "guess":
            return StepResult(result=f"Unknown action: {call.name}. Use 'guess' with a number.")

        guess = call.args.get("number", 0)
        if not isinstance(guess, int):
            try:
                guess = int(guess)
            except (ValueError, TypeError):
                return StepResult(result=f"Invalid guess: {guess}. Please provide an integer.", reward=-0.1)

        distance = abs(guess - self._secret)
        max_distance = self._max_val - self._min_val
        closeness_reward = 1.0 - distance / max_distance if max_distance > 0 else 0.0

        if guess == self._secret:
            self._done = True
            return StepResult(
                result=f"Correct! The number was {self._secret}. You got it in {self._guesses} guesses!",
                reward=1.0,
                done=True,
                metadata={"guesses": self._guesses},
            )
        elif guess < self._secret:
            return StepResult(result=f"{guess} is too low. Try higher.", reward=closeness_reward)
        else:
            return StepResult(result=f"{guess} is too high. Try lower.", reward=closeness_reward)

    def get_metrics(self) -> dict[str, float]:
        return {"guesses": float(self._guesses)}

    def state(self) -> State:
        return State(step_count=self._guesses)


@dataclass
class GuessNumberEnvConfig(BaseEnvConfig):
    """Configuration for GuessNumberEnv."""

    tool_class: ClassVar[type[RLEnvironment]] = GuessNumberEnv
    min_val: int = 1
    max_val: int = 100


_GUESS_PATTERN = re.compile(r"<guess>\s*\[?([^<\]]+)\]?\s*</guess>", re.IGNORECASE)


class WordleTextEnv(TextRLEnvironment):
    """Wordle game as a text environment (TextArena Wordle-v0 compatible).

    The model outputs guesses inside ``<guess>WORD</guess>`` XML tags.
    Feedback uses spaced positional G/Y/X characters::

        Feedback:
        C R A N E
        G X Y X X

    Where G = correct position, Y = wrong position, X = not in word.

    Uses ``response_role="user"`` so feedback appears as a user message
    in the conversation, creating a natural back-and-forth.
    """

    config_name = "wordle"
    response_role = "user"

    _valid_words: ClassVar[set[str] | None] = None

    @classmethod
    def _load_valid_words(cls) -> set[str]:
        if cls._valid_words is None:
            nltk.download("words", quiet=True)
            cls._valid_words = {w.lower() for w in nltk_words.words()}
        return cls._valid_words

    def __init__(self, max_guesses: int = 6, error_allowance: int = 1, **kwargs: Any):
        self._max_guesses = max_guesses
        self._error_allowance = error_allowance
        self._secret_word = ""
        self._guesses: list[str] = []
        self._total_turns = 0
        self._turns_with_tags = 0
        self._consecutive_errors = 0
        self._done = False
        self._words = self._load_valid_words()

    async def _reset(self, **kwargs: Any) -> StepResult:
        self._guesses = []
        self._total_turns = 0
        self._turns_with_tags = 0
        self._consecutive_errors = 0
        self._done = False

        if "word" not in kwargs:
            raise ValueError("WordleTextEnv requires 'word' in env_config (e.g. {'word': 'crane'})")
        self._secret_word = kwargs["word"].upper()

        return StepResult(result="")

    def _format_reward(self) -> float:
        """Average format compliance across all turns, weighted 0.2 (matches prime-rl)."""
        return 0.2 * (self._turns_with_tags / self._total_turns) if self._total_turns > 0 else 0.0

    def _rubric_reward_on_error(self) -> float:
        """Compute reward as prime-rl's rubric would on error termination.

        partial_answer: 0.2*greens + 0.1*yellows from the last valid guess's scoring.
        format_reward: 0.2 * (turns_with_tags / total_turns).
        """
        partial = 0.0
        if self._guesses:
            last_scoring = self._get_scoring(self._guesses[-1], self._secret_word)
            partial = 0.2 * last_scoring.count("G") + 0.1 * last_scoring.count("Y")
        return partial + self._format_reward()

    def _handle_invalid(self, reason: str) -> StepResult:
        """Handle an invalid move: allow one retry, then end the game (matches TextArena error_allowance=1)."""
        self._consecutive_errors += 1
        if self._consecutive_errors <= self._error_allowance:
            return StepResult(result=f"Invalid move: {reason} Please resubmit a valid move.")
        # Exceeded error allowance — game over with rubric-style partial credit
        self._done = True
        return StepResult(result=f"Invalid move: {reason}", reward=self._rubric_reward_on_error(), done=True)

    async def text_step(self, text: str) -> StepResult:
        self._total_turns += 1
        matches = list(_GUESS_PATTERN.finditer(text))
        match = matches[-1] if matches else None
        if not match:
            return self._handle_invalid("Please submit a 5-letter word inside <guess>...</guess> tags.")

        self._turns_with_tags += 1

        raw_guess = match.group(1).strip()
        if not raw_guess.isalpha():
            return self._handle_invalid(
                f"Your guess '{raw_guess}' contains invalid characters. Please submit a single word."
            )
        if len(raw_guess) != 5:
            return self._handle_invalid(
                f"Your guess '{raw_guess}' is {len(raw_guess)} letters. Please submit a 5-letter word."
            )

        guess = raw_guess.upper()

        if guess in self._guesses:
            return self._handle_invalid(f"You have already guessed '{guess}' before. Please try a different word.")

        if guess.lower() not in self._words:
            return self._handle_invalid(f"'{guess}' is not an English word.")

        # Valid move — reset consecutive error count
        self._consecutive_errors = 0

        self._guesses.append(guess)
        scoring = self._get_scoring(guess, self._secret_word)
        n = len(self._guesses)
        remaining = self._max_guesses - n

        feedback = self._format_feedback(guess, scoring)

        format_reward = self._format_reward()

        if guess == self._secret_word:
            self._done = True
            return StepResult(
                result=f"\n{feedback}\nCongratulations! You guessed the word correctly!",
                reward=1.0 + 1.0 / self._total_turns + format_reward,
                done=True,
            )

        num_greens = scoring.count("G")
        num_yellows = scoring.count("Y")
        reward = 0.2 * num_greens + 0.1 * num_yellows + format_reward

        if remaining <= 0:
            self._done = True
            return StepResult(
                result=f"\n{feedback}\nYou used all {self._max_guesses} guesses. The word was {self._secret_word}.",
                reward=reward,
                done=True,
            )

        # Intermediate guess: partial credit + format reward
        return StepResult(result=f"\n{feedback}\nYou have {remaining} guesses left.", reward=reward)

    @staticmethod
    def _format_feedback(guess: str, scoring: list[str]) -> str:
        return f"{' '.join(guess)}\n{' '.join(scoring)}"

    @staticmethod
    def _get_scoring(guess: str, secret: str) -> list[str]:
        """Produce G/Y/X positional scoring list."""
        result = ["X"] * len(guess)
        remaining = list(secret)

        for i, (g, s) in enumerate(zip(guess, secret)):
            if g == s:
                result[i] = "G"
                remaining[i] = ""

        for i, g in enumerate(guess):
            if result[i] == "G":
                continue
            if g in remaining:
                result[i] = "Y"
                remaining[remaining.index(g)] = ""

        return result

    def get_metrics(self) -> dict[str, float]:
        solved = self._guesses and self._guesses[-1] == self._secret_word
        scorings = [self._get_scoring(g, self._secret_word) for g in self._guesses]
        total_greens = sum(s.count("G") for s in scorings)
        total_yellows = sum(s.count("Y") for s in scorings)
        return {
            "guesses": float(len(self._guesses)),
            "invalid_attempts": float(self._consecutive_errors),
            "solved": 1.0 if solved else 0.0,
            "total_greens": float(total_greens),
            "total_yellows": float(total_yellows),
        }

    def state(self) -> State:
        return State(step_count=len(self._guesses))


@dataclass
class WordleTextEnvConfig(BaseEnvConfig):
    """Configuration for WordleTextEnv."""

    tool_class: ClassVar[type[RLEnvironment]] = WordleTextEnv
    max_guesses: int = 6
