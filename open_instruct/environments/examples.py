"""Example environments for testing and demonstration."""

import random

from openenv.core.env_server.types import State

from .base import EnvCall, RLEnvironment, StepResult


class CounterEnv(RLEnvironment):
    """Simple counter environment. Increment to reach target, then submit."""

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

    def __init__(self, target: int = 5, **kwargs):
        self._target = target
        self._current = 0
        self._step_count = 0
        self._done = False
        self._task_id: str | None = None

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return cls._tool_definitions

    async def reset(self, task_id: str | None = None, **kwargs) -> tuple[StepResult, list[dict]]:
        self._current = 0
        self._step_count = 0
        self._done = False
        self._task_id = task_id

        if task_id and task_id.isdigit():
            self._target = int(task_id)

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
        return State(episode_id=self._task_id, step_count=self._step_count)


class GuessNumberEnv(RLEnvironment):
    """Number guessing game. Guess a secret number between min and max."""

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

    def __init__(self, min_val: int = 1, max_val: int = 100, **kwargs):
        self._min_val = min_val
        self._max_val = max_val
        self._secret = 0
        self._guesses = 0
        self._done = False
        self._task_id: str | None = None

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return cls._tool_definitions

    async def reset(self, task_id: str | None = None, **kwargs) -> tuple[StepResult, list[dict]]:
        self._task_id = task_id
        if task_id and task_id.isdigit():
            self._secret = int(task_id)
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
        return State(episode_id=self._task_id, step_count=self._guesses)
