"""Example environments for testing and demonstration."""

import random

from .base import RLEnvironment, StepResult, ToolCall, register_env


@register_env("counter")
class CounterEnv(RLEnvironment):
    """Simple counter environment. Increment to reach target, then submit."""

    max_steps = 20

    # Tool definitions shared between get_tool_definitions() and reset()
    _tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "increment",
                "description": "Increment the counter by a given amount",
                "parameters": {
                    "type": "object",
                    "properties": {"amount": {"type": "integer", "description": "Amount to increment by"}},
                    "required": ["amount"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "decrement",
                "description": "Decrement the counter by a given amount",
                "parameters": {
                    "type": "object",
                    "properties": {"amount": {"type": "integer", "description": "Amount to decrement by"}},
                    "required": ["amount"],
                },
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

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        """Return tool definitions for prompt injection."""
        return cls._tool_definitions

    async def reset(self, task_id: str | None = None, **kwargs) -> StepResult:
        self._current = 0
        self._step_count = 0

        if task_id and task_id.isdigit():
            self._target = int(task_id)

        return StepResult(
            observation=f"Counter is at {self._current}. Reach {self._target} to win.",
            tools=self._tool_definitions,
            info={"target": self._target},
        )

    @staticmethod
    def _parse_amount(tool_args: dict) -> int:
        amount = tool_args.get("amount", 1)
        if isinstance(amount, int):
            return amount
        try:
            return int(amount)
        except (ValueError, TypeError):
            return 1

    async def step(self, tool_call: ToolCall) -> StepResult:
        self._step_count += 1

        if tool_call.name == "increment":
            amount = self._parse_amount(tool_call.args)
            self._current += amount
            obs = f"Counter incremented by {amount}, now at {self._current}."
            reward = -0.1
            done = False
        elif tool_call.name == "decrement":
            amount = self._parse_amount(tool_call.args)
            self._current -= amount
            obs = f"Counter decremented by {amount}, now at {self._current}."
            reward = -0.1
            done = False
        elif tool_call.name == "submit":
            if self._current == self._target:
                obs = f"Success! Counter is {self._current}, target was {self._target}."
                reward = 1.0
            else:
                obs = f"Wrong! Counter is {self._current}, target was {self._target}."
                reward = -0.5
            done = True
        else:
            obs = f"Unknown action: {tool_call.name}"
            reward = -0.5
            done = True

        return StepResult(
            observation=obs, reward=reward, done=done, info={"current": self._current, "target": self._target}
        )

    def get_metrics(self) -> dict[str, float]:
        return {
            "step_count": float(self._step_count),
            "final_value": float(self._current),
            "reached_target": 1.0 if self._current == self._target else 0.0,
        }


@register_env("guess_number")
class GuessNumberEnv(RLEnvironment):
    """Number guessing game. Guess a secret number between min and max."""

    max_steps = 10

    # Tool definitions shared between get_tool_definitions() and reset()
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

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        """Return tool definitions for prompt injection."""
        return cls._tool_definitions

    async def reset(self, task_id: str | None = None, **kwargs) -> StepResult:
        if task_id and task_id.isdigit():
            self._secret = int(task_id)
        else:
            self._secret = random.randint(self._min_val, self._max_val)
        self._guesses = 0

        return StepResult(
            observation=f"Guess a number between {self._min_val} and {self._max_val}.",
            tools=self._tool_definitions,
            info={"min": self._min_val, "max": self._max_val},
        )

    async def step(self, tool_call: ToolCall) -> StepResult:
        self._guesses += 1

        if tool_call.name != "guess":
            return StepResult(
                observation=f"Unknown action: {tool_call.name}. Use 'guess' with a number.", reward=-0.1, done=True
            )

        guess = tool_call.args.get("number", 0)

        if not isinstance(guess, int):
            try:
                guess = int(guess)
            except (ValueError, TypeError):
                return StepResult(
                    observation=f"Invalid guess: {guess}. Please provide an integer.", reward=-0.1, done=False
                )

        # Closeness reward: 1.0 for exact match, 0.0 for maximally wrong
        distance = abs(guess - self._secret)
        max_distance = self._max_val - self._min_val
        closeness_reward = 1.0 - distance / max_distance if max_distance > 0 else 0.0

        if guess == self._secret:
            return StepResult(
                observation=f"Correct! The number was {self._secret}. You got it in {self._guesses} guesses!",
                reward=1.0,
                done=True,
                info={"guesses": self._guesses},
            )
        elif guess < self._secret:
            return StepResult(observation=f"{guess} is too low. Try higher.", reward=closeness_reward, done=False)
        else:
            return StepResult(observation=f"{guess} is too high. Try lower.", reward=closeness_reward, done=False)

    def get_metrics(self) -> dict[str, float]:
        return {"guesses": float(self._guesses)}
