"""Pinned ReasoningGym rewards for prepared, explicitly allowlisted task families."""

import copy
import importlib
import math
import re
from importlib import metadata

from open_instruct.miles.rewards import candidate_verifiers, task_verifiers

TASKS = frozenset(
    {
        "base_conversion",
        "family_relationships",
        "game_of_life_halting",
        "arc_agi",
        "rearc",
        "chain_sum",
        "leg_counting",
        "needle_haystack",
        "simple_equations",
        "word_sorting",
    }
)


class ReasoningGymVerifier:
    get_config_class = staticmethod(lambda: task_verifiers.RewardConfig)

    def __init__(self, verifier_config=None):
        if metadata.version("reasoning-gym") != "0.1.25":
            raise ValueError("This contract requires reasoning-gym==0.1.25")
        self.library = importlib.import_module("reasoning_gym")
        self.scorers = {}

    async def async_call(self, tokens, prediction, label, **kwargs):
        entry = copy.deepcopy(label)
        task = entry["metadata"]["source_dataset"]
        if task not in TASKS or not isinstance(entry.get("answer"), str) or not entry["answer"].strip():
            raise ValueError("Unqualified ReasoningGym task or missing reference")
        final = candidate_verifiers.answer_text(prediction)
        tagged = re.findall(r"<answer>(.*?)</answer>", final, re.S)
        boxes = re.findall(r"\\boxed\{([^{}]+)\}", final)
        answer = tagged[-1].strip() if tagged else boxes[-1].strip() if boxes else final
        if task in {"arc_agi", "rearc"}:
            grid = entry["metadata"]["output"]
            if (
                not isinstance(grid, list)
                or not grid
                or not all(isinstance(row, list) for row in grid)
                or not grid[0]
                or any(len(row) != len(grid[0]) for row in grid)
                or any(type(x) is not int or not 0 <= x <= 9 for row in grid for x in row)
            ):
                raise ValueError("Invalid ReasoningGym grid target")
            entry["metadata"]["output"] = tuple(tuple(row) for row in grid)
        if task == "game_of_life_halting":
            gold = entry["answer"].strip().lower()
            if gold not in {"true", "false"}:
                raise ValueError("Invalid boolean reference")
            result = float(answer.lower() in {"true", "false"} and answer.lower() == gold)
        else:
            if task not in self.scorers:
                self.scorers[task] = self.library.get_score_answer_fn(task)
            result = float(self.scorers[task](answer=answer, entry=entry))
        if not math.isfinite(result) or not 0 <= result <= 1:
            raise ValueError("Invalid ReasoningGym reward")
        return candidate_verifiers.Result(result, diagnostics={"kind": "reasoning_gym_0.1.25", "status": "ok"})
