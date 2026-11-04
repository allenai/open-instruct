"""Exact, local Sudoku grading. No reference-solution matching or code execution."""

import contextlib
import json
import re

from open_instruct.miles.rewards import task_verifiers


def grid(value, *, blanks=False):
    return (
        isinstance(value, list)
        and len(value) == 9
        and all(
            isinstance(row, list)
            and len(row) == 9
            and all(type(cell) is int and (0 if blanks else 1) <= cell <= 9 for cell in row)
            for row in value
        )
    )


def valid_solution(answer, givens):
    if not grid(answer) or not grid(givens, blanks=True):
        return False
    digits = set(range(1, 10))
    units = answer + [list(column) for column in zip(*answer, strict=True)]
    units += [[answer[r + i][c + j] for i in range(3) for j in range(3)] for r in (0, 3, 6) for c in (0, 3, 6)]
    return all(set(unit) == digits for unit in units) and all(
        not givens[r][c] or givens[r][c] == answer[r][c] for r in range(9) for c in range(9)
    )


class SudokuVerifier:
    get_config_class = staticmethod(lambda: task_verifiers.RewardConfig)

    def __init__(self, verifier_config=None):
        pass

    async def async_call(self, tokens, prediction, label, **kwargs):
        # A broken target is a preparation fault, not an incorrect model answer.
        givens = json.loads(label) if isinstance(label, str) else label
        if not grid(givens, blanks=True):
            raise ValueError("Sudoku target must be a 9x9 grid with 0 for blanks")
        answers = re.findall(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
        score = False
        if len(answers) == 1 and len(answers[0]) <= 4096:
            with contextlib.suppress(ValueError, RecursionError):
                score = valid_solution(json.loads(answers[0]), givens)
        return task_verifiers.RewardResult(float(score))
