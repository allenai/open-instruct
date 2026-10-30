"""Pinned MIT SynLogic Sudoku puzzles, rendered in English with unchanged givens."""

import json

from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import sudoku

DATASET = "MiniMaxAI/SynLogic"
REVISION = "bb4297b82b9c28ef39249f48386cefd5e856618a"


def sudoku_rows(rows):
    result = []
    for index, row in enumerate(rows):
        if row["data_source"] != "sudoku":
            continue
        game = json.loads(row["extra_info"]["game_data_str"])
        metadata = game["metadata"]
        givens = [[0 if cell == "X" else cell for cell in line] for line in metadata["original_sudoku"]]
        if not sudoku.valid_solution(metadata["complete_sudoku"], givens):
            raise InputError(f"SynLogic hard/train row {index}: invalid supplied Sudoku solution or givens")
        prompt = (
            "Solve this Sudoku. Each row, column, and 3x3 box must contain the digits 1 through 9 exactly once. "
            "Keep all given digits unchanged; 0 denotes an empty cell.\n"
            + "\n".join(" ".join(str(cell) for cell in line) for line in givens)
            + "\nReturn the completed grid as a JSON array of nine arrays of nine integers, "
            "inside a single <answer>...</answer> block. You may reason before the answer."
        )
        result.append(
            {
                "question": prompt,
                "ground_truth": json.dumps(givens),
                "original_row": index,
                "source_metadata": {
                    "source_config": "hard",
                    "source_split": "train",
                    "source_task": "sudoku",
                    "source_id": row["extra_info"]["index"],
                    "difficulty": game["difficulty"],
                    "prompt_rendering": "synlogic-sudoku-english-v1",
                },
            }
        )
    return result
