"""Grade sampled math responses with the Qwen2.5-Math evaluation harness.

The EOPD paper (arXiv 2603.07079, App. C) reports Avg@8 / Pass@8 with the answer extraction and
equivalence checker of ``QwenLM/Qwen2.5-Math/evaluation`` (``parser.extract_answer`` and
``grader.math_equal``). This module wraps that harness so an exported checkpoint can be scored
the way the paper's Table 2 was, on the prompt sets rendered by
``scripts/miles/prepare_eopd_math_prompts.py`` (JSONL rows with ``input``, ``label`` and
``metadata.dataset``). The harness is imported lazily from a checkout (``attach_harness``); the
pure-Python pieces (ground-truth normalisation, aggregation) are importable without it so they
can be unit-tested.
"""

import json
import multiprocessing
import sys
from collections import defaultdict
from concurrent import futures
from pathlib import Path

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

# Our rendered set name -> the harness ``data_name`` whose extraction / stripping rules apply.
HARNESS_DATA_NAME = {
    "math_500": "math",
    "aime24": "aime24",
    "aime25": "aime24",
    "amc23": "amc23",
    "minerva_math": "minerva_math",
    "olympiadbench": "olympiadbench",
}
# parser.STRIP_EXCEPTIONS in the harness: ground truths kept verbatim apart from three aliases.
STRIP_EXCEPTIONS = ("carp_en", "minerva_math")
DEFAULT_SETS = ("math_500", "amc23", "minerva_math", "olympiadbench", "aime24", "aime25")

# EOPD Table 2, OPD column (Avg@8, Pass@8) for the two paper students.
PAPER_OPD = {
    "qwen3-4b-base": {
        "math_500": (78.81, 90.80),
        "amc23": (57.33, 80.00),
        "minerva_math": (40.08, 54.00),
        "olympiadbench": (42.10, 58.80),
        "aime24": (18.33, 26.67),
        "aime25": (12.08, 30.00),
    },
    "qwen3-1.7b-base": {
        "math_500": (67.76, 84.80),
        "amc23": (39.06, 70.00),
        "minerva_math": (29.83, 47.06),
        "olympiadbench": (30.09, 51.56),
        "aime24": (8.33, 20.00),
        "aime25": (6.25, 16.67),
    },
}


def attach_harness(checkout: Path):
    """Put ``<checkout>/evaluation`` on ``sys.path`` and return ``(extract_answer, strip_string, math_equal)``."""
    evaluation = Path(checkout) / "evaluation"
    if not (evaluation / "grader.py").exists():
        raise FileNotFoundError(f"{evaluation} is not a Qwen2.5-Math checkout (no grader.py)")
    sys.path.insert(0, str(evaluation))
    import grader  # noqa: PLC0415  # harness module, only importable from the checkout
    import parser as harness_parser  # noqa: PLC0415

    return harness_parser.extract_answer, harness_parser.strip_string, grader.math_equal


def load_prompts(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for index, row in enumerate(rows):
        if "input" not in row or "label" not in row:
            raise ValueError(f"{path}:{index} lacks `input`/`label`; expected a prepare_eopd_math_prompts.py set")
    return rows


def normalize_ground_truth(label: str, data_name: str, strip_string) -> str:
    """Mirror ``parser.parse_ground_truth``'s post-processing for a final-answer string."""
    if data_name in STRIP_EXCEPTIONS:
        return label.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    return strip_string(label, skip_unit=data_name == "carp_en")


def olympiadbench_first_answers(rows: list[dict], repo: str, revision: str) -> dict[int, str] | None:
    """``final_answer[0].strip("$")`` per source row, as the harness grades OlympiadBench.

    Our rendered label joins every final answer with ", "; the harness only checks the first.
    Returns None (caller falls back to the label) when the source dataset cannot be loaded.
    """
    try:
        import datasets  # noqa: PLC0415  # optional: only the grading job has it
    except ImportError:
        return None
    try:
        source = datasets.load_dataset(repo, None, revision=revision, split="test")
    except Exception as error:  # network / cache problems must not sink the whole eval
        logger.warning("could not load %s@%s for OlympiadBench answers: %s", repo, revision, error)
        return None
    wanted = {row["metadata"]["source_row"] for row in rows}
    return {i: str(source[i]["final_answer"][0]).strip("$") for i in wanted}


def _equal(args):
    extract_answer, math_equal, text, data_name, ground_truth = args
    prediction = extract_answer(text, data_name)
    return prediction, bool(math_equal(prediction, ground_truth, timeout=False))


def grade(
    responses: list[dict], data_name: str, harness, *, timeout: float = 3.0, workers: int = 8
) -> tuple[list[dict], int]:
    """Attach ``prediction`` / ``correct`` to each response (``text``, ``ground_truth``).

    With ``workers > 0`` each comparison runs in a pebble worker process under the harness's 3 s
    budget (``evaluate.py`` uses ``ProcessPool(timeout=3)``; pebble kills a worker that overruns,
    unlike ``concurrent.futures``); a timeout scores False. ``workers=0`` grades inline without a
    timeout (tests). Returns the graded rows and the count of timeouts.
    """
    extract_answer, _, math_equal = harness
    timeouts = 0
    if workers <= 0:
        for row in responses:
            row["prediction"], row["correct"] = _equal(
                (extract_answer, math_equal, row["text"], data_name, row["ground_truth"])
            )
        return responses, 0
    import pebble  # noqa: PLC0415  # only the grading job needs it

    # fork: workers inherit the harness modules on sys.path; spawn/forkserver could not import them.
    with pebble.ProcessPool(max_workers=workers, context=multiprocessing.get_context("fork")) as pool:
        pending = [
            (
                pool.schedule(
                    _equal,
                    args=[(extract_answer, math_equal, r["text"], data_name, r["ground_truth"])],
                    timeout=timeout,
                ),
                r,
            )
            for r in responses
        ]
        for future, row in pending:
            try:
                row["prediction"], row["correct"] = future.result()
            except futures.TimeoutError:
                row["prediction"], row["correct"] = None, False
                timeouts += 1
            except Exception as error:  # a grader crash on one sample scores it wrong
                logger.warning("grader error on %s: %s", row.get("sample_id"), error)
                row["prediction"], row["correct"] = None, False
    return responses, timeouts


def aggregate(responses: list[dict]) -> dict:
    """Avg@n (mean correctness over every sample) and Pass@n (any sample correct) per problem set."""
    by_problem: dict = defaultdict(list)
    for row in responses:
        by_problem[row["prompt_index"]].append(row)
    n = max(len(v) for v in by_problem.values())
    if any(len(v) != n for v in by_problem.values()):
        raise ValueError("every problem must have the same number of samples")
    correct = [r["correct"] for r in responses]
    per_sample_index = []
    for k in range(n):
        column = [sorted(v, key=lambda r: r["sample_index"])[k]["correct"] for v in by_problem.values()]
        per_sample_index.append(100.0 * sum(column) / len(column))
    return {
        "problems": len(by_problem),
        "samples_per_problem": n,
        "avg_at_n": 100.0 * sum(correct) / len(correct),
        "pass_at_n": 100.0 * sum(any(r["correct"] for r in v) for v in by_problem.values()) / len(by_problem),
        "per_sample_index_acc": per_sample_index,
        "truncated_fraction": sum(r.get("finish_reason") == "length" for r in responses) / len(responses),
        "mean_response_tokens": sum(r.get("response_tokens", 0) for r in responses) / len(responses),
    }


def summary_table(results: dict[str, dict], paper: dict | None) -> str:
    lines = [
        "| Set | Problems | Avg@n | Pass@n | Paper Avg@8 | Paper Pass@8 | Truncated |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        p = (paper or {}).get(name)
        lines.append(
            f"| {name} | {r['problems']} | {r['avg_at_n']:.2f} | {r['pass_at_n']:.2f} | "
            f"{p[0] if p else '-'} | {p[1] if p else '-'} | {100 * r['truncated_fraction']:.1f}% |"
        )
    if paper and all(name in paper for name in results):
        ours_avg = sum(r["avg_at_n"] for r in results.values()) / len(results)
        ours_pass = sum(r["pass_at_n"] for r in results.values()) / len(results)
        paper_avg = sum(paper[n][0] for n in results) / len(results)
        paper_pass = sum(paper[n][1] for n in results) / len(results)
        lines.append(f"| **mean** | | {ours_avg:.2f} | {ours_pass:.2f} | {paper_avg:.2f} | {paper_pass:.2f} | |")
    return "\n".join(lines)
