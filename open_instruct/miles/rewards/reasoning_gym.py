"""Pinned ReasoningGym rewards for prepared, explicitly allowlisted task families."""

import ast
import copy
import importlib
import math
import re
import signal
import time
from dataclasses import dataclass
from fractions import Fraction
from importlib import metadata

from open_instruct import logger_utils
from open_instruct.miles.rewards import deterministic_verifiers, rewards

logger = logger_utils.setup_logger(__name__)

TASKS = frozenset(
    [
        "ab",
        "acre",
        "advanced_geometry",
        "aiw",
        "arc_1d",
        "arc_agi",
        "base_conversion",
        "basic_arithmetic",
        "bf",
        "binary_alternation",
        "binary_matrix",
        "bitwise_arithmetic",
        "boxnet",
        "caesar_cipher",
        "calendar_arithmetic",
        "chain_sum",
        "circuit_logic",
        "codeio",
        "coin_flip",
        "color_cube_rotation",
        "complex_arithmetic",
        "count_bits",
        "count_primes",
        "countdown",
        "course_schedule",
        "cryptarithm",
        "decimal_arithmetic",
        "decimal_chain_sum",
        "dice",
        "emoji_mystery",
        "family_relationships",
        "figlet_font",
        "fraction_simplification",
        "futoshiki",
        "game_of_life",
        "game_of_life_halting",
        "gcd",
        "graph_color",
        "group_anagrams",
        "gsm_symbolic",
        "intermediate_integration",
        "isomorphic_strings",
        "jugs",
        "kakurasu",
        "knight_swap",
        "knights_knaves",
        "largest_island",
        "lcm",
        "leg_counting",
        "letter_counting",
        "letter_jumble",
        "list_functions",
        "mahjong_puzzle",
        "manipulate_matrix",
        "maze",
        "mini_sudoku",
        "modulo_grid",
        "n_queens",
        "needle_haystack",
        "number_filtering",
        "number_format",
        "number_sequence",
        "number_sorting",
        "palindrome_generation",
        "palindrome_partitioning",
        "polynomial_equations",
        "polynomial_multiplication",
        "pool_matrix",
        "power_function",
        "prime_factorization",
        "products",
        "propositional_logic",
        "puzzle24",
        "quantum_lock",
        "ransom_note",
        "rearc",
        "rectangle_count",
        "rotate_matrix",
        "rotten_oranges",
        "rubiks_cube",
        "rush_hour",
        "self_reference",
        "sentence_reordering",
        "shortest_path",
        "simple_equations",
        "simple_geometry",
        "simple_integration",
        "sokoban",
        "spell_backward",
        "spiral_matrix",
        "string_insertion",
        "string_manipulation",
        "string_splitting",
        "string_synthesis",
        "sudoku",
        "survo",
        "syllogism",
        "time_intervals",
        "tower_of_hanoi",
        "tsumego",
        "word_ladder",
        "word_sequence_reversal",
        "word_sorting",
        "zebra_puzzles",
    ]
)
DEFAULT_EXACT_TASKS = frozenset(
    [
        "acre",
        "aiw",
        "arc_1d",
        "base_conversion",
        "basic_arithmetic",
        "bf",
        "binary_alternation",
        "caesar_cipher",
        "count_bits",
        "count_primes",
        "course_schedule",
        "gcd",
        "isomorphic_strings",
        "largest_island",
        "lcm",
        "leg_counting",
        "letter_counting",
        "list_functions",
        "mahjong_puzzle",
        "maze",
        "modulo_grid",
        "number_filtering",
        "number_sequence",
        "ransom_note",
        "rotate_matrix",
        "rotten_oranges",
        "simple_equations",
        "simple_geometry",
        "string_manipulation",
        "string_splitting",
        "string_synthesis",
        "syllogism",
        "word_sequence_reversal",
    ]
)


STATE_TASKS = {
    "rubiks_cube": ("cube_size", "scramble_moves"),
    "graph_color": ("puzzle",),
    "propositional_logic": ("premises", "variables"),
    "rush_hour": ("board_config",),
    "boxnet": ("initial_state",),
}
LITERAL_TASKS = {"binary_matrix", "n_queens", "spiral_matrix", "string_insertion"}
SYMBOLIC_TASKS = {"simple_integration", "intermediate_integration", "polynomial_multiplication"}


def expression_syntax(text, *, symbolic=False, bitwise=False):
    """Reject executable syntax before a pinned native scorer's expression parser.

    The accepted grammar contains numeric constants, arithmetic and (for symbolic
    tasks) single-letter symbols and explicitly named mathematical functions.
    In particular there are no attributes, subscripts, strings or arbitrary calls.
    This also protects dataset expressions, not just model responses.
    """
    tree = ast.parse(text.strip(), mode="eval")
    operators = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.UAdd, ast.USub)
    if bitwise:
        operators += (ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor, ast.Invert)
    functions = {
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "sqrt",
        "Abs",
        "Integral",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Load, *operators)):
            continue
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            continue
        if (
            symbolic
            and isinstance(node, ast.Name)
            and (re.fullmatch("[a-zA-Z]", node.id) or node.id in functions | {"pi"})
        ):
            continue
        if (
            symbolic
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in functions
            and not node.keywords
        ):
            continue
        raise ValueError("Unsupported expression syntax")


def arithmetic_value(node):
    """Evaluate the four-operation puzzle grammar exactly, without eval."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = arithmetic_value(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = arithmetic_value(node.left), arithmetic_value(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("Expected arithmetic using +, -, * and /")


def validate_target(entry):
    if not isinstance(entry, dict) or not isinstance(entry.get("metadata"), dict):
        raise ValueError("malformed_reasoning_gym_entry")
    task = entry["metadata"].get("source_dataset")
    if task not in TASKS:
        raise ValueError("unqualified_reasoning_gym_family:" + str(task))
    if task in STATE_TASKS:
        if any(key not in entry["metadata"] for key in STATE_TASKS[task]):
            raise ValueError("missing_reasoning_gym_state:" + task)
    elif not isinstance(entry.get("answer"), str) or (
        not entry["answer"].strip() and task not in {"ab", "polynomial_equations"}
    ):
        raise ValueError("missing_reasoning_gym_answer")
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
    # Native eval/parse_expr must never see executable source metadata either.
    if task == "number_sorting":
        ast.literal_eval(entry["answer"])
    if task == "bitwise_arithmetic":
        expression_syntax(entry["metadata"]["problem"], bitwise=True)
    if task in SYMBOLIC_TASKS:
        expression_syntax(entry["answer"], symbolic=True)
        if task.endswith("integration"):
            expression_syntax(entry["metadata"]["integrand"], symbolic=True)
    if task == "prime_factorization":
        [int(factor.strip()) for factor in entry["answer"].split("×") if factor.strip()]
    if task == "coin_flip" and not math.isfinite(float(Fraction(entry["answer"].replace(",", "")))):
        raise ValueError("Invalid coin-flip reference")
    return task


@dataclass
class ReasoningGymConfig:
    timeout_seconds: float = 5.0


class VerificationTimeout(BaseException):
    """Pass through native scorers' broad Exception handlers to the deadline guard."""


def timeout_result(error):
    logger.warning("ReasoningGym verifier timeout; assigning zero reward: %s", error)
    return deterministic_verifiers.Result(
        0.0,
        diagnostics={
            "kind": "reasoning_gym_0.1.25",
            "status": "timeout",
            "error": str(error),
            "failure_policy": "zero",
        },
    )


class ReasoningGymVerifier:
    get_config_class = staticmethod(lambda: ReasoningGymConfig)

    def __init__(self, verifier_config=None):
        self.config = verifier_config or ReasoningGymConfig()
        if not 0 < self.config.timeout_seconds <= 30:
            raise ValueError("ReasoningGym timeout must be in (0, 30] seconds")
        if metadata.version("reasoning-gym") != "0.1.25":
            raise ValueError("This contract requires reasoning-gym==0.1.25")
        self.library = importlib.import_module("reasoning_gym")
        self.default_scorer = importlib.import_module("reasoning_gym.dataset").ProceduralDataset.score_answer
        self.scorers = {}

    async def async_call(self, tokens, prediction, label, **kwargs):
        validate_target(label)
        try:
            result = await rewards.isolated_verifier_call(
                {
                    "factory": "open_instruct.miles.rewards.reasoning_gym.ReasoningGymVerifier",
                    "config": {"timeout_seconds": self.config.timeout_seconds},
                },
                tokens,
                prediction,
                label,
                **kwargs,
            )
        except rewards.MathVerifierTimeout as error:
            return timeout_result(error)
        if result.diagnostics.get("status") == "timeout":
            logger.warning("ReasoningGym verifier timeout; assigning zero reward: %s", result.diagnostics["error"])
        return result

    def __call__(self, tokenized_prediction, prediction, label, **kwargs):
        validate_target(label)

        def expired(signum, frame):
            raise VerificationTimeout("ReasoningGym response exceeded its verification budget")

        previous = signal.signal(signal.SIGALRM, expired)
        timer = signal.setitimer(signal.ITIMER_REAL, self.config.timeout_seconds)
        start = time.monotonic()
        try:
            try:
                result = self.score(prediction, label)
                # Some native scorers use bare except, which can swallow the alarm.
                if time.monotonic() - start >= self.config.timeout_seconds:
                    return timeout_result("Native scorer exceeded its verification budget")
                return result
            finally:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                finally:
                    signal.signal(signal.SIGALRM, previous)
                    signal.setitimer(signal.ITIMER_REAL, *timer)
        except VerificationTimeout as error:
            return timeout_result(error)

    def score(self, prediction, label):
        entry = copy.deepcopy(label)
        task = entry["metadata"]["source_dataset"]
        final = deterministic_verifiers.answer_text(prediction)
        tagged = re.findall(r"<answer>(.*?)</answer>", final, re.S)
        boxes = re.findall(r"\\boxed\{([^{}]+)\}", final)
        # Multiple explicit answers cannot be resolved by picking the last one.
        ambiguous = len(tagged) > 1 or len(boxes) > 1
        answer = tagged[-1].strip() if tagged else boxes[-1].strip() if boxes else final
        if tagged and boxes:
            # One answer may use both wrappers: <answer>\boxed{12}</answer>.
            nested_box = re.fullmatch(r"\\boxed\{([^{}]+)\}", answer)
            answer = nested_box[1].strip() if nested_box else answer
            if answer != boxes[0].strip():
                ambiguous = True
        if ambiguous or not final.strip():
            return deterministic_verifiers.Result(
                0.0,
                diagnostics={"kind": "reasoning_gym_0.1.25", "status": "ok", "scoring_protocol": "exact_default_v2"},
            )
        if task in {"ab", "polynomial_equations"} and answer == '""':
            answer = ""
        try:
            if task == "prime_factorization":
                [int(factor.strip()) for factor in answer.split("×") if factor.strip()]
            if task == "coin_flip" and not math.isfinite(float(Fraction(answer.replace(",", "")))):
                raise ValueError("Nonfinite probability")
            if task == "game_of_life":
                cells = deterministic_verifiers.strict_json(answer)
                if not isinstance(cells, list) or any(not isinstance(row, list) for row in cells):
                    raise ValueError("Expected a matrix of cells")
            if task == "boxnet":
                plans = deterministic_verifiers.strict_json(answer)
                if not isinstance(plans, list) or any(not isinstance(plan, dict) for plan in plans):
                    raise ValueError("Expected a list of action maps")
                for plan in plans:
                    for agent, action in plan.items():
                        if not re.fullmatch(r"Agent\[\d+(?:\.\d+)?,\s*\d+(?:\.\d+)?\]", agent) or not isinstance(
                            action, str
                        ):
                            raise ValueError("Invalid action map")
                        match = re.fullmatch(
                            r"move\((box_\w+), (target_\w+|square\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\])\)", action
                        )
                        if match is None:
                            raise ValueError("Invalid move")
                        if (
                            match[3] is not None
                            and f"{float(match[3])}_{float(match[4])}" not in entry["metadata"]["initial_state"]
                        ):
                            raise ValueError("Move outside the grid")
            if task == "puzzle24":
                # Native 0.1.25 checks only number ranges and truncates the total.
                expression_syntax(answer)
                tree = ast.parse(answer, mode="eval")
                constants = [
                    node.value
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
                ]
                if sorted(constants) != sorted(entry["metadata"]["numbers"]):
                    raise ValueError("Answer must use the puzzle's numbers exactly once")
                if arithmetic_value(tree.body) != 24:
                    raise ValueError("Expression must equal exactly 24")
            if (
                task in LITERAL_TASKS
                and answer != entry.get("answer")
                and not (task == "n_queens" and answer in entry["metadata"]["valid_answers"])
            ):
                # Canonical literals preserve native alternate-format partial credit,
                # while arbitrary expressions never reach the library's eval calls.
                answer = repr(ast.literal_eval(answer))
            if task in SYMBOLIC_TASKS | {"countdown", "puzzle24"}:
                expression_syntax(answer, symbolic=task in SYMBOLIC_TASKS)
        except (SyntaxError, ValueError, TypeError, RecursionError, OverflowError, ZeroDivisionError) as error:
            return deterministic_verifiers.Result(
                0.0, diagnostics={"kind": "reasoning_gym_0.1.25", "status": "parse_error", "error": str(error)}
            )
        if task in {"arc_agi", "rearc"}:
            grid = entry["metadata"]["output"]
            entry["metadata"]["output"] = tuple(tuple(row) for row in grid)
        if task == "game_of_life_halting":
            gold = entry["answer"].strip().lower()
            if gold not in {"true", "false"}:
                raise ValueError("Invalid boolean reference")
            result = float(answer.lower() in {"true", "false"} and answer.lower() == gold)
        else:
            if task not in self.scorers:
                self.scorers[task] = self.library.get_score_answer_fn(task)
            scorer = self.scorers[task]
            # The library default rewards any answer containing the gold string,
            # including wrong numbers. Only task-specific scorers may grade softly.
            if getattr(scorer, "__func__", None) is self.default_scorer:
                result = float(answer.strip() == entry["answer"].strip())
            else:
                result = float(scorer(answer=answer, entry=entry))
        if ambiguous:
            result = 0.0
        if not math.isfinite(result) or not 0 <= result <= 1:
            raise ValueError("Invalid ReasoningGym reward")
        return deterministic_verifiers.Result(
            result,
            diagnostics={"kind": "reasoning_gym_0.1.25", "status": "ok", "scoring_protocol": "exact_default_v2"},
        )
