import os
import re
import subprocess
import tempfile
import time

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def prepare_validation_program(validation_program, positive_pred="eastbound", negative_pred="westbound"):
    """
    Normalise predicate names in the validation program by replacing the
    domain-specific positive and negative predicates with the canonical
    names ``pos`` and ``neg``.
    """
    validation_program = re.sub(rf"\b{positive_pred}\b", "pos", validation_program)
    validation_program = re.sub(rf"\b{negative_pred}\b", "neg", validation_program)
    return validation_program


def prepare_validation_program_isomorphic(validation_program, positive_pred="eastbound", negative_pred="westbound"):
    """
    Normalise predicate and constant names for isomorphic evaluation.

    Replaces domain-specific head predicates with ``pos``/``neg`` and
    anonymises train/car constants (``train`` -> ``mytrain``, ``car`` ->
    ``mycar``) so that structurally equivalent rules with different constant
    names are judged as correct.
    """
    # anonymize train and car instances, and head predicates
    validation_program = re.sub(rf"\b{positive_pred}\b", "pos", validation_program)
    validation_program = re.sub(rf"\b{negative_pred}\b", "neg", validation_program)
    # replace train with mytrain and car with mycar
    # trains must follow a digit pattern train\d+ and cars must follow a pattern car\d+_\d+
    validation_program = validation_program.replace("(train", "(mytrain")
    validation_program = validation_program.replace("(car", "(mycar").replace(", car", ", mycar")
    return validation_program


def evaluate_prediction(prediction, validation_program, eval_config, timeout=5, isomorphic=True):
    """
    Evaluate a predicted rule against the SLR-Bench validation program.

    Executes the predicted rule together with the task's labelled examples via
    SWI-Prolog (``swipl``) and returns the fraction of examples classified
    correctly, providing precise partial-credit feedback for RLVR training.

    Args:
        prediction: The predicted rule string.
        validation_program: Logic program defining positive/negative examples.
        eval_config: Dict with ``positive_predicate`` and ``negative_predicate`` keys.
        timeout: Maximum seconds to wait for the Prolog subprocess.
        isomorphic: If True, normalise constant names before evaluation.

    Returns:
        Dict with keys ``is_correct`` (bool), ``partial_score`` (float in [0,1]),
        ``syntax_valid`` (bool), and ``error`` (str or None).
    """

    # Extract configuration
    positive_pred = eval_config.get("positive_predicate", "eastbound")
    negative_pred = eval_config.get("negative_predicate", "westbound")

    if positive_pred not in prediction:
        p = prediction.replace("\n", " ")
        return {
            "is_correct": False,
            "partial_score": 0.0,
            "syntax_valid": False,
            "error": f"Invalid Syntax: Logic Rule not found for symbol '{positive_pred}': {p}",
        }

    pos_examples = re.findall(rf"{positive_pred}\(([^)]+)\)", validation_program)
    neg_examples = re.findall(rf"{negative_pred}\(([^)]+)\)", validation_program)

    # Determine arity by counting commas in first example plus 1
    arity = 1  # default to unary
    if pos_examples:
        arity = pos_examples[0].count(",") + 1
    elif neg_examples:
        arity = neg_examples[0].count(",") + 1

    # Create variables based on arity
    vars = ", ".join([f"X{i}" for i in range(1, arity + 1)])

    symbolic_judge = f"""
% Dynamic evaluation predicates
check({vars}) :- pos({vars}), {positive_pred}({vars}).      % positive covered
check({vars}) :- neg({vars}), \\+ {positive_pred}({vars}).  % negative rejected
% Count successful checks
check_count(Count) :-
    (setof(({vars}), ((pos({vars}); neg({vars})), check({vars})), CorrectExamples) ->
        length(CorrectExamples, Count)
    ;
        Count = 0
    ).
check_all :- forall((pos({vars});neg({vars})), check({vars})).
    """
    # Add the rule to evaluate
    if isomorphic:
        validation_program = prepare_validation_program_isomorphic(validation_program, positive_pred, negative_pred)
    else:
        validation_program = prepare_validation_program(validation_program, positive_pred, negative_pred)

    pos_negs = validation_program.count("pos(") + validation_program.count("neg(")
    validation_program = "\n".join(sorted(validation_program.splitlines()))
    full_program = validation_program + "\n\n" + symbolic_judge + "\n\n" + prediction + "\n\n"

    with tempfile.NamedTemporaryFile(suffix=".pl", mode="w", delete=False) as f:
        f.write(full_program)
        temp_file = f.name

    result = None
    try:
        eval_start_time = time.time()
        # Execute the Prolog program
        cmd = ["swipl", "-s", temp_file, "-g", "check_count(Count), writeln(Count)", "-t", "halt"]
        result = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
        partial_score = 0.0 if result.stdout.strip() == "" else int(result.stdout.strip())
        partial_score = partial_score / pos_negs if pos_negs > 0 else 0.0

        is_correct = partial_score == 1.0

        error = f'{result.stderr} -> Eval Rule "{prediction}"' if result.stderr else None
        t1 = time.time()

        return {
            "is_correct": is_correct,
            "partial_score": partial_score,
            "syntax_valid": True,
            "error": error,
            "exec_time1": t1 - eval_start_time,
        }

    except subprocess.TimeoutExpired:
        r = prediction.replace("\n", " ")
        logger.warning(f"[SLR Reward Model] Evaluation timed out after {timeout} seconds for rule: '{r}'")
        return {
            "is_correct": False,
            "partial_score": 0.0,
            "syntax_valid": False,
            "error": f"Evaluation timed out after {timeout} seconds for rule: '{r}'",
        }
    except Exception as e:
        logger.warning(f"[SLR Reward Model] Error evaluating rule '{prediction}': {e}")
        return {
            "is_correct": False,
            "partial_score": 0.0,
            "syntax_valid": False,
            "error": f"Error evaluating rule '{prediction}' returns: '{result.stdout.strip() if result else 'No error message'}' with error: {e}",
        }
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
