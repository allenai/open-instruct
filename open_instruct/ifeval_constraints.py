"""Parse IFEval targets and screen known unsuitable IF training constraints.

This training admission policy leaves evaluation scoring and other constraint
families, including explicit multilingual constraints, alone. The N<=3 cutoff is
the audited tiny-limit training policy; admitting larger limits or other constraint
families does not establish that their targets or grading are correct.
"""

import ast
import json


def parse_constraint_bundle(target) -> dict:
    """Decode canonical bundles and nested JSON/Python singleton wrappers.

    Return a fresh dictionary with validated parallel instruction_id/kwargs arrays;
    null kwargs entries become empty dictionaries. Invalid targets raise ValueError.
    """
    value = target
    for _ in range(32):
        if isinstance(value, dict):
            break
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, RecursionError):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError, TypeError, RecursionError) as error:
                    raise ValueError(
                        "IFEval target must encode a constraint dictionary (JSON or Python literal)"
                    ) from error
        elif isinstance(value, list):
            if len(value) != 1:
                raise ValueError("IFEval target wrapper must contain exactly one constraint bundle")
            value = value[0]
        else:
            raise ValueError("IFEval target must be a dictionary, encoded dictionary, or singleton-list wrapper")
    else:
        raise ValueError("IFEval target exceeds 32 nested encoding/wrapper levels")

    ids, kwargs = value.get("instruction_id"), value.get("kwargs")
    if not isinstance(ids, list) or not ids:
        raise ValueError("IFEval target instruction_id must be a nonempty array")
    if not isinstance(kwargs, list) or not kwargs:
        raise ValueError("IFEval target kwargs must be a nonempty array")
    if len(ids) != len(kwargs):
        raise ValueError(f"IFEval target instruction_id/kwargs lengths differ ({len(ids)} != {len(kwargs)})")
    for index, (instruction_id, args) in enumerate(zip(ids, kwargs)):
        if not isinstance(instruction_id, str) or not instruction_id.strip():
            raise ValueError(f"IFEval target instruction_id[{index}] must be a nonempty string")
        if args is not None and not isinstance(args, dict):
            raise ValueError(f"IFEval target kwargs[{index}] for {instruction_id!r} must be a dictionary or null")
    return {**value, "instruction_id": list(ids), "kwargs": [dict(args or {}) for args in kwargs]}


def _positive_integer(value):
    return type(value) is int and value > 0


def if_training_issues(target) -> list[str]:
    """Return deduplicated stable reason codes, in constraint/field order.

    Malformed bundles return ``malformed_target``; callers needing parse details
    should call parse_constraint_bundle directly. No checker or RNG is invoked.
    """
    try:
        bundle = parse_constraint_bundle(target)
    except ValueError:
        return ["malformed_target"]
    issues = []
    for instruction_id, args in zip(bundle["instruction_id"], bundle["kwargs"]):
        if instruction_id == "count:lowercase_counting":
            if not _positive_integer(args.get("N")):
                issues.append("lowercase_counting_invalid_n")
            elif args["N"] <= 3:
                issues.append("lowercase_counting_tiny_n")
        elif instruction_id == "letters:letter_counting":
            if not _positive_integer(args.get("N")):
                issues.append("letter_counting_invalid_n")
            if args.get("relation") not in ("less than", "at least"):
                issues.append("letter_counting_invalid_relation")
            elif args["relation"] == "less than" and _positive_integer(args.get("N")) and args["N"] <= 3:
                issues.append("letter_counting_tiny_less_than")
        elif instruction_id == "length_constraints:nth_paragraph_first_word":
            num, nth = args.get("num_paragraphs"), args.get("nth_paragraph")
            if not _positive_integer(num):
                issues.append("nth_paragraph_first_word_invalid_num_paragraphs")
            if not _positive_integer(nth):
                issues.append("nth_paragraph_first_word_invalid_nth_paragraph")
            if _positive_integer(num) and _positive_integer(nth) and nth > num:
                issues.append("nth_paragraph_first_word_nth_exceeds_num_paragraphs")
            if args.get("first_word") is None:
                issues.append("nth_paragraph_first_word_missing_first_word")
            elif not isinstance(args["first_word"], str) or not args["first_word"].strip():
                issues.append("nth_paragraph_first_word_invalid_first_word")
    return list(dict.fromkeys(issues))
