"""Adapters and independent reference checks for the first long-context release."""

import csv
import io
import re

from open_instruct.miles.rewards import long_context as rewards

# The upstream task-to-metric map is explicit; unknown tasks are never guessed.
F1_TASKS = {"T5.1", "T5.2", "T6.2", "T7.1", "T7.2", "T7.3", "T9.1", "T9.2"}
ACCURACY_TASKS = {"T3.1", "T3.2", "T11.1", "T11.2"}
DEFERRED_TASKS = {"T2.1", "T2.2", "T6.3", "T6.1", "T8.1", "T8.2", "T8.3", "T10.1", "T10.2"}
ADAPTERS = {"lc_transduction_v1", "lc_docqa_v1", "lc_golong_v1"}


def target(metric, answers, extraction="final"):
    result = {"version": 1, "metric": metric, "answers": answers, "extraction": extraction}
    rewards.validate_target(result)
    return result


def transduction_reference(raw):
    """Recompute all twelve released task variants from the input, not the label."""
    task = raw["metadata"]["task_type"]
    prompt = raw["messages"][0]["content"]
    body = prompt.split("Here is the real ", 1)[1]
    if task.startswith("csv_"):
        before, table = body.split("Input:\n\n", 1)
        instructions = None
        if "\n\nReordering Instructions:" in table:
            table, instructions = table.split("\n\nReordering Instructions:", 1)
        rows = list(csv.reader(io.StringIO(table.strip())))
        if not rows or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("Malformed source CSV")
        if task == "csv_kv_lookup":
            lookup = dict(re.findall(r"(?m)^([av]\d+)=(\S+)$", before))
            out = [rows[0]] + [
                [row[0], *[" ".join(lookup[k] for k in cell.split("+")) for cell in row[1:]]] for row in rows[1:]
            ]
        elif task in {"csv_permutation_homogeneous", "csv_permutation_heterogeneous"} and instructions:
            row_order = re.search(r"New row order: (.*)", instructions)[1].split(",")
            col_order = re.search(r"New col order: (.*)", instructions)[1].split(",")
            by_row = {row[0]: row for row in rows[1:]}
            if (
                set(row_order) != set(by_row)
                or len(row_order) != len(by_row)
                or sorted(col_order) != sorted(rows[0][1:])
            ):
                raise ValueError("Invalid source permutation")
            columns = [rows[0].index(c) for c in col_order]
            out = [["", *col_order]] + [[r, *[by_row[r][c] for c in columns]] for r in row_order]
        else:
            raise ValueError("Unsupported CSV task")
        stream = io.StringIO()
        csv.writer(stream, lineterminator="\n").writerows(out)
        return stream.getvalue().strip()
    if task.endswith("var_expand"):
        definitions, sequence = body.split("Expressions:\n\n", 1)
        lookup = dict(re.findall(r"(?m)^([0-9a-f]+)=(\S+)$", definitions))
    else:
        sequence = body.split("Input:\n\n", 1)[1]
        lookup = {}
    results = []
    for i, line in enumerate(sequence.strip().splitlines(), 1):
        match = re.fullmatch(r"(?:\[(\d+)\])?(.*)", line)
        index, expression = match.groups()
        if task.startswith("unnumbered_") != (index is None):
            raise ValueError("Invalid source index")
        prefix = f"[{index}]" if index else ""
        if task.endswith("sum"):
            if re.fullmatch(r"\d+(?:[+-]\d+)+", expression) is None:
                raise ValueError("Unsupported arithmetic expression")
            value = sum(map(int, re.findall(r"[+-]?\d+", expression)))
            answer = f"{expression}={value}"
        elif task.endswith("uuid_sort"):
            values = expression.split(",")
            if any(re.fullmatch(r"[0-9a-f]{8}", value) is None for value in values):
                raise ValueError("Invalid hexadecimal token")
            answer = ",".join(sorted(values))
        elif task.endswith("var_expand"):
            keys = expression.split("+")
            if len(keys) != 2:
                raise ValueError("Invalid expansion")
            answer = " ".join(lookup[k] for k in keys)
        else:
            raise ValueError("Unknown transduction task")
        results.append((int(index) if index else i, prefix + answer))
    if sorted(i for i, _ in results) != list(range(1, len(results) + 1)):
        raise ValueError("Source indices must form a complete sequence")
    return "\n".join(answer for _, answer in sorted(results))


def adapt(raw, kind, messages):
    if kind == "lc_transduction_v1":
        ms = raw["messages"]
        if len(ms) != 2 or ms[0]["role"] != "user" or ms[1]["role"] != "assistant":
            raise ValueError("Expected one user prompt and one reference answer")
        reference = transduction_reference(raw)
        if reference != ms[1]["content"].strip():
            raise ValueError("Independent transduction reference mismatch")
        return (
            messages({"prompt": ms[:1]}),
            target("line_accuracy", [reference]),
            "long_context",
            raw["metadata"]["task_type"],
        )
    ms = messages(raw)
    gt = raw["reward_model"]["ground_truth"]
    if kind == "lc_docqa_v1":
        ability = raw["ability"]
        if ability == "doc-mc":
            match = re.fullmatch(r"The correct answer is \(([A-Z])\)\.", gt)
            if not match:
                raise ValueError("Unsupported DocQA choice label")
            t = target("choice", [match[1]], "choice")
        elif ability in {"doc-math", "doc-qa"}:
            prefix = "Therefore, the answer is "
            if not gt.startswith(prefix):
                raise ValueError("Unsupported DocQA reference wrapper")
            value = gt[len(prefix) :].removesuffix(".")
            t = target("decimal" if ability == "doc-math" else "docqa", [value], "therefore")
        else:
            raise ValueError("Unsupported DocQA ability")
        return ms, t, "long_context", raw["data_source"]
    if kind != "lc_golong_v1":
        raise ValueError("Unknown long-context adapter")
    source = raw["data_source"]
    mode = raw["extra_info"]["reward_mode"]
    if source == "math_longcot_math_verify":
        if not isinstance(gt["golden_label"], str) or not gt["golden_label"].strip():
            raise ValueError("Empty math reference")
        return ms, gt["golden_label"], "math", mode
    if source == "EM":
        t = target("exact", [gt["golden_label"]])
    elif source == "selection":
        t = target("choice", [gt["golden_label"]], "choice")
    elif source == "longbench_pro":
        code = mode.split()[0]
        if code in DEFERRED_TASKS:
            raise ValueError("deferred_native_recall_or_partial_order_reward:" + mode)
        if code in {"T4.1", "T4.2"}:
            raise ValueError("deferred_summary_proxy_reward:" + mode)
        if code in ACCURACY_TASKS:
            # Some tasks use text answers rather than choice letters. Require
            # the whole answer, instead of upstream first-line-only credit.
            t = target("docqa", gt["doc_ids"], "answer_marker")
        elif code in F1_TASKS:
            t = target("lines_f1", gt["doc_ids"], "answer_marker")
        elif code in {"T1.1", "T1.2"}:
            t = target("ndcg", gt["doc_ids"], "answer_marker")
        else:
            raise ValueError("Unknown GoLong task:" + mode)
    else:
        raise ValueError("deferred_unimplemented_contract:" + source)
    return ms, t, "long_context", mode
