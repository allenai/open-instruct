"""Score extra-eval generations (tools/format_probe.py --eval-files) with two graders and summarize.

- mv: Open-Instruct MathVerifier, as in training and the in-loop evals (a response cut at the length cap scores 0).
- oe: olmo-eval's Minerva scorer (MathExtractor.extract_answer + is_equiv, any extracted vs any gold), loaded from an
  olmo-eval checkout. Golds as olmo-eval builds them: every answer MathExtractor finds in the MATH reference solution,
  AIME answers with leading zeros stripped, the OCW answer string. Like olmo-eval, truncated responses are still scored.

Reports per set: mean accuracy, MATH macro-average over its 7 subjects (olmo-eval's minerva_math suite aggregation),
AIME pass@k (unbiased, from the repeats of each problem), the same with responses over 16,384 tokens counted wrong
(olmo-eval's pass_at_32_rlzero length), truncation rate and mean length.

Usage: PYTHONPATH=open-instruct python score_extra_evals.py RAW_DIR EVAL_DATA_DIR OLMO_EVAL_EXTRACT_PY [--pool 32] [--self-check]
"""

import argparse
import collections
import importlib.util
import json
import math
import statistics
from multiprocessing import Pool
from pathlib import Path

EXTRACT = None
VERIFIER = None


def init(extract_path):
    global EXTRACT, VERIFIER
    spec = importlib.util.spec_from_file_location("olmo_eval_extract_math", extract_path)
    EXTRACT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(EXTRACT)
    from open_instruct.ground_truth_utils import MathVerifier

    VERIFIER = MathVerifier()


def golds(name, meta, label):
    if name == "minerva_math":
        return EXTRACT.MathExtractor.extract_answer(meta["reference_solution"]) or [label]
    if name == "aime2026":
        return [label.lstrip("0") or "0"]
    return [label]


def score(item):
    name, meta, label, text, truncated = item
    mv = 0.0 if truncated else float(VERIFIER([], text, label).score)
    extracted = EXTRACT.MathExtractor.extract_answer(text) or []
    oe = 0.0
    for answer in extracted:
        if any(EXTRACT.is_equiv(str(answer).strip(), str(gold).strip()) for gold in golds(name, meta, label)):
            oe = 1.0
            break
    return mv, oe


def pass_at_k(n, c, k):
    c = int(round(c))
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("extract_py", type=Path)
    parser.add_argument("--pool", type=int, default=32)
    parser.add_argument("--self-check", action="store_true", help="Grade the gold answers themselves instead of generations")
    args = parser.parse_args()

    meta = {}
    for path in args.data.glob("eval-*.jsonl"):
        name = path.stem.removeprefix("eval-")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            meta[row["metadata"]["prepared_sample_id"]] = (name, row["metadata"], row["label"])

    rows = []  # (name, base_id, subject, tokens, truncated, text)
    if args.self_check:
        for sample_id, (name, m, label) in meta.items():
            text = m["reference_solution"] if name == "minerva_math" else f"The answer is \\boxed{{{label}}}."
            rows.append((name, sample_id.split("#")[0], m.get("subject"), 0, False, text, sample_id))
    else:
        for path in sorted(args.raw.glob("*.raw.json")):
            for prompt in json.loads(path.read_text())["prompts"]:
                name, m, _ = meta[prompt["id"]]
                for response in prompt["responses"]:
                    rows.append((name, prompt["id"].split("#")[0], m.get("subject"), response["tokens"], (response["finish"] or {}).get("type") == "length", response["text"], prompt["id"]))
    items = [(r[0], meta[r[6]][1], meta[r[6]][2], r[5], r[4]) for r in rows]
    with Pool(args.pool, initializer=init, initargs=(str(args.extract_py),)) as pool:
        scores = pool.map(score, items, chunksize=16)

    summary = {}
    for name in sorted({r[0] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r[0] == name]
        entry = {
            "responses": len(idx),
            "problems": len({rows[i][1] for i in idx}),
            "acc_mv": round(100 * statistics.mean(scores[i][0] for i in idx), 2),
            "acc_oe": round(100 * statistics.mean(scores[i][1] for i in idx), 2),
            "truncated_pct": round(100 * statistics.mean(rows[i][4] for i in idx), 2),
            "mean_tokens": round(statistics.mean(rows[i][3] for i in idx)),
        }
        if name == "minerva_math":
            by_subject = collections.defaultdict(list)
            for i in idx:
                by_subject[rows[i][2]].append(scores[i])
            entry["subjects_oe"] = {s: round(100 * statistics.mean(v[1] for v in vals), 2) for s, vals in sorted(by_subject.items())}
            entry["macro_oe"] = round(statistics.mean(entry["subjects_oe"].values()), 2)
            entry["macro_mv"] = round(statistics.mean(100 * statistics.mean(v[0] for v in vals) for vals in by_subject.values()), 2)
        if name in ("aime2026", "minerva_ocw"):
            groups = collections.defaultdict(list)
            for i in idx:
                groups[rows[i][1]].append((scores[i], rows[i][3]))
            for grader, g in (("mv", 0), ("oe", 1)):
                for k in (1, 4, 8, 16, 32):
                    if all(len(v) >= k for v in groups.values()):
                        entry[f"pass@{k}_{grader}"] = round(100 * statistics.mean(pass_at_k(len(v), sum(s[g] for s, _ in v), k) for v in groups.values()), 2)
                entry[f"pass@1_{grader}_16k"] = round(100 * statistics.mean(statistics.mean(s[g] if t <= 16384 else 0.0 for s, t in v) for v in groups.values()), 2)
        summary[name] = entry
    out = args.raw / ("self-check-summary.json" if args.self_check else "summary.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
