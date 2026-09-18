"""Build auditable response-prefix galleries from scored before/after JSONL files.

Extends the recovered September 12 Avery word-trie experiment. Scores are copied,
never recomputed. See response_trees.md for the manifest and interpretation.
"""

import argparse
import base64
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

VIEWS = ("response", "answer", "code")
DEPTHS = (12, 28, 60)


def entropy(counts):
    total = sum(counts)
    return -sum(n / total * math.log2(n / total) for n in counts if n) if total else 0.0


def build_tree(samples, field="response", depth=28):
    """A path-compressed word trie, retaining *all* terminal samples.

    A response ending at an internal node is an additional branch for entropy
    and verdict information gain. Sample IDs are never artificial trie branches.
    """
    if depth < 1:
        raise ValueError("Prefix depth must be positive")
    if len({s["sample"] for s in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs")
    by_id = {s["sample"]: s for s in samples}

    def node():
        return {"ids": [], "terminal": [], "children": {}}

    root = node()
    for s in sorted(samples, key=lambda s: s["sample"]):
        current = root
        current["ids"].append(s["sample"])
        for word in re.findall(r"\S+", s.get(field, ""))[:depth]:
            current = current["children"].setdefault(word, node())
            current["ids"].append(s["sample"])
        current["terminal"].append(s["sample"])

    def verdict_entropy(ids):
        passed = sum(by_id[i]["score"] for i in ids)
        return entropy([passed, len(ids) - passed])

    def compress(current, label):
        words = [label] if label else []
        while len(current["children"]) == 1 and not current["terminal"]:
            word, current = next(iter(current["children"].items()))
            words.append(word)
        groups = [c["ids"] for c in current["children"].values()]
        if current["terminal"]:
            groups.append(current["terminal"])
        n = len(current["ids"])
        ig = verdict_entropy(current["ids"]) - sum(len(g) / n * verdict_entropy(g) for g in groups) if n else 0
        return {
            "label": " ".join(words) or "(root)",
            "n": n,
            "passed": sum(by_id[i]["score"] for i in current["ids"]),
            "h": entropy([len(g) for g in groups]),
            "ig": max(0.0, ig),
            "terminal": current["terminal"],
            "children": [compress(c, w) for w, c in sorted(current["children"].items())],
        }

    return compress(root, "")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def samples_for(rows, identifier):
    matched = [r for r in rows if r["id"] == identifier]
    if len(matched) == 1 and "sampled" in matched[0]:
        # Adapter for the original GSM8K eval files; preserve their exact verdict.
        matched = [
            {**s, "sample": i, "response": s["text"], "score": s["correct"]}
            for i, s in enumerate(matched[0]["sampled"])
        ]
    keys = ("sample", "response", "answer", "code", "tokens", "finish", "cap", "score", "prompt_hash", "prompt_tokens")
    out = []
    for row in matched:
        if row["score"] not in (0, 1, False, True):
            raise ValueError(f"Nonbinary verdict: {identifier}")
        s = {k: row[k] for k in keys if k in row}
        s["score"] = int(s["score"])
        s["available_views"] = ["response", *[v for v in ("answer", "code") if v in row]]
        for key in VIEWS:
            s.setdefault(key, "")
        out.append(s)
    return sorted(out, key=lambda s: s["sample"])


def validate_pair(before, after, count):
    for group in (before, after):
        if [s["sample"] for s in group] != list(range(count)):
            raise ValueError("Missing or duplicate samples; expected IDs 0 through samples-1")
    for key in ("prompt_hash", "prompt_tokens", "cap"):
        values = [s.get(key) for s in before + after]
        if len(set(values)) != 1:
            raise ValueError(f"Prompt/generation contract changed: {key}")


def make_gallery(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    sources = {}
    cache = {}

    def read(relative, jsonl=False):
        path = (manifest_path.parent / relative).resolve()
        if path not in cache:
            raw = path.read_bytes()
            sources[str(path)] = hashlib.sha256(raw).hexdigest()
            cache[path] = read_jsonl(path) if jsonl else json.loads(raw)
        return cache[path]

    contracts = {stage: read(path) for stage, path in manifest.get("contracts", {}).items()}
    if contracts:
        if set(contracts) != {"before", "after"}:
            raise ValueError("Both checkpoint contracts are required")
        keys = (
            "tokenizer",
            "template",
            "flags",
            "data_hash",
            "seed",
            "temperature",
            "top_p",
            "top_k",
            "samples",
            "cap",
        )
        for key in keys:
            if key not in contracts["before"] or contracts["before"][key] != contracts["after"].get(key):
                raise ValueError(f"Evaluation contracts differ or omit {key}")
        if contracts["before"]["samples"] != manifest["samples"]:
            raise ValueError("Sample count disagrees with evaluation contract")
    prompts = {r["id"]: r["query"] for r in read(manifest["prompts"], jsonl=True)} if "prompts" in manifest else {}
    cases = []
    for spec in manifest["cases"]:
        pair = {stage: samples_for(read(spec[stage], jsonl=True), spec["id"]) for stage in ("before", "after")}
        validate_pair(pair["before"], pair["after"], manifest["samples"])
        case = {k: v for k, v in spec.items() if k not in ("before", "after")}
        case["prompt"] = spec.get("prompt", prompts.get(spec["id"]))
        if not case["prompt"]:
            raise ValueError(f"Missing prompt for {spec['id']}")
        case["stages"] = {}
        for stage, samples in pair.items():
            views = [v for v in VIEWS if any(v in s["available_views"] for s in pair["before"] + pair["after"])]
            case["stages"][stage] = {
                "samples": samples,
                "trees": {v: {str(d): build_tree(samples, v, d) for d in DEPTHS} for v in views},
                "summary": {
                    "passed": sum(s["score"] for s in samples),
                    "n": len(samples),
                    "mean_tokens": sum(s["tokens"] for s in samples) / len(samples),
                    "capped": sum(s.get("finish") == "length" for s in samples),
                    "no_answer": sum(not s["answer"].strip() for s in samples) if "answer" in views else None,
                },
            }
        cases.append(case)
    return {
        "title": manifest["title"],
        "labels": manifest["labels"],
        "contract_label": manifest["contract_label"],
        "cases": cases,
        "provenance": {"manifest": str(manifest_path.resolve()), "sha256": sources, "contracts": contracts},
    }


def render_gallery(data, template):
    # Gzip only reduces transfer size: the inspectable JSON and all full responses
    # remain saved beside the gallery. No network request is needed to open it.
    packed = base64.b64encode(gzip.compress(json.dumps(data, ensure_ascii=False).encode(), mtime=0)).decode()
    return template.replace("__RESPONSE_TREE_DATA__", packed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = make_gallery(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "response-trees.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    template = Path(__file__).with_name("assets") / "response_trees.html"
    fragment = render_gallery(data, template.read_text())
    if len(fragment.encode()) >= 1_000_000:
        raise ValueError("Inline gallery exceeds 1 MB; select fewer examples")
    target = args.output / "rl-response-trees.html"
    target.write_text(fragment)
    print(f"Wrote {len(data['cases'])} cases to {target} ({len(fragment.encode()):,} bytes)")


if __name__ == "__main__":
    main()
