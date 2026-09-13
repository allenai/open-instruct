"""Audit retained policy-refresh responses without a serving runtime."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch


def check_transition(before, response):
    tokens = response["output_ids"]
    meta = response["meta_info"]
    old_count = len(before["output_ids"])
    if not 0 < old_count < len(tokens):
        raise ValueError("Transition must retain a nonempty prefix and generate a nonempty suffix")
    scores = meta["output_token_logprobs"]
    if len(scores) != len(tokens) or [s[1] for s in scores] != tokens:
        raise ValueError("Behavior log-probabilities must align one-to-one with output tokens")
    if tokens[:old_count] != before["output_ids"] or [s[0] for s in scores[:old_count]] != before["behavior_logprobs"]:
        raise ValueError("Original behavior tokens or log-probabilities were overwritten")
    if not np.isfinite([s[0] for s in scores]).all():
        raise ValueError("Non-finite behavior log-probability")
    spans = meta.get("weight_versions", [])
    if (
        len(spans) != 2
        or spans[0]["start"] != 0
        or spans[0]["end"] != old_count
        or spans[1]["start"] != old_count
        or spans[1]["end"] != len(tokens)
        or spans[0]["version"] == spans[1]["version"]
    ):
        raise ValueError("Policy-version spans do not match the observed pause boundary")
    return old_count


def routes(response, shape):
    raw = base64.b64decode(response["meta_info"]["routed_experts"], validate=True)
    values = np.frombuffer(raw, dtype=np.int32)
    width = int(np.prod(shape[1:]))
    if not width or values.size % width:
        raise ValueError("Malformed expert route payload")
    return values.reshape(-1, *shape[1:])


def analyze(root):
    root = Path(root)
    report = []
    for path in sorted((root / "trace").glob("*-0.json")):
        before = json.loads(path.read_text())
        rid = before["rid"]
        response = json.loads((root / f"{rid}.json").read_text())
        reference = json.loads((root / f"{rid}-reference.json").read_text())
        kept = check_transition(before, response)
        old = torch.load(path.with_suffix(".routes.pt"), map_location="cpu", weights_only=True).numpy()
        current = routes(response, old.shape)
        fresh = routes(reference, old.shape)
        prefix_rows = len(before["prompt"]) + kept - 1
        if old.shape[0] != prefix_rows or current.shape[0] != len(before["prompt"]) + len(response["output_ids"]) - 1:
            raise ValueError("Expert route rows do not align with forwarded token positions")
        # These prefixes have exactly the same token IDs, unlike later greedy
        # continuations that could diverge due to numerical ties.
        report.append(
            {
                "rid": rid,
                "old_tokens": kept,
                "new_tokens": len(response["output_ids"]) - kept,
                "original_behavior_preserved": True,
                "refreshed_vs_old_prefix_route_agreement": float((current[:prefix_rows] == old).mean()),
                "refreshed_vs_fresh_prefill_prefix_route_agreement": float(
                    (current[:prefix_rows] == fresh[:prefix_rows]).mean()
                ),
            }
        )
    if not report:
        raise ValueError("No interrupted responses found")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / "transition-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
