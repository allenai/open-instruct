"""Audit trusted, locally archived production rollout tensors on CPU.

This checks saved payload consistency, not independent before/after-refresh
preservation or numerical equivalence. Only load artifacts from trusted runs:
the native torch files contain Python objects and require weights_only=False.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def version(value):
    if type(value) not in (int, str) or not str(value).isascii() or not str(value).isdigit():
        raise ValueError("Invalid policy version")
    return int(value)


def audit_sample(sample, model):
    tokens, length = sample["tokens"], sample["response_length"]
    if not 0 < length < len(tokens) or sample["status"] not in ("completed", "truncated"):
        raise ValueError("Unfinished sample or invalid prompt/response lengths")
    scores = np.asarray(sample["rollout_log_probs"])
    if scores.shape != (length,) or not np.isfinite(scores).all():
        raise ValueError("Missing, nonfinite or misaligned behavior log probabilities")
    provenance = sample["metadata"]["policy_refresh"]
    if sample["train_metadata"]["policy_refresh"] != provenance:
        raise ValueError("Serving and training provenance disagree")
    replay = version(provenance["replay_version"])
    spans = provenance["spans"]
    cursor, previous = 0, -1
    canonical = []
    for span in spans:
        number = version(span["version"])
        start, end = span["start"], span["end"]
        if type(start) is not int or type(end) is not int or start != cursor or not start < end <= length:
            raise ValueError("Token spans have a gap, overlap or invalid boundary")
        if not previous <= number <= replay:
            raise ValueError("Token versions regress or exceed the final replay version")
        canonical.append(dict(version=number, start=start, end=end))
        cursor, previous = end, number
    if cursor != length or previous != replay:
        raise ValueError("Token spans do not cover the response through its final replay version")
    offset = len(tokens) - length
    absolute = [
        dict(version=version(s["version"]), start=s["abs_start"] - offset, end=s["abs_end"] - offset)
        for call in sample["weight_versions"]
        for s in call
    ]
    if absolute != canonical:
        raise ValueError("Token spans disagree with the staleness ledger")
    routes = sample["rollout_routed_experts"]
    if model["model_type"] != "olmo3moe":
        raise ValueError("Route layout audit currently supports olmo3moe only")
    expected = (len(tokens) - 1, model["num_hidden_layers"], model["num_experts_per_tok"])
    if not isinstance(routes, np.ndarray) or routes.dtype != np.int32 or routes.shape != expected:
        raise ValueError("Route payload has incorrect dtype or token/layer/top-k shape")
    dense = model.get("dense_layers_indices") or []
    if any(type(layer) is not int or not 0 <= layer < expected[1] for layer in dense):
        raise ValueError("Invalid dense-layer layout")
    layers = [layer for layer in range(expected[1]) if layer not in dense]
    if not layers:
        raise ValueError("No routed layers to audit")
    for start in range(0, len(routes), 1024):
        selected = routes[start : start + 1024, layers]
        if (selected < 0).any() or (selected >= model["n_routed_experts"]).any():
            raise ValueError("Routing payload contains an invalid expert ID")
        ordered = np.sort(selected, axis=-1)
        if (ordered[..., 1:] == ordered[..., :-1]).any():
            raise ValueError("Routing payload contains duplicate experts for one token/layer")
    return {
        "index": sample["index"],
        "group_index": sample["group_index"],
        "prompt_tokens": offset,
        "response_tokens": length,
        "status": sample["status"],
        "first_version": canonical[0]["version"],
        "replay_version": replay,
        "mixed": len({span["version"] for span in canonical}) > 1,
        "historical_prefix_tokens": sum(s["end"] - s["start"] for s in canonical if s["version"] < replay),
        "route_bytes": routes.nbytes,
        "route_sha256": hashlib.sha256(routes.tobytes()).hexdigest(),
    }


def audit(root, model, output):
    root, output = Path(root), Path(output)
    if output == root or root.resolve() in output.resolve().parents:
        raise ValueError("Keep audit output outside the source artifacts")
    output.mkdir(parents=True, exist_ok=False)
    flow = [json.loads(line) for line in (root / "checkpoints/rollout_flow.jsonl").read_text().splitlines()]
    results, identities = [], set()
    with (output / "samples.jsonl").open("w") as stream:
        for collection in flow:
            rollout_id = collection["rollout_id"]
            path = root / "rollouts" / f"{rollout_id}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["rollout_id"] != rollout_id:
                raise ValueError("Rollout file identity disagrees with consumed collection")
            records = [audit_sample(sample, model) for sample in payload["samples"]]
            counts = Counter(record["group_index"] for record in records)
            for record in records:
                identity = record["group_index"], record["index"]
                if identity in identities:
                    raise ValueError("Duplicate consumed sample identity")
                identities.add(identity)
                stream.write(json.dumps({"rollout_id": rollout_id, **record}) + "\n")
            summary = {
                "rollout_id": rollout_id,
                "samples": len(records),
                "response_tokens": sum(r["response_tokens"] for r in records),
                "mixed_responses": sum(r["mixed"] for r in records),
            }
            if any(summary[key] != collection[key] for key in ("samples", "response_tokens", "mixed_responses")):
                raise ValueError("Saved sample payloads disagree with consumed rollout accounting")
            with path.open("rb") as source:
                summary.update(file_sha256=hashlib.file_digest(source, "sha256").hexdigest())
            summary["group_sizes"] = dict(Counter(counts.values()))
            results.append(summary)
            del payload
    if not results:
        raise ValueError("No consumed rollout payloads")
    report = {
        "status": "saved_payload_consistency_passed",
        "model_configuration": model,
        "collections": results,
        "scope": "All consumed samples and full prompt/response route payloads. Dense route padding excluded. Does not prove original behavior scores survived refresh unchanged, engine acknowledgments, or training correctness; combine with publication, trainer and request-timing audits.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("model_config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = audit(args.root, json.loads(args.model_config.read_text()), args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
