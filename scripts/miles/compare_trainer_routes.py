"""Compare fixed-token native scorer routes with their own SGLang observations."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from scripts.miles import compare_update_zero, update_zero_capture


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def logical_routes(routes, backend):
    # The original olmo-direct Megatron architecture splits each HF block into
    # an even sequence layer and an odd MLP layer (see exporter.output_names).
    # The v1 native capture faithfully records those physical module indices.
    if backend == "olmo_core":
        return routes
    result = {}
    for physical, value in routes.items():
        logical, parity = divmod(physical, 2)
        compare_update_zero.require(parity == 1, "Megatron router must belong to an odd physical MLP layer")
        result[logical] = value
    return result


def load_trainers(root, cohort, backend):
    payload = json.loads((root / f"trainer-{cohort}-inputs.json").read_text())
    expected = {x["case_id"]: x for x in payload["cases"]}
    found, provenance = {}, []
    files = sorted((root / "trainer" / cohort).glob(f"trainer-{backend}-rank*.json"))
    compare_update_zero.require(len(files) == 2, "Expected both EP2 native scorer rank files")
    ranks = []
    for summary_path in files:
        summary = json.loads(summary_path.read_text())
        path = summary_path.with_suffix(".pt")
        compare_update_zero.require(compare_update_zero.sha256(path) == summary["sha256"], "Native file hash mismatch")
        data = torch.load(path, map_location="cpu", weights_only=True)
        compare_update_zero.require(data["backend"] == backend and data["world_size"] == 2, "Native topology differs")
        compare_update_zero.require(
            data["optimizer_calls"] == 0 and data["payload_sha256"] == digest(payload),
            "Native input provenance differs",
        )
        ranks.append(data["rank"])
        provenance.append(
            {"summary": summary, "runtime": data.get("runtime"), "router_parameters": data["router_parameters"]}
        )
        for case in data["cases"]:
            case_id = case["case_id"]
            compare_update_zero.require(
                case_id in expected and case_id not in found, "Duplicate or unexpected native case"
            )
            compare_update_zero.require(
                case["input_ids_sha256"] == digest(expected[case_id]["input_ids"]), "Native token hash mismatch"
            )
            compare_update_zero.require(
                case["loss_mask"] == expected[case_id]["loss_mask"], "Native response mask mismatch"
            )
            found[case_id] = dict(case, routes=logical_routes(case["routes"], backend))
    compare_update_zero.require(
        sorted(ranks) == [0, 1] and found.keys() == expected.keys(), "Incomplete native cohort"
    )
    return payload, found, provenance


def serving_scores(response, case):
    tokens = case["input_ids"]
    entries = response["meta_info"]["input_token_logprobs"]
    compare_update_zero.require(
        len(entries) == len(tokens) and [x[1] for x in entries] == tokens,
        "Serving input-logprob token alignment differs",
    )
    start = len(tokens) - case["response_length"]
    scores = torch.tensor([x[0] for x in entries[start:]], dtype=torch.float32)
    compare_update_zero.require(torch.isfinite(scores).all(), "Nonfinite serving response scores")
    return scores


def aligned_route(route, positions=None):
    ids, weights, logits = (route[name] for name in ("topk_ids", "topk_weights", "logits"))
    if positions is not None:
        ids, weights, logits = ids[positions], weights[positions], logits[positions]
    order = ids.argsort(-1)
    return update_zero_capture.route_record(logits, ids.gather(-1, order), weights.gather(-1, order))


def compare_case(native, serving, response, case):
    positions = native["positions"]
    expected_positions = update_zero_capture.request_positions(case["input_ids"])
    compare_update_zero.require(positions == expected_positions, "Native observed token positions differ")
    serving_layers = {int(name.split(".")[2]): value for name, value in serving["routes"].items()}
    compare_update_zero.require(
        native["routes"].keys() == serving_layers.keys(), "Native/serving routed-layer inventory differs"
    )
    layers = {}
    for layer, route in native["routes"].items():
        delta = compare_update_zero.route_difference(
            aligned_route(serving_layers[layer], positions), aligned_route(route)
        )
        delta["changed_set_input_positions"] = [positions[i] for i in delta["changed_set_positions"]]
        delta["weight_comparison_note"] = (
            "Expert-ID-sorted slots; differing selected sets still require membership-aware interpretation."
        )
        layers[layer] = delta
    actual, expected = native["log_probs"].float(), serving_scores(response, case)
    compare_update_zero.require(actual.shape == expected.shape, "Scorer response axis differs")
    target_positions = list(range(len(case["input_ids"]) - len(actual), len(case["input_ids"])))
    active = torch.tensor(case["loss_mask"], dtype=torch.bool)
    return {
        "case_id": case["case_id"],
        "input_ids_sha256": digest(case["input_ids"]),
        "input_length": len(case["input_ids"]),
        "observed_input_positions": positions,
        "response_length": case["response_length"],
        "layers": layers,
        "response_log_probs": compare_update_zero.tensor_difference(expected, actual, target_positions),
        "active_response_log_probs": compare_update_zero.tensor_difference(
            expected[active],
            actual[active],
            [p for p, keep in zip(target_positions, active.tolist(), strict=True) if keep],
        )
        if active.any()
        else None,
        "native_alignment_max_abs": native["alignment_max_abs"],
        "native_controls": native.get("controls"),
    }


def collect(root, backend):
    root = Path(root)
    completion = json.loads((root / "trainer-route-complete.json").read_text())
    compare_update_zero.require(
        completion.get("completed") and completion.get("optimizer_calls") == 0 and completion.get("backwards") == 0,
        "Native scorer protocol is incomplete",
    )
    output = {"root": str(root), "backend": backend, "cohorts": {}, "completion": completion}
    for cohort, phase in (("prefixes", "published"), ("retained", "retained")):
        payload, records, provenance = load_trainers(root, cohort, backend)
        cases = []
        for case in payload["cases"]:
            capture_id = f"{phase}-{case['case_id']}-capture"
            serving = compare_update_zero.load_capture(root, capture_id, case["input_ids"])
            response = json.loads((root / f"{capture_id}-response.json").read_text())
            observer = compare_update_zero.response_control(root, phase, case["case_id"])
            compare_update_zero.require(
                observer["same_output_text"] and observer["exact_logprob_fields"],
                "Serving capture changed its control response",
            )
            cases.append(compare_case(records[case["case_id"]], serving, response, case))
        output["cohorts"][cohort] = {
            "cases": cases,
            "native_provenance": provenance,
            "layer_mapping": "Core block i maps to HF i; original Megatron physical MLP 2*i+1 maps to HF i",
            "payload_sha256": digest(payload),
        }
    output["observations_verified"] = True
    output["cleanup"] = json.loads((root / "cleanup.json").read_text()) if (root / "cleanup.json").is_file() else None
    output["scope"] = (
        "Same-token, rank-strided diagnostic forwards; sampled input-position routes and full response LPs. Differences are measurements, not automatic correctness failures."
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--backend", choices=("olmo_core", "megatron"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    marker = args.root / "trainer-route-complete.json"
    while not marker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Native route completion marker not ready: {marker}")
        print(json.dumps({"waiting_for": str(marker)}), flush=True)
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    torch.set_num_threads(1)
    args.output.write_text(json.dumps(collect(args.root, args.backend), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
