"""Resume fresh-server checks from five saved real Core publications.

This diagnostic never republishes or changes weights, and never writes qualified.json.
It retains every failed replica comparison rather than stopping at the first one.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import httpx
import torch
from scripts.miles import qualify_math_weight_publication as gate


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_subset(actual, reference):
    """Compare all historically saved fields; newer audits also include norms."""
    if isinstance(reference, dict):
        return isinstance(actual, dict) and all(
            key in actual and audit_subset(actual[key], value) for key, value in reference.items()
        )
    if isinstance(reference, list):
        if reference and isinstance(reference[0], dict) and "name" in reference[0]:
            by_name = {row["name"]: row for row in actual}
            return len(by_name) == len(actual) and all(
                row["name"] in by_name and audit_subset(by_name[row["name"]], row) for row in reference
            )
        return len(actual) == len(reference) and all(
            audit_subset(a, b) for a, b in zip(actual, reference, strict=True)
        )
    return actual == reference


def load_publications(folder, model, precision, runtime_lock):
    protocol = json.loads((folder / "protocol.json").read_text())
    if Path(protocol["source"]).resolve() != model.resolve() or protocol["precision"] != precision:
        raise ValueError("Saved publication model/precision mismatch")
    if protocol["source_model_sha256"] != digest(model / "model.safetensors"):
        raise ValueError("Saved publication model hash mismatch")
    if protocol["runtime_lock_sha256"] != digest(runtime_lock):
        raise ValueError("Saved publication runtime mismatch")
    if protocol["variants"] != [0, 1, 2, 2, 0] or protocol["tolerances"] != {
        key: list(value) for key, value in gate.TOLERANCES.items()
    }:
        raise ValueError("Saved publication protocol mismatch")
    progress = json.loads((folder / "publication-progress.json").read_text())
    if len(progress) != 5:
        raise ValueError("Require all five saved publications")
    paths = [folder / "protocol.json", folder / "publication-progress.json", folder / "initial.json"]
    initial = json.loads(paths[-1].read_text())
    observed = {}
    for version, variant in enumerate(protocol["variants"], 1):
        report = progress[version - 1]
        if (report["version"], report["variant"], report["tensors"]) != (version, variant, 23441):
            raise ValueError("Incomplete publication name coverage")
        expected_split = {
            f"model.layers.{layer}.mlp.experts.0.{projection}.weight"
            for layer in range(1, 16)
            for projection in ("down_proj", "gate_proj", "up_proj")
        }
        if set(report["forced_split_names"]) != expected_split or report["buckets"] < 45:
            raise ValueError("Missing split-bucket coverage")
        path = folder / f"publication-{version}-variant-{variant}.json"
        current = json.loads(path.read_text())
        paths.append(path)
        if len(current["outputs"]) != 3 or len(current["audits"]) != 3:
            raise ValueError("Missing publication replica")
        reference = initial if variant == 0 else observed.get(variant)
        if reference is not None:
            if current["audits"] != reference["audits"]:
                raise ValueError("Saved repeated/restored parameter mismatch")
            for a, b in zip(current["outputs"], reference["outputs"], strict=True):
                gate.compare_scores(a, b, precision, unchanged=precision == "bf16")
        observed[variant] = current
    return protocol, observed, {str(p): digest(p) for p in paths}


def saved_full_cap_inputs(outputs, probes):
    """Recover exact frozen conditional requests; never regenerate continuations."""
    if len(outputs) != 25 or len(probes) != 16:
        raise ValueError("Missing full-cap cases or prompt snapshot")
    cases, offset = [], 0
    for count in (1, 8, 16):
        requests = []
        for output, prefix in zip(outputs[offset : offset + count], probes[:count], strict=True):
            meta = output["meta_info"]
            tokens = [entry[1] for entry in meta["input_token_logprobs"]]
            if (
                meta["completion_tokens"] != 16384
                or not meta.get("candidate20_sha256")
                or tokens[: len(prefix)] != prefix
                or len(tokens) != len(prefix) + 16384
            ):
                raise ValueError("Invalid saved full-cap conditional provenance")
            requests.append(tokens)
        cases.append(requests)
        offset += count
    return cases


def probe_saved_full_cap(fleet, label, observed, probes):
    """Score the original c1/8/16 long inputs on fresh servers, without decoding."""
    outputs, audits = [], []
    with httpx.Client(timeout=1800) as client:
        for index, (engine, directory) in enumerate(zip(fleet.clients, fleet.directories, strict=True)):
            rows = []
            cases = saved_full_cap_inputs(observed["outputs"][index], probes)
            for tokens in cases:
                response = client.post(
                    engine.server_url + "/generate",
                    json={
                        "input_ids": tokens,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                        "return_logprob": True,
                        "logprob_start_len": 0,
                    },
                )
                response.raise_for_status()
                current = response.json()
                if not isinstance(current, list) or len(current) != len(tokens):
                    raise ValueError("Missing saved-conditional diagnostic responses")
                for output, request in zip(current, tokens, strict=True):
                    if [x[1] for x in output["meta_info"]["input_token_logprobs"]] != request:
                        raise ValueError("Saved-conditional diagnostic token alignment mismatch")
                rows.extend(current)
            outputs.append(rows)
            paths = sorted(directory.glob("audit-*.json"), key=lambda path: path.stat().st_mtime_ns)
            audit = json.loads(paths[-1].read_text())
            if getattr(fleet.args, "nvfp4_kv", False):
                pools = list(directory.glob("pool-audit-*.json"))
                if len(pools) != 1:
                    raise ValueError("Missing actual native KV/recurrent audit")
                gate.qualify_math_nvfp4_kv.verify_audit(json.loads(pools[0].read_text()))
                gate.qualify_math_quantization.verify_inventory(
                    [module for module in audit if module["class"] != "RadixAttention"], fleet.args.precision
                )
            audits.append(audit)
    gate.save(fleet.args.output / f"{label}.json", {"outputs": outputs, "audits": audits})
    return outputs, audits


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    if getattr(args, "native_config_receipt", None):
        os.environ["OI_PUBLICATION_NATIVE_CONFIG_RECEIPT"] = str(args.native_config_receipt.resolve())
    gate.configure_transport_environment()
    if torch.cuda.device_count() != 4 or any("B300" not in torch.cuda.get_device_name(i) for i in range(4)):
        raise ValueError("Requires exactly four approved B300 GPUs")
    protocol, observed, hashes = load_publications(
        args.resume_publications, args.model, args.precision, Path("/opt/core-rl/sources/runtime.lock.json")
    )
    gate.save(
        args.output / "resume-receipt.json",
        {
            "source": str(args.resume_publications),
            "artifacts_sha256": hashes,
            "publications_reused": 5,
            "publications_rerun": 0,
            "runtime_pins_unchanged": True,
            "tolerances_unchanged": gate.TOLERANCES,
            "diagnostic_only": True,
            "saved_full_cap_scores": getattr(args, "saved_full_cap_scores", False),
            "nvfp4_kv": getattr(args, "nvfp4_kv", False),
        },
    )
    changed = {0: args.model, **{v: gate.fresh_checkpoint(args, v) for v in (1, 2)}}
    checks, failures = [], []
    for variant in args.fresh_variants:
        with gate.Fleet(args, changed[variant], f"fresh-{variant}") as fleet:
            if getattr(args, "saved_full_cap_scores", False):
                if not protocol.get("full_cap_prompt_snapshot_sha256"):
                    raise ValueError("Saved full-cap prompt provenance missing")
                reference, audits = probe_saved_full_cap(
                    fleet, f"fresh-{variant}", observed[variant], protocol["probes"]
                )
                repeated, repeated_audits = probe_saved_full_cap(
                    fleet, f"fresh-{variant}-repeat", observed[variant], protocol["probes"]
                )
            else:
                reference, audits = fleet.probe(f"fresh-{variant}", protocol["probes"])
                repeated, repeated_audits = fleet.probe(f"fresh-{variant}-repeat", protocol["probes"])
        if getattr(args, "native_config_receipt", None):
            for directory in fleet.directories:
                source = json.loads(args.native_config_receipt.read_text())
                directories = [directory]
                if source.get("additional_interventions"):
                    directories.append(directory / "gate-native-intervention")
                for destination in directories:
                    receipt = json.loads((destination / "native-intervention-post.json").read_text())
                    if receipt["matched_calls"] < 2:
                        raise ValueError("Recorded native key was not exercised repeatedly")
        gate.save(
            args.output / f"fresh-{variant}-outputs.json",
            {"diagnostic_only": True, "outputs": reference, "repeated": repeated, "audits": audits},
        )
        for rank in range(3):
            record = {"variant": variant, "replica": rank, "passed": False}
            try:
                if not audit_subset(audits[rank], observed[variant]["audits"][rank]):
                    raise ValueError("Fresh/published parameter hashes differ")
                if audits[rank] != repeated_audits[rank]:
                    raise ValueError("Fresh repetition changed parameter hashes")
                gate.compare_scores(
                    repeated[rank], reference[rank], args.precision, unchanged=args.precision == "bf16"
                )
                record["comparison"] = gate.compare_scores(
                    observed[variant]["outputs"][rank],
                    reference[rank],
                    args.precision,
                    unchanged=variant == 0 and args.precision == "bf16",
                )
                record["passed"] = True
            except ValueError as error:
                record["error"] = str(error)
                failures.append(record)
            checks.append(record)
        gate.save(args.output / "fresh-check-progress.json", checks)
    gate.save(
        args.output / "diagnostic-complete.json",
        {
            "passed": not failures,
            "checks": checks,
            "diagnostic_only": True,
            "note": "Full current audits include non-quantized modules missing from historical audits. No refresh qualification receipt is created.",
        },
    )
    if failures:
        raise ValueError(f"{len(failures)} saved-publication/fresh replica checks failed; see preserved diagnostics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "output", "resume-publications", "reuse-mutation-checkpoints"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "nvfp4"), required=True)
    parser.add_argument("--native-config-receipt", type=Path)
    parser.add_argument("--saved-full-cap-scores", action="store_true")
    parser.add_argument("--nvfp4-kv", action="store_true")
    parser.add_argument("--fresh-variants", type=int, choices=(0, 1, 2), nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    args.gpu, args.port = None, None
    run(args)


if __name__ == "__main__":
    main()
