"""Exact reward-client replay of the sealed H086 control bundle; no policy scoring."""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import requests

from open_instruct.miles.rewards import h080, if_gate_yes, judge_server

ROOT = Path("/weka/oe-eval-default/abhishekr/h086")
MANIFEST_SHA = "0d3ef7daee21c60d174496533e1f41d13ea0cd5b1775f21b44c0316c00c97244"
LABELS_SHA = "46a1814268cb8874d3db608763d549a141a13dea3c8678321104c237c275190f"


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_inputs(requests_path, labels_path, *, manifest_sha=MANIFEST_SHA, labels_sha=LABELS_SHA):
    manifest_bytes = requests_path.with_name("manifest.json").read_bytes()
    if sha(manifest_bytes) != manifest_sha or sha(labels_path.read_bytes()) != labels_sha:
        raise ValueError("H086 manifest/labels pin mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest["configuration"]["service"] != h080.SERVICE:
        raise ValueError("H086 service configuration pin mismatch")
    raw = requests_path.read_bytes()
    if sha(raw) != manifest["artifacts"]["requests.jsonl"]:
        raise ValueError("H086 requests artifact hash mismatch")
    reqs = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(reqs) != 540 or [r.get("control") for r in reqs] != ["b2"] * 200 + ["a"] * 300 + ["e"] * 40:
        raise ValueError("expected 200 fresh, 300 off-topic, 40 hand controls in stored order")
    indexed = {r["request_id"]: r for r in reqs}
    if len(indexed) != 540:
        raise ValueError("duplicate H086 request id")
    labels = {}
    for line in labels_path.read_text().splitlines():
        r = json.loads(line)
        key = r["request_id"]
        if key in labels or key not in indexed or r["request_sha256"] != indexed[key]["sha256"]:
            raise ValueError("duplicate, unknown or mismatched H086 label")
        if r["label"] not in ("yes", "partial", "no") or r["parse_failure"]:
            raise ValueError("null/missing H086 labels cannot qualify")
        labels[key] = r
    if set(labels) != set(indexed):
        raise ValueError("missing H086 labels")
    for r in reqs:
        identity = [r["cell"], r["task"], r["native_id"], r["draw"], r.get("control"), r.get("duplicate_of")]
        if sha(encoded(identity)) != r["request_id"]:
            raise ValueError("H086 request id mismatch")
        if sha(encoded({k: v for k, v in r.items() if k != "sha256"})) != r["sha256"]:
            raise ValueError("H086 request hash mismatch")
        rendered = h080.render_prompt(r["conversation"], r["response"])
        if rendered.encode() != r["prompt"].encode() or sha(rendered.encode()) != sha(r["prompt"].encode()):
            raise ValueError(f"adapter prompt differs: {r['request_id']}")
    return reqs, labels, manifest


async def replay(reqs, labels, client):
    semaphore = asyncio.Semaphore(32)

    async def grade(row):
        async with semaphore:
            try:
                record = await client.grade(row["conversation"], row["response"])
            except Exception as error:
                record = {"label": None, "status": "fatal", "error": f"{type(error).__name__}: {error}"}
        label = record["label"]
        return {
            "request_id": row["request_id"],
            "request_sha256": row["sha256"],
            "expected": labels[row["request_id"]]["label"],
            "identical": label is not None and label == labels[row["request_id"]]["label"],
            **record,
        }

    results = await asyncio.gather(*(grade(r) for r in reqs))
    identical = sum(r["identical"] for r in results)
    return {
        "PASS": len(results) == 540 and identical == 540,
        "requests": len(results),
        "identical": identical,
        "results": results,
    }


async def qualify(
    service,
    endpoint,
    report_path,
    *,
    requests_path=ROOT / "requests/requests.jsonl",
    labels_path=ROOT / "run/labels.jsonl",
):
    # New process => new replay, never resume a partial qualification.
    report_path = Path(report_path)
    if report_path.exists():
        raise FileExistsError(f"qualification report already exists: {report_path}")
    config = if_gate_yes.client_config(service, endpoint)
    try:
        reqs, labels, manifest = load_inputs(Path(requests_path), Path(labels_path))
        report = await replay(reqs, labels, if_gate_yes.VerdictClient(config))
        report["requests_sha256"] = manifest["artifacts"]["requests.jsonl"]
    except Exception as error:
        report = {"PASS": False, "requests": 0, "identical": 0, "error": f"{type(error).__name__}: {error}"}
    report.update(
        manifest_sha256=MANIFEST_SHA,
        labels_sha256=LABELS_SHA,
        prompt_sha256=h080.PROMPT_SHA256,
        service=service,
        client=dataclasses.asdict(config),
        code_sha256={
            p.name: sha(p.read_bytes())
            for p in (
                Path(__file__),
                *(
                    Path(__file__).with_name(name + ".py")
                    for name in (
                        "if_gate_yes",
                        "h080",
                        "judge_server",
                        "general_judge",
                        "judge_registry",
                        "judge_metrics",
                        "rewards",
                        "service",
                    )
                ),
                Path(__file__).parents[1] / "infrastructure/infra_timeouts.py",
            )
        },
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    if not report["PASS"]:
        raise RuntimeError(f"adapter qualification failed: {report['identical']}/540; see {report_path}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, default=ROOT / "requests/requests.jsonl")
    parser.add_argument("--labels", type=Path, default=ROOT / "run/labels.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="false")
    env.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
    # Same immutable command and environment as execution.cluster's managed judge.
    with (args.out / "judge-server.log").open("x") as log:
        proc = subprocess.Popen(
            judge_server.command(h080.SERVICE, args.port), stdout=log, stderr=subprocess.STDOUT, env=env
        )
        try:
            started = time.monotonic()
            while time.monotonic() - started < 3600:
                if proc.poll() is not None:
                    raise RuntimeError("judge exited; see judge-server.log")
                try:
                    if requests.get(f"http://127.0.0.1:{args.port}/health", timeout=5).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(5)
            else:
                raise RuntimeError("judge startup exceeded 3600 seconds")
            report = asyncio.run(
                qualify(
                    h080.SERVICE,
                    f"http://127.0.0.1:{args.port}/v1",
                    args.out / "qualification.json",
                    requests_path=args.requests,
                    labels_path=args.labels,
                )
            )
            print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
