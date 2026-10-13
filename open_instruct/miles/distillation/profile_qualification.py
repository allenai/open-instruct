"""Qualify matched scoring and measure bounded serving workloads."""

import asyncio
import gzip
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib import request

from olmo_sglang import kernel_profile
from transformers import AutoTokenizer

from open_instruct.miles.distillation import alignment, cancellation, hooks, numerical_audit, overnight, panel, teacher


def extend(tokens, length):
    if not tokens or length < 1:
        raise ValueError("Need nonempty tokens and a positive length")
    return (tokens * math.ceil(length / len(tokens)))[:length]


def workloads(items, tokenizer, saved):
    """Retain real reference contexts and label synthetic length stress separately."""
    ordered = sorted(items, key=lambda item: item["rendered"]["metadata"]["opd_teacher_prompt_tokens"])
    selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    cases = []
    for index, item in enumerate(selected):
        row = item["rendered"]
        for kind, text in (
            ("bare", row["input"]),
            ("answer", row["metadata"]["opd_answer_prompt"]),
            ("full", row["metadata"]["opd_teacher_prompt"]),
        ):
            prefix = tokenizer.encode(text, add_special_tokens=False)
            cases.append(
                dict(
                    name=f"{index}-{kind}",
                    prefix=prefix,
                    tokens=extend(saved["tokens"], 16384),
                    kind=kind,
                    sample_id=item["sample_id"],
                    synthetic_response=True,
                )
            )
    # SGLang reserves five positions and rejects equality at its input limit.
    for length in (1024, 8191, 8192, 8193, 16383, 16384, 16385, 32760, 65528):
        response = min(16384, length - 1)
        cases.append(
            dict(
                name=f"boundary-{length}",
                kind="boundary",
                synthetic_response=True,
                prefix=extend(saved["prefix"], length - response),
                tokens=extend(saved["tokens"], response),
            )
        )
    return cases, selected


def command(model, port, role):
    service = dict(
        source=model,
        tensor_parallel_size=1,
        max_context_length=32768 if role == "student" else 65536,
        concurrency=8 if role == "student" else 4,
    )
    cmd = teacher.command(service, port)
    cmd += ["--weight-version", "0", "--random-seed", "130064"]
    if role == "student":
        cmd.remove("--disable-cuda-graph")
        cmd.remove("--disable-radix-cache")
        cmd += [
            "--cuda-graph-backend-decode",
            "full",
            "--cuda-graph-max-bs-decode",
            "8",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--mamba-radix-cache-strategy",
            "extra_buffer",
        ]
        cmd[cmd.index("--max-total-tokens") + 1] = "524288"
        cmd[cmd.index("--max-mamba-cache-size") + 1] = "128"
    return cmd


@contextmanager
def server(model, device, role, mode, root, profile):
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", 0))
        port = sock.getsockname()[1]
    label = f"{mode}-{role}-{device}"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=device,
        OLMO_SGLANG_CORE_COMPAT="rounding",
        OLMO_SGLANG_REQUIRE_CORE_ROUNDING="1",
    )
    for key in (kernel_profile.PROFILE, kernel_profile.MODE, kernel_profile.DIGEST, kernel_profile.REPORT):
        env.pop(key, None)
    if mode in {"record", "strict"}:
        env.update(
            {
                kernel_profile.PROFILE: str(profile),
                kernel_profile.MODE: mode,
                kernel_profile.REPORT: str(root / f"{label}-usage.json"),
            }
        )
        if mode == "record":
            env[kernel_profile.REASON] = "Full-context profile preparation; no training"
        else:
            env.pop(kernel_profile.REASON, None)
            env[kernel_profile.DIGEST] = kernel_profile.digest(json.loads(profile.read_text()))
    cmd = command(model, port, role)
    numerical_audit.append(root / "commands.jsonl", dict(label=label, command=cmd))
    start = time.monotonic()
    with (root / f"{label}.log").open("w") as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        endpoint = f"http://127.0.0.1:{port}"
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{label} exited during startup")
                if time.monotonic() - start > 1200:
                    raise TimeoutError(f"{label} startup timeout")
                try:
                    with request.urlopen(endpoint + "/health", timeout=2):
                        break
                except OSError:
                    time.sleep(2)
            numerical_audit.append(root / "startup.jsonl", dict(label=label, seconds=time.monotonic() - start))
            yield endpoint
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def score(endpoint, cases):
    result = panel.post(
        endpoint + "/generate",
        {
            "input_ids": [case["prefix"] + case["tokens"] for case in cases],
            "sampling_params": {"max_new_tokens": 0, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
        timeout=900,
    )
    if not isinstance(result, list) or len(result) != len(cases):
        raise ValueError("Scoring batch has incorrect cardinality")
    return [
        alignment.extract_scores(output, case["prefix"] + case["tokens"], len(case["tokens"]))
        for case, output in zip(cases, result, strict=True)
    ]


def measure(endpoint, role, mode, cases, root):
    records = {}
    capacity = 32768 if role == "student" else 65536
    eligible = [case for case in cases if len(case["prefix"]) + len(case["tokens"]) <= capacity]
    # Flush between requests so timed repeats do not become cached-prefix lookups.
    for case in eligible:
        for repeat in range(2):
            panel.post(endpoint + "/flush_cache", {})
            start = time.monotonic()
            values = score(endpoint, [case])[0]
            elapsed = time.monotonic() - start
            name = case["name"]
            numerical_audit.append(
                root / "timings.jsonl",
                dict(
                    mode=mode,
                    role=role,
                    case=name,
                    batch=1,
                    repeat=repeat,
                    input_tokens=len(case["prefix"]) + len(case["tokens"]),
                    seconds=elapsed,
                    phase="first_request" if repeat == 0 else "warm",
                ),
            )
            if repeat:
                numerical_audit.append(
                    root / "comparisons.jsonl",
                    dict(
                        kind="repeat",
                        mode=mode,
                        role=role,
                        case=name,
                        **numerical_audit.compare(records[name], values),
                    ),
                )
            records[name] = values
    mixed = sorted(
        (case for case in eligible if case["kind"] != "boundary"),
        key=lambda case: len(case["prefix"]) + len(case["tokens"]),
    )
    for batch in (2, 4, 8) if role == "student" else (2, 4):
        chosen = [mixed[max(0, min(len(mixed) - 1, (i + 1) * len(mixed) // batch - 1))] for i in range(batch)]
        for repeat in range(3):
            panel.post(endpoint + "/flush_cache", {})
            start = time.monotonic()
            results = score(endpoint, chosen)
            elapsed = time.monotonic() - start
            numerical_audit.append(
                root / "timings.jsonl",
                dict(
                    mode=mode,
                    role=role,
                    case="mixed",
                    batch=batch,
                    repeat=repeat,
                    input_tokens=sum(len(c["prefix"]) + len(c["tokens"]) for c in chosen),
                    seconds=elapsed,
                    phase="first_request" if repeat == 0 else "warm",
                ),
            )
            for case, values in zip(chosen, results, strict=True):
                numerical_audit.append(
                    root / "comparisons.jsonl",
                    dict(
                        kind="single_vs_batch",
                        mode=mode,
                        role=role,
                        case=case["name"],
                        batch=batch,
                        **numerical_audit.compare(records[case["name"]], values),
                    ),
                )
    with gzip.open(root / f"scores-{mode}-{role}.json.gz", "wt") as stream:
        json.dump(records, stream, allow_nan=False)
    if role == "student":
        for repeat in range(2):
            panel.post(endpoint + "/flush_cache", {})
            start = time.monotonic()
            outputs = panel.post(
                endpoint + "/generate",
                {
                    "input_ids": [case["prefix"] for case in mixed[:1]] * 8,
                    "return_logprob": True,
                    "sampling_params": {
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "max_new_tokens": 1024,
                        "ignore_eos": True,
                        "sampling_seed": 130064,
                    },
                },
                timeout=900,
            )
            elapsed = time.monotonic() - start
            numerical_audit.append(
                root / "timings.jsonl",
                dict(
                    mode=mode,
                    role=role,
                    case="decode",
                    batch=8,
                    repeat=repeat,
                    output_tokens=sum(len(o["meta_info"]["output_token_logprobs"]) for o in outputs),
                    seconds=elapsed,
                    phase="first_request" if repeat == 0 else "warm",
                ),
            )
            with gzip.open(root / f"decode-{mode}-{repeat}.json.gz", "wt") as stream:
                json.dump(outputs, stream, allow_nan=False)
        for case in eligible:
            values = score(endpoint, [case])[0]
            numerical_audit.append(
                root / "comparisons.jsonl",
                dict(
                    kind="after_decode",
                    mode=mode,
                    role=role,
                    case=case["name"],
                    **numerical_audit.compare(records[case["name"]], values),
                ),
            )
    return records


async def exercise_hook(student, teacher_endpoint, row, tokens, tokenizer, model, root):
    service = dict(
        source=model,
        endpoint=teacher_endpoint,
        concurrency=4,
        request_timeout=900,
        max_context_length=65536,
        objective="cancel_answer_positive",
        kl_coef=1.0,
        calibrate_initial=True,
    )
    sample = SimpleNamespace(
        metadata=row["metadata"],
        response_length=len(tokens),
        tokens=tokenizer.encode(row["input"], add_special_tokens=False) + tokens,
        # Deliberately different from prefill: verify transport never overwrites behavior.
        rollout_log_probs=[-2.0] * len(tokens),
        train_metadata={},
        weight_versions=[SimpleNamespace(to_dicts=lambda: [{"version": "0"}])],
    )
    args = SimpleNamespace(
        hf_checkpoint=model,
        fully_async=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=int(student.rsplit(":", 1)[1]),
        opd_kl_coef=1.0,
    )
    old_env = os.environ.get("OI_MILES_OPD_TEACHER")
    old_grade = hooks.rewards.registered_reward

    async def grade(*args, **kwargs):
        return 0.0

    os.environ["OI_MILES_OPD_TEACHER"] = json.dumps(service)
    hooks.rewards.registered_reward = grade
    try:
        await hooks.reward(args, sample)
        assert sample.rollout_log_probs == [-2.0] * len(tokens)
        bare = sample.train_metadata["opd_student_prefill_log_probs"]
        null, _ = cancellation.token_pressure(
            bare,
            sample.train_metadata["opd_initial_bare_log_probs"],
            sample.train_metadata["opd_initial_bare_log_probs"],
        )
        numerical_audit.append(
            root / "hook-check.jsonl",
            dict(
                student_version=sample.train_metadata["opd_student_prefill_version"],
                anchor=sample.train_metadata["opd_pressure_anchor"],
                matched_gap=sample.train_metadata["opd_initial_matched_prefill_gap_abs_mean"],
                null_pressure_abs_mean=null.abs().mean().item(),
                behavior_preserved=True,
                synthetic_behavior_control=True,
                actual_pressure=sample.train_metadata["opd_pressure"],
            ),
        )
    finally:
        hooks.rewards.registered_reward = old_grade
        if old_env is None:
            os.environ.pop("OI_MILES_OPD_TEACHER", None)
        else:
            os.environ["OI_MILES_OPD_TEACHER"] = old_env
        hooks.resources.cache_clear()


def summarize(root):
    comparisons = list(overnight.rows(root / "comparisons.jsonl"))
    strict = [r for r in comparisons if r.get("mode") == "strict"]
    timings = list(overnight.rows(root / "timings.jsonl"))
    throughput = []
    for role in ("student", "teacher"):
        for case, batch in sorted({(r["case"], r["batch"]) for r in timings if r["role"] == role}):
            modes = {}
            for mode in ("baseline", "strict"):
                rows = [
                    r
                    for r in timings
                    if (r["role"], r["case"], r["batch"], r["mode"], r["phase"]) == (role, case, batch, mode, "warm")
                ]
                if rows:
                    modes[mode] = statistics.median(
                        r.get("output_tokens", r.get("input_tokens")) / r["seconds"] for r in rows
                    )
            throughput.append(
                dict(
                    role=role,
                    case=case,
                    batch=batch,
                    tokens_per_second=modes,
                    strict_over_baseline=modes["strict"] / modes["baseline"] if len(modes) == 2 else None,
                )
            )
    result = dict(
        comparisons=len(strict),
        all_exact=all(r["mae"] == 0 for r in strict),
        max_mae=max(r["mae"] for r in strict),
        throughput=throughput,
    )
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    root = Path(spec["output"]["root"])
    root.mkdir(parents=True, exist_ok=False)
    model = spec["model"]["source"]
    saved = json.loads(Path(spec["launch"]["env"]["OI_MILES_OPD_AUDIT_INPUT"]).read_text())
    if saved["same_checkpoint"] != model:
        raise ValueError("Qualification checkpoint differs from saved tokens")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(devices) != 2:
        raise ValueError("Full-context qualification requires two GPUs")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    items, provenance = overnight.candidates(tokenizer)
    cases, selected = workloads(items, tokenizer, saved)
    (root / "inputs.json").write_text(json.dumps(dict(cases=cases, provenance=provenance)))
    (root / "runtime.json").write_bytes(Path("/opt/core-rl/runtime/miles/runtime.lock.json").read_bytes())
    profile = root / "kernel-profile.json"
    # Recording has extra observation work. A fresh unpinned process provides the
    # ordinary-serving baseline; profile collection is timed separately.
    with server(model, devices[0], "teacher", "record", root, profile) as endpoint:
        measure(endpoint, "teacher", "record", cases, root)
    kernel_profile.validate(json.loads(profile.read_text()))
    with server(model, devices[0], "teacher", "baseline", root, profile) as endpoint:
        measure(endpoint, "teacher", "baseline", cases, root)
    with server(model, devices[1], "student", "baseline", root, profile) as endpoint:
        measure(endpoint, "student", "baseline", cases, root)
    with server(model, devices[0], "teacher", "strict", root, profile) as teacher_endpoint:
        teacher_scores = measure(teacher_endpoint, "teacher", "strict", cases, root)
        with server(model, devices[1], "student", "strict", root, profile) as endpoint:
            student_scores = measure(endpoint, "student", "strict", cases, root)
            for name, values in student_scores.items():
                numerical_audit.append(
                    root / "comparisons.jsonl",
                    dict(
                        mode="strict",
                        kind="student_vs_teacher_bare_weights",
                        case=name,
                        **numerical_audit.compare(values, teacher_scores[name]),
                    ),
                )
            asyncio.run(
                exercise_hook(
                    endpoint, teacher_endpoint, selected[1]["rendered"], saved["tokens"], tokenizer, model, root
                )
            )
        with server(model, devices[1], "teacher", "strict", root, profile) as endpoint:
            for case in cases:
                values = score(endpoint, [case])[0]
                numerical_audit.append(
                    root / "comparisons.jsonl",
                    dict(
                        mode="strict",
                        kind="cross_engine_teacher",
                        case=case["name"],
                        **numerical_audit.compare(values, teacher_scores[case["name"]]),
                    ),
                )
    result = summarize(root)
    print(json.dumps(result), flush=True)
    if not result["all_exact"]:
        raise RuntimeError("Full-context qualification found scoring differences; inspect comparisons")


if __name__ == "__main__":
    main()
