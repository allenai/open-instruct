"""Screen a finishing-prone failure cohort, then run bounded native distillation."""

import argparse
import asyncio
import copy
import gzip
import hashlib
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from urllib import request

from transformers import AutoTokenizer

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.distillation import panel, qualification, teacher
from open_instruct.miles.execution import launch
from open_instruct.miles.rewards import rewards

DATA = Path("/weka/oe-adapt-default/robertb/open-instruct/data/bread-h045-validation-20261008-mathverify")
BREAD = Path("/weka/olmo-3p5-checkpoints/scratch/robertb/open-instruct/runs/bread-h045-validation-20261009-v2")
SPORK = Path("/weka/olmo-3p5-checkpoints/scratch/robertb/open-instruct/runs/h045-answer-exit-treatment-20261010")
SEED = 130064
CAP = 16384


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def classify(attempts):
    if any(a["reward"] > 0 for a in attempts):
        return "has_success"
    if len(attempts) != 8:
        return "incomplete"
    return "finished_wrong" if sum(a["finish"] == "stop" for a in attempts) >= 6 else "truncation_heavy"


def split(eligible):
    if len(eligible) < 48:
        raise ValueError(f"Only {len(eligible)} qualified questions; need 48 without relaxing eligibility")
    ordered = sorted(eligible, key=lambda r: r["sample_id"])
    random.Random(SEED).shuffle(ordered)
    heldout_count = min(32, max(16, len(ordered) // 4))
    return ordered[heldout_count : heldout_count + 96], ordered[:heldout_count]


def candidates(tokenizer, limit=512):
    preferred = set()
    gate_file = BREAD / "checkpoints/bread/groups.jsonl"
    if gate_file.exists():
        for row in rows(gate_file):
            if row.get("event") == "zero_group_ready":
                preferred.add(row["sample_id"])
    selected = []
    excluded = []
    heldout = {r["metadata"]["prepared_sample_id"] for r in rows(DATA / "eval.jsonl")}
    identities, questions = set(), set()
    # Keep only the best bounded candidates in memory, not the entire reference corpus.
    for row in rows(DATA / "train.jsonl"):
        metadata = row["metadata"]
        identity, question = metadata["prepared_sample_id"], metadata["query"]
        if identity in heldout:
            raise ValueError("Candidate source overlaps original heldout identities")
        if identity in identities or question in questions:
            excluded.append({"sample_id": identity, "reason": "duplicate source identity or question"})
            continue
        identities.add(identity)
        questions.add(question)
        rank = (identity not in preferred, hashlib.sha256(f"{SEED}:{identity}".encode()).hexdigest())
        if len(selected) >= limit and rank >= selected[-1][0]:
            continue
        bare = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        wrapped = {"sample_id": identity, "row": row, "gate": {"failure_kind": "pending"}}
        try:
            template = {"input": bare, "metadata": {"query": question}}
            rendered = qualification.render_row(wrapped, template, tokenizer.backend_tokenizer, True)
        except ValueError as error:
            excluded.append({"sample_id": identity, "reason": str(error)})
            continue
        selected.append(
            (rank, {"sample_id": identity, "rendered": rendered, "previously_all_zero": identity in preferred})
        )
        selected.sort(key=lambda r: r[0])
        selected = selected[:limit]
    return [value for _, value in selected], {
        "source_questions": len(identities),
        "historical_zero_id_count": len(preferred),
        "selected_previous_zero": sum(v["previously_all_zero"] for _, v in selected),
        "candidate_count": len(selected),
        "context_exclusions": excluded,
        "source_manifest_sha256": hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
    }


async def screen(items, endpoints, tokenizer, root, deadline):
    limits = [asyncio.Semaphore(8) for _ in endpoints]
    reward_args = SimpleNamespace(
        olmo_core=SimpleNamespace(
            reward_config=str(root / "verifiers.json"), reward_zero_truncated=True, reward_final_answer_only=True
        )
    )
    eligible, heavy = [], []
    counts = {}
    traces = root / "screen-responses.jsonl"

    async def attempt(item, index, endpoint_index):
        rendered = item["rendered"]
        prefix = rendered["metadata"]["prompt_token_ids"]
        seed = int(hashlib.sha256(f"{SEED}:{item['sample_id']}:{index}".encode()).hexdigest()[:8], 16)
        async with limits[endpoint_index]:
            started = time.monotonic()
            result = await asyncio.to_thread(
                panel.post,
                endpoints[endpoint_index] + "/generate",
                {
                    "input_ids": prefix,
                    "return_logprob": True,
                    "sampling_params": {
                        "max_new_tokens": CAP,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "sampling_seed": seed,
                        "skip_special_tokens": False,
                        "no_stop_trim": True,
                    },
                },
                1200,
            )
            info = result["meta_info"]
            finish = info["finish_reason"]["type"]
            if finish not in ("stop", "length"):
                raise ValueError(f"Unexpected screen finish: {finish}")
            tokens = [entry[1] for entry in info["output_token_logprobs"]]
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            sample = SimpleNamespace(
                metadata=copy.deepcopy(rendered["metadata"]),
                tokens=prefix + tokens,
                response_length=len(tokens),
                response=text,
                prompt=rendered["input"],
                status="completed" if finish == "stop" else "truncated",
            )
            value = await rewards.registered_reward(reward_args, sample)
        diagnostics = sample.metadata.get("verifier_diagnostics", {})
        if any(d.get("status") == "timeout" for d in diagnostics.values()):
            raise ValueError("Verifier timeout cannot establish an all-wrong screen")
        record = {
            "sample_id": item["sample_id"],
            "draw": index,
            "seed": seed,
            "finish": finish,
            "reward": value,
            "response": text,
            "tokens": tokens,
            "logprobs": [entry[0] for entry in info["output_token_logprobs"]],
            "seconds": time.monotonic() - started,
            "verifier_diagnostics": diagnostics,
        }
        with traces.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return {k: record[k] for k in ("draw", "seed", "finish", "reward", "seconds")}

    async def group(item, offset):
        endpoint = offset % len(endpoints)
        attempts = await asyncio.gather(*(attempt(item, i, endpoint) for i in range(2)))
        if not any(a["reward"] > 0 for a in attempts):
            attempts.extend(await asyncio.gather(*(attempt(item, i, endpoint) for i in range(2, 8))))
        outcome = classify(attempts)
        item["screen"] = {
            "outcome": outcome,
            "attempts": attempts,
            "finished": sum(a["finish"] == "stop" for a in attempts),
        }
        with (root / "screen-groups.jsonl").open("a") as stream:
            stream.write(json.dumps({k: item[k] for k in ("sample_id", "previously_all_zero", "screen")}) + "\n")
        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome == "finished_wrong":
            eligible.append(item)
        elif outcome == "truncation_heavy":
            heavy.append(item)

    for offset in range(0, len(items), len(endpoints) * 2):
        if time.monotonic() >= deadline or len(eligible) >= 128:
            break
        await asyncio.gather(
            *(group(item, offset + i) for i, item in enumerate(items[offset : offset + len(endpoints) * 2]))
        )
        write(
            root / "screen-progress.json",
            {"counts": counts, "eligible": len(eligible), "remaining_seconds": max(0, deadline - time.monotonic())},
        )
        print(json.dumps({"stage": "screen", "counts": counts, "eligible": len(eligible)}), flush=True)
    return eligible, heavy


def prepare_run(document, eligible, heavy, root):
    train, heldout = split(eligible)
    data = root / "data"
    data.mkdir()

    def render(item, training):
        row = copy.deepcopy(item["rendered"])
        row["metadata"].update(
            opd_screen_failure_kind="finished_wrong",
            opd_screen_sha256=qualification.digest(json.dumps(item["screen"], sort_keys=True).encode()),
        )
        if not training:
            for name in ("opd_teacher_prompt", "opd_answer_prompt"):
                row["metadata"].pop(name, None)
        return row

    partitions = {
        "train": [render(item, True) for item in train],
        "train_diagnostic": [render(item, False) for item in train[:16]],
        "heldout": [render(item, False) for item in heldout],
    }
    for name, values in partitions.items():
        (data / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in values))
    # Separate truncation-heavy cohort is retained for later study, not pooled into the headline.
    (root / "truncation-heavy.jsonl").write_text("".join(json.dumps(item) + "\n" for item in heavy))
    panel_rows = [render(item, True) for item in train[:4] + heldout[:4]]
    write(root / "panel.json", {"evaluation_rollout_ids": [0, 15, 63], "rows": panel_rows})
    write(
        root / "cohort.json",
        {
            "train": [i["sample_id"] for i in train],
            "heldout": [i["sample_id"] for i in heldout],
            "diagnostic": [i["sample_id"] for i in train[:16]],
            "screen_is_not_baseline": True,
            "eligible_total": len(eligible),
            "truncation_heavy_total": len(heavy),
        },
    )
    document["data"] = {
        "prompt_data": str(data / "train.jsonl"),
        "reward_config": str(root / "verifiers.json"),
        "eval_prompt_data": [
            "train_diagnostic",
            str(data / "train_diagnostic.jsonl"),
            "heldout",
            str(data / "heldout.jsonl"),
        ],
        "in_sample_eval": ["train_diagnostic"],
        "seed": SEED,
        "shuffle": True,
    }
    return document


def archive(root, output):
    output.mkdir(parents=True, exist_ok=True)
    for source in [root / "teacher-scores.jsonl", root / "preparation/screen-responses.jsonl"]:
        if source.exists():
            with (
                source.open("rb") as reader,
                gzip.open(output / (source.name + ".gz"), "wb", compresslevel=1) as writer,
            ):
                shutil.copyfileobj(reader, writer, 1024 * 1024)
    if (root / "diagnostics").exists():
        shutil.copytree(root / "diagnostics", output / "diagnostics", dirs_exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    document = json.loads(args.config.read_text())
    run_root = Path(document["output"]["root"])
    root = run_root / "preparation"
    if root.exists():
        raise ValueError("Use a fresh run root; automatic replay of a partial overnight run is disabled")
    root.mkdir(parents=True)
    started = time.monotonic()
    servers, handles = [], []
    try:
        workflow = json.loads((SPORK / "workflow.json").read_text())
        checkpoint = json.loads((SPORK / "checkpoints/core/rollout_0000063/complete.json").read_text())
        if checkpoint["clock"]["completed_steps"] != 64 or workflow["status"] != "complete":
            raise ValueError("SPORK update-64 checkpoint is incomplete")
        model = str(SPORK / "export-hf")
        if document["model"]["source"] != model or document["teacher"]["source"] != model:
            raise ValueError("Student and frozen teacher must use the identical SPORK export")
        if workflow["result"]["export_hf"] != model:
            raise ValueError("SPORK export provenance differs")
        write(
            root / "checkpoint-provenance.json",
            {
                "source": model,
                "clock": checkpoint["clock"],
                "workflow": workflow,
                "config": json.loads((Path(model) / "config.json").read_text()),
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        items, inventory = candidates(tokenizer)
        write(root / "inventory.json", inventory)
        write(
            root / "verifiers.json",
            {"openr1_math": {"factory": "open_instruct.miles.rewards.openr1_verifier.OpenR1MathVerifier"}},
        )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4").split(",")
        if len(visible) != 5:
            raise ValueError("Overnight qualification requires five visible GPUs")
        ports = []
        with ExitStack() as stack:
            for _ in visible:
                sock = stack.enter_context(socket.socket())
                sock.bind(("0.0.0.0", 0))
                ports.append(sock.getsockname()[1])
        endpoints = [f"http://127.0.0.1:{port}" for port in ports]
        service = {"source": model, "tensor_parallel_size": 1, "max_context_length": 32768, "concurrency": 8}
        for device, port in zip(visible, ports, strict=True):
            cmd = teacher.command(service, port)
            cmd.remove("--disable-cuda-graph")
            cmd.extend(
                [
                    "--cuda-graph-backend-decode",
                    "full",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    "--cuda-graph-max-bs-decode",
                    "8",
                    "--random-seed",
                    str(SEED),
                ]
            )
            handle = (root / f"screen-server-{device}.log").open("w")
            handles.append(handle)
            servers.append(
                subprocess.Popen(
                    cmd,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        pending = set(endpoints)
        ready_deadline = time.monotonic() + 900
        while pending and time.monotonic() < ready_deadline:
            if any(server.poll() is not None for server in servers):
                raise RuntimeError("Screening server exited; inspect server logs")
            for endpoint in list(pending):
                try:
                    with request.urlopen(endpoint + "/health", timeout=2):
                        pending.remove(endpoint)
                except OSError:
                    pass
            if pending:
                time.sleep(2)
        if pending:
            raise TimeoutError("Screening startup exceeded 15 minutes")

        # Common weights/path, no cache, fixed seeds; save decode/prefill checks before training.
        async def calibrate():
            ids = items[0]["rendered"]["metadata"]["prompt_token_ids"]
            result = await asyncio.to_thread(
                panel.post,
                endpoints[0] + "/generate",
                {
                    "input_ids": ids,
                    "return_logprob": True,
                    "sampling_params": {
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "max_new_tokens": 1024,
                        "sampling_seed": SEED,
                        "skip_special_tokens": False,
                    },
                },
            )
            entries = result["meta_info"]["output_token_logprobs"]
            tokens = [v[1] for v in entries]
            limit = asyncio.Semaphore(1)
            scored = [await panel.score(endpoint + "/generate", ids, tokens, limit) for endpoint in endpoints[:2]]
            write(
                root / "numerical-calibration.json",
                {
                    "prefix": ids,
                    "tokens": tokens,
                    "decode": [v[0] for v in entries],
                    "same_server_prefill": scored[0],
                    "other_server_prefill": scored[1],
                    "same_checkpoint": model,
                },
            )
            same = sum(abs(v[0] - p) for v, p in zip(entries, scored[0], strict=True)) / len(tokens)
            across = sum(abs(a - b) for a, b in zip(*scored, strict=True)) / len(tokens)
            write(
                root / "numerical-calibration-summary.json",
                {
                    "decode_prefill_mean_absolute_nats": same,
                    "across_server_prefill_mean_absolute_nats": across,
                    "note": "Engineering guardrails, not thresholds for meaningful teacher information.",
                },
            )
            if same > 0.1 or across > 0.005:
                raise ValueError("Numerical calibration exceeds guardrails; preserve diagnostics and do not train")

        asyncio.run(calibrate())
        eligible, heavy = asyncio.run(
            screen(items, endpoints, tokenizer, root, min(started + 5400, time.monotonic() + 5400))
        )
    finally:
        for server in servers:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
        for server in servers:
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        for handle in handles:
            handle.close()
        if sys.exc_info()[0] is not None:
            archive(run_root, Path("/output/data-archives"))
            launch.collect_results(run_root, "/output")
    try:
        document = prepare_run(document, eligible, heavy, root)
        document["core"]["max_run_seconds"] = max(1, int(8 * 3600 - (time.monotonic() - started) - 600))
        document["launch"]["env"]["OI_MILES_OPD_PANEL"] = str(root / "panel.json")
        os.environ["OI_MILES_OPD_PANEL"] = str(root / "panel.json")
        write(run_root / "training-run.json", document)
        RunSpec.from_dict(document, config_path=run_root / "training-run.json").plan()
        print(json.dumps({"stage": "train", "cohort": json.loads((root / "cohort.json").read_text())}), flush=True)
        subprocess.run(
            [sys.executable, "-m", "open_instruct.miles.execution.cluster", str(run_root / "training-run.json")],
            check=True,
        )
    finally:
        archive(run_root, Path("/output/data-archives"))
        launch.collect_results(run_root, "/output")


if __name__ == "__main__":
    main()
