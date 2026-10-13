"""Trace cached decode against fresh-prefix scoring at identical token positions."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from urllib import request

import torch
from olmo_sglang import kernel_profile

from open_instruct.miles.distillation import alignment, numerical_audit, panel, teacher


def worker(spec, mode):
    root = Path(spec["output"]["root"]) / mode
    root.mkdir(parents=True, exist_ok=False)
    saved = json.loads(Path(spec["launch"]["env"]["OI_MILES_OPD_AUDIT_INPUT"]).read_text())
    if saved["same_checkpoint"] != spec["model"]["source"]:
        raise ValueError("Mismatched checkpoint")
    tuning = json.loads(Path(spec["launch"]["env"]["OI_MILES_OPD_AUDIT_TUNING"]).read_text())
    replay = root / "replay.json"
    replay.write_text(json.dumps(tuning["core-eager" if mode == "full" else "rounding-graphs"]))
    profile = root / "profile.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "olmo_sglang.kernel_profile",
            "import-replay",
            str(replay),
            str(profile),
            "--model",
            saved["same_checkpoint"],
            "--arithmetic",
            mode,
        ],
        check=True,
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cmd = teacher.command(
        {"source": saved["same_checkpoint"], "tensor_parallel_size": 1, "max_context_length": 32768, "concurrency": 8},
        port,
    )
    cmd.remove("--disable-cuda-graph")
    force_source = spec["launch"]["env"].get("OI_MILES_OPD_DECODE_FORCE_SOURCE")
    if force_source:
        cmd.append("--disable-overlap-schedule")
    cmd.extend(
        [
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--random-seed",
            "130064",
            "--forward-hooks",
            json.dumps(
                [
                    {
                        "name": "positions",
                        "target_modules": ["model"],
                        "hook_factory": "open_instruct.miles.distillation.audit_decode:factory",
                        "config": {"root": str(root)},
                    },
                    {
                        "name": "logits",
                        "target_modules": ["logits_processor"],
                        "hook_factory": "open_instruct.miles.distillation.audit_decode:head_factory",
                        "config": {"root": str(root)},
                    },
                ]
            ),
        ]
    )
    env = {
        **os.environ,
        "OLMO_SGLANG_CORE_COMPAT": "1" if mode == "full" else "rounding",
        "OLMO_SGLANG_DIAGNOSTIC_OVERRIDE_REASON": "Fixed-position decode trace, no training",
        "OLMO_SGLANG_KERNEL_PROFILE": str(profile),
        "OLMO_SGLANG_KERNEL_PROFILE_MODE": "strict",
        "OLMO_SGLANG_KERNEL_PROFILE_SHA256": kernel_profile.digest(json.loads(profile.read_text())),
        "OLMO_SGLANG_KERNEL_PROFILE_REPORT": str(root / "profile-usage.json"),
    }
    (root / "command.json").write_text(
        json.dumps({"command": cmd, "profile_sha256": env["OLMO_SGLANG_KERNEL_PROFILE_SHA256"]}, indent=2)
    )
    log = (root / "server.log").open("w")
    server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError("Decode trace server exited")
            try:
                with request.urlopen(url + "/health", timeout=2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Decode trace server startup") from None
                time.sleep(2)
        prefix = saved["prefix"]
        # Prime hook installation explicitly, even if serving warmup was skipped.
        panel.post(
            url + "/generate", {"input_ids": prefix, "sampling_params": {"temperature": 0, "max_new_tokens": 1}}
        )
        offsets = [0, 1, 2, 63, 64, 127, 128, 255, 511, 1023]
        positions = [len(prefix) + i - 1 for i in offsets]
        control = root / "control.json"
        settings = {"label": "decode", "positions": positions}
        forced = None
        if force_source:
            prior = json.loads((Path(force_source) / mode / "generation.json").read_text())
            forced = [v[1] for v in prior["meta_info"]["output_token_logprobs"]]
            if len(forced) != 1024:
                raise ValueError("Fixed-token trace expects 1024 saved continuation tokens")
            settings.update(forced_tokens=forced, prefix_length=len(prefix))
        control.write_text(json.dumps(settings))
        result = panel.post(
            url + "/generate",
            {
                "input_ids": prefix,
                "return_logprob": True,
                "sampling_params": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_new_tokens": 1024,
                    "sampling_seed": 130064,
                    "skip_special_tokens": False,
                    "ignore_eos": bool(forced),
                },
            },
        )
        (root / "generation.json").write_text(json.dumps(result))
        entries = result["meta_info"]["output_token_logprobs"]
        tokens = [v[1] for v in entries]
        if forced is not None and tokens != forced:
            raise ValueError("Teacher-forced decode did not reproduce the saved token sequence")
        for offset, position in zip(offsets, positions, strict=True):
            if offset >= len(tokens):
                continue
            control.write_text(json.dumps({"label": "prefill", "positions": [position]}))
            fresh = panel.post(
                url + "/generate",
                {
                    "input_ids": prefix + tokens[:offset],
                    "return_logprob": True,
                    "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 1},
                },
            )
            (root / f"prefill-{position}.json").write_text(json.dumps(fresh))
            cached = torch.load(root / f"decode-{position}.pt", weights_only=True)
            full = torch.load(root / f"prefill-{position}.pt", weights_only=True)
            if cached["input_token"] != full["input_token"] or cached["position"] != full["position"]:
                raise ValueError("Mismatched trace positions")
            a, b = cached["logits"].float().log_softmax(-1), full["logits"].float().log_softmax(-1)
            layers = {}
            for name, v in cached["states"].items():
                w = full["states"][name]
                layers[name] = {
                    "max_abs": (v.float() - w.float()).abs().max().item(),
                    "mean_abs": (v.float() - w.float()).abs().mean().item(),
                    "different_fraction": (v != w).float().mean().item(),
                }
            token = tokens[offset]
            summary = {
                "mode": mode,
                "offset": offset,
                "position": position,
                "token": token,
                "decode_api_logprob": entries[offset][0],
                "teacher_forced": forced is not None,
                "decode_logits_logprob": a[token].item(),
                "prefill_logits_logprob": b[token].item(),
                "kl_decode_to_prefill": (a.exp() * (a - b)).sum().item(),
                "greedy_equal": a.argmax().item() == b.argmax().item(),
                "layers": layers,
            }
            numerical_audit.append(root / "comparisons.jsonl", summary)
            print(json.dumps({k: v for k, v in summary.items() if k != "layers"}), flush=True)
        control.unlink()
        if forced is not None:
            observed = [json.loads(line) for line in (root / "forced-logprobs.jsonl").read_text().splitlines()]
            if [x["offset"] for x in observed] != list(range(len(tokens))):
                raise ValueError("Missing or duplicate forced scoring positions")
            ids = prefix + tokens
            response = panel.post(
                url + "/generate",
                {
                    "input_ids": ids,
                    "return_logprob": True,
                    "logprob_start_len": 0,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 0},
                },
            )
            prefill_scores = alignment.extract_scores(response, ids, len(tokens))
            decode_scores = [x["logprob"] for x in observed]
            comparison = numerical_audit.compare(decode_scores, prefill_scores)
            (root / "sequence-comparison.json").write_text(
                json.dumps(
                    {
                        "tokens": tokens,
                        "prefix": prefix,
                        "decode_raw_logprobs": decode_scores,
                        "prefill_logprobs": prefill_scores,
                        "comparison": comparison,
                        "teacher_forced": True,
                        "source": force_source,
                    }
                )
            )
            print(json.dumps({"mode": mode, "fixed-sequence-comparison": comparison}), flush=True)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--mode", choices=["rounding", "full"])
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.mode:
        worker(spec, args.mode)
        return
    if spec["launch"]["env"].get("OI_MILES_OPD_DECODE_QK_PROBE") == "1":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "/opt/core-rl/serving-tests/test_kda_backend.py",
                "-k",
                "packed_kda_decode",
                "-p",
                "no:cacheprovider",
            ],
            check=True,
        )
    root = Path(spec["output"]["root"])
    root.mkdir(parents=True, exist_ok=False)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(devices) != 2:
        raise ValueError("Expected two GPUs")
    processes = [
        subprocess.Popen(
            [sys.executable, __file__, str(args.spec), "--mode", mode],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
        )
        for device, mode in zip(devices, ["rounding", "full"], strict=True)
    ]
    codes = [p.wait() for p in processes]
    if any(codes):
        raise RuntimeError(f"Decode trace workers failed: {codes}")
    with tarfile.open("/output/selected-decode-traces.tar.gz", "w:gz") as archive:
        roots = [("current", root)]
        source = spec["launch"]["env"].get("OI_MILES_OPD_DECODE_FORCE_SOURCE")
        if source:
            roots.append(("original", Path(source)))
        for label, folder in roots:
            for path in folder.glob("*/*.pt"):
                archive.add(path, arcname=str(Path(label) / path.relative_to(folder)))


if __name__ == "__main__":
    main()
