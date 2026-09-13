"""Fresh SGLang process loading a completed dense readiness run's HF export."""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from miles.utils.hf_config import HF_EXPORT_COMPLETE_MARKER
from scripts.miles import audit_workflow


def exercise(root, output):
    root, output = Path(root), Path(output)
    state = json.loads((root / "workflow.json").read_text())
    if state["status"] != "complete":
        raise ValueError("Fresh reload requires a completed training workflow")
    model = root / "export-hf"
    if not (model / HF_EXPORT_COMPLETE_MARKER).is_file():
        raise ValueError("HF export is not committed")
    spec = json.loads((root / "run-spec.json").read_text())
    if spec["inference"]["max_context_length"] != 4096:
        raise ValueError("This bounded reload exercise is for the dense 4096-context profile")
    final = spec["training"]["num_rollouts"] - 1
    retained = audit_workflow.load_rollout(root / "rollouts" / f"eval_{final}.pt")["samples"]
    if not retained:
        raise ValueError("Missing retained final evaluation")
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "31000",
        "--tp-size",
        "1",
        "--context-length",
        "4096",
        "--max-total-tokens",
        "32768",
        "--mem-fraction-static",
        "0.2",
        "--max-running-requests",
        "4",
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--disable-radix-cache",
        "--trust-remote-code",
        "--skip-server-warmup",
    ]
    process = None
    started = time.monotonic()
    try:
        with (output / "server.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with httpx.Client(base_url="http://127.0.0.1:31000", timeout=300) as client:
            deadline = time.monotonic() + 900
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Reload server exited with {process.returncode}")
                try:
                    if client.get("/health", timeout=5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Fresh reload server did not become healthy")
                time.sleep(2)
            startup_seconds = time.monotonic() - started
            results = []
            for sample in retained:
                length = sample["response_length"]
                prompt = sample["tokens"][:-length]
                response = client.post(
                    "/generate",
                    json={
                        "input_ids": prompt,
                        "return_logprob": True,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 64},
                    },
                )
                response.raise_for_status()
                payload = response.json()
                scored = payload["meta_info"]["output_token_logprobs"]
                if not scored or any(not math.isfinite(row[0]) for row in scored):
                    raise ValueError("Fresh reload produced empty or nonfinite log probabilities")
                generated = [row[1] for row in scored]
                reference = sample["tokens"][-length:][: len(generated)]
                results.append(
                    {
                        "id": sample["metadata"]["prepared_sample_id"],
                        "text": payload["text"],
                        "token_ids": generated,
                        "exact_prefix_matches_final_eval": generated == reference,
                        "reference_reward": sample["reward"],
                        "finish_reason": payload["meta_info"]["finish_reason"],
                    }
                )
            report = {
                "passed": True,
                "model": str(model),
                "command": command,
                "startup_seconds": startup_seconds,
                "samples": results,
                "limits": "Fresh load and finite greedy prefix generation only; prefix differences are reported, not a numerical-equivalence or reward gate.",
            }
            (output / "reload.json").write_text(json.dumps(report, indent=2) + "\n")
            return report
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/output"))
    args = parser.parse_args()
    print(json.dumps(exercise(args.root, args.output), indent=2))
