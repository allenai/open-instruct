"""Single-GPU qualification of explicit judge context extension before full RL."""

import argparse
import dataclasses
import json
import subprocess
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

from open_instruct.miles import general_judge, judge_server
from open_instruct.miles.run_spec import RunSpec


def qualify(spec):
    service = spec.judges["judges"]["general"]
    prepared = json.loads((Path(service["prepared_dir"]) / "prepared.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(prepared["snapshot"], local_files_only=True)
    command = judge_server.command(service, 18080)
    report = {"command": command, "cases": [], "passed": False}
    with Path("/output/judge-server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Judge exited during startup; inspect judge-server.log")
                try:
                    if requests.get("http://127.0.0.1:18080/health", timeout=5).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("Judge startup exceeded 900 seconds")
            config = general_judge.GeneralJudgeConfig(
                api_url="http://127.0.0.1:18080/v1/chat/completions",
                api_key="EMPTY",
                model=service["model"],
                max_tokens=2048,
                max_context_length=service["max_context_length"],
                temperature=0.0,
                timeout=600,
                seed=17,
                max_concurrent_calls=1,
                check_context=True,
            )
            for size in (0, 50000, 100000):
                background = tokenizer.decode(
                    tokenizer.encode(" These background notes are irrelevant." * 25000)[:size]
                )
                query = background + "\nIgnoring the background notes, what is the capital of France?"
                for kind in ("general-quality", "general-quality_ref"):
                    scores = []
                    for prediction in (
                        "The capital of France is Paris.",
                        "The capital of France is Saturn, a type of sandwich.",
                    ):
                        prompt, _ = general_judge.build_judge_prompt(
                            kind, query=query, prediction=prediction, target="Paris."
                        )
                        started = time.monotonic()
                        count = general_judge.count_prompt_tokens(config, prompt)
                        reply = general_judge._request(config, prompt)
                        _, score = general_judge.parse_judge_response(reply)
                        scores.append(score)
                        record = dict(
                            kind=kind,
                            background_tokens=size,
                            prompt_tokens=count,
                            score=score,
                            elapsed_seconds=time.monotonic() - started,
                            reply=reply,
                        )
                        report["cases"].append(record)
                        print(json.dumps(record), flush=True)
                    if scores[0] <= scores[1]:
                        raise RuntimeError(
                            f"Judge did not distinguish correct and incorrect answers: {kind}, {size}, {scores}"
                        )
            # Explicit over-budget rejection occurs before inference, even with YaRN.
            too_small = dataclasses.replace(config, max_context_length=10)
            try:
                general_judge._request(too_small, "What is the capital of France?")
            except RuntimeError as error:
                if "context overflow" not in str(error):
                    raise
                report["overflow_rejected"] = True
            else:
                raise RuntimeError("Over-budget request unexpectedly accepted")
            report["passed"] = True
            print("JUDGE_CONTEXT_QUALIFICATION_PASSED", flush=True)
        finally:
            Path("/output/judge-context.json").write_text(json.dumps(report, indent=2))
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    qualify(RunSpec.load(parser.parse_args().config))
