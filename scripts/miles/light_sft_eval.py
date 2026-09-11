"""Native 128-question curve plus separately retained historical full-test endpoints."""

import copy
import dataclasses
import json
import re
import time
from pathlib import Path

from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.eval_config import EvalDatasetConfig
from scripts.miles.prepare_gsm8k_parity import digest, json_bytes, write_immutable


def independent_score(response, label):
    """Independent implementation of historical last-signed-number exact match."""
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", re.sub(r"(\d),(\d)", r"\1\2", response))
    return float(bool(numbers) and numbers[-1] == label)


def summarize(samples, prompts, proofs, step):
    if len(samples) != len(prompts) or len(prompts) != len(proofs):
        raise ValueError("Historical full-test completion count differs")
    rows = []
    for sample, prompt, proof in zip(samples, prompts, proofs, strict=True):
        if sample.metadata["prepared_sample_id"] != prompt["id"] or sample.prompt != prompt["input"]:
            raise ValueError("Full-test question identity or raw prompt changed")
        ids = sample.tokens[: -sample.response_length]
        if digest(json_bytes(ids)) != proof["token_ids_sha256"]:
            raise ValueError("Full-test prompt tokenization differs from frozen requests")
        versions = sample.weight_versions
        if not versions or any(str(version) != str(step) for version in versions):
            raise ValueError("Full-test samples do not use the required published policy")
        if sample.status.name not in ("COMPLETED", "TRUNCATED") or not 0 < sample.response_length <= 512:
            raise ValueError("Full-test sample failed or exceeded the historical response cap")
        score = independent_score(sample.response, prompt["label"])
        reward = sample.reward["score"] if isinstance(sample.reward, dict) else sample.reward
        if float(reward) != score:
            raise ValueError("Independent historical full-test scoring disagrees with registered verifier")
        rows.append(
            {
                "id": prompt["id"],
                "label": prompt["label"],
                "response": sample.response,
                "tokens": sample.tokens,
                "response_length": sample.response_length,
                "score": score,
                "truncated": sample.status.name == "TRUNCATED",
                "weight_versions": versions,
            }
        )
    correct = sum(row["score"] for row in rows)
    return {
        "step": step,
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "capped": sum(row["truncated"] for row in rows),
        "samples": rows,
        "format": "Historical raw completion prompt; greedy, max512, stops Question: and double-newline",
        "transport": "MILES /generate with proven raw-prompt token IDs; historical client used /completions",
    }


class HistoricalEvaluation:
    def __init__(self, input):
        if getattr(input.args, "eval_uses_snapshots", False):
            raise ValueError("Historical evaluation requires the inline serving policy, not snapshots")
        self.native = InferenceRolloutFn(input)
        self.constructor_input = input
        self.root = Path(input.args.prompt_data).parent

    def offline_function(self):
        # MILES assigns router endpoints after extension construction.
        # Copy live transport arguments only when evaluating.
        args = copy.deepcopy(self.constructor_input.args)
        if not args.sglang_router_ip or not args.sglang_router_port:
            raise ValueError("Historical evaluation requires a live router endpoint")
        args.rollout_stop = ["Question:", "\n\n"]
        args.eval_datasets = [
            EvalDatasetConfig(
                name="historical-full-test",
                path=str(self.root / "offline/prompts.jsonl"),
                input_key="input",
                label_key="label",
                metadata_key="metadata",
                n_samples_per_eval_prompt=1,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                max_response_len=512,
            )
        ]
        return InferenceRolloutFn(dataclasses.replace(self.constructor_input, args=args))

    async def __call__(self, input):
        if not input.evaluation:
            raise ValueError("HistoricalEvaluation is an evaluation-only extension")
        if input.generate_state is not None or input.hf_dir is not None:
            raise ValueError("Historical evaluation cannot inherit a snapshot generation state")
        native = await self.native(input)
        if input.rollout_id not in (0, 199):
            return native
        step = 0 if input.rollout_id == 0 else 200
        native_rows = native.data["gsm8k"]
        native_report = {
            "step": step,
            "count": len(native_rows["samples"]),
            "correct": sum(float(reward) for reward in native_rows["rewards"]),
            "capped": sum(native_rows["truncated"]),
            "samples": [
                {
                    "id": sample.metadata["prepared_sample_id"],
                    "response": sample.response,
                    "tokens": sample.tokens,
                    "weight_versions": sample.weight_versions,
                    "reward": reward,
                }
                for sample, reward in zip(native_rows["samples"], native_rows["rewards"], strict=True)
            ],
        }
        write_immutable(self.root / f"core/native-{step}.json", json_bytes(native_report))
        summary = {key: value for key, value in native_report.items() if key != "samples"}
        print("LIGHT_SFT_NATIVE_EVAL", json.dumps(summary), flush=True)
        started = time.monotonic()
        output = await self.offline_function()(input)
        prompts = [json.loads(line) for line in (self.root / "offline/prompts.jsonl").read_text().splitlines()]
        proofs = json.loads((self.root / "offline/token-proofs.json").read_text())
        report = summarize(output.data["historical-full-test"]["samples"], prompts, proofs, step)
        report["elapsed_seconds"] = time.monotonic() - started
        write_immutable(self.root / f"core/offline-{step}.json", json_bytes(report))
        native.metrics = {
            **(native.metrics or {}),
            "eval/historical-full-test/accuracy": report["accuracy"],
            "eval/historical-full-test/seconds": report["elapsed_seconds"],
        }
        return native
