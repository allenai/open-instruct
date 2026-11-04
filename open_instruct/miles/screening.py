"""Exploratory fixed-policy studies of prepared MILES sources and registered rewards.

This is an optional research tool, not a required stage of the normal RL pipeline.
Run with an already serving, fixed-weight SGLang checkpoint. No trainer, optimizer,
zero-variance filter or separate offline verifier dispatch is involved. Per-source
uniform samples are reported separately, with the training fractions recorded.
"""

import argparse
import asyncio
import copy
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from importlib import metadata as package_metadata
from pathlib import Path
from types import SimpleNamespace

import aiohttp

from open_instruct.miles.configuration import specs
from open_instruct.miles.datasets import run_data
from open_instruct.miles.errors import InputError
from open_instruct.miles.infrastructure import artifacts
from open_instruct.miles.rewards import judge_registry, judging, rewards


def digest(value):
    return hashlib.sha256(run_data._encoded(value)).hexdigest()


def select_prompts(prepared, count, seed):
    """Sample without replacement, or select every admitted row when count is None."""
    if count is not None and count < 1:
        raise InputError("prompts-per-source must be positive")
    if prepared.get("prompt_mixture"):
        path = Path(prepared["prompt_mixture"])
        sources = json.loads(path.read_text())["sources"]
        sources = [{**s, "path": str(path.parent / s["path"])} for s in sources]
    else:
        sources = [{"name": "train", "fraction": 1.0, "path": prepared["prompt_data"]}]
    selected, populations = [], []
    for source in sources:
        rows = run_data._rows(Path(source["path"]).read_bytes())
        rng = random.Random(f"{seed}:{source['name']}")
        indices = range(len(rows)) if count is None else sorted(rng.sample(range(len(rows)), min(count, len(rows))))
        populations.append({"name": source["name"], "rows": len(rows), "fraction": source["fraction"]})
        for index in indices:
            row = copy.deepcopy(rows[index])
            # Attribution is supplied by the mixture provider at training runtime.
            row["metadata"]["source_name"] = source["name"]
            selected.append(
                {
                    "id": digest([source["name"], index, rows[index]]),
                    "source": source["name"],
                    "row_index": index,
                    "row": row,
                }
            )
    return selected, populations


async def score_completion(config, prompt, completion):
    """The same reward bridge used by MILES; verifier exceptions remain unknown."""
    metadata = copy.deepcopy(prompt["row"]["metadata"])
    sample = SimpleNamespace(
        prompt=prompt["row"]["input"],
        response=completion["response"],
        metadata=metadata,
        tokens=completion["token_ids"],
        response_length=len(completion["token_ids"]),
        status="truncated" if completion["truncated"] else "completed",
    )
    args = SimpleNamespace(**config.miles, olmo_core=config.core)
    try:
        score = await rewards.registered_reward(args, sample)
        return {"score": score, "status": "scored", "metadata": metadata}
    except Exception as error:
        return {
            "score": None,
            "status": "scoring_error",
            "error": type(error).__name__,
            "detail": str(error),
            "metadata": metadata,
        }


def summarize(records, ks=(1, 4, 8, 16)):
    """Finite-sample pass/mixed estimates, with incomplete groups kept separate."""
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["source"], record["id"])].append(record)
    sources = defaultdict(list)
    for (source, identity), rows in grouped.items():
        scored = [r for r in rows if r.get("score") is not None]
        n = len(rows)
        values = [r["score"] for r in scored]
        complete = len(scored) == n
        binary = complete and all(v in (0.0, 1.0) for v in values)
        successes = sum(v == 1 for v in values)
        metrics = {}
        for k in ks:
            if k > n:
                continue
            denominator = math.comb(n, k)
            variation = 1 - sum(math.comb(c, k) if c >= k else 0 for c in Counter(values).values()) / denominator
            metrics[str(k)] = {
                "pass": 1 - (math.comb(n - successes, k) if n - successes >= k else 0) / denominator
                if binary
                else None,
                "varying_reward": variation if complete else None,
            }
        sources[source].append(
            {
                "id": identity,
                "attempted": n,
                "scored": len(scored),
                "complete": complete,
                "successes": successes if binary else None,
                "rewards": values,
                "metrics": metrics,
                "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
                "truncated": sum(r.get("truncated", False) for r in rows),
            }
        )
    result = {}
    for source, groups in sources.items():
        metrics = {}
        for k in map(str, ks):
            metrics[k] = {}
            for field in ("pass", "varying_reward"):
                values = [
                    g["metrics"][k][field] for g in groups if k in g["metrics"] and g["metrics"][k][field] is not None
                ]
                metrics[k][field] = sum(values) / len(values) if values else None
                metrics[k][field + "_groups"] = len(values)
        result[source] = {
            "prompts": len(groups),
            "complete_groups": sum(g["complete"] for g in groups),
            "unresolved_groups": sum(not g["complete"] for g in groups),
            "metrics": metrics,
            "success_histogram": dict(Counter(g["successes"] for g in groups if g["successes"] is not None)),
            "groups": groups,
        }
    return result


def generation_settings(config):
    # These are MILES rollout settings, not an independent screening profile.
    options = config.miles
    return {
        "temperature": options.get("rollout_temperature", 1.0),
        "top_p": options.get("rollout_top_p", 1.0),
        "top_k": options.get("rollout_top_k", -1),
        "max_new_tokens": options["rollout_max_response_len"],
        "stop": options.get("rollout_stop"),
        "stop_token_ids": options.get("rollout_stop_token_ids"),
        "skip_special_tokens": options.get("rollout_skip_special_tokens", False),
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }


async def run(options):
    spec = specs.load(options.config)
    # A one-turn screen must not masquerade as coverage of an interactive task.
    planned = spec.compile()
    for key in ("custom_generate_function_path", "custom_reward_post_process_path", "rollout_function_path"):
        if planned.miles.get(key):
            raise InputError(f"Screen requires an explicitly supported rollout/reward path; found {key}")
    if planned.miles.get("custom_rm_path") not in (None, "open_instruct.miles.rewards.rewards.registered_reward"):
        raise InputError("Screen requires the registered MILES reward bridge")
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = run_data.prepare_run_data(spec, options.tokenizer.resolve(), output / "prepared")
    config = spec.compile({**prepared, "hf_checkpoint": str(options.tokenizer.resolve())})
    resolved = judge_registry.registry() if spec.judges["judging"]["bindings"] else judging.registry(spec.judges)
    if resolved is None:
        raise InputError("Named rewards require the same resolved judge registry used by training")
    judge_registry.validate_data(prepared, resolved)
    prompts, populations = select_prompts(prepared, options.prompts_per_source, options.seed)
    if any(p["row"]["metadata"].get("tools") or p["row"].get("tools") for p in prompts):
        raise InputError("Interactive tool rows require an environment screen, not single-turn generation")
    tokenizer = run_data._tokenizer(options.tokenizer)
    sampling = generation_settings(config)
    timeout = aiohttp.ClientTimeout(total=options.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(options.server.rstrip("/") + "/get_model_info") as response:
            response.raise_for_status()
            model_info = await response.json()
        if model_info.get("model_path") != spec.model["source"]:
            raise InputError("Serving checkpoint differs from model.source in the training run file")
        contract = {
            "schema_version": 1,
            "model": spec.model,
            "server_model_info": model_info,
            "prepared_manifest": json.loads(Path(prepared["manifest"]).read_text()),
            "sampling": sampling,
            "reward_policy": dataclasses.asdict(config.core),
            "judge_registry": resolved,
            "draws": options.draws,
            "seed": options.seed,
            "populations": populations,
            "prompt_ids": [p["id"] for p in prompts],
            "screen_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reward_bridge_sha256": hashlib.sha256(Path(rewards.__file__).read_bytes()).hexdigest(),
            "application_python_sha256": digest(
                {
                    str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path(__file__).parents[1].rglob("*.py"))
                }
            ),
            "packages": [
                list(v) for v in sorted({(d.metadata["Name"], d.version) for d in package_metadata.distributions()})
            ],
            "verifier_environment": {
                key: os.environ.get(key)
                for key in (
                    "OI_MILES_MATH_TIMEOUT_POLICY",
                    "OI_MILES_FORMAT_TIMEOUT_POLICY",
                    "OI_MILES_CODE_FAILURE_POLICY",
                )
            },
        }
        path = output / "contract.json"
        if path.exists() and json.loads(path.read_text()) != contract:
            raise InputError("Screen contract changed; use a new output directory")
        artifacts.atomic_json(path, contract)
        contract_hash = digest(contract)
        (output / "draws").mkdir(exist_ok=True)
        semaphore = asyncio.Semaphore(options.concurrency)

        async def draw(prompt, number):
            path = output / "draws" / f"{prompt['id']}-{number:03d}.json"
            if path.exists():
                previous = json.loads(path.read_text())
                if previous.get("contract_sha256") != contract_hash:
                    raise InputError(f"Draw contract mismatch: {path}")
                return previous
            record = {
                "id": prompt["id"],
                "source": prompt["source"],
                "draw": number,
                "contract_sha256": contract_hash,
                "score": None,
                "status": "generation_error",
            }
            async with semaphore:
                try:
                    ids = tokenizer.encode(prompt["row"]["input"], add_special_tokens=False)
                    params = {**sampling, "sampling_seed": int(digest([options.seed, prompt["id"], number])[:8], 16)}
                    payload = {"input_ids": ids, "sampling_params": params, "return_logprob": True}
                    async with session.post(options.server.rstrip("/") + "/generate", json=payload) as response:
                        response.raise_for_status()
                        value = await response.json()
                    meta = value["meta_info"]
                    token_ids = [v[1] for v in meta["output_token_logprobs"]]
                    if len(token_ids) != meta["completion_tokens"]:
                        raise InputError("Missing generated token IDs; cannot reproduce the training reward input")
                    reason = meta["finish_reason"]["type"]
                    if reason not in {"stop", "length"}:
                        raise InputError(f"Unresolved generation finish reason: {reason}")
                    completion = {
                        "response": value["text"],
                        "token_ids": token_ids,
                        "completion_tokens": len(token_ids),
                        "truncated": reason == "length",
                    }
                    record.update(completion)
                    record.update(await score_completion(config, prompt, completion))
                except Exception as error:
                    record.update(error=type(error).__name__, detail=str(error))
                artifacts.atomic_json(path, record)
            return record

        records = await asyncio.gather(*(draw(p, n) for p in prompts for n in range(options.draws)))
    report = {
        "contract_sha256": contract_hash,
        "sources": summarize(records),
        "limits": "Per-source results. Pass@k uses complete binary-reward groups only; unknowns are not zero. No mixture-weight or transfer claim.",
    }
    artifacts.atomic_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="The same schema_version=1 run file used for training")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Local HF tokenizer for that checkpoint")
    parser.add_argument("--server", required=True, help="Fixed-weight SGLang server URL")
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--prompts-per-source", type=int, default=96)
    selection.add_argument(
        "--all",
        dest="prompts_per_source",
        action="store_const",
        const=None,
        help="Generate draws for every row in every prepared source, without subsampling",
    )
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=3600)
    options = parser.parse_args(argv)
    if min(options.draws, options.concurrency, options.request_timeout) <= 0 or (
        options.prompts_per_source is not None and options.prompts_per_source <= 0
    ):
        parser.error("Counts, concurrency and timeout must be positive")
    options.output.mkdir(parents=True, exist_ok=True)
    with (options.output / ".screen.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another screening process owns this output directory")
        asyncio.run(run(options))


if __name__ == "__main__":
    main()
