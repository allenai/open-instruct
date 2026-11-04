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
from collections import Counter
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


def jsonl_rows(path):
    """Read JSONL one record at a time (Unicode line separators are not records)."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_prompts(prepared, count, seed):
    """Return a lazy selection and population counts, preserving historical sample IDs."""
    if count is not None and count < 1:
        raise InputError("prompts-per-source must be positive")
    if prepared.get("prompt_mixture"):
        path = Path(prepared["prompt_mixture"])
        sources = json.loads(path.read_text())["sources"]
        sources = [{**s, "path": str(path.parent / s["path"])} for s in sources]
    else:
        sources = [{"name": "train", "fraction": 1.0, "path": prepared["prompt_data"]}]
    populations = [
        {"name": s["name"], "rows": sum(1 for _ in jsonl_rows(s["path"])), "fraction": s["fraction"]} for s in sources
    ]

    def selected():
        for source, population in zip(sources, populations, strict=True):
            size = population["rows"]
            # All-row selection never builds an index list. Fixed-size selection keeps
            # only the requested indices and preserves the previous random.sample rule.
            indices = None
            if count is not None and count < size:
                indices = set(random.Random(f"{seed}:{source['name']}").sample(range(size), count))
            for index, row in enumerate(jsonl_rows(source["path"])):
                if indices is not None and index not in indices:
                    continue
                identity = digest([source["name"], index, row])
                row["metadata"]["source_name"] = source["name"]
                yield {"id": identity, "source": source["name"], "row_index": index, "row": row}

    return selected(), populations


def write_prompt_manifest(prompts, path):
    """Validate the entire selection before generation, spooling prompt identities to disk."""
    count = 0
    checksum = hashlib.sha256()
    with path.open("wb") as stream:
        for prompt in prompts:
            if prompt["row"]["metadata"].get("tools") or prompt["row"].get("tools"):
                raise InputError("Interactive tool rows require an environment screen, not single-turn generation")
            data = run_data._encoded(prompt)
            stream.write(data)
            checksum.update(data)
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": "prompts.jsonl", "sha256": checksum.hexdigest(), "rows": count}


async def run_workers(jobs, work, concurrency):
    """Only workers are tasks; even cached draws cannot create an unbounded task backlog."""
    iterator = iter(jobs)

    async def worker():
        for job in iterator:
            await work(*job)
            # Cached results and test transports may not otherwise yield.
            await asyncio.sleep(0)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


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


def summarize_group(identity, rows, ks=(1, 4, 8, 16)):
    """Finite-sample estimates from one prompt's compact draw records."""
    values = [r["score"] for r in rows if r.get("score") is not None]
    n = len(rows)
    complete = len(values) == n
    binary = complete and all(v in (0.0, 1.0) for v in values)
    successes = sum(v == 1 for v in values)
    metrics = {}
    for k in ks:
        if k > n:
            continue
        denominator = math.comb(n, k)
        variation = 1 - sum(math.comb(c, k) if c >= k else 0 for c in Counter(values).values()) / denominator
        metrics[str(k)] = {
            "pass": 1 - (math.comb(n - successes, k) if n - successes >= k else 0) / denominator if binary else None,
            "varying_reward": variation if complete else None,
        }
    return {
        "id": identity,
        "attempted": n,
        "scored": len(values),
        "complete": complete,
        "successes": successes if binary else None,
        "rewards": values,
        "metrics": metrics,
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
        "truncated": sum(r.get("truncated", False) for r in rows),
    }


class SourceSummary:
    """Accumulate scalar statistics, never response text or a list of prompt groups."""

    def __init__(self, ks=(1, 4, 8, 16)):
        self.ks = ks
        self.sources = {}

    def add(self, source, group):
        if source not in self.sources:
            self.sources[source] = {
                "prompts": 0,
                "complete_groups": 0,
                "unresolved_groups": 0,
                "completion_tokens": 0,
                "truncated": 0,
                "success_histogram": Counter(),
                "metrics": {str(k): {field: [0.0, 0] for field in ("pass", "varying_reward")} for k in self.ks},
            }
        result = self.sources[source]
        result["prompts"] += 1
        result["complete_groups"] += group["complete"]
        result["unresolved_groups"] += not group["complete"]
        result["completion_tokens"] += group["completion_tokens"]
        result["truncated"] += group["truncated"]
        if group["successes"] is not None:
            result["success_histogram"][group["successes"]] += 1
        for k, fields in group["metrics"].items():
            for field, value in fields.items():
                if value is not None:
                    result["metrics"][k][field][0] += value
                    result["metrics"][k][field][1] += 1

    def finish(self):
        result = copy.deepcopy(self.sources)
        for source in result.values():
            source["success_histogram"] = dict(source["success_histogram"])
            for fields in source["metrics"].values():
                for field in ("pass", "varying_reward"):
                    total, count = fields[field]
                    fields[field] = total / count if count else None
                    fields[field + "_groups"] = count
        return result


def draw_path(output, prompt, number):
    # Avoid millions of directory entries in one directory on full-corpus studies.
    return output / "draws" / prompt["id"][:2] / f"{prompt['id']}-{number:03d}.json"


def read_draw(path, prompt, number, contract_hash):
    record = json.loads(path.read_text())
    expected = {"contract_sha256": contract_hash, "id": prompt["id"], "source": prompt["source"], "draw": number}
    if any(record.get(key) != value for key, value in expected.items()):
        raise InputError(f"Draw contract/identity mismatch: {path}")
    return record


def summarize_saved(output, prompt_path, draws, contract_hash):
    """Stream one prompt group at a time; publish a small summary and JSONL group evidence."""
    summary = SourceSummary()
    temporary = output / "groups.pending.jsonl"
    count = 0
    checksum = hashlib.sha256()
    with temporary.open("wb") as stream:
        for prompt in jsonl_rows(prompt_path):
            rows = []
            for number in range(draws):
                record = read_draw(draw_path(output, prompt, number), prompt, number, contract_hash)
                rows.append(
                    {
                        key: record.get(key, default)
                        for key, default in (("score", None), ("completion_tokens", 0), ("truncated", False))
                    }
                )
                del record
            group = {"source": prompt["source"], **summarize_group(prompt["id"], rows)}
            summary.add(prompt["source"], group)
            data = run_data._encoded(group)
            stream.write(data)
            checksum.update(data)
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / "groups.jsonl")
    return summary.finish(), {"path": "groups.jsonl", "sha256": checksum.hexdigest(), "rows": count}


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
    pending_prompts = output / "prompts.pending.jsonl"
    selection = write_prompt_manifest(prompts, pending_prompts)
    prompt_path = output / selection["path"]
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
            "schema_version": 2,
            "model": spec.model,
            "server_model_info": model_info,
            "prepared_manifest": json.loads(Path(prepared["manifest"]).read_text()),
            "sampling": sampling,
            "reward_policy": dataclasses.asdict(config.core),
            "judge_registry": resolved,
            "draws": options.draws,
            "seed": options.seed,
            "populations": populations,
            "selection": selection,
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
        if prompt_path.exists() and file_digest(prompt_path) != selection["sha256"]:
            raise InputError("Saved prompt manifest changed; use a new output directory")
        pending_prompts.replace(prompt_path)
        artifacts.atomic_json(path, contract)
        contract_hash = digest(contract)
        (output / "draws").mkdir(exist_ok=True)

        async def draw(prompt, number):
            path = draw_path(output, prompt, number)
            if path.exists():
                read_draw(path, prompt, number, contract_hash)
                return
            record = {
                "id": prompt["id"],
                "source": prompt["source"],
                "draw": number,
                "contract_sha256": contract_hash,
                "score": None,
                "status": "generation_error",
            }
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

        jobs = ((prompt, n) for prompt in jsonl_rows(prompt_path) for n in range(options.draws))
        await run_workers(jobs, draw, options.concurrency)
    sources, groups = summarize_saved(output, prompt_path, options.draws, contract_hash)
    report = {
        "contract_sha256": contract_hash,
        "schema_version": 2,
        "sources": sources,
        "groups": groups,
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
