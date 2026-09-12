"""Strict named reward routing shared by the driver and remote reward workers."""

import asyncio
import dataclasses
import json
import os
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from open_instruct.miles import general_judge, judging


def registry():
    raw = os.environ.get(judging.REGISTRY_ENV)
    if not raw:
        return None
    value = json.loads(raw)
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported named judge registry")
    return value


def bound(name):
    value = registry()
    return value is not None and name in value["bindings"]


def resolve_request(args, name, query, prediction, target):
    value = registry()
    if value is None:
        return None
    binding = value["bindings"][name]
    service, rubric = value["judges"][binding["judge"]], value["rubrics"][binding["rubric"]]
    kind = judging.PROFILES[rubric["profile"]]
    if kind != "general-quality" and (target is None or target == ""):
        raise ValueError(f"{name} requires a reference target")
    prompt, digest = general_judge.build_judge_prompt(kind, query=query, prediction=prediction, target=target)
    config = general_judge.GeneralJudgeConfig(
        api_url=service["endpoint"].rstrip("/") + "/chat/completions",
        api_key="EMPTY",
        model=service["model"],
        max_tokens=rubric["max_response_tokens"],
        max_context_length=service["max_context_length"],
        temperature=rubric["temperature"],
        timeout=service["timeout"],
        seed=getattr(args, "seed", 1),
        max_concurrent_calls=service["max_concurrent_calls"],
        check_context=service["mode"] == "managed",
    )
    return config, prompt, digest, "yes-no" if kind.endswith("general_verifier") else "score-1-to-10", binding, digest


def validate_data(prepared, value):
    paths = [prepared["prompt_data"], *prepared.get("eval_prompt_data", [])[1::2]]
    seen = set()
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                metadata = row.get("metadata", {})
                for verifier in metadata.get("verifiers", []):
                    name = verifier["name"]
                    if name not in value["bindings"]:
                        if name.startswith("general-"):
                            raise ValueError(f"Missing named judge binding for {name}")
                        continue
                    seen.add(name)
                    if not isinstance(metadata.get("judge_query"), str) or not metadata["judge_query"]:
                        raise ValueError("Judged data requires the preserved full metadata.judge_query")
                    profile = value["rubrics"][value["bindings"][name]["rubric"]]["profile"]
                    if profile != "open-instruct/general-quality" and verifier.get("target") in (None, ""):
                        raise ValueError(f"{name} requires reference targets")
    if unused := set(value["bindings"]) - seen:
        raise ValueError(f"Judge bindings absent from train/eval data: {sorted(unused)}")


async def probe(output):
    """Require model identity and correct > incorrect grades from the driver node."""
    reports = []
    value = registry()
    for name, binding in value["bindings"].items():
        service = value["judges"][binding["judge"]]

        def models(service=service):
            with urllib.request.urlopen(service["endpoint"].rstrip("/") + "/models", timeout=10) as response:
                return json.load(response)

        advertised = await asyncio.to_thread(models)
        if service["model"] not in {m["id"] for m in advertised["data"]}:
            raise RuntimeError("Judge endpoint advertises the wrong model")
        scores = []
        for response in ("The capital of France is Paris.", "The capital of France is Saturn, a type of sandwich."):
            sample = SimpleNamespace(response=response, metadata={"judge_query": "What is the capital of France?"})
            score = await general_judge.general_judge_score(
                SimpleNamespace(seed=1), sample, name=name, target="Paris."
            )
            scores.append(score)
            reports.append(
                {
                    "binding": name,
                    "response": response,
                    "score": score,
                    "diagnostics": sample.metadata["verifier_diagnostics"],
                }
            )
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(reports, indent=2))
        if scores[0] <= scores[1]:
            raise RuntimeError(f"Judge known-answer contrast failed for {name}")
    return reports


@dataclasses.dataclass
class NamedJudgeConfig:
    name: str


class NamedJudgeVerifier:
    """Trusted registry marker; the reward bridge supplies the entire sample metadata."""

    get_config_class = staticmethod(lambda: NamedJudgeConfig)

    def __init__(self, verifier_config):
        self.name = verifier_config.name

    async def async_call(self, *args, **kwargs):
        raise RuntimeError("Named judges must be called through the MILES sample-aware reward bridge")
