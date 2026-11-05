"""Versioned candidate adapters: preserve prompts and make omissions reviewable.

Inputs are local JSONL hygiene exports, never downloaded or silently resampled.
Factories are selected by reviewed adapter code, not dataset-supplied imports.
"""

import argparse
import copy
import hashlib
import importlib
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from open_instruct.miles.rewards import candidate_verifiers as local
from open_instruct.miles.rewards import litmus, nvidia_constraints, reasoning_gym

FACTORIES = {
    "candidate": "open_instruct.miles.rewards.candidate_verifiers.CandidateVerifier",
    "math": "open_instruct.ground_truth_utils.MathVerifier",
    "structured": "open_instruct.miles.rewards.structured_outputs.StructuredOutputVerifier",
    "instruction": "open_instruct.miles.rewards.nvidia_constraints.InstructionVerifier",
    "calendar": "open_instruct.miles.rewards.nvidia_constraints.CalendarVerifier",
    "reasoning_gym": "open_instruct.miles.rewards.reasoning_gym.ReasoningGymVerifier",
    "litmus": "open_instruct.miles.rewards.litmus.LitmusVerifier",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def messages(raw):
    params = raw.get("responses_create_params") or {}
    if params.get("tools"):
        raise ValueError("requires_tool_environment")
    value = params.get("input", raw.get("input", raw.get("prompt", raw.get("question", raw.get("problem")))))
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value or value[-1].get("role") != "user":
        raise ValueError("requires_text_conversation_ending_with_user")
    result = copy.deepcopy(value)
    for message in result:
        if message.get("role") not in {"system", "user", "assistant"} or set(message) - {"role", "content", "type"}:
            raise ValueError("unsupported_message_fields_or_role")
        content = message.get("content")
        if isinstance(content, list):
            if any(part.get("type") not in {"text", "input_text"} for part in content):
                raise ValueError("requires_nontext_environment")
            content = "\n".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise ValueError("requires_text_content")
        role = message["role"]
        message.clear()
        message.update(role=role, content=content)
    return result


def adapt(raw, kind, options=None):
    options = options or {}
    ms = messages(raw)
    meta = raw.get("verifier_metadata") or {}
    if kind == "nemo_v1":
        # Structural dispatch only, never a topic/success/difficulty filter.
        if "schema_str" in raw:
            kind = "structured_v1"
        elif "verifier" in raw:
            kind = "format_v1"
        elif "exp_cal_state" in raw:
            kind = "calendar_v1"
        elif (
            "instruction_id_list" in raw
            or "instruction_id_list" in meta
            or "instruction_id_list" in (raw.get("args") or {})
        ):
            kind = "instruction_v1"
        elif raw.get("options") and "expected_answer" in raw:
            kind = "mcqa_v1"
        elif raw.get("verifier_type") == "math" or "math_with_judge" in (raw.get("agent_ref") or {}).get("name", ""):
            raise ValueError("math_or_judge_requires_explicit_adapter")
        else:
            raise ValueError("unsupported_reward_contract")
    if kind == "structured_v1":
        structured = importlib.import_module("open_instruct.miles.rewards.structured_outputs")
        target = {
            "format": raw["schema_type"],
            "schema": json.loads(raw["schema_str"]),
            "schema_policy": options.get("schema_policy", "nemo_strict_v1"),
            "coerce_types": True,
        }
        structured.validate_target(target)
        name = "structured"
        stratum = kind + ":" + target["format"]
    elif kind == "format_v1":
        target = {"kind": "format", "value": copy.deepcopy(raw["verifier"])}
        local.validate_target(target)
        name, stratum = "candidate", kind
    elif kind == "instruction_v1":
        original = raw.get("args") or (meta if "instruction_id_list" in meta else raw)
        target = {
            "family": "nemo",
            "target": {
                "instruction_id_list": original["instruction_id_list"],
                "kwargs": original.get("kwargs", original.get("instruction_kwargs")),
                "grading_mode": original.get("grading_mode", "all"),
            },
        }
        nvidia_constraints.instruction_contract(json.dumps(target, sort_keys=True))
        name, stratum = "instruction", kind + ":" + "+".join(sorted(target["target"]["instruction_id_list"]))
    elif kind == "calendar_v1":
        target = copy.deepcopy(raw["exp_cal_state"])
        nvidia_constraints.calendar_contract(target)
        name, stratum = "calendar", kind
    elif kind == "litmus_v02_v1":
        target = {k: raw[k] for k in ("expected_answer", "answer_type", "output_regex", "match") if k in raw}
        litmus.validate_target(target)
        name, stratum = "litmus", kind + ":" + str(raw.get("method")) + ":" + str(raw.get("property"))
    elif kind == "reasoning_gym_v1":
        target = copy.deepcopy(raw.get("entry", raw))
        task = target.get("metadata", {}).get("source_dataset")
        if task not in reasoning_gym.TASKS:
            raise ValueError("unqualified_reasoning_gym_family:" + str(task))
        if not isinstance(target.get("answer"), str) or not target["answer"].strip():
            raise ValueError("missing_reasoning_gym_answer")
        name, stratum = "reasoning_gym", kind + ":" + task
    elif kind == "mcqa_v1":
        choices = sorted(
            {key for option in raw.get("options", []) for key, value in option.items() if value is not None}
        )
        displayed = sorted(set(re.findall(r"(?m)^\s*([A-Z])\s*[:.)]", ms[-1]["content"])))
        if choices and choices != displayed:
            raise ValueError("displayed_choices_mismatch")
        target = {
            "kind": options.get("choice_contract", "mcqa_final_choice"),
            "value": str(raw["expected_answer"]).strip().upper(),
            "choices": choices or displayed,
        }
        if target["kind"] not in {"mcqa", "mcqa_final_choice"}:
            raise ValueError("unknown_choice_contract")
        local.validate_target(target)
        name, stratum = "candidate", kind
    elif kind == "math_v1":
        target = raw.get("expected_answer", raw.get("answer", raw.get("solution")))
        if type(target) not in (str, int, float) or not str(target).strip():
            raise ValueError("missing_math_reference")
        if re.search(r"\b(prove|proof|show that)\b", ms[-1]["content"], re.I):
            raise ValueError("proof_requires_judged_contract")
        target = str(target).strip()
        name, stratum = "math", kind
    elif kind == "candidate_v1":
        target = copy.deepcopy(raw["target"])
        local.validate_target(target)
        name, stratum = "candidate", kind + ":" + target["kind"]
    else:
        raise ValueError("unknown_adapter:" + kind)
    return ms, target, name, stratum


def checked_file(descriptor, root):
    path = (root / descriptor["path"]).resolve()
    if file_hash(path) != descriptor["sha256"]:
        raise ValueError("Input hash mismatch: " + str(path))
    return path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(plan_path, output, tokenizer):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get("schema_version") != 1 or not plan.get("sources"):
        raise ValueError("Expected versioned source plan")
    output.mkdir(parents=True, exist_ok=False)
    catalog = {"schema_version": 1, "sources": {}}
    for name, spec in plan["sources"].items():
        if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name) is None:
            raise ValueError("Invalid source name")
        root = output / name
        root.mkdir()
        # A prior hygiene receipt is required; this operation does not rerun decon.
        checked_file(spec["hygiene_receipt"], plan_path.parent)
        counts, strata, registry, identities, prompts = Counter(), Counter(), {}, set(), Counter()
        with (root / "prompts.jsonl").open("w") as rows, (root / "dispositions.jsonl").open("w") as dispositions:
            for descriptor in spec["inputs"]:
                path = checked_file(descriptor, plan_path.parent)
                if not descriptor.get("revision") or descriptor.get("split") not in {"train", "unassigned"}:
                    raise ValueError("Inputs require pinned revision and train/unassigned split")
                for index, line in enumerate(path.open()):
                    wrapper = json.loads(line)
                    raw = wrapper.get("raw_record", wrapper)
                    identity = wrapper.get("id", f"{descriptor['sha256']}:{index}")
                    if identity in identities:
                        raise ValueError("Repeated input identity: " + identity)
                    identities.add(identity)
                    counts["input_rows"] += 1
                    try:
                        ms, target, verifier, stratum = adapt(raw, spec["adapter"], spec.get("options"))
                    except (ValueError, KeyError, TypeError) as error:
                        reason = f"{type(error).__name__}: {error}"
                        counts["omitted_rows"] += 1
                        dispositions.write(json.dumps({"id": identity, "status": "omitted", "reason": reason}) + "\n")
                        continue
                    text = tokenizer.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
                    tokens = len(tokenizer.encode(text, add_special_tokens=False))
                    registry[verifier] = {"factory": FACTORIES[verifier]}
                    row = {
                        "input": text,
                        "label": target,
                        "metadata": {
                            "sample_id": identity,
                            "source_name": name,
                            "subtask": stratum,
                            "query": ms[-1]["content"],
                            "messages": ms,
                            "raw_record_sha256": digest(raw),
                            "hygiene_status": wrapper.get("hygiene_status", wrapper.get("status")),
                            "hygiene_labels": wrapper.get("labels", []),
                            "prompt_tokens": tokens,
                            "provenance": wrapper.get("provenance", descriptor),
                            "verifiers": [{"name": verifier, "target": target}],
                        },
                    }
                    rows.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    counts["prepared_rows"] += 1
                    counts["max_prompt_tokens"] = max(counts["max_prompt_tokens"], tokens)
                    strata[stratum] += 1
                    prompts[digest(ms)] += 1
        write(root / "verifiers.json", registry)
        entry = {
            "status": "prepared",
            "adapter": spec["adapter"],
            "counts": dict(counts),
            "strata": dict(strata),
            "repeated_prompt_rows": sum(n - 1 for n in prompts.values()),
            "hygiene_receipt": spec["hygiene_receipt"],
            "input_descriptors": spec["inputs"],
        }
        for key, path in [("prompt_data", root / "prompts.jsonl"), ("reward_config", root / "verifiers.json")]:
            entry[key] = {"path": str(path.relative_to(output)), "sha256": file_hash(path)}
        catalog["sources"][name] = entry
    write(output / "plan.json", plan)
    write(output / "catalog.json", catalog)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prepare(args.plan, args.output, tokenizer)


if __name__ == "__main__":
    main()
