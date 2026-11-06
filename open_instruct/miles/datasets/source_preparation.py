"""Versioned dataset adapters: preserve prompts and make omissions reviewable.

Inputs are local JSONL hygiene exports, never downloaded or silently resampled.
Factories are selected by reviewed adapter code, not dataset-supplied imports.
"""

import argparse
import copy
import dataclasses
import hashlib
import importlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from jsonschema.exceptions import SchemaError
from transformers import AutoTokenizer

from open_instruct import ifeval_constraints
from open_instruct.miles.rewards import complete_code, litmus, nvidia_constraints, reasoning_gym, structured_tool_calls
from open_instruct.miles.rewards import deterministic_verifiers as local

FACTORIES = {
    "complete_code": "open_instruct.miles.rewards.complete_code.CompleteCodeVerifier",
    "reference_judge": "open_instruct.miles.rewards.judge_registry.NamedJudgeVerifier",
    "deterministic": "open_instruct.miles.rewards.deterministic_verifiers.DeterministicVerifier",
    "math": "open_instruct.ground_truth_utils.MathVerifier",
    "structured": "open_instruct.miles.rewards.structured_outputs.StructuredOutputVerifier",
    "instruction": "open_instruct.miles.rewards.nvidia_constraints.InstructionVerifier",
    "ifeval": "open_instruct.miles.rewards.task_verifiers.ManifestIFVerifier",
    "calendar": "open_instruct.miles.rewards.nvidia_constraints.CalendarVerifier",
    "reasoning_gym": "open_instruct.miles.rewards.reasoning_gym.ReasoningGymVerifier",
    "litmus": "open_instruct.miles.rewards.litmus.LitmusVerifier",
}


class PlanError(Exception):
    """The plan itself is wrong; unlike an unsupported row, this stops preparation."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dolci_prompt_messages(value):
    """Recover the unambiguous single-user form of Dolci's flattened chat.

    The publisher joins messages with newlines and adds ``role: `` prefixes.
    That serialization cannot distinguish extra turns from quoted role lines;
    require structured messages for those cases instead of guessing boundaries.
    """
    if value.startswith("user: "):
        content = value[len("user: ") :]
        if re.search(r"\n(?:user|assistant|system|developer):", content):
            raise ValueError("ambiguous_dolci_conversation: supply structured chat messages")
        if not content.strip():
            raise ValueError("empty_dolci_user_message")
        return [{"role": "user", "content": content}]
    if re.match(r"^\s*(?:user|assistant|system|developer):", value, flags=re.IGNORECASE):
        raise ValueError("unsupported_dolci_role_serialization: supply structured chat messages")
    return [{"role": "user", "content": value}]


def messages(raw, *, dolci_prompt=False, structured_tool_answer=False):
    params = raw.get("responses_create_params") or {}
    if not isinstance(params, dict):
        raise ValueError("requires_responses_parameters_object")
    if params.get("tools") and not structured_tool_answer:
        raise ValueError("requires_tool_environment")
    instructions = params.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError("requires_text_instructions")
    system_prompt = raw.get("system_prompt")
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise ValueError("requires_text_system_prompt")
    value = params.get("input", raw.get("input", raw.get("prompt", raw.get("question", raw.get("problem")))))
    if isinstance(value, str):
        value = dolci_prompt_messages(value) if dolci_prompt else [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value or not all(isinstance(message, dict) for message in value):
        raise ValueError("requires_text_conversation_ending_with_user")
    if value[-1].get("role") != "user":
        raise ValueError("requires_text_conversation_ending_with_user")
    result = copy.deepcopy(value)
    for message in result:
        if message.get("role") not in {"system", "developer", "user", "assistant"} or set(message) - {
            "role",
            "content",
            "type",
        }:
            raise ValueError("unsupported_message_fields_or_role")
        content = message.get("content")
        if isinstance(content, list):
            if any(not isinstance(part, dict) or part.get("type") not in {"text", "input_text"} for part in content):
                raise ValueError("requires_nontext_environment")
            content = "\n".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise ValueError("requires_text_content")
        role = message["role"]
        message.clear()
        message.update(role=role, content=content)
    # Match NeMo Gym's Responses-to-chat conversion, using the system role our
    # chat templates render. Preserve the order and text of the leading context.
    leading_parts = [text for text in (system_prompt, instructions) if text]
    leading_count = 0
    for message in result:
        if message["role"] not in {"system", "developer"}:
            break
        leading_parts.append(message["content"])
        leading_count += 1
    result = result[leading_count:]
    if any(message["role"] == "developer" for message in result):
        raise ValueError("requires_leading_developer_messages")
    if leading_parts:
        result.insert(0, {"role": "system", "content": "\n\n".join(leading_parts)})
    return result


def validate_keyword_option(kind, options):
    policy = options.get("keyword_inclusion", "preserve")
    if policy not in {"preserve", "at_least_once_v1"}:
        raise PlanError("Unknown keyword_inclusion policy: " + str(policy))
    if policy != "preserve" and kind != "dolci_if_v1":
        raise PlanError("keyword_inclusion correction requires the dolci_if_v1 adapter")


def dolci_if_target(raw):
    if raw.get("dataset", ["ifeval"]) != ["ifeval"]:
        raise ValueError("dolci_if_v1 requires a single ifeval route")
    ground_truth = raw.get("ground_truth")
    if not isinstance(ground_truth, list) or len(ground_truth) != 1:
        raise ValueError("dolci_if_v1 requires one ground_truth entry")
    return ifeval_constraints.parse_constraint_bundle(ground_truth[0])


def corrected_dolci_target(raw, ms, policy):
    target = dolci_if_target(raw)
    issues = ifeval_constraints.if_training_issues(target)
    if issues:
        raise ValueError(f"Unsuitable IF training target: {issues}")
    if policy == "at_least_once_v1":
        for index, (name, args) in enumerate(zip(target["instruction_id"], target["kwargs"], strict=True)):
            if name != "keywords:word_once":
                continue
            keyword = args.get("keyword")
            if not isinstance(keyword, str) or re.fullmatch(r"[A-Za-z]+", keyword) is None or set(args) != {"keyword"}:
                raise ValueError("Keyword inclusion correction requires the audited alphabetic keyword shape")
            # This repair is for the audited generator wording, not arbitrary
            # instructions which might explicitly require exactly one occurrence.
            instruction = f"Include keyword {keyword.strip()} in your response."
            if instruction not in ms[-1]["content"]:
                raise ValueError("Keyword inclusion correction requires the original inclusion instruction")
            target["instruction_id"][index] = "keywords:frequency"
            target["kwargs"][index] = {"keyword": keyword, "frequency": 1, "relation": "at least"}
    return target


def adapt(raw, kind, options=None):
    options = options or {}
    validate_keyword_option(kind, options)
    if kind == "routed_v1":
        value = raw
        for key in options["field"].split("."):
            value = value[key]
        route = options["routes"].get(value)
        if not route or route["adapter"] == "routed_v1":
            raise ValueError("unmapped_or_recursive_route:" + str(value))
        return adapt(raw, route["adapter"], route.get("options"))
    if kind == "acecode_v1" and raw.get("context_messages"):
        # AceCode's context is part of the original task, not generated solutions.
        raw = copy.deepcopy(raw)
        context = raw["context_messages"]
        if not isinstance(context, list) or not isinstance(raw.get("question"), str):
            raise ValueError("AceCode requires context messages and a text question")
        raw["input"] = context
    tool_answer = (kind == "structured_v1" or (kind == "nemo_v1" and "schema_str" in raw)) and raw.get(
        "response_mode"
    ) == "tool_call"
    ms = messages(raw, dolci_prompt=kind == "dolci_if_v1", structured_tool_answer=tool_answer)
    meta = raw.get("verifier_metadata") or {}
    if kind == "nemo_v1":
        agent = raw.get("agent_ref")
        agent = agent.get("name") if isinstance(agent, dict) else agent
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
        elif raw.get("verifier_type") == "math" or "math_with_judge" in str(agent or ""):
            raise ValueError("math_or_judge_requires_explicit_adapter")
        else:
            raise ValueError("unsupported_reward_contract")
    if kind in {"acecode_v1", "livecodebench_v1"}:
        if kind == "acecode_v1":
            target = {"mode": "assert", "tests": copy.deepcopy(raw["test_cases"])}
        else:
            target = complete_code.livecodebench_target(meta["unit_tests"])
        complete_code.validate_target(target)
        name, stratum = "complete_code", kind + ":" + target["mode"]
    elif kind == "structured_v1":
        structured = importlib.import_module("open_instruct.miles.rewards.structured_outputs")
        target = {
            "format": raw["schema_type"],
            "schema": json.loads(raw["schema_str"]),
            "schema_policy": options.get("schema_policy", "nemo_strict_v1"),
            "coerce_types": True,
        }
        if tool_answer:
            structured_tool_calls.tools_for_template(raw)
            target.update(
                response_mode="tool_call",
                tool_name=raw.get("tool_name"),
                tool_payload_key=raw.get("tool_payload_key"),
                tool_call_format="olmo3_pythonic_v1",
            )
        structured.validate_target(target)
        # These native parsers have fixed root shapes. A schema demanding the
        # opposite shape cannot be satisfied by any response in that format.
        root_type = target["schema"].get("type")
        allowed_types = {root_type} if isinstance(root_type, str) else set(root_type or [])
        parsed_type = {"csv": "array", "toml": "object"}.get(target["format"])
        if parsed_type and allowed_types and parsed_type not in allowed_types:
            raise ValueError(f"unrepresentable_{target['format']}_root:{root_type}")
        name = "structured"
        stratum = f"{kind}:tool_call" if tool_answer else f"{kind}:{raw['schema_type']}"
    elif kind == "format_v1":
        target = {"kind": "format", "value": copy.deepcopy(raw["verifier"])}
        local.validate_target(target)
        name = "deterministic"
        stratum = kind + ":" + str(raw["verifier"]["type"])
        if raw["verifier"].get("pattern_id"):
            stratum += ":" + raw["verifier"]["pattern_id"]
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
        name, stratum = "instruction", kind + ":" + "+".join(sorted(original["instruction_id_list"]))
    elif kind == "dolci_if_v1":
        target = corrected_dolci_target(raw, ms, options.get("keyword_inclusion", "preserve"))
        name, stratum = "ifeval", kind + ":" + "+".join(sorted(target["instruction_id"]))
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
        task = reasoning_gym.validate_target(target)
        name, stratum = "reasoning_gym", kind + ":" + task
    elif kind == "mcqa_v1":
        choices_data = raw.get("options", [])
        if not isinstance(choices_data, list) or any(not isinstance(option, dict) for option in choices_data):
            raise ValueError("MCQA options must be a list of dictionaries")
        choices = sorted({key for option in choices_data for key, value in option.items() if value is not None})
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
        name, stratum = "deterministic", kind
    elif kind == "math_v1":
        target = (
            raw[options["reference_field"]]
            if "reference_field" in options
            else raw.get("expected_answer", raw.get("answer"))
        )
        if type(target) not in (str, int, float) or not str(target).strip():
            raise ValueError("missing_math_reference")
        if re.search(r"\b(prove|proof|show that)\b", ms[-1]["content"], re.I):
            raise ValueError("proof_requires_judged_contract")
        target = str(target).strip()
        name, stratum = "math", kind
    elif kind == "reference_judge_v1":
        target = raw[options.get("reference_field", "expected_answer")]
        if not isinstance(target, str) or not target.strip():
            raise ValueError("Judged reference requires nonempty text")
        name, stratum = "reference_judge", kind
    elif kind == "deterministic_v1":
        target = copy.deepcopy(raw["target"])
        local.validate_target(target)
        name, stratum = "deterministic", kind + ":" + target["kind"]
    else:
        # A mistyped adapter would otherwise omit every row and still report "prepared".
        raise PlanError("unknown_adapter:" + str(kind))
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
    catalog: dict = {"schema_version": 1, "sources": {}}
    for name, spec in plan["sources"].items():
        if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name) is None:
            raise ValueError("Invalid source name")
        validate_keyword_option(spec["adapter"], spec.get("options") or {})
        root = output / name
        root.mkdir()
        # A prior hygiene receipt is required; this operation does not rerun decon.
        hygiene = checked_file(spec["hygiene_receipt"], plan_path.parent)
        shutil.copyfile(hygiene, root / "hygiene-receipt.json")
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
                    except (ValueError, KeyError, TypeError, SchemaError) as error:
                        reason = f"{type(error).__name__}: {error}"
                        counts["omitted_rows"] += 1
                        dispositions.write(json.dumps({"id": identity, "status": "omitted", "reason": reason}) + "\n")
                        continue
                    template_options = {}
                    if verifier == "structured" and target.get("response_mode") == "tool_call":
                        if "<function_calls>" not in tokenizer.chat_template:
                            raise PlanError("Structured tool answers require the Olmo Pythonic tool-call template")
                        template_options["tools"] = structured_tool_calls.tools_for_template(raw)
                    text = tokenizer.apply_chat_template(
                        ms, tokenize=False, add_generation_prompt=True, **template_options
                    )
                    tokens = len(tokenizer.encode(text, add_special_tokens=False))
                    registry[verifier] = {"factory": FACTORIES[verifier]}
                    if verifier == "complete_code":
                        try:
                            config = complete_code.Config(**spec.get("code_service", {}))
                            complete_code.validate_config(config)
                        except (TypeError, ValueError) as error:
                            raise PlanError(f"Invalid code_service plan configuration: {error}") from error
                        registry[verifier]["config"] = dataclasses.asdict(config)
                    if verifier == "reference_judge":
                        registry[verifier]["config"] = {"name": verifier}
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
                    if spec["adapter"] == "dolci_if_v1":
                        original = dolci_if_target(raw)
                        changes = [
                            {
                                "name": "dolci_keyword_inclusion_v1",
                                "constraint_index": i,
                                "from": before,
                                "to": after,
                                "from_kwargs": original["kwargs"][i],
                                "to_kwargs": target["kwargs"][i],
                            }
                            for i, (before, after) in enumerate(
                                zip(original["instruction_id"], target["instruction_id"], strict=True)
                            )
                            if before != after
                        ]
                        row["metadata"]["original_target_sha256"] = digest(original)
                        row["metadata"]["target_transformations"] = changes
                        counts["transformed_rows"] += bool(changes)
                        counts["transformed_constraints"] += len(changes)
                    if template_options:
                        row["metadata"]["tools"] = template_options["tools"]
                    if verifier == "reference_judge":
                        row["metadata"]["judge_query"] = json.dumps(ms, ensure_ascii=False)
                    rows.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    counts["prepared_rows"] += 1
                    counts["max_prompt_tokens"] = max(counts["max_prompt_tokens"], tokens)
                    strata[stratum] += 1
                    prompts[digest(ms)] += 1
        write(root / "verifiers.json", registry)
        entry = {
            "status": "prepared",
            "adapter": spec["adapter"],
            "options": spec.get("options") or {},
            "counts": dict(counts),
            "strata": dict(strata),
            "repeated_prompt_rows": sum(n - 1 for n in prompts.values()),
            "hygiene_receipt": {
                "path": str((root / "hygiene-receipt.json").relative_to(output)),
                "sha256": file_hash(hygiene),
            },
            "input_descriptors": spec["inputs"],
        }
        for key, path in [
            ("prompt_data", root / "prompts.jsonl"),
            ("reward_config", root / "verifiers.json"),
            ("dispositions", root / "dispositions.jsonl"),
        ]:
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
