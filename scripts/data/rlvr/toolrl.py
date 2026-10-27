"""Convert the pinned ToolRL 4K source to prompt-only Open Instruct RL JSONL.

Run from the repository root with ``python -m scripts.data.rlvr.toolrl --help``.
Only preparation is performed: no tools, training, or Hub uploads are invoked.
"""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from urllib import request

from open_instruct import toolrl_utils

REVISION = "8cee13ec0ca72f0461da372a93a6fd8140dbb840"
SOURCE_PATH = "dataset/rlla_4k_raw/rlla_rl.json"
SOURCE_URL = f"https://raw.githubusercontent.com/qiancheng0/ToolRL/{REVISION}/{SOURCE_PATH}"
SOURCE_SHA256 = "f61cf5820afa8ced1a7fcc6b5078d3ccdfdf85623e7ec70783a1a63591620ff0"


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parameter_schema(source):
    """Translate the source's Python-like type labels into JSON Schema."""
    result = {key: value for key, value in source.items() if key != "type"}
    kind = source.get("type", "string")
    aliases = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "dict": "object",
        "list": "array",
        "set": "array",
    }
    if isinstance(kind, list):
        result["type"] = [aliases.get(item.lower(), item.lower()) for item in kind]
        return result
    bare = kind.split(", optional", 1)[0].split(", default=", 1)[0]
    if bare.startswith("List["):
        result.update(type="array", items=parameter_schema({"type": bare[5:-1]}))
    elif bare.startswith("Tuple["):
        result.update(
            type="array", prefixItems=[parameter_schema({"type": item.strip()}) for item in bare[6:-1].split(",")]
        )
    elif bare.startswith("Union["):
        result["type"] = [parameter_schema({"type": item.strip()})["type"] for item in bare[6:-1].split(",")]
    elif bare.startswith("Callable["):
        # Offline examples encode functions as strings; no function is executed.
        result["type"] = "string"
        result["description"] = result.get("description", "") + f" (Source type: {kind})"
    else:
        result["type"] = aliases.get(bare.lower(), bare.lower())
    if result["type"] not in ("string", "integer", "number", "boolean", "array", "object", "null") and not isinstance(
        result["type"], list
    ):
        raise ValueError(f"Unsupported source parameter type: {kind}")
    return result


def source_tools(instruction):
    try:
        menu = instruction.split("**Available Tools**", 1)[1].split("**Steps for Each Turn**", 1)[0]
    except IndexError as error:
        raise ValueError("Missing source tool menu") from error
    pattern = r"(?:^|\n)\d+\. Name: (.*?)\nDescription: (.*?)\nParameters: (.*?)(?=\n\d+\. Name: |\Z)"
    tools = []
    for name, description, raw in re.findall(pattern, menu.strip(), re.DOTALL):
        parameters = json.loads(raw)
        properties = {key: parameter_schema(value) for key, value in parameters.items()}
        required = [
            key
            for key, value in parameters.items()
            if "optional" not in str(value.get("type", ""))
            and "default" not in value
            and "default=" not in str(value.get("type", ""))
        ]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {"type": "object", "properties": properties, "required": required},
                },
            }
        )
    if not tools or len({tool["function"]["name"] for tool in tools}) != len(tools):
        raise ValueError("Missing or duplicate tool definitions")
    return tools


def canonical_calls(calls, tools):
    schemas = {tool["function"]["name"]: tool["function"]["parameters"]["properties"] for tool in tools}
    result = []
    for call in calls:
        if call["name"] not in schemas:
            raise ValueError("Reference/history calls an unavailable function")
        parameters = {}
        for key, value in call["parameters"].items():
            if key not in schemas[call["name"]]:
                raise ValueError("Reference/history uses an undeclared parameter")
            raw = value if isinstance(value, str) else json.dumps(value, allow_nan=False)
            parameters[key] = toolrl_utils.parameter_value(raw, schemas[call["name"]][key])
        result.append({"name": call["name"], "parameters": parameters})
    return result


def source_messages(history, tools):
    header = "**Dialogue Records History**"
    if not history.startswith(header):
        raise ValueError("Missing dialogue history header")
    remaining = history[len(header) :].strip()
    messages, reasoning = [], None
    while remaining:
        match = re.match(r"<(user|think|tool_call|obs|response)>(.*?)</\1>", remaining, re.DOTALL)
        if not match:
            raise ValueError("Malformed dialogue history")
        tag, content = match[1], match[2].strip()
        if tag == "think":
            if reasoning is not None:
                raise ValueError("Repeated history reasoning")
            reasoning = content
        elif tag in ("tool_call", "response"):
            if reasoning is None:
                raise ValueError("History assistant turn lacks reasoning")
            message = {
                "role": "assistant",
                "content": content if tag == "response" else "",
                "reasoning_content": reasoning,
            }
            if tag == "tool_call":
                calls = canonical_calls(toolrl_utils.parse_calls(match[0]), tools)
                message["tool_calls"] = [
                    {"type": "function", "function": {"name": call["name"], "arguments": call["parameters"]}}
                    for call in calls
                ]
            messages.append(message)
            reasoning = None
        else:
            if reasoning is not None:
                raise ValueError("Unfinished history reasoning")
            messages.append({"role": "user" if tag == "user" else "tool", "content": content})
        remaining = remaining[match.end() :].strip()
    if reasoning is not None or not messages or messages[-1]["role"] not in ("user", "tool"):
        raise ValueError("History must end at a user/tool turn")
    return messages


def convert_rows(rows, *, eval_count=80, seed=31415, think_prefix_in_prompt=False):
    """Retain structurally valid examples; identical inputs stay in one partition.

    Unlike upstream's row shuffle, grouping also prevents identical dialogue inputs
    with different tool menus from crossing the train/eval boundary. Whole groups
    can make the requested eval count slightly larger. This is a new split, not a
    reproduction of the paper's split or a decontamination check against BFCL.
    """
    if type(eval_count) is not int or not 0 < eval_count < len(rows):
        raise ValueError("eval_count must be positive and smaller than the source row count")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    groups = {}
    converted = []
    rejected = []
    call_counts = Counter()
    normalized_reference_rows = []
    verifier = "toolrl_think_prefix" if think_prefix_in_prompt else "toolrl"
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip() for key in ("instruction", "input", "output")
        ):
            raise ValueError(f"Row {index}: expected nonempty instruction, input, and output strings")
        try:
            _, calls = toolrl_utils.reference_parts(row["output"])
            tools = source_tools(row["instruction"])
            messages = source_messages(row["input"], tools)
            source_calls = calls
            calls = canonical_calls(calls, tools)
            if encoded(source_calls) != encoded(calls):
                normalized_reference_rows.append(index)
            target = {"format": "olmo-toolrl-v1", "calls": calls, "tools": tools}
        except ValueError as error:
            rejected.append(
                {"source_row": index, "reason": str(error), "reference_sha256": sha(row["output"].encode())}
            )
            continue
        group = sha(row["input"].encode())
        groups.setdefault(group, []).append(index)
        call_counts[len(calls)] += 1
        converted.append(
            {
                "messages": messages,
                "tools": tools,
                "ground_truth": target,
                "dataset": "toolrl",
                "metadata": {
                    "prepared_sample_id": f"toolrl:{REVISION}:{index}",
                    "source_revision": REVISION,
                    "source_row": index,
                    "dialogue_group": group,
                    "reference_call_count": len(calls),
                    "verifiers": [{"name": verifier, "target": target, "weight": 1.0}],
                },
            }
        )
    if eval_count >= len(converted):
        raise ValueError("Too few valid references for the requested eval_count")
    held_out = set()
    for group in sorted(groups, key=lambda key: sha(f"{seed}:{key}".encode())):
        if len(held_out) >= eval_count:
            break
        held_out.update(groups[group])
    if len(held_out) == len(converted):
        raise ValueError("Grouping leaves no training rows; reduce eval_count or use more distinct dialogues")
    partitions = {"train": [], "eval": []}
    for row in converted:
        partitions["eval" if row["metadata"]["source_row"] in held_out else "train"].append(row)
    summary = {
        "source_rows": len(rows),
        "retained_rows": len(converted),
        "rejected_rows": rejected,
        "normalized_reference_rows": normalized_reference_rows,
        "target_format": "olmo-toolrl-v1",
        "unique_dialogues": len(groups),
        "reference_call_counts": dict(sorted(call_counts.items())),
        "partition_rows": {split: len(values) for split, values in partitions.items()},
        "eval_count_requested": eval_count,
        "split_seed": seed,
        "split_method": "sha256-seeded-dialogue-groups-v1",
        "verifier": verifier,
        "bfcl_overlap_audited": False,
    }
    return partitions, summary


def prepare(source: bytes, output: Path, **options):
    if sha(source) != SOURCE_SHA256:
        raise ValueError("Source SHA256 does not match the pinned ToolRL release")
    rows = json.loads(source)
    if not isinstance(rows, list) or len(rows) != 4000:
        raise ValueError("Expected the pinned 4000-row ToolRL source")
    partitions, summary = convert_rows(rows, **options)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        artifacts = {}
        for split, values in partitions.items():
            raw = b"".join(encoded(row) for row in values)
            filename = f"{split}.jsonl"
            (staging / filename).write_bytes(raw)
            artifacts[split] = {"path": filename, "sha256": sha(raw), "records": len(values)}
        manifest = {
            "schema_version": 1,
            "source": {"url": SOURCE_URL, "revision": REVISION, "sha256": SOURCE_SHA256},
            "summary": summary,
            "artifacts": artifacts,
            "miles": {
                "input_key": "messages",
                "tools_key": "tools",
                "tool_format": "olmo-toolrl-v1",
                "metadata_key": "metadata",
                "label_key": "ground_truth",
                "apply_chat_template": True,
                # Required compatibility identifier for the existing manifest importer;
                # the importer resolves current open_instruct verifier factories.
                "custom_rm_path": "olmo_miles.rl.rewards.registered_reward",
            },
        }
        (staging / "rl-manifest.json").write_bytes(encoded(manifest))
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local copy of pinned rlla_rl.json; otherwise download it")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument(
        "--think-prefix-in-prompt",
        action="store_true",
        help="Select the verifier for a model chat template whose generation prompt ends in <think>",
    )
    args = parser.parse_args()
    if args.input:
        source = args.input.read_bytes()
    else:
        with request.urlopen(SOURCE_URL, timeout=60) as response:
            source = response.read()
    manifest = prepare(
        source,
        args.output_dir,
        eval_count=args.eval_count,
        seed=args.seed,
        think_prefix_in_prompt=args.think_prefix_in_prompt,
    )
    print(json.dumps(manifest["summary"], indent=2))
    print(f"MILES data.rl_manifest: {args.output_dir.resolve() / 'rl-manifest.json'}")


if __name__ == "__main__":
    main()
