#!/usr/bin/env python3
"""
Quick test runner for CodeViewTool using the first example from a JSONL conversation
that contains <tool_call> blocks for the 'str_replace_editor' tool.

Usage:
  python -u coding-agent/test_code_view_tool_from_jsonl.py \
    --jsonl /weka/oe-adapt-default/saurabhs/repos/open-instruct-3/coding-agent/data/old/ft_hermes_search_swesmith_think_atk_ru_rc_SYSTEM_WITH_TOOL_FIND.jsonl \
    --api http://localhost:1234/view_file \
    --index 0

Notes:
- The --api argument should point directly to the view endpoint (CodeViewTool posts to the exact URL provided).
- If no --repo is provided, the script will attempt to infer it from the tool_call path using keys in open_instruct/code/docker_images.json. If inference fails, CodeViewTool may default to "testbed" which often does not exist.
"""

import argparse
import json
import re
from typing import Any, Dict, Optional, List

from open_instruct.tool_utils.tool_vllm import CodeViewTool


def infer_repo_name_from_block(block: str) -> Optional[str]:
    """Attempt to infer repo_name from known docker_images.json keys based on common paths.

    Heuristic:
    - If path includes "/testbed/starlette/" -> "encode/starlette"
    - If path includes "/testbed/gunicorn/" -> "benoitc/gunicorn"
    - Otherwise return None and rely on --repo
    """
    try:
        inner = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", block, re.DOTALL).group(1)
        obj = json.loads(inner)
        path = obj.get("arguments", {}).get("path", "") or ""
    except Exception:
        return None
    if "/testbed/starlette/" in path or path.endswith("/testbed/docs/config.md"):
        return "encode/starlette"
    if "/testbed/gunicorn/" in path:
        return "benoitc/gunicorn"
    return None


def load_jsonl_record(jsonl_path: str, index: int) -> Dict[str, Any]:
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Index {index} out of range for file {jsonl_path}")


def iter_all_strings(obj: Any):
    """Yield all string values recursively from a nested JSON-like object."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_all_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_all_strings(v)


def build_first_valid_view_block(prompt: str) -> Optional[str]:
    """
    Scan all <tool_call> blocks within the given prompt, parse the inner JSON, and
    return the first block (tags + JSON) whose command == 'view'.
    """
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", prompt, re.DOTALL)
    for inner in blocks:
        try:
            obj = json.loads(inner)
            if obj.get("arguments", {}).get("command") == "view":
                return f"<tool_call>\n{inner}\n</tool_call>"
        except Exception:
            continue
    return None


def find_first_valid_view_block_in_record(record: Dict[str, Any]) -> Optional[str]:
    """
    Search across all strings in the record for <tool_call> blocks and return the first
    valid JSON block whose command == 'view'.
    """
    for s in iter_all_strings(record):
        if "<tool_call>" in s and "</tool_call>" in s:
            block = build_first_valid_view_block(s)
            if block:
                return block
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        default="/weka/oe-adapt-default/saurabhs/repos/open-instruct-3/coding-agent/data/old/ft_hermes_search_swesmith_think_atk_ru_rc_SYSTEM_WITH_TOOL_FIND.jsonl",
        help="Path to JSONL dataset containing conversations with <tool_call> blocks.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Record index within the JSONL to use.",
    )
    parser.add_argument(
        "--api",
        default="http://localhost:1234/view_file",
        help="Full URL to the view endpoint that accepts POST requests (CodeViewTool posts to this URL).",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Optional repo_name override; defaults to CodeViewTool's internal inference (often 'testbed').",
    )
    args = parser.parse_args()

    record = load_jsonl_record(args.jsonl, args.index)
    # Pick the first valid JSON tool call with command == 'view' across the entire record.
    first_view_block = find_first_valid_view_block_in_record(record)
    if not first_view_block:
        raise RuntimeError("Failed to locate a valid JSON tool call with command == 'view' in the record")

    # Construct the minimal prompt that contains one tool call; CodeViewTool will parse it.
    minimal_prompt = f"<think>\n\n</think>\n\n{first_view_block}"

    repo_name = args.repo or infer_repo_name_from_block(first_view_block)
    tool = CodeViewTool(api_endpoint=args.api, repo_name=repo_name, start_str="<search>", end_str="</search>")
    result = tool(minimal_prompt)

    print("Called:", result.called)
    print("Timeout:", result.timeout)
    if result.error:
        print("Error:", result.error)
    print("Runtime:", f"{result.runtime:.2f}s")
    print("\n--- Tool Output ---\n")
    print(result.output)


if __name__ == "__main__":
    main()

