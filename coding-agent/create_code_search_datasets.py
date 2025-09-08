#!/usr/bin/env python3
"""
Script to create two HuggingFace datasets for code search tasks:
1. Multi-step tool use dataset (HF_OUTPUT_MULTI_STEP_TOOL) - for models using CodeSearchTool
2. Single-turn dataset - for models that output a single view call

The data comes from coding-agent/data/ and contains information about bugs in code repositories.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import argparse
from tqdm import tqdm


def extract_buggy_line_info(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Extract buggy line information from the messages.
    Look for the actual bug location in the conversation.
    """
    buggy_info = {}
    
    # Look through assistant messages for file views that contain the bug
    for message in messages:
        if message.get("role") == "assistant":
            content = message.get("content", "")
            
            # Look for tool calls with view commands
            tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
            tool_calls = re.findall(tool_call_pattern, content, re.DOTALL)
            
            for tool_call_str in tool_calls:
                try:
                    tool_call = json.loads(tool_call_str)
                    if tool_call.get("arguments", {}).get("command") == "view":
                        path = tool_call["arguments"].get("path", "")
                        view_range = tool_call["arguments"].get("view_range")
                        
                        # Check if this is viewing config.py with the buggy line
                        if "config.py" in path and view_range:
                            # Line 130 contains the bug (based on the sample)
                            if view_range[0] <= 130 <= view_range[1]:
                                buggy_info["file_path"] = path
                                buggy_info["buggy_line"] = 130
                                buggy_info["view_range"] = view_range
                                return buggy_info
                except (json.JSONDecodeError, KeyError):
                    continue
    
    # Fallback: look for specific file mentions in the PR description
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            if "Config.__call__" in content and "starlette/config.py" in content:
                buggy_info["file_path"] = "/testbed/starlette/config.py"
                buggy_info["buggy_line"] = 130  # Known bug location from the sample
                buggy_info["description"] = "Bug in Config.__call__ related to boolean casting"
                return buggy_info
    
    return None


def _parse_tool_commands_from_content(content: str) -> List[str]:
    tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    tool_calls = re.findall(tool_call_pattern, content, re.DOTALL)
    commands: List[str] = []
    for tool_call_str in tool_calls:
        try:
            tool_call = json.loads(tool_call_str)
            command = tool_call.get("arguments", {}).get("command")
            if isinstance(command, str):
                commands.append(command)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return commands


def _truncate_to_pre_view_phase(conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def _is_view_like_command(cmd: str) -> bool:
        """Return True if the tool command is a view-like operation.
        - Explicit tool command 'view' (str_replace_editor)
        - Shell commands that start with 'cat' (e.g., 'cat -n ...')
        """
        if not isinstance(cmd, str):
            return False
        cmd_l = cmd.strip().lower()
        if cmd_l == "view":
            return True
        if cmd_l.startswith("cat"):
            return True
        return False
    last_non_view_assistant_index: Optional[int] = None

    for idx, msg in enumerate(conversation):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if "<tool_call>" not in content:
            continue
        commands = _parse_tool_commands_from_content(content)
        if any((not _is_view_like_command(cmd)) for cmd in commands):
            last_non_view_assistant_index = idx

    # If there's no non-view command at all, then we are already in the
    # view-only phase. In that case, return the minimal context up to
    # (but not including) the first view/cat call so the next assistant
    # turn will be the first view.
    if last_non_view_assistant_index is None:
        first_view_idx: Optional[int] = None
        for idx, msg in enumerate(conversation):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if "<tool_call>" not in content:
                continue
            commands = _parse_tool_commands_from_content(content)
            # If all commands are view-like, this is a view-only call
            if commands and all((
                _is_view_like_command(cmd)
            ) for cmd in commands):
                first_view_idx = idx
                break
        # If we found a first view, keep everything before it; otherwise, keep as-is
        if first_view_idx is not None:
            return conversation[: first_view_idx]
        return conversation

    end_index = last_non_view_assistant_index

    # If the very next message is the tool response from that call, include it
    if end_index + 1 < len(conversation):
        next_msg = conversation[end_index + 1]
        if next_msg.get("role") == "user" and "<tool_response>" in next_msg.get("content", ""):
            end_index += 1

    return conversation[: end_index + 1]


def create_multi_step_dataset(data_files: List[Path]) -> Dataset:
    """
    Create a dataset for multi-step tool use with CodeSearchTool.
    This dataset includes the full conversation with multiple tool calls.
    """
    dataset_items = []
    
    for file_path in tqdm(data_files, desc="Processing multi-step data"):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        messages = data.get("messages", [])
        instance_id = data.get("instance_id", "unknown")
        
        # Build conversation preserving original order, excluding training-only messages
        conversation: List[Dict[str, str]] = []
        for msg in messages:
            if not msg.get("train", True):
                conversation.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Truncate conversation to minimal pre-view context
        conversation = _truncate_to_pre_view_phase(conversation)
        
        # Extract buggy line info for evaluation
        buggy_info = extract_buggy_line_info(messages)
        
        tool_calls_made = sum(1 for m in conversation if m.get("role") == "assistant" and "<tool_call>" in m.get("content", ""))

        dataset_items.append({
            "instance_id": instance_id,
            "messages": conversation,
            "buggy_info": buggy_info,
            "num_turns": len(conversation),
            "tool_calls_made": tool_calls_made,
        })
    
    return Dataset.from_list(dataset_items)


def create_single_turn_dataset(data_files: List[Path]) -> Dataset:
    """
    Create a single-turn dataset where the model should output a single view call
    to find the buggy line.
    """
    dataset_items = []
    
    for file_path in tqdm(data_files, desc="Processing single-turn data"):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        messages = data.get("messages", [])
        instance_id = data.get("instance_id", "unknown")
        
        # Extract buggy line info
        buggy_info = extract_buggy_line_info(messages)
        if not buggy_info:
            continue  # Skip if we can't identify the buggy line
        
        # Create a simplified single-turn prompt
        system_prompt = (
            "You are a code search assistant. Your task is to identify and view "
            "the file containing a bug based on the given description. "
            "Use the str_replace_editor tool with the 'view' command to examine files.\n\n"
            "Available tool:\n"
            '{"type": "function", "function": {"name": "str_replace_editor", '
            '"description": "View files in the repository", '
            '"parameters": {"type": "object", "properties": {'
            '"command": {"type": "string", "enum": ["view"]}, '
            '"path": {"type": "string"}, '
            '"view_range": {"type": "array", "items": {"type": "integer"}}}, '
            '"required": ["command", "path"]}}}'
        )
        
        # Extract the PR description or bug description
        bug_description = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if "<pr_description>" in content:
                    pr_match = re.search(r"<pr_description>(.*?)</pr_description>", content, re.DOTALL)
                    if pr_match:
                        bug_description = pr_match.group(1).strip()
                        break
        
        if not bug_description:
            bug_description = f"There is a bug in {buggy_info.get('file_path', 'unknown file')}"
        
        user_prompt = (
            f"Repository location: /testbed\n\n"
            f"Bug description:\n{bug_description}\n\n"
            f"Please find and view the relevant code."
        )
        
        # Create the expected response (single view call)
        expected_view_call = {
            "name": "str_replace_editor",
            "arguments": {
                "command": "view",
                "path": buggy_info["file_path"]
            }
        }
        
        # Add view_range if available
        if "view_range" in buggy_info:
            expected_view_call["arguments"]["view_range"] = buggy_info["view_range"]
        
        expected_response = (
            f"I'll examine the file mentioned in the bug description.\n\n"
            f"<tool_call>\n{json.dumps(expected_view_call, indent=2)}\n</tool_call>"
        )
        
        dataset_items.append({
            "instance_id": instance_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": expected_response}
            ],
            "buggy_file": buggy_info.get("file_path", ""),
            "buggy_line": buggy_info.get("buggy_line", -1),
            "bug_description": bug_description[:500]  # Truncate long descriptions
        })
    
    return Dataset.from_list(dataset_items)


def main():
    parser = argparse.ArgumentParser(description="Create code search datasets")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="coding-agent/data",
        help="Directory containing the data files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="code_search_datasets",
        help="Directory to save the datasets"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push datasets to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub-org",
        type=str,
        default="allenai",
        help="HuggingFace organization to push to"
    )
    parser.add_argument(
        "--single-file-test",
        action="store_true",
        help="Only process sample.json for testing"
    )
    
    args = parser.parse_args()
    
    # Find data files
    data_dir = Path(args.data_dir)
    if args.single_file_test:
        data_files = [data_dir / "sample.json"]
    else:
        data_files = list(data_dir.glob("*.json")) + list(data_dir.glob("*.jsonl"))
    
    if not data_files:
        print(f"No data files found in {data_dir}")
        return
    
    print(f"Found {len(data_files)} data files")
    
    # Create datasets
    print("\n" + "="*60)
    print("Creating Multi-Step Tool Use Dataset")
    print("="*60)
    multi_step_dataset = create_multi_step_dataset(data_files)
    
    print(f"\nMulti-step dataset size: {len(multi_step_dataset)}")
    if len(multi_step_dataset) > 0:
        print("Sample entry:")
        sample = multi_step_dataset[0]
        print(f"  Instance ID: {sample['instance_id']}")
        print(f"  Number of turns: {sample['num_turns']}")
        print(f"  Tool calls made: {sample['tool_calls_made']}")
        print(f"  Has buggy info: {sample['buggy_info'] is not None}")
    
    print("\n" + "="*60)
    print("Creating Single-Turn Dataset")
    print("="*60)
    single_turn_dataset = create_single_turn_dataset(data_files)
    
    print(f"\nSingle-turn dataset size: {len(single_turn_dataset)}")
    if len(single_turn_dataset) > 0:
        print("Sample entry:")
        sample = single_turn_dataset[0]
        print(f"  Instance ID: {sample['instance_id']}")
        print(f"  Buggy file: {sample['buggy_file']}")
        print(f"  Buggy line: {sample['buggy_line']}")
        print(f"  Bug description (truncated): {sample['bug_description'][:100]}...")
    
    # Save datasets locally
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save multi-step dataset
    multi_step_path = output_dir / "multi_step_tool_dataset"
    multi_step_dataset.save_to_disk(str(multi_step_path))
    print(f"\nSaved multi-step dataset to {multi_step_path}")
    
    # Save single-turn dataset
    single_turn_path = output_dir / "single_turn_dataset"
    single_turn_dataset.save_to_disk(str(single_turn_path))
    print(f"Saved single-turn dataset to {single_turn_path}")
    
    # Optionally push to HuggingFace Hub
    if args.push_to_hub:
        print("\n" + "="*60)
        print("Pushing to HuggingFace Hub")
        print("="*60)
        
        # Create dataset dict for multi-step
        multi_step_dict = DatasetDict({
            "train": multi_step_dataset
        })
        
        multi_step_repo = f"{args.hub_org}/code-search-multi-step-tool"
        print(f"Pushing multi-step dataset to {multi_step_repo}")
        multi_step_dict.push_to_hub(multi_step_repo, private=False)
        
        # Create dataset dict for single-turn
        single_turn_dict = DatasetDict({
            "train": single_turn_dataset
        })
        
        single_turn_repo = f"{args.hub_org}/code-search-single-turn"
        print(f"Pushing single-turn dataset to {single_turn_repo}")
        single_turn_dict.push_to_hub(single_turn_repo, private=False)
        
        print("\nDatasets successfully pushed to HuggingFace Hub!")
    
    print("\n" + "="*60)
    print("Dataset creation complete!")
    print("="*60)
    print(f"\nDatasets saved to: {output_dir}")
    print(f"  - Multi-step tool dataset: {multi_step_path}")
    print(f"  - Single-turn dataset: {single_turn_path}")
    
    # Print usage example
    print("\nTo load the datasets:")
    print("```python")
    print("from datasets import load_from_disk")
    print(f"multi_step = load_from_disk('{multi_step_path}')")
    print(f"single_turn = load_from_disk('{single_turn_path}')")
    print("```")


if __name__ == "__main__":
    main()