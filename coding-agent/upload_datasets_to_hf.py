#!/usr/bin/env python3
"""
Script to process all code search data and upload to HuggingFace Hub.
Uploads to:
- saurabh5/rlvr-code-view-tool (multi-step)
- saurabh5/rlvr-code-view-single-turn (single-turn)
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import argparse


# HuggingFace repository names
HF_OUTPUT_MULTI_STEP_TOOL = "saurabh5/rlvr-code-view-tool"
HF_OUTPUT_SINGLE_STEP = "saurabh5/rlvr-code-view-single-turn"


def _normalize_repo_name(extra: Dict[str, Any]) -> Optional[str]:
    """Try to construct an owner/repo string from various possible keys."""
    candidates = [
        extra.get("repo_full_name"),
        extra.get("repo_name"),
        extra.get("gh_repo"),
    ]
    for c in candidates:
        if isinstance(c, str) and "/" in c and len(c) > 1:
            return c
    owner = extra.get("repo_owner") or extra.get("owner")
    name = extra.get("repo") or extra.get("name") or extra.get("repo_short_name")
    if isinstance(owner, str) and isinstance(name, str) and owner and name:
        return f"{owner}/{name}"
    return None


def _extract_patches(extra: Dict[str, Any]) -> Optional[List[str]]:
    """Extract patches as a list of unified diffs."""
    if extra is None:
        return None
    patches = extra.get("patches")
    if isinstance(patches, list):
        return [str(p) for p in patches if isinstance(p, (str, bytes))]
    single = extra.get("patch") or extra.get("diff") or extra.get("unified_diff")
    if isinstance(single, (str, bytes)):
        return [str(single)]
    meta = extra.get("patch_metadata")
    if isinstance(meta, dict):
        return _extract_patches(meta)
    return None


def build_tool_context(instance_id: str, entry: Dict[str, Any], yaml_ground_truth: Optional[Dict[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Build per-instance tool_context with repo_name, base_commit, and patches."""
    extra: Optional[Dict[str, Any]] = None
    if yaml_ground_truth and instance_id in yaml_ground_truth:
        extra = yaml_ground_truth[instance_id] or {}
    if (not extra) and isinstance(entry.get("extra_fields"), dict):
        extra = entry.get("extra_fields")
    if not isinstance(extra, dict):
        return None
    repo_name = _normalize_repo_name(extra)
    base_commit = extra.get("base_commit") or extra.get("commit") or extra.get("sha")
    patches = _extract_patches(extra)
    if not repo_name and not base_commit and not patches:
        return None
    tc: Dict[str, Any] = {}
    if repo_name:
        tc["repo_name"] = repo_name
    if base_commit:
        tc["base_commit"] = base_commit
    if patches:
        tc["patches"] = patches
    return tc or None


def extract_buggy_info_from_yaml(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract buggy info from YAML data structure."""
    buggy_info = {}
    
    if "extra_fields" in data:
        extra = data["extra_fields"]
        if "bug_fn_file" in extra:
            # Store the original file path without /testbed prefix
            buggy_info["bug_fn_file"] = extra["bug_fn_file"]
            buggy_info["file_path"] = f"/testbed/{extra['bug_fn_file']}"
            buggy_info["buggy_function"] = extra.get("bug_fn", "")
            
            # Extract line range if available
            if "line_start" in extra:
                buggy_info["line_start"] = extra["line_start"]
                buggy_info["buggy_line"] = extra["line_start"]
            if "line_end" in extra:
                buggy_info["line_end"] = extra["line_end"]
            if "line_start" in extra and "line_end" in extra:
                buggy_info["view_range"] = [extra["line_start"], extra["line_end"]]
            
            return buggy_info
    
    return None


def extract_buggy_info_from_messages(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract buggy line information from conversation messages."""
    buggy_info = {}
    
    # Look for file views in assistant messages
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
                        
                        # Store the first significant file view
                        if path and "config.py" in path:  # Example heuristic
                            buggy_info["file_path"] = path
                            if view_range:
                                buggy_info["view_range"] = view_range
                                buggy_info["buggy_line"] = view_range[0]
                            return buggy_info
                except (json.JSONDecodeError, KeyError):
                    continue
    
    # Fallback: extract from PR description
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            # Look for file mentions
            if "config.py" in content.lower():
                buggy_info["file_path"] = "/testbed/starlette/config.py"
                buggy_info["description"] = content[:500]
                return buggy_info
    
    return buggy_info


def process_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
    """Process a JSONL file and extract all entries."""
    entries = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc=f"Reading {file_path.name}")):
            try:
                data = json.loads(line.strip())
                if "messages" in data:
                    entries.append(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return entries


def process_yaml_file(file_path: Path) -> List[Dict[str, Any]]:
    """Process a YAML file and extract entries."""
    entries = []
    
    with open(file_path, 'r') as f:
        try:
            data = yaml.safe_load(f)
            if isinstance(data, list):
                for item in data:
                    # Convert YAML format to our expected format
                    if "messages" in item:
                        entries.append(item)
                    elif "extra_fields" in item:
                        # This is the post_instances format
                        # We need to reconstruct messages from the data
                        entry = {
                            "instance_id": item.get("id", "unknown"),
                            "messages": [],
                            "extra_fields": item.get("extra_fields", {})
                        }
                        entries.append(entry)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
    
    return entries


def create_multi_step_dataset(entries: List[Dict[str, Any]], yaml_ground_truth: Dict[str, Dict[str, Any]] = None) -> Dataset:
    """Create multi-step dataset from entries."""
    dataset_items = []
    
    for entry in tqdm(entries, desc="Processing multi-step entries"):
        messages = entry.get("messages", [])
        if not messages:
            continue
            
        instance_id = entry.get("instance_id", entry.get("id", f"entry_{len(dataset_items)}"))
        
        # Filter out training flags while preserving order
        conversation = []
        for msg in messages:
            if not msg.get("train", True):  # Include if train is False or not present
                conversation.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Truncate conversation to pre-view phase (keep up to last non-view tool call and its response)
        def _parse_tool_commands_from_content(content: str):
            tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
            tool_calls = re.findall(tool_call_pattern, content, re.DOTALL)
            commands = []
            for tool_call_str in tool_calls:
                try:
                    tool_call = json.loads(tool_call_str)
                    cmd = tool_call.get("arguments", {}).get("command")
                    if isinstance(cmd, str):
                        commands.append(cmd)
                except Exception:
                    continue
            return commands

        def _truncate_to_pre_view_phase(conv: List[Dict[str, str]]):
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
            last_non_view_idx = None
            for idx, m in enumerate(conv):
                if m.get("role") != "assistant":
                    continue
                content = m.get("content", "")
                if "<tool_call>" not in content:
                    continue
                cmds = _parse_tool_commands_from_content(content)
                if any(not _is_view_like_command(c) for c in cmds):
                    last_non_view_idx = idx
            # If there was no non-view command, trim up to the first view-only call
            if last_non_view_idx is None:
                first_view_idx = None
                for idx, m in enumerate(conv):
                    if m.get("role") != "assistant":
                        continue
                    content = m.get("content", "")
                    if "<tool_call>" not in content:
                        continue
                    cmds = _parse_tool_commands_from_content(content)
                    if cmds and all(_is_view_like_command(c) for c in cmds):
                        first_view_idx = idx
                        break
                if first_view_idx is not None:
                    return conv[: first_view_idx]
                return conv
            end_idx = last_non_view_idx
            if end_idx + 1 < len(conv):
                nxt = conv[end_idx + 1]
                if nxt.get("role") == "user" and "<tool_response>" in nxt.get("content", ""):
                    end_idx += 1
            return conv[: end_idx + 1]

        conversation = _truncate_to_pre_view_phase(conversation)
        
        if not conversation:
            continue
        
        # Extract buggy info
        buggy_info = None
        if "extra_fields" in entry:
            buggy_info = extract_buggy_info_from_yaml(entry)
        if not buggy_info:
            buggy_info = extract_buggy_info_from_messages(messages)
        
        # Count tool calls on truncated conversation
        tool_calls = sum(1 for msg in conversation 
                        if msg.get("role") == "assistant" and "<tool_call>" in msg.get("content", ""))
        
        # Create ground_truth JSON string with search info
        ground_truth_data = {}
        
        # First priority: check yaml_ground_truth mapping
        if yaml_ground_truth and instance_id in yaml_ground_truth:
            yaml_extra = yaml_ground_truth[instance_id]
            if "bug_fn_file" in yaml_extra:
                ground_truth_data["bug_fn_file"] = yaml_extra["bug_fn_file"]
            if "line_start" in yaml_extra:
                ground_truth_data["line_start"] = yaml_extra["line_start"]
            if "line_end" in yaml_extra:
                ground_truth_data["line_end"] = yaml_extra["line_end"]
        
        # Second priority: check extra_fields in the entry itself
        elif "extra_fields" in entry:
            extra = entry["extra_fields"]
            if "bug_fn_file" in extra:
                ground_truth_data["bug_fn_file"] = extra["bug_fn_file"]
            if "line_start" in extra:
                ground_truth_data["line_start"] = extra["line_start"]
            if "line_end" in extra:
                ground_truth_data["line_end"] = extra["line_end"]
        
        # Third priority: extract from buggy_info (fallback)
        elif buggy_info:
            ground_truth_data["bug_fn_file"] = buggy_info.get("file_path", "").replace("/testbed/", "")
            if "view_range" in buggy_info and buggy_info["view_range"]:
                ground_truth_data["line_start"] = buggy_info["view_range"][0]
                ground_truth_data["line_end"] = buggy_info["view_range"][1]
            elif "buggy_line" in buggy_info:
                ground_truth_data["line_start"] = buggy_info["buggy_line"]
                ground_truth_data["line_end"] = buggy_info["buggy_line"]
        
        ground_truth = json.dumps(ground_truth_data) if ground_truth_data else "{}"

        # Build tool_context consistent with tool_vllm and api expectations
        tool_context = build_tool_context(instance_id, entry, yaml_ground_truth)
        
        dataset_items.append({
            "instance_id": instance_id,
            "messages": conversation,
            "ground_truth": ground_truth,  # JSON-serialized search info
            "num_turns": len(conversation),
            "tool_calls_made": tool_calls,
            "dataset": "code_search",  # Add static dataset column
            "tool_context": tool_context
        })
    
    return Dataset.from_list(dataset_items)


def create_single_turn_dataset(entries: List[Dict[str, Any]], yaml_ground_truth: Dict[str, Dict[str, Any]] = None) -> Dataset:
    """Create single-turn dataset from entries."""
    dataset_items = []
    
    for entry in tqdm(entries, desc="Processing single-turn entries"):
        instance_id = entry.get("instance_id", entry.get("id", f"entry_{len(dataset_items)}"))
        
        # Extract buggy info - first from YAML ground truth, then from entry
        buggy_info = None
        
        # Check if we have YAML ground truth for this instance
        if yaml_ground_truth and instance_id in yaml_ground_truth:
            yaml_extra = yaml_ground_truth[instance_id]
            buggy_info = {
                "bug_fn_file": yaml_extra.get("bug_fn_file", ""),
                "file_path": f"/testbed/{yaml_extra.get('bug_fn_file', '')}",
                "line_start": yaml_extra.get("line_start"),
                "line_end": yaml_extra.get("line_end"),
                "buggy_line": yaml_extra.get("line_start"),
                "view_range": [yaml_extra.get("line_start"), yaml_extra.get("line_end")] if "line_start" in yaml_extra and "line_end" in yaml_extra else None
            }
        elif "extra_fields" in entry:
            buggy_info = extract_buggy_info_from_yaml(entry)
        
        if not buggy_info:
            buggy_info = extract_buggy_info_from_messages(entry.get("messages", []))
        
        if not buggy_info:
            continue  # Skip entries without identifiable bugs
        
        # Create simplified prompt
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
        
        # Extract bug description
        bug_description = buggy_info.get("description", "")
        if not bug_description:
            # Try to extract from messages
            for msg in entry.get("messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if "<pr_description>" in content:
                        pr_match = re.search(r"<pr_description>(.*?)</pr_description>", content, re.DOTALL)
                        if pr_match:
                            bug_description = pr_match.group(1).strip()
                            break
                    elif len(content) > 50:
                        bug_description = content[:500]
                        break
        
        if not bug_description:
            bug_description = f"Bug in {buggy_info.get('file_path', 'unknown file')}"
        
        user_prompt = (
            f"Repository location: /testbed\n\n"
            f"Bug description:\n{bug_description[:1000]}\n\n"
            f"Please find and view the relevant code."
        )
        
        # Create expected response
        expected_view_call = {
            "name": "str_replace_editor",
            "arguments": {
                "command": "view",
                "path": buggy_info["file_path"]
            }
        }
        
        if "view_range" in buggy_info:
            expected_view_call["arguments"]["view_range"] = buggy_info["view_range"]
        
        expected_response = (
            f"I'll examine the relevant file.\n\n"
            f"<tool_call>\n{json.dumps(expected_view_call, indent=2)}\n</tool_call>"
        )

        # Build tool_context for this instance
        tool_context = build_tool_context(instance_id, entry, yaml_ground_truth)
        
        dataset_items.append({
            "instance_id": instance_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": expected_response}
            ],
            "dataset": "code_search",  # Add static dataset column
            "ground_truth": json.dumps({  # Add ground_truth with search info
                "bug_fn_file": buggy_info.get("bug_fn_file", buggy_info.get("file_path", "").replace("/testbed/", "")),
                "line_start": buggy_info.get("line_start", buggy_info.get("buggy_line", -1)),
                "line_end": buggy_info.get("line_end", buggy_info.get("buggy_line", -1))
            }),
            "tool_context": tool_context
        })
    
    return Dataset.from_list(dataset_items)


def main():
    parser = argparse.ArgumentParser(description="Upload code search datasets to HuggingFace")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="coding-agent/data",
        help="Directory containing the data files"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only process sample.json for testing"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Process data but don't upload to HuggingFace"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # First, load YAML data to get ground truth info
    yaml_ground_truth = {}
    yaml_files = list(data_dir.glob("*.yaml")) + list(data_dir.glob("*.yml"))
    for yaml_file in yaml_files:
        print(f"\nLoading ground truth from {yaml_file.name}...")
        with open(yaml_file, 'r') as f:
            try:
                data = yaml.safe_load(f)
                if isinstance(data, list):
                    for item in data:
                        if "id" in item and "extra_fields" in item:
                            yaml_ground_truth[item["id"]] = item["extra_fields"]
                    print(f"  Loaded ground truth for {len(yaml_ground_truth)} entries")
            except yaml.YAMLError as e:
                print(f"  Error: {e}")
    
    # Collect all entries
    all_entries = []
    
    if args.test_only:
        # Only process sample.json
        sample_file = data_dir / "sample.json"
        if sample_file.exists():
            with open(sample_file, 'r') as f:
                data = json.load(f)
                all_entries.append(data)
            print(f"Loaded 1 entry from sample.json")
    else:
        # Process all data files
        
        # Process JSONL files
        jsonl_files = list(data_dir.glob("*.jsonl"))
        for jsonl_file in jsonl_files:
            print(f"\nProcessing {jsonl_file.name}...")
            entries = process_jsonl_file(jsonl_file)
            all_entries.extend(entries)
            print(f"  Loaded {len(entries)} entries")
        
        # Process JSON files
        json_files = list(data_dir.glob("*.json"))
        for json_file in json_files:
            print(f"\nProcessing {json_file.name}...")
            with open(json_file, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_entries.extend(data)
                    else:
                        all_entries.append(data)
                    print(f"  Loaded entry from {json_file.name}")
                except json.JSONDecodeError as e:
                    print(f"  Error: {e}")
        
        # Process YAML files
        yaml_files = list(data_dir.glob("*.yaml")) + list(data_dir.glob("*.yml"))
        for yaml_file in yaml_files:
            print(f"\nProcessing {yaml_file.name}...")
            entries = process_yaml_file(yaml_file)
            all_entries.extend(entries)
            print(f"  Loaded {len(entries)} entries")
    
    print(f"\nTotal entries loaded: {len(all_entries)}")
    
    if not all_entries:
        print("No entries found!")
        return
    
    # Create datasets
    print("\n" + "="*60)
    print("Creating Multi-Step Dataset")
    print("="*60)
    multi_step_dataset = create_multi_step_dataset(all_entries, yaml_ground_truth)
    print(f"Multi-step dataset size: {len(multi_step_dataset)}")
    
    print("\n" + "="*60)
    print("Creating Single-Turn Dataset")
    print("="*60)
    single_turn_dataset = create_single_turn_dataset(all_entries, yaml_ground_truth)
    print(f"Single-turn dataset size: {len(single_turn_dataset)}")
    
    # Create train/validation splits
    multi_step_dict = DatasetDict({
        "train": multi_step_dataset
    })
    
    single_turn_dict = DatasetDict({
        "train": single_turn_dataset
    })
    
    # Upload to HuggingFace
    if not args.no_upload:
        print("\n" + "="*60)
        print("Uploading to HuggingFace Hub")
        print("="*60)
        
        print(f"\nUploading multi-step dataset to {HF_OUTPUT_MULTI_STEP_TOOL}...")
        try:
            multi_step_dict.push_to_hub(HF_OUTPUT_MULTI_STEP_TOOL, private=False)
            print(f"✅ Successfully uploaded to {HF_OUTPUT_MULTI_STEP_TOOL}")
        except Exception as e:
            print(f"❌ Error uploading multi-step dataset: {e}")
        
        print(f"\nUploading single-turn dataset to {HF_OUTPUT_SINGLE_STEP}...")
        try:
            single_turn_dict.push_to_hub(HF_OUTPUT_SINGLE_STEP, private=False)
            print(f"✅ Successfully uploaded to {HF_OUTPUT_SINGLE_STEP}")
        except Exception as e:
            print(f"❌ Error uploading single-turn dataset: {e}")
        
        print("\n" + "="*60)
        print("Upload Complete!")
        print("="*60)
        print(f"\nDatasets available at:")
        print(f"  Multi-step: https://huggingface.co/datasets/{HF_OUTPUT_MULTI_STEP_TOOL}")
        print(f"  Single-turn: https://huggingface.co/datasets/{HF_OUTPUT_SINGLE_STEP}")
    else:
        print("\n" + "="*60)
        print("Processing complete (upload skipped)")
        print("="*60)
        
    # Print sample entries
    print("\n" + "="*60)
    print("Sample Entries")
    print("="*60)
    
    if len(multi_step_dataset) > 0:
        print("\nMulti-step sample:")
        sample = multi_step_dataset[0]
        print(f"  Instance: {sample['instance_id']}")
        print(f"  Dataset: {sample.get('dataset', 'N/A')}")
        print(f"  Ground truth: {sample.get('ground_truth', 'N/A')[:100]}...")
        print(f"  Turns: {sample['num_turns']}")
        print(f"  Tool calls: {sample['tool_calls_made']}")
    
    if len(single_turn_dataset) > 0:
        print("\nSingle-turn sample:")
        sample = single_turn_dataset[0]
        print(f"  Instance: {sample['instance_id']}")
        print(f"  Dataset: {sample.get('dataset', 'N/A')}")
        print(f"  Ground truth: {sample.get('ground_truth', 'N/A')[:100]}...")


if __name__ == "__main__":
    main()