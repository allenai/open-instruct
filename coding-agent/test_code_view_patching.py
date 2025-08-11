#!/usr/bin/env python3
"""
Test script for /view_file patch application logic.

- Loads two instances from the YAML (default: PySnooper_10007 and PySnooper_10008)
- Calls the API baseline (no patch) then with patch, validates expected changes appear
- Runs two concurrent requests for the same repo with different patches to test locking
- Verifies repo state resets by calling baseline again and checking original content

Usage:
  python -u coding-agent/test_code_view_patching.py \
    --api http://localhost:1234 \
    --yaml coding-agent/data/post_instances_final.yaml \
    --ids PySnooper_10007 PySnooper_10008
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import requests
import yaml


def load_instances(yaml_path: str, ids: List[str]) -> List[Dict[str, Any]]:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    wanted = set(ids)
    results = []
    for item in data:
        if str(item.get("id")) in wanted:
            results.append(item)
    if len(results) != len(ids):
        missing = wanted - {str(x.get("id")) for x in results}
        raise RuntimeError(f"Could not find all instances in YAML. Missing: {sorted(missing)}")
    return results


def build_requests_from_entry(entry: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    extra = entry.get("extra_fields", {})
    repo_name = extra.get("repo_name")
    if not repo_name:
        # Some YAMLs have repo under this key name
        repo_name = extra.get("repo_full_name")
    if not repo_name:
        raise RuntimeError(f"repo_name not found in extra_fields for id={entry.get('id')}")
    file_path = extra["bug_fn_file"]
    line_start = int(extra.get("line_start", 1))
    line_end = int(extra.get("line_end", line_start))
    # Expand a bit to give context
    view_range = [max(1, line_start - 2), max(line_end + 2, line_start)]
    patch = extra.get("patch")
    if not patch:
        raise RuntimeError(f"patch not found in extra_fields for id={entry.get('id')}")

    base_payload = {
        "repo_name": repo_name,
        "path": file_path,
        "view_range": view_range,
    }
    patch_payload = {
        **base_payload,
        "patches": [patch],
    }
    return base_payload, patch_payload


def post_view(api: str, payload: Dict[str, Any]) -> str:
    r = requests.post(f"{api}/view_file", json=payload, timeout=120)
    r.raise_for_status()
    content = r.json().get("content", "")
    if not isinstance(content, str):
        raise RuntimeError("Invalid response format: missing 'content'")
    return content


def assert_contains(content: str, pattern: str, msg: str):
    if pattern not in content:
        raise AssertionError(f"{msg}: expected to find '{pattern}' in response")


def assert_not_contains(content: str, pattern: str, msg: str):
    if pattern in content:
        raise AssertionError(f"{msg}: expected to NOT find '{pattern}' in response")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:1234", help="Base URL for the code API")
    parser.add_argument(
        "--yaml",
        default="coding-agent/data/post_instances_final.yaml",
        help="Path to post_instances YAML",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=["PySnooper_10007", "PySnooper_10008"],
        help="Instance IDs to test (preferably from the same repo)",
    )
    args = parser.parse_args()

    print(f"Loading instances: {args.ids}")
    entries = load_instances(args.yaml, args.ids)

    # Build payloads
    payloads = []
    for e in entries:
        base, patchy = build_requests_from_entry(e)
        payloads.append((e["id"], base, patchy))

    # Sanity baseline for first instance
    inst_id, base_payload, patch_payload = payloads[0]
    print(f"Baseline (no patch) for {inst_id}...")
    base_content = post_view(args.api, base_payload)

    # Now with patch for first instance
    print(f"With patch for {inst_id}...")
    patched_content = post_view(args.api, patch_payload)

    # Validate the patch took effect for known patterns
    # Heuristics for PySnooper patches in sample:
    if patch_payload["path"].endswith("pycompat.py"):
        # Expect total_seconds addition to show up
        assert_contains(
            patched_content,
            "total_seconds()",
            f"{inst_id}: pycompat.py patched content",
        )
    if patch_payload["path"].endswith("tracer.py"):
        # Expect assignment to be 'self.path = path'
        assert_contains(
            patched_content,
            "self.path = path",
            f"{inst_id}: tracer.py patched content",
        )

    # After patch call, confirm reset by calling baseline again (should not include patch-only strings)
    print(f"Post-patch baseline reset check for {inst_id}...")
    base_content2 = post_view(args.api, base_payload)
    if base_payload["path"].endswith("pycompat.py"):
        assert_not_contains(
            base_content2,
            "total_seconds()",
            f"{inst_id}: reset check for pycompat.py",
        )
    if base_payload["path"].endswith("tracer.py"):
        # Original code had pycompat.text_type(path)
        assert_contains(
            base_content2,
            "pycompat.text_type(path)",
            f"{inst_id}: reset check for tracer.py",
        )

    # Concurrency test: run both instances with patches at the same time
    print("\nConcurrency test: applying two patches concurrently on same repo...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = []
        for inst_id_i, _, patch_payload_i in payloads:
            futures.append(ex.submit(post_view, args.api, patch_payload_i))
        results = []
        for fut in as_completed(futures):
            results.append(fut.result())
    print(f"Concurrency test completed in {time.time() - t0:.2f}s")

    # Validate each returned content contains some expected signal from its own patch
    for (inst_id_i, _, patch_payload_i), content in zip(payloads, results):
        if patch_payload_i["path"].endswith("pycompat.py"):
            assert_contains(content, "total_seconds()", f"{inst_id_i}: concurrent pycompat")
        if patch_payload_i["path"].endswith("tracer.py"):
            assert_contains(content, "self.path = path", f"{inst_id_i}: concurrent tracer")

    # Final reset verification: baseline both again
    print("\nFinal reset verification...")
    for inst_id_i, base_payload_i, _ in payloads:
        content = post_view(args.api, base_payload_i)
        if base_payload_i["path"].endswith("pycompat.py"):
            assert_not_contains(content, "total_seconds()", f"{inst_id_i}: final reset pycompat")
        if base_payload_i["path"].endswith("tracer.py"):
            assert_contains(content, "pycompat.text_type(path)", f"{inst_id_i}: final reset tracer")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()

