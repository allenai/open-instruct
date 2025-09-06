#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Gemma3 preference grading shards into a single HF dataset.

What this does (in order):
1) Concatenate the 10 JSONL shards named parsed_responses_chunked.part_{aa..aj}_fixed(.jsonl)
2) Compute ratings_average per row (length == #models), where
      ratings_average[j] = mean(
          ratings_helpfulness[j],
          ratings_honesty[j],
          ratings_instruction[j],
          ratings_truthfulness[j]
      )
   ONLY if *all* of those entries are non-null. If any rating entry is null,
   we do NOT compute ratings_average and we mark is_valid_row=False.
   We also keep global and per-row counts of null rating entries.
3) Set is_valid_row:
      - False if ANY rating value in any of the four aspect lists is null
      - False if all entries of ratings_average are exactly equal (within 1e-9)
      - True otherwise
4) If is_valid_row:
      - Pick highest-rated index from ratings_average (tie-break by seeded choice)
        -> create `chosen` by appending assistant msg of that index to original prompt list
        -> set chosen_model, chosen_rating
      - Pick lowest-rated index similarly -> rejected, rejected_model, rejected_rating
5) Rename column:
      - old "prompt" (list of chat messages) -> "prompt_msgs"
      - new "prompt" -> first user turn content from prompt_msgs (fallback to "")
6) Upload to Hugging Face Hub as `allenai/preference-gemma3-judge`

Printed stats include:
- Total rows
- Total individual NULL rating entries
- Valid vs invalid row counts (and reasons)
- Tie-break counts for best/worst

Note on averaging (very important and confirmed):
- The average is computed **across the four aspects at the same model index**.
  Example: for index 0, average the 0-th elements of the four rating lists.

Dependencies:
  pip install datasets huggingface_hub orjson tqdm

"""

import argparse
import json
import orjson
import os
import sys
import re
import glob
import math
import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Hugging Face
try:
    from datasets import Dataset
    from huggingface_hub import HfApi, create_repo
except Exception as e:
    print("Hint: pip install datasets huggingface_hub")
    raise


def load_jsonl_lines(path: str):
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def dump_jsonl_lines(path: str, rows):
    with open(path, "wb") as f:
        for row in rows:
            f.write(orjson.dumps(row))
            f.write(b"\n")


def first_user_content(prompt_msgs: List[Dict[str, Any]]) -> str:
    if isinstance(prompt_msgs, list):
        for msg in prompt_msgs:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
    # fallback
    if prompt_msgs and isinstance(prompt_msgs[0], dict):
        return str(prompt_msgs[0].get("content", ""))
    return ""


def equal_all(values: List[float], tol: float = 1e-9) -> bool:
    if not values:
        return False
    v0 = values[0]
    return all(abs(v - v0) <= tol for v in values[1:])


def nan_or_none(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def get_seed_for_row(prompt_id: str) -> int:
    # stable tie-breaking per row
    h = hashlib.sha256((prompt_id or "no-id").encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def compute_ratings_average(row: Dict[str, Any]) -> Tuple[Optional[List[float]], int, bool]:
    """
    Returns (ratings_average or None, null_count, has_any_null)
    """
    aspects = [
        row.get("ratings_helpfulness"),
        row.get("ratings_honesty"),
        row.get("ratings_instruction"),
        row.get("ratings_truthfulness"),
    ]
    # Count nulls across all aspects
    null_count = 0
    has_any_null = False

    # Basic validation: all aspect lists should exist and be lists
    if any(not isinstance(a, list) for a in aspects):
        # Treat as totally invalid
        for a in aspects:
            if isinstance(a, list):
                null_count += sum(1 for x in a if nan_or_none(x))
        return None, null_count, True

    # Length agreement across aspects
    lens = [len(a) for a in aspects]
    if len(set(lens)) != 1:
        # If lengths differ, count null-like as 0? We mark invalid.
        for a in aspects:
            null_count += sum(1 for x in a if nan_or_none(x))
        return None, null_count, True

    n = lens[0]
    # Scan for nulls
    for a in aspects:
        for x in a:
            if nan_or_none(x):
                has_any_null = True
                null_count += 1

    if has_any_null:
        return None, null_count, True

    # Compute averages per index
    avgs = []
    for j in range(n):
        vals = [aspects[0][j], aspects[1][j], aspects[2][j], aspects[3][j]]
        # all non-null guaranteed
        avgs.append(sum(vals) / 4.0)
    return avgs, null_count, False


def pick_indices_from_avgs(avgs: List[float], seed: int) -> Tuple[int, int, int, int]:
    """
    Returns (best_idx, worst_idx, num_best_ties, num_worst_ties)
    Tie-breaking is done by seeded RNG for reproducibility.
    """
    rnd = random.Random(seed)
    max_val = max(avgs)
    min_val = min(avgs)

    best_cands = [i for i, v in enumerate(avgs) if abs(v - max_val) <= 1e-9]
    worst_cands = [i for i, v in enumerate(avgs) if abs(v - min_val) <= 1e-9]

    best_idx = rnd.choice(best_cands)
    worst_idx = rnd.choice(worst_cands)

    return best_idx, worst_idx, len(best_cands), len(worst_cands)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Directory containing the 10 shards")
    p.add_argument("--output-dir", default=None, help="Where to write merged and processed files (default: input dir)")
    p.add_argument("--merged-out", default="merged_raw.jsonl", help="Concatenated raw JSONL")
    p.add_argument("--processed-out", default="processed_for_hf.jsonl", help="Processed JSONL ready for HF")
    p.add_argument("--repo-id", default="allenai/preference-gemma3-judge", help="HF dataset repo id")
    p.add_argument("--hf-token", default=None, help="HF token (or env HF_TOKEN/HUGGINGFACEHUB_API_TOKEN)")
    p.add_argument("--private", action="store_true", help="Create/push as private dataset")
    p.add_argument("--skip-upload", action="store_true", help="Skip pushing to HF")
    args = p.parse_args()

    input_dir = args.input_dir
    out_dir = args.output_dir or input_dir
    os.makedirs(out_dir, exist_ok=True)

    # Find shards (support with or without .jsonl)
    expected_suffixes = ["aa","ab","ac","ad","ae","af","ag","ah","ai","aj"]
    candidates = []
    for sfx in expected_suffixes:
        # Try the exact names with both patterns
        for pat in [
            f"parsed_responses_chunked.part_{sfx}_fixed.jsonl",
            f"parsed_responses_chunked.part_{sfx}_fixed",
        ]:
            full = os.path.join(input_dir, pat)
            if os.path.isfile(full):
                candidates.append(full)
                break
        else:
            # fallback: glob
            g = glob.glob(os.path.join(input_dir, f"parsed_responses_chunked.part_{sfx}_fixed*"))
            if g:
                candidates.append(sorted(g)[0])
            else:
                print(f"WARNING: Could not find shard for suffix '{sfx}'")

    if len(candidates) != 10:
        print("ERROR: Did not resolve all 10 shards. Found:\n  " + "\n  ".join(candidates))
        sys.exit(1)

    merged_path = os.path.join(out_dir, args.merged_out)
    processed_path = os.path.join(out_dir, args.processed_out)

    # 1) Concatenate shards to merged_raw.jsonl
    print("Concatenating shards ->", merged_path)
    with open(merged_path, "wb") as fout:
        for path in candidates:
            for line in open(path, "rb"):
                if line.strip():
                    fout.write(line if line.endswith(b"\n") else line + b"\n")

    # 2-5) Process rows -> processed.jsonl
    stats = {
        "total_rows": 0,
        "total_null_rating_entries": 0,
        "valid_rows": 0,
        "invalid_rows_any_null": 0,
        "invalid_rows_all_equal": 0,
        "tie_breaks_best": 0,
        "tie_breaks_worst": 0,
    }

    print("Processing rows and computing fields ->", processed_path)
    with open(processed_path, "wb") as fout:
        for row in tqdm(load_jsonl_lines(merged_path), unit="rows"):
            stats["total_rows"] += 1

            # Prepare working copy
            r = dict(row)

            # Compute ratings_average + null counts
            avgs, null_count, has_any_null = compute_ratings_average(r)
            stats["total_null_rating_entries"] += null_count

            # Default flags
            is_valid = True
            reason_all_equal = False

            # Any null rating => invalid
            if has_any_null:
                is_valid = False

            # If we have averages, check if all equal
            if is_valid and avgs is not None and len(avgs) > 0:
                if equal_all(avgs, tol=1e-9):
                    is_valid = False
                    reason_all_equal = True

            # Attach computed fields
            r["ratings_average"] = avgs  # may be None
            r["ratings_nulls_total"] = null_count
            r["is_valid_row"] = bool(is_valid)

            # If valid, produce chosen/rejected
            if is_valid and avgs is not None:
                # sanity check lengths
                models = r.get("instruct_models") or []
                responses = r.get("model_responses") or []
                if not (isinstance(models, list) and isinstance(responses, list) and len(models) == len(responses) == len(avgs)):
                    # length mismatch -> invalidate
                    r["is_valid_row"] = False
                    is_valid = False

            if is_valid and avgs is not None:
                seed = get_seed_for_row(r.get("prompt_id", ""))
                best_idx, worst_idx, best_ties, worst_ties = pick_indices_from_avgs(avgs, seed)
                if best_ties > 1:
                    stats["tie_breaks_best"] += 1
                if worst_ties > 1:
                    stats["tie_breaks_worst"] += 1

                # Build chosen/rejected conversations from *original* prompt (list of messages)
                orig_prompt_list = r.get("prompt")
                if not isinstance(orig_prompt_list, list):
                    orig_prompt_list = []

                def append_assistant(convo: List[Dict[str, Any]], content: str):
                    new_conv = [dict(m) for m in convo]  # shallow copy
                    new_conv.append({"role": "assistant", "content": content})
                    return new_conv

                models = r["instruct_models"]
                responses = r["model_responses"]

                r["chosen"] = append_assistant(orig_prompt_list, responses[best_idx])
                r["chosen_model"] = models[best_idx]
                r["chosen_rating"] = avgs[best_idx]

                r["rejected"] = append_assistant(orig_prompt_list, responses[worst_idx])
                r["rejected_model"] = models[worst_idx]
                r["rejected_rating"] = avgs[worst_idx]

            # Update stats on validity
            if r["is_valid_row"]:
                stats["valid_rows"] += 1
            else:
                if reason_all_equal:
                    stats["invalid_rows_all_equal"] += 1
                elif has_any_null:
                    stats["invalid_rows_any_null"] += 1
                else:
                    # catch-all
                    stats["invalid_rows_any_null"] += 1

            # 5) Rename prompt -> prompt_msgs, set new prompt = first user turn content
            orig_prompt = r.get("prompt")
            r["prompt_msgs"] = orig_prompt if isinstance(orig_prompt, list) else []
            r["prompt"] = first_user_content(r["prompt_msgs"])
            pid = r.get("prompt_id", "")
            if isinstance(pid, str):
                # Take everything before the first occurrence of '-request'
                r["source"] = re.sub(r"-request.*$", "", pid)
            else:
                r["source"] = ""

            # Write processed row
            fout.write(orjson.dumps(r))
            fout.write(b"\n")

    # Print stats
    print("\n=== OUTPUT STATS ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # 6) Upload to HF
    if args.skip_upload:
        print("\nSkipping upload (--skip-upload set). Done.")
        return

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        print("\nERROR: No Hugging Face token provided. Use --hf-token or set HF_TOKEN/HUGGINGFACEHUB_API_TOKEN.")
        sys.exit(2)

    repo_id = args.repo_id
    private = args.private

    print(f"\nPushing dataset to HF Hub: {repo_id} (private={private})")
    api = HfApi(token=token)
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    except Exception as e:
        print("Note: repo may already exist or you may lack permissions. Continuing if it exists...")
        pass

    # Load processed jsonl into datasets.Dataset and push
    ds = Dataset.from_json(processed_path, split="train")
    ds.push_to_hub(repo_id, token=token)
    print("Upload complete.")


if __name__ == "__main__":
    main()
