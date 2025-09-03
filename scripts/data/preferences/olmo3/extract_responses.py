#!/usr/bin/env python3
"""
Join parsed judgments with model selections and model generations.

Inputs:
- One or more parsed_judgments.jsonl files (from the judge dirs)
  Each row must contain: {"prompt_id": str, "ratings_<aspect>": ..., ...}
- /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/selected_models_FINAL.jsonl
  Each row must contain: {"prompt_id": str, "instruct_models": [4 model names]}
- /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/combined-outputs/{model}.jsonl
  Each line has at least: {"custom_id": str, "messages": [ ... ]}
  - The desired response is the *last* message with role == "assistant".
  - The "prompt" is all messages *before* that last assistant turn.
  - We match parsed_judgments.prompt_id to model_file.custom_id (exact match).

Output:
- JSONL with rows:
  {
    "prompt_id": str,
    "instruct_models": [str, str, str, str],
    "prompt": [ {role, content, ...}, ... ],    # messages up to last assistant
    "model_responses": [str, str, str, str],    # same order as instruct_models
    "ratings_<aspect>": ...                     # copied from parsed_judgments row
  }

Usage examples:
  python join_judgments_and_generations.py \
    --parsed /path/to/judgeA/parsed_judgments.jsonl \
    --selected /weka/.../selected_models_FINAL.jsonl \
    --combined-dir /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/combined-outputs \
    --out /some/base/parsed_with_responses.jsonl

  # Process two judge files into a single merged output:
  python join_judgments_and_generations.py \
    --parsed /judgeA/parsed_judgments.jsonl \
    --parsed /judgeB/parsed_judgments.jsonl \
    --selected /weka/.../selected_models_FINAL.jsonl \
    --combined-dir /weka/.../combined-outputs \
    --out /base/parsed_with_responses.jsonl

Flags:
  --models-cache N     # max number of model maps to keep in memory (LRU). Default: 8
  --allow-prompt-mismatch  # don’t raise on prompt mismatch; just log and keep the first
  --overwrite          # allow clobbering existing output
"""

import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict

# Progress bars (graceful fallback)
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it=None, total=None, desc=None, unit=None):
        return it if it is not None else iter(())

def json_compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def read_selected_models(path):
    """Return dict: prompt_id -> [model1, model2, model3, model4]."""
    sel = {}
    missing = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            pid = rec.get("prompt_id")
            models = rec.get("instruct_models")
            if not isinstance(pid, str) or not isinstance(models, list) or len(models) != 4 or not all(isinstance(m, str) for m in models):
                missing += 1
                continue
            sel[pid] = models
    if not sel:
        raise SystemExit(f"[FATAL] No valid rows loaded from selected models: {path}")
    if missing:
        print(f"[WARN] Skipped {missing} malformed rows in selected models.", file=sys.stderr)
    return sel

def split_prompt_and_last_assistant(messages):
    """
    Given a list of messages, return (prompt_messages, last_assistant_content).
    Raises if no assistant turn found or messages is not a list.
    """
    if not isinstance(messages, list):
        raise ValueError("messages is not a list")
    last_idx = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "assistant":
            last_idx = i
            break
    if last_idx is None:
        raise ValueError("no assistant message found")
    prompt_msgs = messages[:last_idx]
    content = messages[last_idx].get("content")
    if not isinstance(content, str):
        # Some tool-using traces might store non-string; stringify as fallback
        content = json.dumps(content, ensure_ascii=False) if content is not None else ""
    return prompt_msgs, content

class ModelIndexLRU:
    """
    LRU cache of per-model maps: custom_id -> (prompt_messages, response_text).
    Loads a model file on first access; evicts oldest if over capacity.
    """
    def __init__(self, combined_dir, capacity=8):
        self.combined_dir = combined_dir
        self.capacity = max(1, int(capacity))
        self.cache = OrderedDict()  # model -> dict(custom_id -> (prompt, resp))
        self.load_errors = {}       # model -> error str

    def _load_model(self, model):
        path = os.path.join(self.combined_dir, f"{model}.jsonl")
        m = {}
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    cid = rec.get("custom_id")
                    msgs = rec.get("messages")
                    if not isinstance(cid, str) or not isinstance(msgs, list):
                        continue
                    try:
                        prompt, resp = split_prompt_and_last_assistant(msgs)
                    except Exception:
                        continue
                    m[cid] = (prompt, resp)
        except FileNotFoundError:
            self.load_errors[model] = f"missing file: {path}"
            m = {}
        except Exception as e:
            self.load_errors[model] = f"{type(e).__name__}: {e}"
            m = {}

        # LRU insert
        self.cache[model] = m
        self.cache.move_to_end(model, last=True)
        # evict if needed
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def _ensure_loaded(self, model):
        if model in self.cache:
            self.cache.move_to_end(model, last=True)
            return
        self._load_model(model)

    def get(self, model, custom_id):
        """Return (prompt_messages, response_text) or (None, None) if not found."""
        self._ensure_loaded(model)
        d = self.cache.get(model, {})
        return d.get(custom_id, (None, None))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", action="append", required=True,
                    help="Path to parsed_judgments.jsonl (can be given multiple times)")
    ap.add_argument("--selected", required=True,
                    help="Path to selected_models_FINAL.jsonl")
    ap.add_argument("--combined-dir", required=True,
                    help="Dir containing combined-outputs/{model}.jsonl")
    ap.add_argument("--out", default=None,
                    help="Output JSONL path (default: alongside first --parsed, named parsed_with_responses.jsonl)")
    ap.add_argument("--models-cache", type=int, default=8,
                    help="Max number of model maps kept in memory (LRU). Default: 8")
    ap.add_argument("--allow-prompt-mismatch", action="store_true",
                    help="If set, do not raise on prompt mismatch; log warning and keep the first prompt")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow overwriting existing OUT")
    args = ap.parse_args()

    # Output path default
    if args.out is None:
        base_dir = os.path.dirname(os.path.abspath(args.parsed[0]))
        args.out = os.path.join(base_dir, "parsed_with_responses.jsonl")

    # Safety: no clobbering unless asked
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {args.out} (pass --overwrite)")

    # Load selected models map
    selected = read_selected_models(args.selected)

    # Prepare model index
    lru = ModelIndexLRU(args.combined_dir, capacity=args.models_cache)

    total_rows = 0
    missing_selected = 0
    bad_models = 0
    prompt_mismatch = 0
    not_found_in_model = 0

    with open(args.out + ".tmp", "w", encoding="utf-8") as fout:
        for parsed_path in args.parsed:
            # Count lines for progress bar total (optional: skip total for speed)
            try:
                with open(parsed_path, "r", encoding="utf-8", errors="replace") as f:
                    total_lines = sum(1 for _ in f)
            except Exception:
                total_lines = None

            with open(parsed_path, "r", encoding="utf-8", errors="replace") as fin, \
                 tqdm(total=total_lines, desc=f"Processing {os.path.basename(parsed_path)}", unit="row") as bar:
                for line in fin:
                    if not line.strip():
                        if bar: bar.update(1)
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        if bar: bar.update(1)
                        continue

                    pid = row.get("prompt_id")
                    if not isinstance(pid, str):
                        if bar: bar.update(1)
                        continue

                    models = selected.get(pid)
                    if not (isinstance(models, list) and len(models) == 4 and all(isinstance(m, str) for m in models)):
                        missing_selected += 1
                        if bar: bar.update(1)
                        continue

                    # Collect responses/prompts for these 4 models
                    prompts = []
                    responses = []
                    for m in models:
                        prompt_msgs, resp = lru.get(m, pid)
                        if prompt_msgs is None or resp is None:
                            not_found_in_model += 1
                            prompt_msgs, resp = None, None
                        prompts.append(prompt_msgs)
                        responses.append(resp)

                    # Assert prompts are identical across the four models (if present)
                    # We serialize to compact JSON for stable equality; None allowed
                    non_null_prompts = [p for p in prompts if p is not None]
                    if non_null_prompts:
                        first = json_compact(non_null_prompts[0])
                        mismatch = any(json_compact(p) != first for p in non_null_prompts[1:])
                        if mismatch:
                            prompt_mismatch += 1
                            if not args.allow_prompt_mismatch:
                                raise SystemExit(
                                    f"[FATAL] Prompt mismatch for prompt_id={pid} across models {models}. "
                                    f"Run with --allow-prompt-mismatch to proceed."
                                )
                        # choose the first non-null as canonical
                        prompt_canonical = non_null_prompts[0]
                    else:
                        prompt_canonical = None  # none of the models had the row

                    # Copy ratings_* fields from parsed row
                    ratings_fields = {k: v for k, v in row.items() if k.startswith("ratings_")}

                    out_row = {
                        "prompt_id": pid,
                        "instruct_models": models,
                        "prompt": prompt_canonical,
                        "model_responses": responses,  # aligned with instruct_models
                        **ratings_fields,
                    }
                    fout.write(json_compact(out_row) + "\n")
                    total_rows += 1
                    if bar: bar.update(1)

    # Atomic replace
    os.replace(args.out + ".tmp", args.out)

    # Report model load errors (if any)
    if lru.load_errors:
        print("\n[MODEL LOAD WARNINGS]", file=sys.stderr)
        for model, err in lru.load_errors.items():
            print(f"  {model}: {err}", file=sys.stderr)

    # Summary
    print("\n=== Join Summary ===")
    print(f"Parsed rows written : {total_rows:,}")
    print(f"Missing selection   : {missing_selected:,}  (no instruct_models for some prompt_id)")
    print(f"Missing in models   : {not_found_in_model:,}  (model file had no matching custom_id)")
    print(f"Prompt mismatches   : {prompt_mismatch:,}  ({'allowed' if args.allow_prompt_mismatch else 'fatal if >0'})")
    print(f"Output              : {args.out}")

if __name__ == "__main__":
    main()
