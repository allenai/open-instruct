#!/usr/bin/env python3
"""
Fast join with per-model seek indexes + parallel join workers.

PHASE A (indexing, one-time per model file or when it changes):
  Build a SQLite index for each {model}.jsonl that maps:
    custom_id TEXT PRIMARY KEY -> offset INTEGER  (byte offset of the line)
  Index lives in:  <index_dir>/{model}.sqlite

PHASE B (join):
  Stream one or more parsed_judgments.jsonl files, look up the 4 instruct models for each prompt_id,
  fetch each model's line via the index (seek + read one line), parse messages, extract:
    - prompt (all messages before the last assistant)
    - model_responses (last assistant content) for each of the 4 models
  Assert (optionally) prompts are identical across the 4 models.
  Write a consolidated JSONL row per prompt_id.

Inputs:
- --parsed (repeatable): paths to parsed_judgments.jsonl
- --selected: /weka/.../selected_models_FINAL.jsonl
- --combined-dir: /weka/.../combined-outputs  (contains {model}.jsonl)

Outputs:
- --out: consolidated JSONL (default: alongside first --parsed)
- --index-dir: where per-model indexes go (default: <combined-dir>/_idx)

Speed knobs:
- --index-workers N  (parallel index build)
- --join-workers  N  (process multiple --parsed files in parallel; spawn N processes)
- --expected-rows K  (fixed-size progress bar; skip pre-counting)

Usage (single file):
  python fast_join_with_indexes.py \
    --parsed /path/to/parsed_judgments.jsonl \
    --selected /weka/.../selected_models_FINAL.jsonl \
    --combined-dir /weka/.../combined-outputs \
    --index-workers 8 \
    --join-workers 1 \
    --expected-rows 2500000 \
    --out /some/base/parsed_with_responses.jsonl \
    --overwrite

Usage (multiple parsed files in parallel):
  python fast_join_with_indexes.py \
    --parsed /path/to/part_aa.jsonl \
    --parsed /path/to/part_ab.jsonl \
    ... \
    --selected /weka/.../selected_models_FINAL.jsonl \
    --combined-dir /weka/.../combined-outputs \
    --index-workers 8 \
    --join-workers 5 \
    --expected-rows 2500000 \
    --out /some/base/parsed_with_responses.jsonl \
    --overwrite \
    --allow-prompt-mismatch

Note:
- First run may spend time building indexes; subsequent runs reuse them.
- If you split into 5 chunks and run 5 *separate jobs*, each job can set --expected-rows to ~1/5 of the total.
"""

import argparse
import json
import os
import sys
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# progress bars (graceful fallback)
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it=None, total=None, desc=None, unit=None, **kw):
        return it if it is not None else iter(())

def json_compact(o):
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))

# ---------- Phase A: per-model index building ----------

def ensure_index_for_model(model, combined_dir, index_dir, overwrite=False):
    """
    Build (or reuse) SQLite index mapping custom_id -> byte offset for {model}.jsonl
    Returns the path to the index DB.
    """
    src = os.path.join(combined_dir, f"{model}.jsonl")
    dst = os.path.join(index_dir, f"{model}.sqlite")
    if not overwrite and os.path.exists(dst):
        return dst  # reuse

    os.makedirs(index_dir, exist_ok=True)

    # Create an empty index if source file is missing
    if not os.path.exists(src):
        con = sqlite3.connect(dst)
        with con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("CREATE TABLE IF NOT EXISTS idx (custom_id TEXT PRIMARY KEY, offset INTEGER)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_custom_id ON idx(custom_id)")
        con.close()
        return dst

    con = sqlite3.connect(dst)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.execute("DROP TABLE IF EXISTS idx")
    cur.execute("CREATE TABLE idx (custom_id TEXT PRIMARY KEY, offset INTEGER)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_custom_id ON idx(custom_id)")

    batch = []
    BATCH_SIZE = 5000

    # Read in binary so byte offsets are exact
    with open(src, "rb") as f:
        pbar = tqdm(desc=f"Indexing {model}", unit="line")
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                rec = json.loads(line.decode("utf-8", errors="replace"))
                cid = rec.get("custom_id")
                if isinstance(cid, str):
                    batch.append((cid, offset))
            except Exception:
                pass
            if len(batch) >= BATCH_SIZE:
                cur.executemany("INSERT OR REPLACE INTO idx(custom_id, offset) VALUES (?,?)", batch)
                con.commit()
                batch.clear()
            pbar.update(1)
        pbar.close()

    if batch:
        cur.executemany("INSERT OR REPLACE INTO idx(custom_id, offset) VALUES (?,?)", batch)
        con.commit()
    con.close()
    return dst

def build_indexes(models, combined_dir, index_dir, overwrite=False, workers=1):
    if workers <= 1:
        for m in models:
            ensure_index_for_model(m, combined_dir, index_dir, overwrite=overwrite)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(ensure_index_for_model, m, combined_dir, index_dir, overwrite) for m in models]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="Building indexes", unit="model"):
            pass

# ---------- Utilities for message extraction ----------

def split_prompt_and_last_assistant(messages):
    """
    Given a list of messages, return (prompt_messages, last_assistant_content).
    """
    if not isinstance(messages, list):
        raise ValueError("messages not a list")
    last_i = None
    for i in range(len(messages)-1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "assistant":
            last_i = i
            break
    if last_i is None:
        raise ValueError("no assistant message found")
    prompt_msgs = messages[:last_i]
    content = messages[last_i].get("content")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False) if content is not None else ""
    return prompt_msgs, content

# ---------- ModelReader (seek by offset via SQLite) ----------

class ModelReader:
    """
    Opens {model}.jsonl (binary) and its SQLite index, supports fast lookups by custom_id.
    Each process should keep its own readers.
    """
    def __init__(self, model, combined_dir, index_dir, cache_size=1024):
        self.model = model
        self.path = os.path.join(combined_dir, f"{model}.jsonl")
        self.index_path = os.path.join(index_dir, f"{model}.sqlite")
        self.con = sqlite3.connect(self.index_path, check_same_thread=False)
        self.cur = self.con.cursor()
        # read-optimized PRAGMAs (safe; index is read-only here)
        try:
            self.cur.execute("PRAGMA cache_size=-200000")   # ~800MB if 4KB pages
            self.cur.execute("PRAGMA mmap_size=1073741824")# 1 GiB if supported
        except Exception:
            pass
        self.fh = open(self.path, "rb") if os.path.exists(self.path) else None
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size

    def _cache_put(self, key, val):
        if key in self.cache:
            return
        self.cache[key] = val
        self.cache_order.append(key)
        if len(self.cache_order) > self.cache_size:
            old = self.cache_order.pop(0)
            self.cache.pop(old, None)

    def get_prompt_and_response(self, custom_id):
        """Return (prompt_messages, response_text) or (None, None) if not found."""
        if custom_id in self.cache:
            return self.cache[custom_id]

        row = self.cur.execute("SELECT offset FROM idx WHERE custom_id = ?", (custom_id,)).fetchone()
        if not row or not self.fh:
            self._cache_put(custom_id, (None, None))
            return (None, None)

        offset = int(row[0])
        try:
            self.fh.seek(offset)
            line = self.fh.readline()
            rec = json.loads(line.decode("utf-8", errors="replace"))
            msgs = rec.get("messages")
            if not isinstance(msgs, list):
                self._cache_put(custom_id, (None, None))
                return (None, None)
            prompt, resp = split_prompt_and_last_assistant(msgs)
            self._cache_put(custom_id, (prompt, resp))
            return (prompt, resp)
        except Exception:
            self._cache_put(custom_id, (None, None))
            return (None, None)

    def close(self):
        try: self.cur.close()
        except Exception: pass
        try: self.con.close()
        except Exception: pass
        try:
            if self.fh: self.fh.close()
        except Exception: pass

# ---------- selected_models loader ----------

def read_selected_models(path):
    """Return dict: prompt_id -> [4 models], and the sorted set of all unique models seen."""
    sel = {}
    uniq = set()
    bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                bad += 1
                continue
            pid = rec.get("prompt_id")
            models = rec.get("instruct_models")
            if not isinstance(pid, str) or not isinstance(models, list) or len(models) != 4 or not all(isinstance(m, str) for m in models):
                bad += 1
                continue
            sel[pid] = models
            uniq.update(models)
    if bad:
        print(f"[WARN] skipped {bad} malformed lines in selected_models", file=sys.stderr)
    return sel, sorted(uniq)

# ---------- Worker function for one parsed file ----------

def process_parsed_file(parsed_path, out_part_path, selected_path, combined_dir, index_dir, allow_mismatch):
    """
    Runs in a separate process.
    Each worker keeps its own ModelReader instances.
    """
    selected, models_uni = read_selected_models(selected_path)  # small dict; ok per process
    readers = {}  # model -> ModelReader

    def get_reader(model):
        r = readers.get(model)
        if r is None:
            readers[model] = ModelReader(model, combined_dir, index_dir)
            r = readers[model]
        return r

    written = 0
    not_sel = 0
    miss_model = 0
    mismatches = 0

    with open(parsed_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(out_part_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            pid = row.get("prompt_id")
            if not isinstance(pid, str):
                continue

            models = selected.get(pid)
            if not models:
                not_sel += 1
                continue

            prompts, responses = [], []
            for m in models:
                pmsg, resp = get_reader(m).get_prompt_and_response(pid)
                if pmsg is None or resp is None:
                    miss_model += 1
                prompts.append(pmsg); responses.append(resp)

            non_null = [p for p in prompts if p is not None]
            prompt_canonical = non_null[0] if non_null else None

            if len(non_null) > 1:
                first = json_compact(non_null[0])
                if any(json_compact(p) != first for p in non_null[1:]):
                    mismatches += 1
                    if not allow_mismatch:
                        # Close and fail fast
                        for r in readers.values(): r.close()
                        raise SystemExit(
                            f"[FATAL] Prompt mismatch for prompt_id={pid} in {parsed_path}. "
                            f"Use --allow-prompt-mismatch to continue."
                        )

            ratings_fields = {k: v for k, v in row.items() if k.startswith("ratings_")}
            out_row = {
                "prompt_id": pid,
                "instruct_models": models,
                "prompt": prompt_canonical,
                "model_responses": responses,
                **ratings_fields,
            }
            fout.write(json_compact(out_row) + "\n")
            written += 1

    for r in readers.values():
        r.close()

    return {"file": parsed_path, "written": written, "not_sel": not_sel,
            "miss_model": miss_model, "mismatches": mismatches}

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", action="append", required=True,
                    help="Path to parsed_judgments.jsonl (repeatable)")
    ap.add_argument("--selected", required=True,
                    help="Path to selected_models_FINAL.jsonl")
    ap.add_argument("--combined-dir", required=True,
                    help="Directory with combined-outputs/{model}.jsonl")
    ap.add_argument("--index-dir", default=None,
                    help="Directory to store/reuse per-model indexes (default: <combined-dir>/_idx)")
    ap.add_argument("--index-workers", type=int, default=1,
                    help="Parallel workers to build indexes (default 1)")
    ap.add_argument("--rebuild-indexes", action="store_true",
                    help="Force rebuild of per-model indexes")
    ap.add_argument("--join-workers", type=int, default=8,
                    help="Process multiple --parsed files in parallel (separate processes)")
    ap.add_argument("--out", default=None,
                    help="Output JSONL (default: alongside first --parsed as parsed_with_responses.jsonl)")
    ap.add_argument("--allow-prompt-mismatch", action="store_true",
                    help="Do not stop on prompt mismatch across models; log and proceed")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow overwriting OUT")
    ap.add_argument("--expected-rows", type=int, default=2_500_000,
                    help="Fixed progress bar total across all --parsed files (purely cosmetic)")
    args = ap.parse_args()

    # derive defaults
    if args.out is None:
        base_dir = os.path.dirname(os.path.abspath(args.parsed[0]))
        args.out = os.path.join(base_dir, "parsed_with_responses_fast.jsonl")
    if args.index_dir is None:
        args.index_dir = os.path.join(os.path.abspath(args.combined_dir), "_idx")

    # safety: output overwrite
    if os.path.exists(args.out) and not args.overwrite:
        sys.exit(f"Refusing to overwrite OUT: {args.out} (pass --overwrite)")

    # load selected models & list of unique models
    selected, all_models = read_selected_models(args.selected)
    if not all_models:
        sys.exit("No models found in selected_models file.")
    print(f"Unique models referenced: {len(all_models)}")

    # Phase A: build/reuse indexes
    build_indexes(
        all_models,
        args.combined_dir,
        args.index_dir,
        overwrite=args.rebuild_indexes,
        workers=max(1, args.index_workers),
    )

    # Fixed-size global bar (no pre-counting)
    master_pbar = tqdm(total=args.expected_rows, desc="Joining parsed files", unit="row")

    # Serial join (single worker) or parallel across multiple parsed files
    part_paths = []
    stats = []

    if args.join_workers <= 1 or len(args.parsed) == 1:
        # Single process, stream lines and update bar per row
        written_total = 0
        not_sel_total = 0
        miss_model_total = 0
        mismatches_total = 0

        # Create local readers in this process (lazy per model)
        readers = {}
        def get_reader(model):
            r = readers.get(model)
            if r is None:
                readers[model] = ModelReader(model, args.combined_dir, args.index_dir)
                r = readers[model]
            return r

        with open(args.out + ".tmp", "w", encoding="utf-8") as fout:
            for parsed_path in args.parsed:
                with open(parsed_path, "r", encoding="utf-8", errors="replace") as fin:
                    for line in fin:
                        # cosmetic: tick even on blank/malformed lines
                        master_pbar.update(1)
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue

                        pid = row.get("prompt_id")
                        if not isinstance(pid, str):
                            continue

                        models = selected.get(pid)
                        if not models:
                            not_sel_total += 1
                            continue

                        prompts, responses = [], []
                        for m in models:
                            pmsg, resp = get_reader(m).get_prompt_and_response(pid)
                            if pmsg is None or resp is None:
                                miss_model_total += 1
                            prompts.append(pmsg); responses.append(resp)

                        non_null = [p for p in prompts if p is not None]
                        prompt_canonical = non_null[0] if non_null else None

                        if len(non_null) > 1:
                            first = json_compact(non_null[0])
                            if any(json_compact(p) != first for p in non_null[1:]):
                                mismatches_total += 1
                                if not args.allow_prompt_mismatch:
                                    for r in readers.values(): r.close()
                                    sys.exit(
                                        f"[FATAL] Prompt mismatch for prompt_id={pid}. "
                                        f"Use --allow-prompt-mismatch to continue."
                                    )

                        ratings_fields = {k: v for k, v in row.items() if k.startswith("ratings_")}
                        out_row = {
                            "prompt_id": pid,
                            "instruct_models": models,
                            "prompt": prompt_canonical,
                            "model_responses": responses,
                            **ratings_fields,
                        }
                        fout.write(json_compact(out_row) + "\n")
                        written_total += 1

        for r in readers.values():
            r.close()

        stats.append({"file": "(serial)", "written": written_total, "not_sel": not_sel_total,
                      "miss_model": miss_model_total, "mismatches": mismatches_total})

    else:
        # Parallel: each parsed file in its own process writing <out>.part{i}.tmp
        with ProcessPoolExecutor(max_workers=args.join_workers) as ex:
            futs = []
            for i, p in enumerate(args.parsed):
                part_path = f"{args.out}.part{i}.tmp"
                part_paths.append(part_path)
                fut = ex.submit(
                    process_parsed_file,
                    p, part_path, args.selected, args.combined_dir, args.index_dir, args.allow_prompt_mismatch
                )
                futs.append(fut)

            # progress: advance roughly as files complete (coarse but no pre-counting)
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Joining (parallel)", unit="file"):
                res = fut.result()
                stats.append(res)
                # If you know rows per file, you could add master_pbar.update(rows_here) for smoother progress
                # Here we approximate by evenly dividing expected rows among parsed files:
                if len(args.parsed) > 0:
                    master_pbar.update(max(0, args.expected_rows // len(args.parsed)))

        # Concatenate part outputs
        with open(args.out + ".tmp", "w", encoding="utf-8") as outfp:
            for pp in part_paths:
                with open(pp, "r", encoding="utf-8") as pf:
                    for line in pf:
                        outfp.write(line)
                try:
                    os.remove(pp)
                except Exception:
                    pass

    master_pbar.close()

    # Finalize atomically
    os.replace(args.out + ".tmp", args.out)

    # Summarize
    total_written = sum(s["written"] for s in stats)
    total_not_sel = sum(s["not_sel"] for s in stats)
    total_miss_model = sum(s["miss_model"] for s in stats)
    total_mismatches = sum(s["mismatches"] for s in stats)

    print("\n=== Join Summary ===")
    print(f"Rows written        : {total_written:,}")
    print(f"Missing in selected : {total_not_sel:,}")
    print(f"Missing in model    : {total_miss_model:,}")
    print(f"Prompt mismatches   : {total_mismatches:,}  ({'allowed' if args.allow_prompt_mismatch else 'fatal if >0'})")
    print(f"Output              : {args.out}")
    print(f"Indexes dir         : {args.index_dir}")
    print("Done.")

if __name__ == "__main__":
    main()
