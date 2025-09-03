#!/usr/bin/env python3
"""
Combine mapped JSONL files into one per-prompt_id row with 4 aspects.

INPUT  (dir): .../batch_outputs_mapped/   (each line: {"prompt_id", "aspect", "ratings"})
OUTPUT (file): <base_dir>/parsed_judgments_combined.jsonl

Changes:
- Incomplete groups do NOT count as errors; we pad missing aspects with null.
- Canonical aspects = --expected-aspects OR inferred as union across dataset (must be size 4).

Usage:
  python combine_mapped.py \
    --mapped-dir /weka/.../batch_outputs_mapped \
    [--out <base_dir>/parsed_judgments_combined.jsonl] \
    [--tmp-dir <mapped-dir>/_combine_tmp] \
    [--shards 256] \
    [--expected-aspects instruction,truthfulness,reasoning,helpfulness] \
    [--max-files N] \
    [--overwrite] \
    [--examples 10] \
    [--example-max-chars 400]
"""

import argparse
import json
import os
import sys
import zlib
from collections import defaultdict, Counter

# --- Progress bar (graceful fallback) ---
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it=None, total=None, desc=None, unit=None):
        return it if it is not None else iter(())

def iter_jsonl_files(d):
    with os.scandir(d) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".jsonl"):
                yield e.path

def safe_makedirs(p):
    os.makedirs(p, exist_ok=True)

def open_shard_writers(tmp_dir, n):
    safe_makedirs(tmp_dir)
    writers = []
    for i in range(n):
        fp = os.path.join(tmp_dir, f"shard_{i:03d}.jsonl")
        writers.append(open(fp, "w", encoding="utf-8"))
    return writers

def close_all(files):
    for f in files:
        try: f.close()
        except Exception: pass

def json_compact(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def parse_expected_aspects(s):
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return set(parts) if parts else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapped-dir", required=True)
    ap.add_argument("--out", default=None,
                    help="Output JSONL (default: <parent_of_mapped>/parsed_judgments_combined.jsonl)")
    ap.add_argument("--tmp-dir", default=None,
                    help="Temp directory for shards (default: <mapped-dir>/_combine_tmp)")
    ap.add_argument("--shards", type=int, default=256)
    ap.add_argument("--expected-aspects", default=None,
                    help="Comma-separated list of the 4 expected aspects; missing aspects will be padded with null")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Process at most N mapped files (for sampling)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow overwriting existing OUT/temporary shards")
    ap.add_argument("--examples", type=int, default=10,
                    help="Print up to N error examples at the end")
    ap.add_argument("--example-max-chars", type=int, default=400)
    args = ap.parse_args()

    mapped_dir = args.mapped_dir
    if not os.path.isdir(mapped_dir):
        sys.exit(f"Mapped dir not found: {mapped_dir}")

    base_dir = os.path.dirname(os.path.abspath(mapped_dir))
    out_path = args.out or os.path.join(base_dir, "parsed_judgments_combined.jsonl")
    err_path = os.path.join(base_dir, "parsed_judgments_combined_errors.jsonl")
    tmp_dir = args.tmp_dir or os.path.join(mapped_dir, "_combine_tmp")

    # Refuse to clobber unless --overwrite
    if os.path.exists(out_path) and not args.overwrite:
        sys.exit(f"Refusing to overwrite existing OUT: {out_path} (pass --overwrite to replace)")
    if os.path.exists(tmp_dir):
        if not args.overwrite and any(True for _ in os.scandir(tmp_dir)):
            sys.exit(f"Temp dir not empty: {tmp_dir} (pass --overwrite or choose --tmp-dir)")

    expected_aspects = parse_expected_aspects(args.expected_aspects)
    if expected_aspects and len(expected_aspects) != 4:
        sys.exit(f"--expected-aspects must have exactly 4 entries (got {len(expected_aspects)})")

    # --- PASS 1: shard all mapped rows by prompt_id; gather aspect union ---
    files = list(iter_jsonl_files(mapped_dir))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        sys.exit(f"No .jsonl files found in {mapped_dir}")

    # Clean tmp_dir
    safe_makedirs(tmp_dir)
    for e in os.scandir(tmp_dir):
        try: os.remove(e.path)
        except Exception: pass

    writers = open_shard_writers(tmp_dir, args.shards)

    total_rows = 0
    bad_input_rows = 0  # malformed JSON or missing prompt_id/aspect
    all_aspects = set()
    aspect_freq = Counter()

    for fp in tqdm(files, desc="Pass1: sharding files", unit="file"):
        with open(fp, "r", encoding="utf-8", errors="replace") as fin:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    bad_input_rows += 1
                    continue
                pid = row.get("prompt_id")
                aspect = row.get("aspect")
                ratings = row.get("ratings")
                if not isinstance(pid, str) or not isinstance(aspect, str):
                    bad_input_rows += 1
                    continue
                all_aspects.add(aspect)
                aspect_freq[aspect] += 1
                shard = zlib.crc32(pid.encode("utf-8")) % args.shards
                writers[shard].write(json_compact({"prompt_id": pid, "aspect": aspect, "ratings": ratings}) + "\n")
                total_rows += 1

    close_all(writers)

    # Determine canonical aspects
    canonical_aspects = expected_aspects or (all_aspects if len(all_aspects) == 4 else None)
    if canonical_aspects is None:
        # Fall back to the 4 most common if union != 4 (still pad deterministically)
        canonical_aspects = set([a for a, _ in aspect_freq.most_common(4)])

    # --- PASS 2: group within each shard and write combined output ---
    out_tmp = out_path + ".tmp"
    err_tmp = err_path + ".tmp"
    out_f = open(out_tmp, "w", encoding="utf-8")
    err_f = open(err_tmp, "w", encoding="utf-8")

    errors = 0
    written_groups = 0
    padded_groups = 0
    examples = []

    def record_error(pid, reason, details=None, example_text=None):
        nonlocal errors
        errors += 1
        rec = {"prompt_id": pid, "reason": reason}
        if details is not None:
            rec["details"] = details
        err_f.write(json_compact(rec) + "\n")
        if len(examples) < args.examples:
            snippet = example_text
            if isinstance(snippet, str):
                snippet = snippet.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
                if len(snippet) > args.example_max_chars:
                    snippet = snippet[: args.example_max_chars] + "…"
            examples.append({"prompt_id": pid, "reason": reason, "example": snippet})

    def ratings_ok(r):
        return (r is None) or (isinstance(r, list) and len(r) == 4 and all(isinstance(x, int) for x in r))

    shard_files = [os.path.join(tmp_dir, f) for f in sorted(os.listdir(tmp_dir)) if f.startswith("shard_") and f.endswith(".jsonl")]
    for sfp in tqdm(shard_files, desc="Pass2: combining shards", unit="shard"):
        groups = defaultdict(dict)   # pid -> {aspect: ratings}
        seen_complete = set()

        with open(sfp, "r", encoding="utf-8", errors="replace") as fin:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    record_error(None, "malformed_shard_line")
                    continue

                pid = rec.get("prompt_id")
                aspect = rec.get("aspect")
                ratings = rec.get("ratings")

                if not isinstance(pid, str) or not isinstance(aspect, str):
                    record_error(pid, "invalid_pid_or_aspect", details={"aspect": aspect})
                    continue

                if pid in seen_complete:
                    record_error(pid, "extra_row_after_complete", details={"aspect": aspect})
                    continue

                # Store (overwrites duplicates of the same aspect)
                groups[pid][aspect] = ratings

                # If we reached 4 aspects for this pid, validate & flush
                if len(groups[pid]) == 4:
                    aspects = groups[pid]
                    keys = set(aspects.keys())
                    # If expected aspects were specified, enforce exact match
                    if args.expected_aspects:
                        expected = parse_expected_aspects(args.expected_aspects)
                        if keys != expected:
                            record_error(pid, "unexpected_aspect_set",
                                         details={"got": sorted(keys), "expected": sorted(expected)})
                            seen_complete.add(pid)
                            continue
                    # Validate ratings (allow None or 4-int list)
                    invalid = [a for a, r in aspects.items() if not ratings_ok(r)]
                    if invalid:
                        record_error(pid, "invalid_ratings", details={"bad_aspects": invalid})
                        seen_complete.add(pid)
                        continue

                    # OK: write consolidated row (canonical aspect order)
                    row = {"prompt_id": pid}
                    ordered = sorted(canonical_aspects)
                    for a in ordered:
                        row[f"ratings_{a}"] = aspects.get(a)
                    out_f.write(json_compact(row) + "\n")
                    written_groups += 1
                    seen_complete.add(pid)
                    if pid in groups: del groups[pid]

        # Pad any leftover groups (< 4 aspects) and write them (NOT an error)
        for pid, amap in groups.items():
            if pid in seen_complete:
                continue
            # Prepare row with canonical keys; missing -> None
            row = {"prompt_id": pid}
            ordered = sorted(canonical_aspects)
            # If expected aspects were specified and amap contains unknowns, flag but still pad/write
            if args.expected_aspects:
                unknown = [a for a in amap.keys() if a not in canonical_aspects]
                if unknown:
                    record_error(pid, "unexpected_aspect_set_partial",
                                 details={"got": sorted(amap.keys()), "expected": sorted(canonical_aspects)})
                    # continue writing anyway

            # Validate present ratings; malformed present ratings are errors
            invalid = [a for a, r in amap.items() if not ratings_ok(r)]
            if invalid:
                record_error(pid, "invalid_ratings_partial", details={"bad_aspects": invalid})
                # continue writing anyway with None for invalid aspects
            for a in ordered:
                r = amap.get(a)
                row[f"ratings_{a}"] = r if ratings_ok(r) else None
            out_f.write(json_compact(row) + "\n")
            written_groups += 1
            padded_groups += 1

    out_f.close()
    err_f.close()

    # Atomic finalize outputs
    os.replace(out_tmp, out_path)
    os.replace(err_tmp, err_path)

    # Summary
    print("\n=== Combine Summary ===")
    print(f"Mapped files read : {len(files):,}  (rows sharded: {total_rows:,}, bad input rows: {bad_input_rows:,})")
    print(f"Groups written    : {written_groups:,}  (padded: {padded_groups:,})")
    print(f"Canonical aspects : {sorted(canonical_aspects)}")
    print(f"Errors            : {errors:,}  (see {err_path})")

    # Exit non-zero only if true errors (padding is OK)
    if errors > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
