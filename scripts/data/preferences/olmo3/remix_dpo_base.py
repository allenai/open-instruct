#!/usr/bin/env python3
import os
import sys
import re
import math
import random
import argparse
from typing import List, Dict, Set, Tuple

from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value

# ---------------------- Defaults & Sets ----------------------
BASE_REPO_DEFAULT = "allenai/dpo-base-100k-qwq-judge"
BASE_SPLIT_DEFAULT = "train"
OUT_REPO_DEFAULT = "allenai/dpo-base-100k-qwq-judge-remix"

WC_REPLACEMENT_REPO_DEFAULT = "saumyamalik/filtered-wc-sample-500k-unused-qwq"
VALPY_REPLACEMENT_REPO_DEFAULT = "valpy/if_dpo_verified_permissible"

WC_SOURCES_TO_REPLACE = {
    "Wildchat-1M-gpt-4.1-regenerated-english",
    "Wildchat-1m-gpt-4.1-regeneration-not-english",
    "filtered_wc_sample_500k",
}

SOURCES_TO_PARTIAL_REPLACE_WITH_VALPY = {
    "tulu-3-sft-personas-instruction-following-o3",
    "IF_sft_data_verified_permissive",
    "valpy_if_qwq_reasoning_verified_no_reasoning",
}

RNG_SEED_DEFAULT = 123

# Candidate alternate column names we might map into 'chosen'/'rejected'
CHOSEN_CANDIDATES = ["chosen", "chosen_response", "preferred", "pos", "answer_preferred", "accepted", "chosen_text"]
REJECTED_CANDIDATES = ["rejected", "rejected_response", "dispreferred", "neg", "answer_rejected", "rejected_text"]

# ---------------------- Helpers ----------------------

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "unknown"

def find_first_present(ds: Dataset, candidates: List[str]) -> str | None:
    cols = set(ds.column_names)
    for c in candidates:
        if c in cols:
            return c
        # case-insensitive check
        for name in list(cols):
            if name.lower() == c.lower():
                return name
    return None

def ensure_chosen_rejected(ds: Dataset, label: str) -> Dataset:
    cols = set(ds.column_names)
    # chosen
    if "chosen" not in cols:
        cand = find_first_present(ds, CHOSEN_CANDIDATES)
        if not cand:
            raise ValueError(f"{label}: couldn't find a column to use as 'chosen' (candidates: {CHOSEN_CANDIDATES})")
        if cand != "chosen":
            ds = ds.rename_column(cand, "chosen")
    # rejected
    cols = set(ds.column_names)
    if "rejected" not in cols:
        cand = find_first_present(ds, REJECTED_CANDIDATES)
        if not cand:
            raise ValueError(f"{label}: couldn't find a column to use as 'rejected' (candidates: {REJECTED_CANDIDATES})")
        if cand != "rejected":
            ds = ds.rename_column(cand, "rejected")
    # Drop rows with null/empty chosen/rejected
    def _ok(x):
        c = x.get("chosen", None)
        r = x.get("rejected", None)
        return (c is not None and str(c).strip() != "") and (r is not None and str(r).strip() != "")
    ds = ds.filter(_ok)
    return ds

def add_missing_columns_with_none(ds: Dataset, target_cols: List[str]) -> Dataset:
    # Add any missing columns with None
    missing = [c for c in target_cols if c not in ds.column_names]
    if missing:
        for c in missing:
            ds = ds.add_column(c, [None] * len(ds))
    return ds

def drop_extra_columns(ds: Dataset, keep_cols: List[str]) -> Dataset:
    extra = [c for c in ds.column_names if c not in keep_cols]
    if extra:
        ds = ds.remove_columns(extra)
    return ds

def align_to_base(ds: Dataset, base: Dataset, ensure_source_value: str | None = None) -> Dataset:
    """
    Make ds columns match base EXACTLY (add missing with None, drop extras), then cast to base.features.
    Optionally ensure a 'source' value if ds lacks 'source' (or fill nulls).
    """
    base_cols = list(base.column_names)
    if ensure_source_value is not None:
        if "source" not in ds.column_names:
            ds = ds.add_column("source", [ensure_source_value] * len(ds))
        else:
            # fill null/empty with ensure_source_value
            def _fill_source(x):
                s = x["source"]
                if s is None or (isinstance(s, str) and s.strip() == ""):
                    return {"source": ensure_source_value}
                return {"source": s}
            ds = ds.map(_fill_source)
    ds = add_missing_columns_with_none(ds, base_cols)
    ds = drop_extra_columns(ds, base_cols)
    # Cast types to base.features (will coerce None appropriately)
    try:
        ds = ds.cast(base.features)
    except Exception as e:
        print("WARN: cast to base.features failed; proceeding without cast. Error:", e, file=sys.stderr)
    return ds

def ensure_prompt_id(ds: Dataset, prefix: str) -> Dataset:
    if "prompt_id" not in ds.column_names:
        ds = ds.add_column("prompt_id", [None] * len(ds))
    # Fill any None/empty prompt_id with generated
    def _fill(batch, idx):
        out = []
        for i, pid in enumerate(batch["prompt_id"]):
            if pid is None or (isinstance(pid, str) and pid.strip() == ""):
                out.append(f"{prefix}-{idx[i]}")
            else:
                out.append(pid)
        return {"prompt_id": out}
    ds = ds.map(_fill, with_indices=True, batched=True)
    return ds

def split_counts(total_replace: int, per_source_counts: Dict[str, int]) -> Dict[str, int]:
    if total_replace <= 0 or not per_source_counts:
        return {s: 0 for s in per_source_counts.keys()}
    total_pool = sum(per_source_counts.values())
    # Initial floor allocation
    alloc = {s: min(per_source_counts[s], int(math.floor(total_replace * (per_source_counts[s] / total_pool)))) for s in per_source_counts}
    assigned = sum(alloc.values())
    # Distribute remainder by largest fractional parts while respecting per-source caps
    remainders = []
    for s, cnt in per_source_counts.items():
        exact = total_replace * (cnt / total_pool)
        frac = exact - math.floor(exact)
        remainders.append((frac, s))
    remainders.sort(reverse=True)
    i = 0
    while assigned < total_replace and i < len(remainders):
        s = remainders[i][1]
        if alloc[s] < per_source_counts[s]:
            alloc[s] += 1
            assigned += 1
        i += 1
        if i == len(remainders):
            i = 0  # loop if still remaining
    return alloc

def push_to_hub_quiet(ds: Dataset, repo_id: str, private: bool | None = None):
    kwargs = {}
    if private is not None:
        kwargs["private"] = private
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    print(f"Pushing to hub: {repo_id} (private={kwargs.get('private', 'default')}) ...")
    ds.push_to_hub(repo_id, **kwargs)
    print("✅ Pushed.")

# ---------------------- Main ----------------------

def main():
    parser = argparse.ArgumentParser(description="Remix a DPO base by substituting controlled sources (preserve ALL columns).")
    parser.add_argument("--base_repo", default=BASE_REPO_DEFAULT)
    parser.add_argument("--base_split", default=BASE_SPLIT_DEFAULT)
    parser.add_argument("--wc_repl_repo", default=WC_REPLACEMENT_REPO_DEFAULT)
    parser.add_argument("--valpy_repo", default=VALPY_REPLACEMENT_REPO_DEFAULT)
    parser.add_argument("--out_repo", default=OUT_REPO_DEFAULT)
    parser.add_argument("--seed", type=int, default=RNG_SEED_DEFAULT)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--private", type=str, choices=["true", "false"], default=None, help="Override hub visibility")
    parser.add_argument("--wc_allow_oversample", action="store_true", help="Allow sampling with replacement if WC replacement pool is smaller than needed")
    parser.add_argument("--valpy_chosen_col", type=str, default=None, help="Explicit column name for 'chosen' in valpy repo")
    parser.add_argument("--valpy_rejected_col", type=str, default=None, help="Explicit column name for 'rejected' in valpy repo")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load base (full schema)
    print(f"Loading base: {args.base_repo}:{args.base_split}")
    base = load_dataset(args.base_repo, split=args.base_split)
    base_n = len(base)
    print(f"Base rows: {base_n:,}")

    # Confirm essential columns
    essential = {"source", "chosen", "rejected"}
    missing_essential = essential - set(base.column_names)
    if missing_essential:
        raise ValueError(f"Base dataset missing essential columns: {missing_essential}")

    # ------- STEP 1: Replace all WC_SOURCES_TO_REPLACE with samples from WC replacement repo -------
    print("Loading WC replacement pool:", args.wc_repl_repo)
    wc_pool = load_dataset(args.wc_repl_repo, split="train")
    # Ensure chosen/rejected exist and non-null in pool
    wc_pool = ensure_chosen_rejected(wc_pool, "WC replacement pool")

    # Align pool to base schema; ensure 'source' exists (keep its own if present)
    wc_pool = align_to_base(wc_pool, base)

    # If prompt_id missing/None, synthesize to match base schema expectations
    wc_pool = ensure_prompt_id(wc_pool, prefix="wc")

    # Count how many to replace
    wc_mask = base.filter(lambda x: x["source"] in WC_SOURCES_TO_REPLACE)
    wc_count = len(wc_mask)
    print(f"WC sources to replace: {WC_SOURCES_TO_REPLACE} | count in base: {wc_count:,}")

    if wc_count > 0:
        pool_n = len(wc_pool)
        print(f"WC replacement pool size: {pool_n:,}")
        if pool_n < wc_count and not args.wc_allow_oversample:
            raise ValueError(f"Not enough rows in {args.wc_repl_repo} to replace {wc_count} rows (have {pool_n}). "
                             f"Use --wc_allow_oversample to sample with replacement.")
        # Sample replacements
        if pool_n >= wc_count:
            repl_idx = random.sample(range(pool_n), wc_count)
        else:
            repl_idx = [random.randrange(pool_n) for _ in range(wc_count)]
        wc_replacements = wc_pool.select(repl_idx)

        # Remove those rows from base
        to_keep = base.filter(lambda x: x["source"] not in WC_SOURCES_TO_REPLACE)
        print(f"After removing WC sources: {len(to_keep):,} rows")
        # Combine with replacements
        base_after_wc = concatenate_datasets([to_keep, wc_replacements])
    else:
        print("No WC sources found to replace.")
        base_after_wc = base

    assert len(base_after_wc) == base_n, "Dataset size changed after WC replacement; expected to keep total counts the same."

    # ------- STEP 2: Partially replace three specific sources with valpy data -------
    print("Loading VALPY replacement pool:", args.valpy_repo)
    valpy_pool = load_dataset(args.valpy_repo, split="train")

    # If explicit col names provided, rename them first
    if args.valpy_chosen_col and args.valpy_chosen_col in valpy_pool.column_names and "chosen" not in valpy_pool.column_names:
        valpy_pool = valpy_pool.rename_column(args.valpy_chosen_col, "chosen")
    if args.valpy_rejected_col and args.valpy_rejected_col in valpy_pool.column_names and "rejected" not in valpy_pool.column_names:
        valpy_pool = valpy_pool.rename_column(args.valpy_rejected_col, "rejected")

    # Ensure chosen/rejected exist and non-null
    valpy_pool = ensure_chosen_rejected(valpy_pool, "VALPY replacement pool")

    # Ensure/source value and align to base schema
    valpy_pool = align_to_base(valpy_pool, base_after_wc, ensure_source_value=args.valpy_repo)

    # Ensure prompt_id present/non-null
    valpy_pool = ensure_prompt_id(valpy_pool, prefix="valpy")

    # Determine counts for targeted sources in the current base
    targeted = base_after_wc.filter(lambda x: x["source"] in SOURCES_TO_PARTIAL_REPLACE_WITH_VALPY)
    targeted_total = len(targeted)
    counts_per_source = {}
    for s in SOURCES_TO_PARTIAL_REPLACE_WITH_VALPY:
        counts_per_source[s] = len(base_after_wc.filter(lambda x: x["source"] == s))

    print(f"Targeted sources for partial replacement: {SOURCES_TO_PARTIAL_REPLACE_WITH_VALPY}")
    print("Counts per targeted source:", counts_per_source)
    print(f"Total targeted rows in base: {targeted_total:,}")

    # Replacement budget is limited by valpy pool size
    valpy_n = len(valpy_pool)
    R = min(valpy_n, targeted_total)
    print(f"VALPY pool size: {valpy_n:,} | Will replace R={R:,} rows across targeted sources (proportional).")

    if R > 0 and targeted_total > 0:
        alloc = split_counts(R, counts_per_source)
        print("Per-source replacement allocation:", alloc)

        # Remove allocated rows per source and keep the remainder
        remaining_base_parts = []
        replace_total = 0
        for s, k in alloc.items():
            sub = base_after_wc.filter(lambda x: x["source"] == s)
            n_sub = len(sub)
            if k > 0 and n_sub > 0:
                idx = list(range(n_sub))
                drop_idx = set(random.sample(idx, k)) if k <= n_sub else set(idx)
                keep_idx = [i for i in idx if i not in drop_idx]
                remaining_base_parts.append(sub.select(keep_idx))
                replace_total += min(k, n_sub)
            else:
                if n_sub > 0:
                    remaining_base_parts.append(sub)

        # Add all non-targeted rows
        non_targeted = base_after_wc.filter(lambda x: x["source"] not in SOURCES_TO_PARTIAL_REPLACE_WITH_VALPY)
        remaining_base_parts.append(non_targeted)

        base_minus_alloc = concatenate_datasets(remaining_base_parts) if len(remaining_base_parts) > 1 else remaining_base_parts[0]

        # Select R rows from valpy pool (no replacement)
        if len(valpy_pool) >= R:
            valpy_idx = random.sample(range(len(valpy_pool)), R)
        else:
            valpy_idx = [random.randrange(len(valpy_pool)) for _ in range(R)]
        valpy_take = valpy_pool.select(valpy_idx)

        remixed = concatenate_datasets([base_minus_alloc, valpy_take])
    else:
        print("No partial replacement performed (R=0 or no targeted rows).")
        remixed = base_after_wc

    # Final sanity checks
    assert len(remixed) == base_n, f"Final dataset size mismatch: got {len(remixed)}, expected {base_n}"
    # Ensure chosen/rejected non-null across final
    def _ok_final(x):
        return (x["chosen"] is not None and str(x["chosen"]).strip() != "") and (x["rejected"] is not None and str(x["rejected"]).strip() != "")
    ok_cnt = len(remixed.filter(_ok_final))
    if ok_cnt != len(remixed):
        raise ValueError(f"{len(remixed) - ok_cnt} rows in final dataset have null/empty 'chosen' or 'rejected'")

    # Print a quick before/after source histogram
    def summarize(ds: Dataset, title: str, top_k: int = 30):
        from collections import Counter
        counts = Counter(ds["source"])
        total = sum(counts.values())
        print(f"\n--- Source composition: {title} (n={total}) ---")
        for src, cnt in counts.most_common(top_k):
            pct = 100.0 * cnt / total if total else 0.0
            print(f"{src}\t{cnt}\t{pct:.2f}%")

    summarize(base, "Original base")
    summarize(remixed, "Remixed final")

    # Push or dry-run
    if args.dry_run:
        print("\nDRY-RUN: not pushing to hub.")
    else:
        private_flag = None if args.private is None else (True if args.private.lower() == "true" else False)
        push_to_hub_quiet(remixed, args.out_repo, private=private_flag)

    print("\nAll done. ✅")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
