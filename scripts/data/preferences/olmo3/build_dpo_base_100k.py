#!/usr/bin/env python3
import os
import sys
import re
import random
import argparse
from collections import Counter
from typing import Set, Dict, List

from datasets import load_dataset, Dataset

# ---------------------- Config ----------------------
DEFAULT_QWQ_DATASET = "allenai/preference-qwq-judge"
DEFAULT_GEMMA_DATASET = "allenai/preference-gemma3-judge"
DEFAULT_SPLIT = "train"

LIMITED_SOURCES = {
    "Wildchat-1M-gpt-4.1-regenerated-english",
    "Wildchat-1m-gpt-4.1-regeneration-not-english",
    "filtered_wc_sample_500k",
}
TOTAL_SAMPLE = 100_000
MAX_LIMITED = 35_000
RNG_SEED = 42

TARGET_QWQ_100K = "allenai/dpo-base-100k-qwq-judge"
TARGET_GEMMA_100K = "allenai/dpo-base-100k-gemma3-judge"

UNUSED_NS_QWQ = "saumyamalik"
UNUSED_NS_GEMMA = "saumyamalik"
SUFFIX_QWQ = "unused-qwq"
SUFFIX_GEMMA = "unused-gemma3"

# ---------------------- Utils ----------------------

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "unknown"

def filter_valid(ds: Dataset) -> Dataset:
    if "is_valid_row" not in ds.column_names:
        raise ValueError("Dataset missing 'is_valid_row' column")
    return ds.filter(lambda ex: ex["is_valid_row"] == True, num_proc=os.cpu_count())

def membership_filter(ds: Dataset, keep_ids: Set[str]) -> Dataset:
    # Batched membership check for speed
    if "prompt_id" not in ds.column_names:
        raise ValueError("Dataset missing 'prompt_id' column")
    def _fn(batch):
        pids = batch["prompt_id"]
        return {"_keep_mask": [pid in keep_ids for pid in pids]}
    out = ds.map(_fn, batched=True, batch_size=65536)
    out = out.filter(lambda ex, idx: ex["_keep_mask"], with_indices=True, num_proc=os.cpu_count())
    return out.remove_columns("_keep_mask")

def ensure_required_columns(ds: Dataset, name: str):
    required = {"prompt_id", "is_valid_row", "source"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

def build_prompt_to_source(ds: Dataset) -> Dict[str, str]:
    pids = ds["prompt_id"]
    sources = ds["source"]
    return dict(zip(pids, sources))

def print_source_stats(ds: Dataset, label: str, top_k: int = 50):
    counts = Counter(ds["source"])
    total = sum(counts.values())
    print(f"\n--- Source composition for {label} (n={total}) ---")
    for src, cnt in counts.most_common(top_k):
        pct = 100.0 * cnt / total if total else 0.0
        print(f"{src}\t{cnt}\t{pct:.2f}%")
    limited_count = sum(counts[s] for s in LIMITED_SOURCES if s in counts)
    print(f"Limited-source total (cap={MAX_LIMITED}): {limited_count}")

def sample_with_cap(common_ids: Set[str],
                    id_to_source: Dict[str, str],
                    total_n: int = TOTAL_SAMPLE,
                    cap_sources: Set[str] = LIMITED_SOURCES,
                    cap_n: int = MAX_LIMITED,
                    seed: int = RNG_SEED) -> Set[str]:
    random.seed(seed)
    limited = [pid for pid in common_ids if id_to_source.get(pid) in cap_sources]
    other = [pid for pid in common_ids if id_to_source.get(pid) not in cap_sources]
    print(f"Common ids: {len(common_ids)} | limited-candidate ids: {len(limited)} | other-candidate ids: {len(other)}")

    if len(common_ids) < total_n:
        raise ValueError(f"Not enough common valid prompt_ids: have {len(common_ids)}, need {total_n}.")

    take_limited = min(cap_n, len(limited))
    need_other = total_n - take_limited
    if len(other) < need_other:
        raise ValueError(
            f"Constraint infeasible: need {need_other} from non-limited sources but only {len(other)} available "
            f"(limited candidates={len(limited)}, cap={cap_n})."
        )

    limited_sample = set(random.sample(limited, take_limited)) if take_limited > 0 else set()
    other_sample = set(random.sample(other, need_other)) if need_other > 0 else set()
    sampled = limited_sample | other_sample
    assert len(sampled) == total_n, f"Sampled {len(sampled)} != {total_n}"
    lim_in_sample = sum(1 for pid in sampled if id_to_source.get(pid) in cap_sources)
    assert lim_in_sample <= cap_n, f"Cap violated: {lim_in_sample} > {cap_n}"
    return sampled

def push_to_hub_quiet(ds: Dataset, repo_id: str, private: bool | None = None):
    kwargs = {}
    if private is not None:
        kwargs["private"] = private
    token = os.environ.get("HF_TOKEN")  # optional
    if token:
        kwargs["token"] = token
    print(f"Pushing to hub: {repo_id} (private={kwargs.get('private', 'default')}) ...")
    ds.push_to_hub(repo_id, **kwargs)
    print("✅ Pushed.")

def main():
    parser = argparse.ArgumentParser(description="Build 100k DPO bases + publish unused-by-source splits (ALL columns preserved)")
    parser.add_argument("--qwq", default=DEFAULT_QWQ_DATASET)
    parser.add_argument("--gemma", default=DEFAULT_GEMMA_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--total", type=int, default=TOTAL_SAMPLE)
    parser.add_argument("--cap", type=int, default=MAX_LIMITED)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # ---------------- Lightweight pass (only necessary cols) ----------------
    print(f"Loading (lite) {args.qwq}:{args.split} ...")
    qwq_lite = load_dataset(args.qwq, split=args.split)
    ensure_required_columns(qwq_lite, "QWQ")
    qwq_lite = qwq_lite.remove_columns([c for c in qwq_lite.column_names if c not in {"prompt_id","is_valid_row","source"}])

    print(f"Loading (lite) {args.gemma}:{args.split} ...")
    gem_lite = load_dataset(args.gemma, split=args.split)
    ensure_required_columns(gem_lite, "Gemma3")
    gem_lite = gem_lite.remove_columns([c for c in gem_lite.column_names if c not in {"prompt_id","is_valid_row","source"}])

    qwq_valid_lite = filter_valid(qwq_lite)
    gem_valid_lite = filter_valid(gem_lite)
    print(f"QWQ valid (lite): {len(qwq_valid_lite):,} | Gemma3 valid (lite): {len(gem_valid_lite):,}")

    qwq_ids = set(qwq_valid_lite["prompt_id"])
    gem_ids = set(gem_valid_lite["prompt_id"])
    common_ids = qwq_ids & gem_ids
    print(f"Overlap in valid prompt_id: {len(common_ids):,}")
    print(f"Sample overlapping prompt_ids (first 10): {list(common_ids)[:10]}")

    # Assert LIMITED_SOURCES present in overlapping QWQ data
    qwq_common_lite = membership_filter(qwq_valid_lite, common_ids)
    present_sources = set(qwq_common_lite.unique("source"))
    missing_declared = [s for s in LIMITED_SOURCES if s not in present_sources]
    if missing_declared:
        raise ValueError(
            f"Assertion failed: Some LIMITED_SOURCES are not present in the (valid & overlapping) QWQ data: {missing_declared}"
        )
    print("All LIMITED_SOURCES present in overlapping QWQ data ✔")

    # Map for sampling
    pid_to_src_qwq = build_prompt_to_source(qwq_common_lite)

    # Sample with cap
    sampled_ids = sample_with_cap(
        common_ids=common_ids,
        id_to_source=pid_to_src_qwq,
        total_n=args.total,
        cap_sources=LIMITED_SOURCES,
        cap_n=args.cap,
        seed=args.seed,
    )
    print(f"Sampled {len(sampled_ids)} prompt_ids.")

    # ---------------- Reload FULL datasets to preserve ALL columns ----------------
    print(f"Reloading FULL datasets to preserve all columns...")
    qwq_full = load_dataset(args.qwq, split=args.split)
    gem_full = load_dataset(args.gemma, split=args.split)
    ensure_required_columns(qwq_full, "QWQ (full)")
    ensure_required_columns(gem_full, "Gemma3 (full)")

    # Build the 100k subsets from FULL valid data (all columns kept)
    qwq_full_valid = filter_valid(qwq_full)
    gem_full_valid = filter_valid(gem_full)

    qwq_100k = membership_filter(qwq_full_valid, sampled_ids)
    gem_100k = membership_filter(gem_full_valid, sampled_ids)

    # Assert prompt_ids exactly match
    assert set(qwq_100k["prompt_id"]) == set(gem_100k["prompt_id"]) == sampled_ids, \
        "Mismatch: prompt_ids are not identical across the two 100k subsets"

    # Print source composition (from full, though same as lite)
    print_source_stats(qwq_100k, "QWQ 100k")
    print_source_stats(gem_100k, "Gemma3 100k")

    # Push 100k datasets
    if not args.dry_run:
        push_to_hub_quiet(qwq_100k, TARGET_QWQ_100K)
        push_to_hub_quiet(gem_100k, TARGET_GEMMA_100K)

    # ---------------- Remaining valid rows -> per-source public datasets ----------------
    def push_unused_groups(full_ds_valid: Dataset, model_suffix: str, namespace: str):
        remaining = full_ds_valid.filter(lambda ex: ex["prompt_id"] not in sampled_ids, num_proc=os.cpu_count())
        print(f"\nRemaining valid rows for {model_suffix}: {len(remaining):,}")
        sources = remaining.unique("source")
        print(f"Found {len(sources)} sources in remaining {model_suffix}. Pushing each as public dataset...")
        for src in sources:
            part = remaining.filter(lambda ex: ex["source"] == src, num_proc=os.cpu_count())
            if len(part) == 0:
                continue
            repo_id = f"{namespace}/{slugify(src)}-{model_suffix}"
            print(f"Pushing {len(part):,} rows -> {repo_id} (private=False)")
            if not args.dry_run:
                push_to_hub_quiet(part, repo_id, private=False)

    push_unused_groups(qwq_full_valid, SUFFIX_QWQ, UNUSED_NS_QWQ)
    push_unused_groups(gem_full_valid, SUFFIX_GEMMA, UNUSED_NS_GEMMA)

    print("\nAll done. ✅")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
