#!/usr/bin/env python3
# decontam_old_vs_new_with_pivot.py
# Unique-row contamination counters (old vs new) + NEW-method per-eval pivot heatmap.
# NEW method applies: n-gram only, all-spans heuristic, dataset-specific threshold (verified -> 0.85).

import argparse, os, sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm

# ----------- Configure eval suites and train datasets -----------
EVAL_SUFFIXES = [
    "code_generation_lite",
    "aime_2024",
    "aime_2025",
    "ZebraLogicBench-private",
    "multilingual_mbpp",
    "mmlu",
    "openai_humaneval",
    "gsm8k",
    "hendrycks_math",
    "IFEval",
    "math_dataset",
    "PopQA",
    "alpaca_eval",
    "bbh",
    "truthful_qa",
    "wildguardmix",
    "wildjailbreak",
    "tulu-3-trustllm-jailbreaktrigger-eval",
    "tulu-3-harmbench-eval",
    "tulu-3-do-anything-now-eval",
    "toxigen-prompts",
    "XSTest",
    "SimpleQA",
    "omega-all-test-prompts",
    "MMLU-Pro",
    "gpqa",
    "agi_eval_en",
    "bigcodebench",
    "math_dataset",
    "IFBench_test",
    "hle",
    "SuperGPQA",
    "bbeh",
    "aimo-validation-amc",
    "olympiadbench",
    "cruxeval",
    "codeeditorbench_prompts",
    "wmdp",
    "BeyondAIME",
    "strong_reject_data",
    "bbq_prompts",
]

USABLE_TRAIN = [
   "hamishivi/rlvr_combined_prompts"
]

# ----------- Fast, robust JSON helpers -----------
try:
    import orjson as _fastjson
    def _loads(b: bytes):
        return _fastjson.loads(b)
    def _dumps(o) -> str:
        return _fastjson.dumps(
            o,
            option=_fastjson.OPT_NAIVE_UTC | _fastjson.OPT_SERIALIZE_NUMPY
        ).decode("utf-8")
except Exception:
    import json as _fastjson
    import datetime as _dt
    def _loads(b: bytes):
        return _fastjson.loads(b.decode("utf-8"))
    def _json_default(obj):
        if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
            try: return obj.isoformat()
            except Exception: return str(obj)
        try:
            import numpy as _np
            if isinstance(obj, _np.generic): return obj.item()
            if isinstance(obj, _np.ndarray): return obj.tolist()
        except Exception:
            pass
        if isinstance(obj, (set, tuple)): return list(obj)
        if isinstance(obj, (bytes, bytearray)):
            try: return obj.decode("utf-8", errors="replace")
            except Exception: return str(obj)
        return str(obj)
    def _dumps(o) -> str:
        return _fastjson.dumps(o, default=_json_default, ensure_ascii=False)

# ----------- Optional HuggingFace datasets (only needed if --decontaminate_to_hub or kept/removed ids) -----------
try:
    from datasets import load_dataset, Dataset
except Exception:
    load_dataset, Dataset = None, None

# ----------- Heuristics (all-spans) -----------
GENERIC_STEM_WITH_NL = ["Which","of","the","following","statements","is","correct","?","\n"]
GENERIC_STEM_NO_NL   = ["Which","of","the","following","statements","is","correct","?"]
# Present the answer in LaTex format: \boxed{Your answer}
GENERIC_STEM_MATH = ["Present", "the", "answer", "in", "LaTex", "format", ":", "\\boxed{Your", "answer", "}"]
GENERIC_STEM_MATH = ["Present", "the", "answer", "in", "LaTex", "format", ":", "\\boxed{Your", "answer", "}", "\n"]

def tokens_majority_len1(tokens: List[str]) -> bool:
    if not tokens: return False
    cleaned = [t.strip() for t in tokens if isinstance(t, str) and t.strip()]
    if not cleaned: return False
    return sum(1 for t in cleaned if len(t) == 1) > (len(cleaned) / 2.0)

def is_generic_stem(tokens: List[str]) -> bool:
    return tokens == GENERIC_STEM_WITH_NL or tokens == GENERIC_STEM_NO_NL or tokens == GENERIC_STEM_MATH

def analyze_spans_and_heuristics(matching_tokens, score) -> Tuple[bool, bool, bool]:
    """
    Returns:
      any_generic: True if ANY span equals the generic stem
      any_single_lt09: True if ANY span is majority single-char tokens AND score < 0.9
      drop_all_spans: True if EVERY span triggers (generic OR single-char<0.9)
    """
    if not isinstance(matching_tokens, list) or not matching_tokens:
        return False, False, False
    try:
        s = float(score) if score is not None else None
    except Exception:
        s = None

    def span_is_generic(span: List[str]) -> bool:
        return isinstance(span, list) and is_generic_stem(span)

    def span_is_single_lt09(span: List[str]) -> bool:
        if not isinstance(span, list): return False
        return (s is not None and s < 0.9) and tokens_majority_len1(span)

    generic_flags, single_lt09_flags = [], []
    for span in matching_tokens:
        generic_flags.append(span_is_generic(span))
        single_lt09_flags.append(span_is_single_lt09(span))

    any_generic = any(generic_flags)
    any_single_lt09 = any(single_lt09_flags)
    drop_all_spans = all(g or t for g, t in zip(generic_flags, single_lt09_flags))
    return any_generic, any_single_lt09, drop_all_spans

def drop_by_heuristics_all_spans(matching_tokens, score) -> bool:
    _, _, drop_all = analyze_spans_and_heuristics(matching_tokens, score)
    return drop_all

# ----------- Utility -----------
def idx_prefix(dataset_name: str, index_type: str) -> str:
    return dataset_name.replace("/", "_").lower() + f"_{index_type}"

def is_verified_dataset(train_ds_name: str) -> bool:
    # case-insensitive substring 'verified'
    return "verified" in train_ds_name.lower()

def write_tsv(path: str, header: List[str], rows: List[List[str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

# ----------- Core reader (N-GRAM ONLY) -----------
def read_ngram_maps_for_file(path: str) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Return (before_ngram, after_ngram) maps for a single {train, eval} jsonl file:
      - before_ngram: original_id -> max score across ALL n-gram hits (no heuristics)
      - after_ngram : original_id -> max score across n-gram hits that PASS all-spans heuristic
    Ignores 'results' (vector) and 'train_docs' (exact) for both maps.
    """
    before, after = {}, {}
    if not os.path.exists(path):
        return before, after

    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            # Fast prefilter: skip lines that obviously have no matches without JSON parse
            if b'"matches"' not in raw:
                continue
            try:
                d = _loads(raw)
            except Exception:
                continue
            if not (isinstance(d, dict) and "matches" in d):
                continue
            matches = d.get("matches", [])
            if not isinstance(matches, list):
                continue

            for m in matches:
                oid = m.get("source", {}).get("original_id")
                if not isinstance(oid, int):
                    continue
                score = m.get("score", None)
                try:
                    s = float(score) if score is not None else float("-inf")
                except Exception:
                    s = float("-inf")

                # BEFORE (no heuristics)
                if s > before.get(oid, float("-inf")):
                    before[oid] = s

                # AFTER (keep only if heuristics NOT tripped)
                mt = m.get("matching_tokens", [])
                if not drop_by_heuristics_all_spans(mt, s):
                    if s > after.get(oid, float("-inf")):
                        after[oid] = s

    return before, after

# ---- Top-level worker for mp pools (picklable) ----
def worker_ngram_read(task: Tuple[str, str, str]):
    ds, ev, fpath = task
    before_map, after_map = read_ngram_maps_for_file(fpath)
    return ds, ev, before_map, after_map

# ----------- Optional: push filtered to Hub -----------
def upload_filtered_to_hub(dataset_name: str, removed_ids_new: set, namespace: str) -> None:
    if load_dataset is None or Dataset is None:
        raise RuntimeError("datasets library not available; can't push to hub.")
    ds = load_dataset(dataset_name, split="train")
    n = len(ds)
    keep_idx = [i for i in range(n) if i not in removed_ids_new]
    target = f"{namespace}/{dataset_name.replace('/', '_')}_decontaminated_newheuristics"
    print(f"[upload] {dataset_name}: pushing {len(keep_idx)}/{n} rows to {target}")
    ds.select(keep_idx).push_to_hub(target, private=True)

# ----------- Main -----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Dir with cached *_text_*.jsonl files")
    ap.add_argument("--index_type", choices=["text","vector"], default="text",
                    help="Index name part in cache filenames (usually 'text')")
    # thresholds
    ap.add_argument("--old_floor", type=float, default=0.5,
                    help="Old method: n-gram score > this value (strict >). Default 0.5")
    ap.add_argument("--new_match_threshold", type=float, default=0.5,
                    help="New method base threshold for non-verified datasets (strict >). 'verified' datasets use 0.85 hard override.")
    ap.add_argument("--source_field", type=str, default="source",
                    help="Per-sample field name in train dataset used to decide higher threshold.")
    ap.add_argument("--source_match_substring", type=str, default="if_multi_constraints,All_Puzzles,math_dataset,omega,orz_math,dapo-math,acereason,deepscaler,mathsub",
                    help="Comma-separated, case-insensitive substrings in source field that trigger the higher threshold (0.85).")
    ap.add_argument("--disable_per_sample_source", action="store_true",
                    help="If set, do not use per-sample source-based thresholds (use base threshold everywhere).")
    ap.add_argument("--zebra_eval_min_threshold", type=float, default=0.85,
                    help="If eval name contains 'zebra', enforce at least this threshold.")
    # outputs
    ap.add_argument("--per_eval_counts_tsv", required=True,
                    help="TSV: per train×eval, old vs new unique-row stripped counts")
    ap.add_argument("--per_train_union_counts_tsv", required=True,
                    help="TSV: per train (union across eval), old vs new unique-row stripped counts")
    ap.add_argument("--pivot_new_tsv", required=True,
                    help="TSV: per-eval pivot heatmap using NEW method (n-gram, heuristics, dataset thresholds)")
    ap.add_argument("--kept_removed_ids_dir", type=str, default=None,
                    help="If set, write {train}__NEW_kept_ids.txt and {train}__NEW_removed_ids.txt")
    # optional hub upload
    ap.add_argument("--decontaminate_to_hub", action="store_true",
                    help="If set, push decontaminated splits (NEW method) to HF Hub")
    ap.add_argument("--hub_namespace", type=str, default="saumyamalik",
                    help="HF namespace for uploading decontaminated splits")
    # performance
    ap.add_argument("--workers", type=int, default=-1,
                    help="Parallel file reading (-1 = auto cpu count, 0 = sequential)")
    args = ap.parse_args()
    # If auto, default to using available CPUs for faster cache reads
    if args.workers is None or args.workers < 0:
        try:
            args.workers = max(1, (os.cpu_count() or 1))
        except Exception:
            args.workers = 1

    # Accumulators: per (train, eval) maps (N-GRAM ONLY)
    per_pair_before: Dict[Tuple[str,str], Dict[int,float]] = defaultdict(dict)  # no heuristics
    per_pair_after:  Dict[Tuple[str,str], Dict[int,float]] = defaultdict(dict)  # all-spans kept

    # Build tasks (train_ds, eval, file_path)
    tasks = []
    for ds in USABLE_TRAIN:
        prefix = idx_prefix(ds, args.index_type)
        for ev in EVAL_SUFFIXES:
            fname = f"{prefix}_{ev}.jsonl"
            fpath = os.path.join(args.input_dir, fname)
            tasks.append((ds, ev, fpath))

    # Read files
    if args.workers and args.workers > 0:
        import multiprocessing as mp
        with mp.Pool(processes=args.workers, maxtasksperchild=20) as pool:
            for ds, ev, before_map, after_map in tqdm(
                pool.imap_unordered(worker_ngram_read, tasks, chunksize=max(1, len(tasks)//(args.workers*4) or 1)),
                total=len(tasks),
                desc="Reading n-gram caches"
            ):
                # merge (max)
                dst_b = per_pair_before[(ds, ev)]
                for oid, s in before_map.items():
                    if s > dst_b.get(oid, float("-inf")):
                        dst_b[oid] = s
                dst_a = per_pair_after[(ds, ev)]
                for oid, s in after_map.items():
                    if s > dst_a.get(oid, float("-inf")):
                        dst_a[oid] = s
    else:
        for ds, ev, fpath in tqdm(tasks, desc="Reading n-gram caches"):
            before_map, after_map = read_ngram_maps_for_file(fpath)
            # merge (max) into accumulators
            dst_b = per_pair_before[(ds, ev)]
            for oid, s in before_map.items():
                if s > dst_b.get(oid, float("-inf")):
                    dst_b[oid] = s
            dst_a = per_pair_after[(ds, ev)]
            for oid, s in after_map.items():
                if s > dst_a.get(oid, float("-inf")):
                    dst_a[oid] = s

    # Threshold helpers (optimized)
    # Build per-dataset set of oids whose source contains the configured substring.
    # Additionally, track oids whose source specifically contains 'omega' to apply a stricter threshold.
    matching_oids_by_ds: Dict[str, set] = {}
    omega_oids_by_ds: Dict[str, set] = {}
    if not args.disable_per_sample_source and load_dataset is not None:
        cache_subs = [s.strip().lower() for s in (args.source_match_substring or "").split(",") if s.strip()]
        for ds in USABLE_TRAIN:
            # Candidate oids: union across evals from per_pair_after keys
            candidate_oids = set()
            for ev in EVAL_SUFFIXES:
                candidate_oids.update(per_pair_after.get((ds, ev), {}).keys())

            if not candidate_oids:
                matching_oids_by_ds[ds] = set()
                continue

            # Load only needed rows
            try:
                ds_obj = load_dataset(ds, split="train")
            except Exception as e:
                print(f"[source_map] Could not load {ds}: {e}", file=sys.stderr)
                matching_oids_by_ds[ds] = set()
                continue

            positives = set()
            omega_positives = set()
            field = args.source_field
            for oid in candidate_oids:
                try:
                    row = ds_obj[int(oid)]
                    val = row.get(field) if isinstance(row, dict) else None
                    if isinstance(val, str):
                        low = val.lower()
                        if any(sub in low for sub in cache_subs):
                            positives.add(int(oid))
                        if "omega" in low or "dapo-math" in low:
                            omega_positives.add(int(oid))
                except Exception:
                    continue
            matching_oids_by_ds[ds] = positives
            omega_oids_by_ds[ds] = omega_positives

    def new_threshold_for_sample(ds: str, ev: str, oid: int) -> float:
        """Per-sample threshold for a given eval:
        - If eval name contains 'zebra', enforce at least args.zebra_eval_min_threshold
        - Else if sample source contains 'omega', use 0.95
        - Else if sample matches other configured source substrings, use 0.85
        - Else use args.new_match_threshold
        """
        if args.disable_per_sample_source:
            base = args.new_match_threshold
        else:
            int_oid = int(oid)
            if int_oid in omega_oids_by_ds.get(ds, set()):
                base = 0.95
            elif int_oid in matching_oids_by_ds.get(ds, set()):
                base = 0.85
            else:
                base = args.new_match_threshold
        # Eval-based bump: zebra, mmlu-pro, gpqa, ifbench, bbh, bbeh
        if isinstance(ev, str):
            ev_low = ev.lower()
            if any(k in ev_low for k in ("zebra", "mmlu-pro", "gpqa", "ifbench", "bbh", "bbeh")):
                return max(base, args.zebra_eval_min_threshold)
        return base
    def old_ids_for_pair(ds: str, ev: str) -> set:
        # Old method: n-gram only, score > old_floor, no heuristics.
        return {oid for oid, s in per_pair_before.get((ds, ev), {}).items() if s > args.old_floor}

    def new_ids_for_pair(ds: str, ev: str) -> set:
        # New method: n-gram only, after all-spans heuristic, score > per-sample threshold
        return {oid for oid, s in per_pair_after.get((ds, ev), {}).items() if s > new_threshold_for_sample(ds, ev, oid)}

    # --- Per train×eval counts (unique rows stripped) ---
    per_eval_rows = []
    for ds in USABLE_TRAIN:
        for ev in EVAL_SUFFIXES:
            old_set = old_ids_for_pair(ds, ev)
            new_set = new_ids_for_pair(ds, ev)
            per_eval_rows.append([
                ds, ev, len(old_set), len(new_set), len(new_set) - len(old_set)
            ])
    write_tsv(args.per_eval_counts_tsv,
              ["train_dataset","eval","old_unique_rows_stripped","new_unique_rows_stripped","delta_new_minus_old"],
              per_eval_rows)

    # --- Per train union across evals (unique rows stripped) ---
    per_train_rows = []
    new_removed_by_train: Dict[str, set] = {ds: set() for ds in USABLE_TRAIN}
    for ds in USABLE_TRAIN:
        old_union, new_union = set(), set()
        for ev in EVAL_SUFFIXES:
            old_union |= old_ids_for_pair(ds, ev)
            ids_new = new_ids_for_pair(ds, ev)
            new_union |= ids_new
        new_removed_by_train[ds] = new_union  # for decontamination
        per_train_rows.append([
            ds, len(old_union), len(new_union), len(new_union) - len(old_union),
            ""
        ])
    write_tsv(args.per_train_union_counts_tsv,
              ["train_dataset","old_union_stripped","new_union_stripped","delta_new_minus_old","note"],
              per_train_rows)

    # --- NEW: Per-eval pivot heatmap (NEW method rules) ---
    # Rows: train datasets, Columns: evals, Cell: count of unique train rows stripped under NEW rules.
    pivot_header = ["train_dataset\\eval"] + EVAL_SUFFIXES
    pivot_rows = []
    for ds in USABLE_TRAIN:
        row = [ds]
        for ev in EVAL_SUFFIXES:
            # count unique oids with after-map score > per-sample threshold (n-gram + heuristics applied)
            score_map = per_pair_after.get((ds, ev), {})
            cnt = sum(1 for _oid, s in score_map.items() if s > new_threshold_for_sample(ds, ev, _oid))
            row.append(cnt)
        pivot_rows.append(row)
    write_tsv(args.pivot_new_tsv, pivot_header, pivot_rows)

    # --- Optional: write kept/removed id lists (NEW method) ---
    if args.kept_removed_ids_dir:
        os.makedirs(args.kept_removed_ids_dir, exist_ok=True)
        for ds in USABLE_TRAIN:
            base = ds.replace("/", "_")
            removed = sorted(new_removed_by_train.get(ds, set()))
            removed_path = os.path.join(args.kept_removed_ids_dir, f"{base}__NEW_removed_ids.txt")
            with open(removed_path, "w", encoding="utf-8") as f:
                f.write("\n".join(str(i) for i in removed) + ("\n" if removed else ""))

            # Optionally compute kept IDs if datasets lib is present
            if load_dataset is not None:
                try:
                    n = len(load_dataset(ds, split="train"))
                    kept = [str(i) for i in range(n) if i not in new_removed_by_train[ds]]
                    kept_path = os.path.join(args.kept_removed_ids_dir, f"{base}__NEW_kept_ids.txt")
                    with open(kept_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(kept) + ("\n" if kept else ""))
                except Exception as e:
                    print(f"[kept_ids] Could not load {ds}: {e}", file=sys.stderr)

    # --- Optional: push decontaminated splits to Hub (NEW method) ---
    if args.decontaminate_to_hub:
        if load_dataset is None or Dataset is None:
            raise RuntimeError("--decontaminate_to_hub set, but 'datasets' lib is unavailable.")
        for ds in tqdm(USABLE_TRAIN, desc="Uploading decontaminated (NEW)"):
            try:
                upload_filtered_to_hub(ds, new_removed_by_train.get(ds, set()), args.hub_namespace)
            except Exception as e:
                print(f"[upload] {ds}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
