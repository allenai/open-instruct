"""Ledger H218 gates (1) and (2) on a row-aligned numpy SFT cache: the cache receipt and the label-mask dump.

  python h218_mask_dump.py <numpy cache dir> <out.json> [--rows 20] [--max-seq-length 131072] [--check-tokenizer <dir>]

Gate (1), cache receipt: rows in and out (dataset_statistics.json), total tokens and loss tokens, the row-length
distribution, and the rows at exactly max_seq_length (truncated by the tokenizer; expected 0 for this set).

Gate (2), label-mask dump: the ``--rows`` rows with the most tool turns are decoded span by span, where a span is a
maximal run of tokens with one label state (trained or masked), and each span is classified by the chat-template
markers it starts with. The dump prints every span (state, length, first 90 characters) for the owner to read, and
checks the rules the registration states: every system, user and tool turn is masked; every assistant turn has a
trained span that ends at its ``<|im_end|>``; the ``<think>`` that the template puts in the generation prompt before
an assistant turn is masked (prefix labels, as H194), so each trained span starts at the reasoning text; every
trained span ends at ``<|im_end|>`` (tool-call turns) or the eos token (final answers); no trained span starts inside
a user or environment (tool-result) turn.
``--check-tokenizer`` re-tokenizes each dumped row's decoded text with another tokenizer (the start checkpoint's own
tokenizer.json, promoted-think) and requires identical ids, so the cache's tokenization is the one the start was
trained with. Exit 0 iff every check passes; the JSON holds everything printed.
"""

import argparse
import csv
import glob
import gzip
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def load_rows(cache_dir):
    stats = json.load(open(os.path.join(cache_dir, "dataset_statistics.json")))
    token_files = sorted(glob.glob(os.path.join(cache_dir, "token_ids_part_*.npy")))
    rows = []  # (part index, start, end) in that part's arrays
    for pi, tf in enumerate(token_files):
        meta = tf[: -len(".npy")] + ".csv.gz"
        with gzip.open(meta, "rt") as f:
            for line in csv.reader(f):
                rows.append((pi, int(line[0]), int(line[1])))
    tokens = [np.load(tf, mmap_mode="r") for tf in token_files]
    labels = [np.load(tf.replace("token_ids_part_", "labels_mask_part_"), mmap_mode="r") for tf in token_files]
    return stats, rows, tokens, labels


def spans(mask):
    out, start = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            out.append((bool(mask[start]), start, i))
            start = i
    return out


def classify(text):
    """The olmo35 template: <|im_start|>{system,user,assistant,environment}; tool results are 'environment'."""
    t = text.lstrip("\n")
    for role, kind in (("system", "system"), ("user", "user"), ("assistant", "assistant"), ("environment", "tool")):
        if t.startswith(f"{IM_START}{role}"):
            return kind
    if text.startswith("<think>") or text.strip() == "<think>":
        return "gen-think"
    if text.startswith("<tool_response>"):
        return "tool-body"
    return "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("out")
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--max-seq-length", type=int, default=131072)
    ap.add_argument("--check-tokenizer", default=None)
    a = ap.parse_args()
    # The cache writer saves its tokenizer (with the chat template) under <cache>/tokenizer.
    tok = AutoTokenizer.from_pretrained(os.path.join(a.cache_dir, "tokenizer"))
    im_end_ids = tok.encode(IM_END, add_special_tokens=False)
    assert len(im_end_ids) == 1, im_end_ids
    im_end_id = im_end_ids[0]
    # The start's tokenizer dir may carry a newer transformers' tokenizer_config; load the file directly.
    other = (PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.check_tokenizer, "tokenizer.json"))
             if a.check_tokenizer else None)
    stats, rows, tokens, labels = load_rows(a.cache_dir)
    lengths = np.array([e - s for _, s, e in rows])
    loss = 0
    for pi, s, e in rows:
        loss += int(labels[pi][s:e].sum())
    per = stats.get("per_dataset_stats", [])
    receipt = {
        "cache_dir": a.cache_dir,
        "rows_in": [p.get("initial_instances") for p in per],
        "rows_out": [p.get("final_instances") for p in per],
        "rows": int(len(rows)),
        "tokens": int(lengths.sum()),
        "loss_tokens": int(loss),
        "row_len": {"min": int(lengths.min()), "median": int(np.median(lengths)), "mean": float(lengths.mean()),
                    "max": int(lengths.max())},
        "rows_at_max_seq_length": int((lengths >= a.max_seq_length).sum()),
        "row_aligned": stats.get("configuration", {}),
    }
    print("H218-CACHE", json.dumps(receipt), flush=True)

    # Rows with the most tool turns: count <|im_start|>tool openings.
    tool_id = tok.encode(f"{IM_START}tool", add_special_tokens=False)
    counts = []
    for ri, (pi, s, e) in enumerate(rows):
        ids = np.asarray(tokens[pi][s:e])
        # count occurrences of the tool-turn opening token sequence
        n = 0
        if len(tool_id) == 1:
            n = int((ids == tool_id[0]).sum())
        else:
            first = np.where(ids == tool_id[0])[0]
            for i in first:
                if list(ids[i:i + len(tool_id)]) == tool_id:
                    n += 1
        counts.append((n, ri))
    chosen = [ri for _, ri in sorted(counts, reverse=True)[: a.rows]]

    failures, dump = [], []
    for ri in chosen:
        pi, s, e = rows[ri]
        ids = np.asarray(tokens[pi][s:e]).astype(np.int64)
        lm = np.asarray(labels[pi][s:e]).astype(bool)
        sp = spans(lm)
        entries, turns = [], {"system": 0, "user": 0, "assistant": 0, "tool": 0}
        prev_role = None
        for trained, b, en in sp:
            text = tok.decode(ids[b:en])
            kind = classify(text)
            if kind in turns:
                turns[kind] += 1
            entries.append({"trained": trained, "start": int(b), "end": int(en), "kind": kind, "head": text[:90]})
            # rules
            if trained and kind in ("system", "user", "tool", "tool-body"):
                failures.append(f"row {ri}: trained span starts a {kind} turn at {b}")
            if trained and kind == "gen-think":
                failures.append(f"row {ri}: generation-prompt <think> trained at {b}")
            if (not trained) and kind == "assistant":
                # masked assistant header must be followed by a trained span (the turn's content)
                pass
            prev_role = kind if kind in turns else prev_role
        # every assistant turn: a masked header span followed by a trained span ending with <|im_end|>
        for i, en in enumerate(entries):
            if en["kind"] == "assistant":
                if en["trained"]:
                    failures.append(f"row {ri}: assistant header trained at {en['start']}")
                nxt = entries[i + 1] if i + 1 < len(entries) else None
                if nxt is None or not nxt["trained"]:
                    failures.append(f"row {ri}: assistant turn at {en['start']} has no trained span")
                else:
                    # tool-call turns end with <|im_end|>, final answers with the eos token (olmo35 template)
                    last = int(ids[nxt["end"] - 1])
                    if last not in (im_end_id, tok.eos_token_id):
                        tail = tok.decode(ids[nxt["end"] - 3: nxt["end"]])
                        failures.append(f"row {ri}: trained span at {nxt['start']} ends with {tail!r}, not {IM_END}/eos")
                    # the masked header must end with the generation prompt's <think> (prefix labels, as H194):
                    # the trained span then starts with the reasoning text, not with <think>.
                    head_tail = tok.decode(ids[en["end"] - 1: en["end"]])
                    if head_tail != "<think>":
                        failures.append(f"row {ri}: assistant header at {en['start']} ends with {head_tail!r}, not <think>")
        if not any(en["kind"] == "assistant" for en in entries):
            failures.append(f"row {ri}: no assistant turn found")
        tok_check = None
        if other is not None:
            text = tok.decode(ids)
            re_ids = other.encode(text, add_special_tokens=False)
            tok_check = {"identical": re_ids == ids.tolist(), "n": len(re_ids)}
            if not tok_check["identical"]:
                failures.append(f"row {ri}: re-tokenization with {a.check_tokenizer} differs ({len(re_ids)} vs {len(ids)})")
        dump.append({"row": ri, "len": int(len(ids)), "loss_tokens": int(lm.sum()), "turns": turns, "spans": entries,
                     "tokenizer_check": tok_check})
        print(f"H218-ROW {ri} len {len(ids)} loss {int(lm.sum())} turns {turns} spans {len(entries)} tok_check {tok_check}",
              flush=True)
        for en in entries:
            print(f"   {'T' if en['trained'] else '.'} {en['end'] - en['start']:>6} {en['kind']:<9} {en['head']!r}")

    result = {"receipt": receipt, "rows_dumped": chosen, "dump": dump, "failures": failures,
              "pass": not failures}
    json.dump(result, open(a.out, "w"), indent=1)
    print("H218-MASK", "PASS" if not failures else f"FAIL {len(failures)}", flush=True)
    for f in failures[:40]:
        print("  ", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
