"""Run the same rows through MaxText's SFT preprocessing; compare with open-instruct's dump."""

import collections
import json
import sys

import transformers
from maxtext.input_pipeline import input_pipeline_utils as u

src = sys.argv[1]
FIX = len(sys.argv) > 2 and sys.argv[2] == "fixed"
tok = (transformers.PreTrainedTokenizerFast if FIX else transformers.AutoTokenizer).from_pretrained(
    sys.argv[3] if len(sys.argv) > 3 else "allenai/olmo-3-tokenizer-instruct-dev"
)
print("tokenizer class", type(tok).__name__)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.unk_token_id
mask = u.SFTPromptMasking("messages", completion_only=True, max_target_length=32768, unk_id=-100)
stats = collections.Counter()
examples = []
with open(src) as f:
    rows = [json.loads(line) for line in f]
for r in rows:
    msgs = r["messages"]  # every field (function_calls, functions) reaches the chat template, as in MaxText
    stats["rows"] += 1
    stats["stratum_" + r.get("stratum", "na")] += 1
    roles = [m["role"] for m in msgs]
    stats["multi_turn"] += roles.count("assistant") > 1
    try:
        ex = u.apply_chat_template({"messages": msgs}, tok, "messages")
    except Exception as e:
        stats["mt_error"] += 1
        examples.append((r["i"], "error", str(e)[:200]))
        continue
    ex = u.tokenization(ex, tok, truncation=False, max_length=32768, column_names=["messages"])
    o = mask.map(ex)
    mt_ids, mt_lab = o["inputs"].tolist(), o["targets"].tolist()
    oi_ids, oi_lab = r["input_ids"], r["labels"]
    # MaxText targets are unshifted next-token targets of inputs (same convention as OI labels)
    ids_eq = mt_ids == oi_ids
    tr_mt = {i for i, t in enumerate(mt_lab) if t != -100}
    tr_oi = {i for i, t in enumerate(oi_lab) if t != -100}
    stats["ids_equal"] += ids_eq
    stats["ids_equal_and_mask_equal"] += ids_eq and tr_mt == tr_oi
    if not (ids_eq and tr_mt == tr_oi):
        stats["mismatch_" + r.get("stratum", "na")] += 1
    stats["oi_trained_tokens"] += len(tr_oi)
    stats["mt_trained_tokens"] += len(tr_mt)
    stats["oi_tokens"] += len(oi_ids)
    stats["mt_tokens"] += len(mt_ids)
    if not ids_eq and len(examples) < 4:
        k = next((j for j, (a, b) in enumerate(zip(mt_ids, oi_ids)) if a != b), min(len(mt_ids), len(oi_ids)))
        examples.append(
            (
                r["i"],
                roles,
                len(oi_ids),
                len(mt_ids),
                "first diff @",
                k,
                "OI:",
                repr(tok.decode(oi_ids[max(0, k - 8) : k + 12])),
                "MT:",
                repr(tok.decode(mt_ids[max(0, k - 8) : k + 12])),
            )
        )
    elif ids_eq and tr_mt != tr_oi and len(examples) < 4:
        d = sorted(tr_mt ^ tr_oi)[:12]
        examples.append(
            (r["i"], roles, "mask diff at", d, [("MT" if i in tr_mt else "OI", tok.decode([oi_ids[i]])) for i in d])
        )
print(dict(stats))
for e in examples:
    print(e)
