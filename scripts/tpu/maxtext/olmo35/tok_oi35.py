"""open-instruct SFT tokenization of Dolci-Think-SFT rows with H045's tokenizer and template
(allenai/dolma2-tokenizer-olmo35 @8b971706, chat_template_name olmo35). Keeps N rows from a shuffled stream,
stratified to include multi-turn and tool rows; saves the tokenizer for MaxText.
  tok_oi35.py <N> <out.jsonl> <tokenizer save dir>"""

import json
import sys

from datasets import load_dataset

from open_instruct import dataset_transformation as dt

N, out, tokdir = int(sys.argv[1]), sys.argv[2], sys.argv[3]
tc = dt.TokenizerConfig(
    tokenizer_name_or_path="allenai/dolma2-tokenizer-olmo35",
    tokenizer_revision="8b9717061fae09d5be814373d189919a62a9a00d",
    chat_template_name="olmo35",
)
tok = tc.tokenizer
tok.save_pretrained(tokdir)
print(
    "tokenizer",
    type(tok).__name__,
    "fast",
    tok.is_fast,
    "generation blocks",
    "generation" in (tok.chat_template or ""),
)
ds = load_dataset("allenai/Dolci-Think-SFT", split="train", streaming=True).shuffle(seed=0, buffer_size=20000)
want = {"multi": N // 3, "tool": N // 3, "single": N - 2 * (N // 3)}
got = {k: 0 for k in want}
with open(out, "w") as f:
    for i, row in enumerate(ds):
        roles = [m["role"] for m in row["messages"]]
        k = (
            "tool"
            if any(m.get("function_calls") or m.get("functions") for m in row["messages"])
            or "environment" in roles
            or "tool" in roles
            else "multi"
            if roles.count("assistant") > 1
            else "single"
        )
        if got[k] >= want[k]:
            if all(got[x] >= want[x] for x in want) or i > 400000:
                break
            continue
        try:
            r = dt.sft_tulu_tokenize_and_truncate_v1({"messages": row["messages"]}, tok, max_seq_length=65536)
            ids, lab, dropped = r["input_ids"].tolist(), r["labels"].tolist(), False
        except Exception as e:  # open-instruct drops conversations whose spans it cannot derive
            ids, lab, dropped = [], [], type(e).__name__
        f.write(
            json.dumps(
                {
                    "i": row.get("id", i),
                    "stratum": k,
                    "messages": row["messages"],
                    "input_ids": ids,
                    "labels": lab,
                    "dropped": dropped,
                }
            )
            + "\n"
        )
        got[k] += 1
print("scanned", i + 1, "kept", got)
