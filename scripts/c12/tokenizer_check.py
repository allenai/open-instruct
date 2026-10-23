"""C12 (ledger H125) tokenizer asserts PA and PB.

  python scripts/c12/tokenizer_check.py config                 the tokenizer the cache build loads, before and after
                                                               --reserved_slot_tokens promotion
  python scripts/c12/tokenizer_check.py dir <path>...          a saved C12 tokenizer dir (numpy cache tokenizer/, HF
                                                               export, eval model dir, RL hf_checkpoint): promoted
  python scripts/c12/tokenizer_check.py control-dir <path>...  a saved C11-control tokenizer dir: unpromoted

open-instruct #1896: under transformers 5 a tokenizer-only Dolma 2 repo can load with plain ByteLevel instead of the cl100k
Split regex. "\\n\\n" -> [271] does not detect it (a broken load still gives 271); the tag split does (an unpromoted
<think> becomes 27, 27963, 29 instead of 14023, 771, 29). #1911's promotion re-serializes the loaded backend, so a broken
load would bake ByteLevel into every saved tokenizer.json. So every mode asserts:
  - the loaded backend's pre_tokenizer equals PINNED_PRE_TOKENIZER (vendored from dolma2-tokenizer-olmo35@8b97170's
    Hub tokenizer.json, sha256 56a6699d...), and for dirs, that the saved tokenizer.json's pre_tokenizer equals it too;
  - config: before promotion <think> -> [14023, 771, 29], </think> -> [524, 27963, 29]; after, [100256] and [100266];
  - dir: [100256] and [100266]; control-dir: the unpromoted pieces;
  - "\\n\\n" -> [271], a weak extra only.
The build loads through open_instruct/tokenizer_utils.py load_tokenizer (the #1896 fix, cache version v8), called from
dataset_transformation.get_tokenizer_tulu_v2_2. One JSON line per tokenizer; exits non-zero on any mismatch.
"""

import json
import os
import sys

from transformers import AutoTokenizer

from open_instruct import tokenizer_utils
from open_instruct.dataset_transformation import TokenizerConfig

TOKENIZER = "allenai/dolma2-tokenizer-olmo35"
REVISION = "8b9717061fae09d5be814373d189919a62a9a00d"
PINNED_PRE_TOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {
            "type": "Split",
            "pattern": {
                "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
            },
            "behavior": "Removed",
            "invert": True,
        },
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
    ],
}
UNPROMOTED = {"<think>": [14023, 771, 29], "</think>": [524, 27963, 29], "\n\n": [271]}
PROMOTED = {"<think>": [100256], "</think>": [100266], "\n\n": [271]}


def check(name, tokenizer, expected, saved_json=None):
    got = {text: tokenizer.encode(text, add_special_tokens=False) for text in expected}
    loaded_pre = json.loads(tokenizer.backend_tokenizer.to_str()).get("pre_tokenizer")
    result = {
        "tokenizer": name,
        "encodings_ok": got == expected,
        "loaded_pre_tokenizer_pinned": loaded_pre == PINNED_PRE_TOKENIZER,
        "encodings": got,
        "vocab": len(tokenizer),
    }
    if saved_json is not None:
        with open(saved_json) as f:
            result["saved_pre_tokenizer_pinned"] = json.load(f).get("pre_tokenizer") == PINNED_PRE_TOKENIZER
    result["ok"] = all(v for k, v in result.items() if k.endswith(("_ok", "_pinned")))
    print(json.dumps(result), flush=True)
    return result["ok"]


def config_tokenizer(reserved):
    return TokenizerConfig(
        tokenizer_name_or_path=TOKENIZER,
        tokenizer_revision=REVISION,
        chat_template_name="olmo35",
        reserved_slot_tokens=["<think>", "</think>"] if reserved else None,
    ).tokenizer


def main(argv):
    if argv[:1] == ["config"]:
        raw = tokenizer_utils.load_tokenizer(TOKENIZER, revision=REVISION)
        ok = check(f"{TOKENIZER}@{REVISION} load_tokenizer", raw, UNPROMOTED)
        ok &= check(f"{TOKENIZER}@{REVISION} before promotion", config_tokenizer(False), UNPROMOTED)
        ok &= check(f"{TOKENIZER}@{REVISION} after promotion", config_tokenizer(True), PROMOTED)
        return ok
    if argv[:1] in (["dir"], ["control-dir"]) and len(argv) > 1:
        expected = PROMOTED if argv[0] == "dir" else UNPROMOTED
        return all(
            [
                check(path, AutoTokenizer.from_pretrained(path), expected, os.path.join(path, "tokenizer.json"))
                for path in argv[1:]
            ]
        )
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
