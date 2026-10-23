"""C12 (ledger H125) tokenizer asserts PA and PB.

  python scripts/c12/tokenizer_check.py config            the tokenizer the cache build will use (olmo35 + reserved slots)
  python scripts/c12/tokenizer_check.py dir <path>...     a saved tokenizer dir: numpy cache tokenizer/, HF export,
                                                          eval model dir or RL hf_checkpoint

Asserts encode("<think>") == [100256], encode("</think>") == [100266] (PA, #1911's promotion into reserved slots) and
encode("\\n\\n") == [271] (PB, the cl100k Split regex survives loading; open-instruct #1896). Prints one JSON line per
tokenizer and exits non-zero on any mismatch.
"""

import json
import sys

from transformers import AutoTokenizer

from open_instruct.dataset_transformation import TokenizerConfig

TOKENIZER = "allenai/dolma2-tokenizer-olmo35"
REVISION = "8b9717061fae09d5be814373d189919a62a9a00d"
EXPECTED = {"<think>": [100256], "</think>": [100266], "\n\n": [271]}


def check(name, tokenizer):
    got = {text: tokenizer.encode(text, add_special_tokens=False) for text in EXPECTED}
    ok = got == EXPECTED
    print(json.dumps({"tokenizer": name, "ok": ok, "encodings": got, "vocab": len(tokenizer)}), flush=True)
    return ok


def main(argv):
    if argv[:1] == ["config"]:
        tc = TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER, tokenizer_revision=REVISION, reserved_slot_tokens=["<think>", "</think>"]
        )
        return check(f"{TOKENIZER}@{REVISION}+reserved_slots", tc.tokenizer)
    if argv[:1] == ["dir"] and len(argv) > 1:
        return all([check(path, AutoTokenizer.from_pretrained(path)) for path in argv[1:]])
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
