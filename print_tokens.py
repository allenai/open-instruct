# print_tokens.py
import argparse
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="HF repo or local path")
    parser.add_argument("--ids", nargs="+", type=int, required=False,
                        default=[151710, 151915, 151682, 151930, 151874])
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    print(f"tokenizer.vocab_size (base embeddings): {tok.vocab_size}")
    try:
        # len(tokenizer) includes added tokens
        print(f"len(tokenizer) (with added tokens): {len(tok)}")
    except Exception:
        pass

    added_vocab = set(tok.get_added_vocab().keys())
    for i in args.ids:
        token = None
        note = ""
        try:
            token = tok.convert_ids_to_tokens([i])[0]
        except Exception as e:
            note = f"(convert_ids_to_tokens error: {e})"

        status = []
        if i >= tok.vocab_size:
            status.append(">= vocab_size (likely added token id)")
        if token in added_vocab:
            status.append("token string is an added token")

        status_str = f" [{' ; '.join(status)}]" if status else ""
        print(f"id={i}: token={repr(token)}{status_str} {note}")

if __name__ == "__main__":
    main()