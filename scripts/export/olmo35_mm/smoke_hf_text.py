"""Smoke test of a multimodal SFT checkpoint's text-only HF export (``hf_text``) before any text eval runs on it.

    python smoke_hf_text.py HF_TEXT_DIR [--device cuda]

Checks, in this order, and prints ``TOKENIZER-CHECK OK`` when all pass:
  * generation_config.json stops at <|im_end|> and <|endoftext|> (eos_token_id [100265, 100257]); without 100265
    vLLM runs past the end of every chat turn (H271: BFCL tool calls ran to the token budget);
  * ``<think>`` / ``</think>`` are the single tokens 100256 / 100266 of the promoted Olmo 3.5 tokenizer, and the
    chat template's generation prompt ends with ``<think>``, as in the SFT data;
  * two greedy generations, with finite logits.

From the heredoc of /root/mmsft-eval/export/text_export_job.sh (H271's export job; ledger copy at
olmo-post-training-ledger experiments/mmsft/tpu/export/text_export_job.sh), which ran it on every export.
"""

import argparse
import json
from pathlib import Path

import torch
import transformers

CHAT_EOS = [100265, 100257]  # <|im_end|>, <|endoftext|>
THINK, END_THINK = 100256, 100266


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hf_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    eos = json.loads((args.hf_dir / "generation_config.json").read_text())["eos_token_id"]
    assert eos == CHAT_EOS, f"generation_config eos_token_id {eos} != {CHAT_EOS}"
    # PreTrainedTokenizerFast: AutoTokenizer builds GPT2Tokenizer, whose pre-tokenizer differs (OLMo-core #944).
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(args.hf_dir)
    assert tokenizer.encode("<think>", add_special_tokens=False) == [THINK]
    assert tokenizer.encode("</think>", add_special_tokens=False) == [END_THINK]
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.hf_dir, dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.to(args.device).eval()
    for question in ("What is 17 * 23?", "Write one sentence about the ocean."):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], add_generation_prompt=True, tokenize=False
        )
        ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False)], device=args.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=48, do_sample=False)
            logits = model(ids).logits
        print("PROMPT", repr(text[-80:]))
        print(
            "SMOKE",
            repr(question),
            "finite",
            bool(torch.isfinite(logits).all()),
            repr(tokenizer.decode(out[0, ids.shape[1] :])),
        )
        assert int(ids[0, -1]) == THINK, "the chat prompt must end with <think> (100256)"
        assert bool(torch.isfinite(logits).all())
    print(f"TOKENIZER-CHECK OK: <think>={THINK} </think>={END_THINK}, prompts end with <think>, eos {CHAT_EOS}")


if __name__ == "__main__":
    main()
