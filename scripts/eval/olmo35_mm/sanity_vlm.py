"""Image+question sanity generation through olmo-eval's olmo_core_vlm provider (the provider the vision suites use).

  sanity_vlm.py IMAGE TOKENIZER EXPORT_DIR [EXPORT_DIR ...]

For each OLMo-core multimodal export (model.safetensors + olmo_core_config.json) prints greedy answers to a few
questions about IMAGE with prompt_format=olmo35, thinking=generate (the vision suites' settings), and the
greedy answers on a text-only prompt. Pass a reference export (e.g. the GPU run's, at the same step) as a second dir for a side-by-side. Prints
SANITY-VLM-DONE, or SANITY-VLM-EMPTY-OUTPUT if any answer is empty.

From /root/mmsft-eval/tpu/sanity_vlm.py (H271; ledger copy olmo-post-training-ledger experiments/mmsft/tpu/).
Runs in sanity_job.sh's environment: OLMo-core (#942) and an olmo-eval checkout with the olmo_core_vlm provider.
"""

import gc
import sys
import time

import torch
from olmo_eval.common.types.base import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.olmo_core_vlm.provider import OlmoCoreVLMProvider
from PIL import Image

QUESTIONS = [
    "Describe this image in one sentence.",
    "What utensil is lying on the table to the left of the plate?",
    "What color is the plate?",
]


def main():
    image_path, tok, *dirs = sys.argv[1:]
    image = Image.open(image_path).convert("RGB")
    ok = True
    for d in dirs:
        t0 = time.time()
        prov = OlmoCoreVLMProvider(
            d, tokenizer=tok, prompt_format="olmo35", thinking="generate", device="cuda", batch_size=4
        )
        reqs = [
            LMRequest(request_type=RequestType.CHAT, messages=({"role": "user", "content": q},), images=(image,))
            for q in QUESTIONS
        ]
        reqs.append(
            LMRequest(request_type=RequestType.CHAT, messages=({"role": "user", "content": "What is 17 * 23?"},))
        )
        outs = prov.generate(reqs, SamplingParams(max_tokens=160, temperature=0.0))
        print(f"== {d} (load+generate {time.time() - t0:.0f}s)", flush=True)
        for q, o in zip(QUESTIONS + ["(text) What is 17 * 23?"], outs):
            text = o[0].text
            print(f"SANITY Q: {q}\nSANITY A: {text!r}\n", flush=True)
            ok &= bool(text.strip())
        prov.close()
        del prov
        gc.collect()
        torch.cuda.empty_cache()
    print("SANITY-VLM-DONE" if ok else "SANITY-VLM-EMPTY-OUTPUT", flush=True)


if __name__ == "__main__":
    main()
