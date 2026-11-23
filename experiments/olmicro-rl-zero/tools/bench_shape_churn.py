"""Does the trainer recompile for every new packed-document layout? Cold vs warm timing on realistic packs.

Builds packs from real v2 response lengths (+ prompt tokens), filled to the trainer's token budget, then times
Core fwd+bwd for each pack the first time it is seen (cold) and again (warm), in one process with the given attention
backend. Counts FA4 compile warnings per pack (warnings forced to "always"; Python otherwise prints them once).
Setup mirrors MILES's trainer (fp32 params, bf16 autocast, full activation checkpointing). Single GPU.

Usage (MILES image, from /opt/core-rl): python bench_shape_churn.py CORE_CONFIG ROLLOUT_FLOW --backend flash_4 --out OUT.json
"""

import argparse
import json
import random
import time
import warnings
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("core_config", type=Path)
    parser.add_argument("rollout_flow", type=Path)
    parser.add_argument("--backend", default="flash_4")
    parser.add_argument("--budget", type=int, default=40960)
    parser.add_argument("--packs", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Seeds weights and token ids so arms are comparable")
    args = parser.parse_args()

    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode

    from miles.backends.core_utils import hybrid_models

    lengths = [n + args.prompt_tokens for line in args.rollout_flow.read_text().splitlines()
               for n in json.loads(line)["response_lengths"]]
    rng = random.Random(0)
    rng.shuffle(lengths)
    packs, current = [], []
    for n in lengths:
        n = min(n, args.budget)
        if sum(current) + n > args.budget:
            packs.append(current)
            current = []
            if len(packs) == args.packs:
                break
        current.append(n)

    document = json.loads(args.core_config.read_text())
    model_dict = document.get("model", document)
    hybrid_models._override_attention_backend(model_dict, args.backend)
    config = TransformerConfig.from_dict(model_dict)
    config.lm_head.loss_implementation = LMLossImplementation.default
    torch.manual_seed(args.seed)
    model = config.build(init_device="cuda")
    model.init_weights(device=torch.device("cuda"))
    model = model.float()
    model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)

    def step(doc_lens):
        ids = torch.randint(0, config.vocab_size, (1, sum(doc_lens)), device="cuda")
        lens = torch.tensor([doc_lens], device="cuda", dtype=torch.int32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, doc_lens=lens, max_doc_lens=[max(doc_lens)])
            loss = logits.float().logsumexp(-1).mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        compiles = sum("JitArgument" in str(w.message) or "compile" in str(w.message).lower() for w in caught)
        return seconds, compiles, loss.item()

    step([2048] * 4)  # load-time warm-up on a fixed shape, excluded
    results = {"backend": args.backend, "budget": args.budget, "packs": []}
    for phase in ("cold", "warm"):
        for i, doc_lens in enumerate(packs):
            seconds, compiles, loss = step(doc_lens)
            row = {"phase": phase, "pack": i, "tokens": sum(doc_lens), "docs": len(doc_lens), "max_doc": max(doc_lens),
                   "seconds": round(seconds, 2), "tok_per_s": round(sum(doc_lens) / seconds), "compile_warnings": compiles, "loss": loss}
            results["packs"].append(row)
            print("CHURN " + json.dumps(row), flush=True)
    args.out.write_text(json.dumps(results, indent=1))
    print("CHURN_DONE", flush=True)


if __name__ == "__main__":
    main()
