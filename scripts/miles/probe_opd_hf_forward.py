"""OI's HF backbone/head forward reference, without distributed ZeRO execution.

Preserve Kevin's packed GDN patch, FA4, BF16 backbone and FP32 head arithmetic.
This is a forward reference, not a reproduction of the distributed optimizer.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from open_instruct import grpo_utils, model_utils, qwen3_5_packing_patch


def run(options):
    options.output.mkdir(parents=True, exist_ok=False)
    qwen3_5_packing_patch.patch_qwen3_5_packing()
    model = AutoModelForCausalLM.from_pretrained(
        options.checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_4",
        device_map={"": "cuda:0"},
        local_files_only=True,
    )
    model.config.use_cache = False
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    _, head = grpo_utils.get_causal_lm_backbone_and_lm_head(model)
    panel = json.loads(options.panel.read_text())
    (options.output / "provenance.json").write_text(
        json.dumps(
            dict(
                torch=torch.__version__,
                panel_sha256=hashlib.sha256(options.panel.read_bytes()).hexdigest(),
                attention="flash_attention_4",
                head="FP32 inputs and projection; retain BF16 shared embedding storage",
                scope="HF forward reference; not distributed DeepSpeed execution",
                head_weight_dtype=str(head.weight.dtype),
            )
        )
    )
    with torch.no_grad():
        for repeat in range(2):
            for row in panel:
                ids = torch.tensor([row["tokens"]], device="cuda", dtype=torch.long)
                positions = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
                hidden = grpo_utils.forward_for_liger_hidden_states(model, ids, None, positions)
                flat = hidden.reshape(-1, hidden.shape[-1]).float()
                labels = ids[:, 1:].reshape(-1)
                assert len(flat) == len(labels)
                values = []
                for h, t in zip(torch.chunk(flat, 8), torch.chunk(labels, 8), strict=True):
                    logits = F.linear(h, head.weight.float(), head.bias.float() if head.bias is not None else None)
                    values.append(model_utils.log_softmax_and_gather(logits, t))
                lp = torch.cat(values)[-row["response_length"] :].float().cpu().tolist()
                with (options.output / "scores.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(dict(id=row["id"], wave=f"hf-r{repeat}", logprobs=lp), allow_nan=False) + "\n"
                    )
    (options.output / "complete.json").write_text('{"complete": true}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
