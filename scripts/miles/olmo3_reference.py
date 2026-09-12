"""Generate a tiny Olmo 3 oracle with the release-era Transformers, independently of Core.

Run this in Transformers 4.57.0, then run olmo3_qualification in the MILES runtime.
Only synthetic weights are produced. This is not a released-checkpoint accuracy test.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Olmo3Config
from transformers.models.olmo3 import modeling_olmo3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", help="Optional local tokenizer to attach for Ray/SGLang smoke generation")
    args = parser.parse_args()
    if transformers.__version__ != "4.57.0":
        raise ValueError("Generate the reference in Transformers 4.57.0, the published Think-DPO config's version")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(17)
    torch.set_num_threads(2)
    config = Olmo3Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        sliding_window=16,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        rope_theta=500000,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 32,
            "beta_fast": 32,
            "beta_slow": 1,
            "attention_factor": 1.2079441541679836,
        },
        pad_token_id=0,
        eos_token_id=2,
        bos_token_id=1,
    )
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
    model.save_pretrained(args.output / "hf")
    if args.tokenizer:
        AutoTokenizer.from_pretrained(args.tokenizer).save_pretrained(args.output / "hf")
    records = []
    for length in (1, 15, 16, 17, 31, 32, 33, 64, 128):
        ids = (torch.arange(length).unsqueeze(0) % 250 + 4).long()
        with torch.no_grad():
            logits = model(ids, use_cache=False).logits.float()
        records.append({"tokens": ids, "logits": logits})
    torch.save(records, args.output / "reference.pt")
    source = Path(modeling_olmo3.__file__)
    manifest = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "seed": 17,
        "reference_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "semantics": "YaRN on full-attention layers; unscaled RoPE on sliding-attention layers",
        "lengths": [record["tokens"].shape[1] for record in records],
        "config": config.to_dict(),
    }
    (args.output / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"reference": str(args.output), "lengths": manifest["lengths"]}))


if __name__ == "__main__":
    main()
