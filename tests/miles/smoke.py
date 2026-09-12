import argparse
import asyncio
import json
import sys
from pathlib import Path

import ray
import torch
from miles.utils import arguments
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config

from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.driver import train

parser = argparse.ArgumentParser(description="Synthetic Ray/SGLang/Core smoke fixture; not a learning evaluation.")
parser.add_argument("output", type=Path)
parser.add_argument("--resume", action="store_true")
options = parser.parse_args()
root = options.output
root.mkdir(parents=True, exist_ok=True)
hf = root / "hf"
if not hf.exists():
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    AutoModelForCausalLM.from_config(config).to(torch.bfloat16).save_pretrained(hf)
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{f"t{i}": i for i in range(4, 256)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>"
    ).save_pretrained(hf)
(root / "prompts.jsonl").write_text("".join(json.dumps({"input": "t4 t5 t6", "label": "1"}) + "\n" for _ in range(8)))
config = RunConfig(
    CoreConfig(attention_backend="torch", max_sequence_length=128, activation_checkpointing=False),
    dict(
        hf_checkpoint=str(hf),
        global_batch_size=4,
        rollout_batch_size=1,
        n_samples_per_prompt=4,
        num_rollout=2,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=1,
        colocate=True,
        offload_rollout=False,
        prompt_data=str(root / "prompts.jsonl"),
        input_key="input",
        label_key="label",
        rollout_max_response_len=8,
        sglang_mem_fraction_static=0.2,
        sglang_disable_cuda_graph=True,
        sglang_attention_backend="torch_native",
        sglang_sampling_backend="pytorch",
        custom_rm_path="smoke_reward.score",
        save=str(root / "save"),
        save_interval=1,
        sglang_log_level="warning",
    ),
)
if options.resume:
    config.miles["load"] = str(root / "save")
    config.miles["num_rollout"] = 3
    config.miles["start_rollout_id"] = 2
sys.argv = ["smoke", *config.arguments()]
args = arguments.parse_args()
ray.init(num_gpus=1, num_cpus=6, include_dashboard=False, object_store_memory=256 * 1024 * 1024)
try:
    asyncio.run(train(args))
finally:
    ray.shutdown()
assert (root / "save/core-latest.json").exists()
manifest = json.loads((root / "save/core-latest.json").read_text())
assert manifest["rollout_id"] == (2 if options.resume else 1)
complete = json.loads((root / "save/core" / f"rollout_{manifest['rollout_id']:07d}" / "complete.json").read_text())
assert complete["clock"]["completed_steps"] == (3 if options.resume else 2)
# One optimizer step per collection and zero KL: the standalone scoring pass runs only
# on each process's first update (a check), and the resumed process checks again.
records = [json.loads(line) for line in (root / "save/training_contract_rank0.jsonl").read_text().splitlines()]
modes = [record["scoring_pass"] for record in records if record["event"] == "optimizer"]
sources = [record["source"] for record in records if record["event"] == "scores"]
checks = [record for record in records if record["event"] == "scoring_check"]
expected_modes = ["checked", "skipped"] + (["checked"] if options.resume else [])
assert modes == expected_modes, modes
assert sources == ["standalone", "training_forward"] + (["standalone"] if options.resume else []), sources
assert [check["step"] for check in checks] == [0] + ([2] if options.resume else []), checks
assert all(check["mean_abs"] <= args.olmo_core.scoring_check_tolerance for check in checks), checks
print("MILES_CORE_E2E_PASSED", json.dumps({"scoring_pass": modes, "scoring_checks": checks}))
