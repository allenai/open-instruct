"""Source HF model reference CE on the exact packed TPU first batch, split by document."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributed import checkpoint as dcp
from transformers import AutoModelForCausalLM, AutoTokenizer

SOURCE = Path("/weka/olmo-3p5-checkpoints/protected/olmo35-14t600b-mt100b-lc-sft-20261014/lc/step5961/olmo-core")
HF = "/weka/olmo-3p5-checkpoints/scratch/olmo35-14t600b-mt100b-lc-sft-20261014/olmo35-14t600b-mt100b-lc-sft-20261014-lc/emo/step5961/hf"
OUT = Path("/output")
ROWS = [(100256, [14023, 771, 29]), (100270, [524, 27963, 29])]


def main():
    OUT.mkdir(exist_ok=True)
    reader = dcp.FileSystemReader(str(SOURCE / "model_and_optim"))
    original_read = reader.read_metadata

    def read_metadata(*args, **kwargs):
        metadata = original_read(*args, **kwargs)
        for value in metadata.storage_data.values():
            if not hasattr(value, "transform_descriptors"):
                object.__setattr__(value, "transform_descriptors", None)
        return metadata

    reader.read_metadata = read_metadata
    metadata = reader.read_metadata()
    keys = ["module.embeddings.weight.main", "module.lm_head.w_out.weight.main"]
    state = {key: torch.empty(tuple(metadata.state_dict_metadata[key].size), dtype=torch.float32) for key in keys}
    dcp.load(state, storage_reader=reader)
    matrices = [state[key].reshape(100352, 1024) for key in keys]
    model = AutoModelForCausalLM.from_pretrained(
        HF, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    matrix_refs = [model.get_input_embeddings().weight, model.get_output_embeddings().weight]
    for expected, actual in zip(matrices, matrix_refs, strict=True):
        assert torch.equal(expected[: actual.shape[0]].to(torch.bfloat16), actual.detach()), (
            "HF embedding/head differs from native checkpoint"
        )
    model.resize_token_embeddings(100352, mean_resizing=False)
    matrix_refs = [model.get_input_embeddings().weight, model.get_output_embeddings().weight]
    initial_rows = {}
    with torch.no_grad():
        for key, master, target in zip(keys, matrices, matrix_refs, strict=True):
            bf16 = master.to(torch.bfloat16)
            for row, pieces in ROWS:
                bf16[row] = bf16[pieces].float().mean(0).to(torch.bfloat16)
            target.copy_(bf16)
            initial_rows[key] = hashlib.sha256(bf16[[row for row, _ in ROWS]].float().numpy().tobytes()).hexdigest()
    del state, matrices, matrix_refs
    model = model.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained("/probe/tokenizer")
    receipt = {"source": HF, "initial_rows": initial_rows, "rows": [], "opener_probes": []}
    total, total_tokens = 0.0, 0
    started = time.monotonic()
    with torch.inference_mode():
        for host in range(8):
            batch = np.load(f"/probe/replay/h{host}.step1.npz")
            for row in range(batch["inputs"].shape[0]):
                segments = batch["inputs_segmentation"][row]
                loss_sum, count = 0.0, 0
                for segment in sorted(set(segments[segments > 0].tolist())):
                    idx = np.flatnonzero(segments == segment)
                    positions = batch["inputs_position"][row][idx]
                    assert positions[0] == 0 and (np.diff(positions) == 1).all()
                    ids = torch.tensor(batch["inputs"][row][idx].astype(np.int64), device="cuda")[None]
                    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
                    targets = torch.tensor(batch["targets"][row][idx].astype(np.int64), device="cuda")
                    mask = torch.tensor(batch["targets_segmentation"][row][idx] > 0, device="cuda")
                    for offset in range(0, len(idx), 1024):
                        logits = model.lm_head(hidden[offset : offset + 1024]).float()
                        nll = torch.nn.functional.cross_entropy(
                            logits, targets[offset : offset + 1024], reduction="none"
                        )
                        loss_sum += float((nll * mask[offset : offset + 1024]).sum())
                        count += int(mask[offset : offset + 1024].sum())
                total += loss_sum
                total_tokens += count
                receipt["rows"].append({"host": host, "row": row, "loss_tokens": count, "ce": loss_sum / count})
                print("REF-ROW", host, row, count, loss_sum / count, "seconds", time.monotonic() - started, flush=True)
        for line in Path("/probe/prompts.jsonl").read_text().splitlines()[:32]:
            record = json.loads(line)
            messages = record["messages"]
            end = next(i for i, message in enumerate(messages) if message["role"] == "assistant")
            text = (
                tokenizer.apply_chat_template(messages[:end], tokenize=False, add_generation_prompt=True) + "<think>"
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert ids[-1] == 100256
            hidden = model.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).last_hidden_state[
                :, -1
            ]
            probabilities = model.lm_head(hidden).float().softmax(-1)[0]
            receipt["opener_probes"].append(
                {
                    "id": record["id"],
                    "input_ids": ids,
                    "probabilities": {str(token): float(probabilities[token]) for token in [198, 271, 33413, 36539]},
                }
            )
    receipt.update(ce=total / total_tokens, loss_tokens=total_tokens, seconds=time.monotonic() - started)
    (OUT / "reference-ce.json").write_text(json.dumps(receipt, indent=2))
    print("REFERENCE-CE", receipt["ce"], "tokens", total_tokens, flush=True)


if __name__ == "__main__":
    main()
