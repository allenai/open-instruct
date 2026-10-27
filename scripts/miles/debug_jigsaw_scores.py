"""Rescore a trusted saved image rollout without taking an optimizer step.

Run each mode in a fresh GPU process in the pinned pilot image. The Core mode
uses the MILES model builder/configuration and score extraction, but omits the
optimizer and distributed trainer buffers so the probe fits one H100.
"""

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import olmo_sglang
import sglang
import torch
from miles.backends.core_utils import models, vision_models
from miles.backends.training_utils import loss, parallel
from miles.utils.ft_utils.process_group_utils import GroupInfo
from olmo_core.nn.moe.v2 import olmo3
from olmo_sglang.vision import model as vision_model
from olmo_sglang.vision import processing
from PIL import Image
from torch import distributed as dist
from transformers import AutoConfig

from open_instruct.miles.configuration.config import CoreConfig


def write(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def core(args, samples, checkpoint):
    dist.init_process_group("nccl", init_method="file:///tmp/jigsaw-score-rdzv", rank=0, world_size=1)
    trivial = GroupInfo(rank=0, size=1, group=None)
    parallel.set_parallel_state(
        parallel.ParallelState(
            **{k: trivial for k in ("intra_dp", "intra_dp_cp", "cp", "tp", "pp", "ep", "etp", "indep_dp")}
        )
    )
    options = CoreConfig(
        row_specialization="dynamic",
        router_aux_loss_weight=0.0,
        router_z_loss_weight=0.0,
        max_sequence_length=2048,
        compile_optimizer=True,
    )
    hf = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    cfg = models.model_config_from_hf(hf, options)
    cfg.init_seed = 17
    vision_models.prepare_model_config(cfg, hf, options)
    model = cfg.build(init_device="meta")
    model.init_weights(max_seq_len=2048, max_local_microbatch_size=2048, device=torch.device("cuda"))
    state = vision_models.read_hf_state(checkpoint)
    olmo3.load_olmo3_moe_hf_state(
        model,
        hf.text_config,
        {k.removeprefix("language_model."): v for k, v in state.items() if k.startswith("language_model.")},
    )
    visual = vision_model.OlmoVisionStack(hf, device="cuda", dtype=torch.bfloat16)
    visual.load_weights((k, v) for k, v in state.items() if not k.startswith("language_model."))
    del state
    gc.collect()
    model.eval().requires_grad_(False)
    visual.eval().requires_grad_(False)
    module = SimpleNamespace(model=model, _miles_vision=visual, _miles_image_patch_token_id=hf.image_patch_token_id)
    score_args = SimpleNamespace(
        qkv_format="thd",
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        allgather_cp=False,
        log_probs_chunk_size=-1,
    )
    processor = processing.OlmoVisionProcessor.from_pretrained(checkpoint)
    report = {
        "mode": "core",
        "model_config": cfg.as_config_dict(),
        "records": [],
        "hardware": torch.cuda.get_device_name(),
    }
    write(args.output, report)
    with torch.no_grad():
        for index, sample in enumerate(samples):
            tokens = torch.tensor(sample["tokens"], device="cuda", dtype=torch.long)
            n = sample["response_length"]
            batch = {
                "tokens": tokens[None],
                "response_lengths": [n],
                "multimodal_train_inputs": [{k: v.to("cuda") for k, v in sample["multimodal_train_inputs"].items()}],
            }
            inputs = vision_models.conditioning_inputs(module, batch)
            # Serving encodes images separately; test whether grouping the crops
            # changes the visual features or the resulting response scores.
            images = [Image.open(item["path"]).convert("RGB") for item in sample["metadata"]["vision"]["images"]]
            prompt, prepared = processor.prepare(
                sample["metadata"]["vision"]["prepared"]["placeholder_token_ids"], images
            )
            assert prompt == sample["tokens"][:-n]
            features = torch.cat(
                [visual.encode_images(p.patches.to("cuda", torch.bfloat16), p.pooling.to("cuda")) for p in prepared],
                dim=1,
            )
            positions = batch["multimodal_train_inputs"][0]["image_positions"]
            separate = model.forward_embed(tokens[None]).index_add(1, positions, features)
            embedding_error = (inputs["input_embeddings"].float() - separate.float()).abs()
            row = {
                "index": index,
                "id": sample["metadata"]["prepared_sample_id"],
                "response": sample["response"],
                "response_token_ids": sample["tokens"][-n:],
                "token_text": [
                    processor.tokenizer.decode([x], skip_special_tokens=False) for x in sample["tokens"][-n:]
                ],
                "rollout_log_probs": sample["rollout_log_probs"],
                "separate_image_embedding_max_abs": float(embedding_error.max()),
                "separate_image_embedding_mean_abs": float(embedding_error.mean()),
            }
            for variant, embeddings in (
                ("combined_images", inputs["input_embeddings"]),
                ("separate_images", separate),
            ):
                started = time.perf_counter()
                logits = model(tokens[None], input_embeddings=embeddings, loss_div_factor=tokens.numel())
                scores = loss.get_log_probs_and_entropy(
                    logits,
                    args=score_args,
                    unconcat_tokens=[tokens],
                    total_lengths=[len(tokens)],
                    response_lengths=[n],
                    max_seq_lens=[len(tokens)],
                )["log_probs"][0].float()
                direct = (
                    logits[0, len(tokens) - n - 1 : len(tokens) - 1]
                    .float()
                    .log_softmax(-1)
                    .gather(1, tokens[-n:, None])
                    .flatten()
                )
                error = scores.cpu() - torch.tensor(sample["rollout_log_probs"])
                row[variant] = {
                    "log_probs": scores.tolist(),
                    "trainer_minus_rollout": error.tolist(),
                    "mean_abs": float(error.abs().mean()),
                    "max_abs": float(error.abs().max()),
                    "direct_score_max_abs": float((scores - direct).abs().max()),
                    "seconds": time.perf_counter() - started,
                }
                del logits
            report["records"].append(row)
            write(args.output, report)
            print(json.dumps(row), flush=True)
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["complete"] = True
    write(args.output, report)
    dist.destroy_process_group()


def serving(args, samples, checkpoint):
    olmo_sglang.register()
    engine = sglang.Engine(
        model_path=str(checkpoint),
        trust_remote_code=True,
        enable_multimodal=True,
        context_length=2048,
        max_total_tokens=32768,
        max_running_requests=8,
        max_mamba_cache_size=128,
        mem_fraction_static=0.65,
        disable_radix_cache=True,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        chunked_prefill_size=16384,
        attention_backend=args.attention_backend,
        sampling_backend="pytorch",
        random_seed=17,
        skip_server_warmup=True,
    )
    report = {"mode": "serving", "attention_backend": args.attention_backend, "records": []}
    try:
        for index, sample in enumerate(samples):
            n = sample["response_length"]
            response_ids = sample["tokens"][-n:]
            prompt = sample["metadata"]["vision"]["prepared"]["placeholder_token_ids"]
            images = [item["path"] for item in sample["metadata"]["vision"]["images"]]
            result = engine.generate(
                input_ids=prompt + response_ids,
                image_data=images,
                sampling_params={"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 1},
                return_logprob=True,
                logprob_start_len=len(sample["tokens"]) - n - 1,
            )
            values = result["meta_info"]["input_token_logprobs"][-n:]
            assert [v[1] for v in values] == response_ids, values
            row = {
                "index": index,
                "response_token_ids": response_ids,
                "prefill_log_probs": [v[0] for v in values],
                "rollout_log_probs": sample["rollout_log_probs"],
                "result": result,
            }
            report["records"].append(row)
            write(args.output, report)
            print(json.dumps(row), flush=True)
    finally:
        engine.shutdown()
    report["complete"] = True
    write(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["core", "serving"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", default="torch_native")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = torch.load(args.root / "rollouts/0.pt", map_location="cpu", weights_only=False, mmap=True)["samples"]
    {"core": core, "serving": serving}[args.mode](args, samples, args.root / "prepared/hf")


if __name__ == "__main__":
    main()
