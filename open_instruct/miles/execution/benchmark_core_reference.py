"""Forward-only Core scores for frozen benchmark trajectories; no trainer or optimizer."""

import contextlib
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from miles.backends.core_utils import moe_models
from olmo_core.nn.moe.v2 import olmo3
from safetensors import torch as safetensors_torch
from transformers import AutoConfig

from open_instruct import logger_utils
from open_instruct.miles.execution import workflow
from open_instruct.miles.rollout import benchmark_metrics, benchmark_route_reference

logger = logger_utils.setup_logger(__name__)


def run(args, settings, root):
    started = time.monotonic()
    current = json.loads((root / "run-spec.json").read_text())
    raw, records = benchmark_metrics.reference_records(
        settings["replay_reference"], current["model"], args.rollout_max_context_len
    )
    reference_sha256 = hashlib.sha256(raw).hexdigest()
    route_reference = None
    if settings["score_route_reference"]:
        route_reference = benchmark_route_reference.reference_routes(
            settings["score_route_reference"],
            reference_sha256,
            records,
            settings["score_route_repeat"],
            current["model"],
        )
    checkpoint = Path(args.hf_checkpoint)
    hf = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    model_config = json.loads((checkpoint / "config.json").read_text())
    config = moe_models.model_config_from_hf(hf, args.olmo_core)
    moe_models.prepare_model_config(config, hf, args.olmo_core)
    config.recompute_each_block = False
    model = config.build(init_device="meta")
    logger.info("Core reference initializing model parameters")
    model.init_weights(
        max_seq_len=args.rollout_max_context_len,
        max_local_microbatch_size=args.rollout_max_context_len,
        device=torch.device("cuda"),
    )
    state = {}
    shards = sorted(checkpoint.glob("*.safetensors"))
    logger.info("Core reference loading checkpoint shards=%s", len(shards))
    for shard in shards:
        part = safetensors_torch.load_file(shard)
        if state.keys() & part.keys():
            raise ValueError("Duplicate checkpoint tensor in Core reference")
        state.update(part)
    if not state:
        raise ValueError("Core reference checkpoint has no weights")
    logger.info("Core reference installing checkpoint tensors=%s", len(state))
    olmo3.load_olmo3_moe_hf_state(model, hf, state)
    del state, part
    model.eval()
    head = model.lm_head
    model.lm_head = None
    startup_seconds = time.monotonic() - started
    logger.info("Core reference ready; startup_seconds=%.3f trajectories=%s", startup_seconds, len(records))
    builder = Path(olmo3.__file__)
    report = {
        "diagnostic_only": True,
        "reference_backend": "core",
        "reference": settings["replay_reference"],
        "reference_sha256": reference_sha256,
        "routing": "serving_replay" if route_reference else "natural",
        "route_reference": route_reference[2] if route_reference else None,
        "trajectories": len(records),
        "response_tokens": sum(r["response_tokens"] for r in records),
        "attention_backend": args.olmo_core.attention_backend,
        "core_builder_sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
        "startup_seconds": startup_seconds,
        "scope": "BF16 forward-only Core with MILES model preparation and configured attention; routing mode recorded explicitly, one causal document per request, EP1. No optimizer, publication, packed training or learning-quality claim.",
    }
    workflow.write_json(root / "benchmark-model-config.json", model_config)
    workflow.write_json(
        root / "benchmark-core-provenance.json",
        {
            **report,
            "runtime_lock": json.loads(
                (Path(__file__).resolve().parents[3] / "runtime/miles/runtime.lock.json").read_text()
            ),
        },
    )
    with torch.no_grad(), (root / "benchmark-core-reference.jsonl").open("w", buffering=1) as stream:
        for record in records:
            ids = record["tokens"]
            input_ids = torch.tensor([ids], device="cuda")
            context = contextlib.nullcontext()
            context_started = None
            context_setup_seconds = None
            if route_reference:
                route_root, route_rows, _ = route_reference
                matrix = benchmark_route_reference.load_routes(
                    route_root, route_rows[record["index"]], record, model_config
                )
                batch = {
                    "tokens": input_ids,
                    "total_lengths": [len(ids)],
                    "rollout_routed_experts": [torch.from_numpy(matrix).to(device="cuda")],
                }
                context_started = time.monotonic()
                context = moe_models.replay_context(SimpleNamespace(model=model), batch)
            with context:
                if context_started is not None:
                    context_setup_seconds = time.monotonic() - context_started
                hidden = model(input_ids)
            probability_chunks = []
            for offset in range(record["prompt_tokens"] - 1, len(ids) - 1, 128):
                logits = head(hidden[:, offset : min(offset + 128, len(ids) - 1)])[0].float()
                tokens = torch.tensor(ids[offset + 1 : offset + 1 + len(logits)], device="cuda")
                values = logits.gather(-1, tokens[:, None]).squeeze(-1) - logits.logsumexp(-1)
                probability_chunks.append(values)
            probabilities_tensor = torch.cat(probability_chunks)
            if not torch.isfinite(probabilities_tensor).all():
                raise ValueError("Nonfinite Core reference probabilities")
            probabilities = probabilities_tensor.cpu().tolist()
            if len(probabilities) != record["response_tokens"]:
                raise ValueError("Core reference token/logprob length mismatch")
            stream.write(
                json.dumps(
                    {key: record[key] for key in ("index", "group_index", "prompt_tokens", "response_tokens")}
                    | {"logprobs": probabilities}
                    | (
                        {
                            "route_source_sha256": route_reference[1][record["index"]]["route_sha256"],
                            "route_context_setup_host_seconds": context_setup_seconds,
                        }
                        if route_reference
                        else {}
                    )
                )
                + "\n"
            )
            del hidden
            logger.info("Core reference scored sample=%s tokens=%s", record["index"], len(probabilities))
    report["scoring_seconds"] = time.monotonic() - started - startup_seconds
    return report
