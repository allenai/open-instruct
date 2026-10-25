"""Forward-only Core scores for frozen benchmark trajectories; no trainer or optimizer."""

import hashlib
import json
import time
from pathlib import Path

import torch
from miles.backends.core_utils import moe_models
from olmo_core.nn.moe.v2 import olmo3
from safetensors import torch as safetensors_torch
from transformers import AutoConfig

from open_instruct import logger_utils
from open_instruct.miles.execution import workflow
from open_instruct.miles.rollout import benchmark_metrics

logger = logger_utils.setup_logger(__name__)


def run(args, settings, root):
    started = time.monotonic()
    current = json.loads((root / "run-spec.json").read_text())
    raw, records = benchmark_metrics.reference_records(
        settings["replay_reference"], current["model"], args.rollout_max_context_len
    )
    checkpoint = Path(args.hf_checkpoint)
    hf = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    config = moe_models.model_config_from_hf(hf, args.olmo_core)
    moe_models.prepare_model_config(config, hf, args.olmo_core)
    config.recompute_each_block = False
    model = config.build(init_device="meta")
    model.init_weights(
        max_seq_len=args.rollout_max_context_len,
        max_local_microbatch_size=args.rollout_max_context_len,
        device=torch.device("cuda"),
    )
    state = {}
    for shard in sorted(checkpoint.glob("*.safetensors")):
        part = safetensors_torch.load_file(shard)
        if state.keys() & part.keys():
            raise ValueError("Duplicate checkpoint tensor in Core reference")
        state.update(part)
    if not state:
        raise ValueError("Core reference checkpoint has no weights")
    olmo3.load_olmo3_moe_hf_state(model, hf, state)
    del state, part
    model.eval()
    head = model.lm_head
    model.lm_head = None
    startup_seconds = time.monotonic() - started
    builder = Path(olmo3.__file__)
    report = {
        "diagnostic_only": True,
        "reference_backend": "core",
        "reference": settings["replay_reference"],
        "reference_sha256": hashlib.sha256(raw).hexdigest(),
        "trajectories": len(records),
        "response_tokens": sum(r["response_tokens"] for r in records),
        "attention_backend": args.olmo_core.attention_backend,
        "core_builder_sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
        "startup_seconds": startup_seconds,
        "scope": "BF16 forward-only Core with MILES model preparation and configured attention; natural routing, one causal document per request, EP1. No optimizer, publication, packed training or learning-quality claim.",
    }
    workflow.write_json(root / "benchmark-model-config.json", json.loads((checkpoint / "config.json").read_text()))
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
            hidden = model(torch.tensor([ids], device="cuda"))
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
                )
                + "\n"
            )
            del hidden
            logger.info("Core reference scored sample=%s tokens=%s", record["index"], len(probabilities))
    report["scoring_seconds"] = time.monotonic() - started - startup_seconds
    return report
