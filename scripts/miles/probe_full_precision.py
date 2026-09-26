"""Score fixed serving trajectories through unchanged BF16/FP32 Core models.

Use with the separately audited eager serving reference. No backward, optimizer,
TF32, or checkpoint writes. The weights are loaded from the same source values.
"""

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import torch
from olmo_core import config as core_config
from olmo_core.nn import attention
from olmo_core.nn.moe.v2 import olmo3
from safetensors.torch import load_file
from scripts.miles import benchmark_core_compat as benchmark
from scripts.miles import fp32_reference
from scripts.miles import probe_lm_head_precision as head_probe
from transformers import AutoConfig

from open_instruct.miles.training import fla_compat


def score(args):
    fla_compat.install_kda_triton_compat()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.dtype == "float32":
        fp32_reference.strict_arithmetic()
    samples = json.loads(args.samples.read_text())
    digest = hashlib.sha256(args.samples.read_bytes()).hexdigest()
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    core = olmo3.build_olmo3_moe_config_from_hf_config(
        config,
        dtype=getattr(core_config.DType, args.dtype),
        attention_backend=attention.AttentionBackendName.torch,
        router_aux_loss_weight=0.0,
        router_z_loss_weight=0.0,
    )
    core.recompute_each_block = False
    model = core.build(init_device="meta")
    model.init_weights(max_seq_len=4096, max_local_microbatch_size=4096, device=torch.device("cuda"))
    state = {}
    for shard in sorted(Path(args.model).glob("*.safetensors")):
        state.update(load_file(shard))
    olmo3.load_olmo3_moe_hf_state(model, config, state)
    del state
    model.eval()
    if args.dtype == "float32":
        # The pinned loader assigns checkpoint tensors, replacing the factory's
        # requested dtype. Widen the loaded values without changing the checkpoint.
        model.float()
        fp32_reference.audit_model(model)
    versions = {name: p._version for name, p in model.named_parameters()}
    report = {
        "model": args.model,
        "dtype": args.dtype,
        "samples_sha256": digest,
        "torch": torch.__version__,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "parameter_dtypes": dict(collections.Counter(str(p.dtype) for p in model.parameters())),
        "comparisons": {},
    }
    cache = {}
    with torch.no_grad():
        for path in args.serving_reports:
            serving = json.loads(path.read_text())
            if serving["model"] != args.model or serving["samples_sha256"] != digest:
                raise ValueError("Serving provenance mismatch")
            output = {"rows": [], "aggregate": {}}
            combined = collections.defaultdict(lambda: ([], []))
            for kind in ("rollouts", "forced"):
                for rollout in serving[kind]:
                    row = samples["rows"][rollout["row"]]
                    ids = row["input_ids"] + rollout["output_ids"]
                    key = tuple(ids)
                    if key not in cache:
                        start = time.perf_counter()
                        logits = model(torch.tensor([ids], device="cuda"))
                        scores = head_probe.selected_scores(logits, ids, len(row["input_ids"]) - 1)
                        cache[key] = scores
                        del logits
                        print(
                            "CORE_ROW",
                            args.dtype,
                            path.stem,
                            kind,
                            rollout["row"],
                            time.perf_counter() - start,
                            flush=True,
                        )
                    expected = cache[key]
                    actual = torch.tensor(rollout["logprobs"])
                    output["rows"].append({"kind": kind, "row": rollout["row"], "core_logprobs": expected.tolist()})
                    combined[kind][0].append(actual)
                    combined[kind][1].append(expected)
            for kind, (actual, expected) in combined.items():
                output["aggregate"][kind] = head_probe.probability_statistics(torch.cat(actual), torch.cat(expected))
            report["comparisons"][path.stem] = output
            benchmark.write_json(args.output, report)
            print("FULL_PRECISION_RESULT", args.dtype, path.stem, json.dumps(output["aggregate"]), flush=True)
    if versions != {name: p._version for name, p in model.named_parameters()}:
        raise ValueError("Parameters mutated during frozen scoring")
    report["parameter_versions_unchanged"] = True
    benchmark.write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--serving-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
