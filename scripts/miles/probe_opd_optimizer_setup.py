"""Execute pinned OI tiled loss on CPU; simulate only the count collective and DP averaging."""

import argparse
import ast
import enum
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--oi-source", type=Path, required=True, help="Trusted pinned directory containing grpo_fast.py and grpo_utils.py"
)
options = parser.parse_args()
SOURCE = options.oi_source


def extract(path, names, namespace, methods=False):
    tree = ast.parse(path.read_text())
    candidates = list(ast.walk(tree)) if methods else tree.body
    selected = [
        node for node in candidates if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


ns = {"torch": torch, "enum": enum}
extract(
    SOURCE / "grpo_utils.py",
    ["GRPOLossType", "DPPODivergenceType", "TiledGRPOLMHeadLoss", "tiled_grpo_lm_head_loss"],
    ns,
)
extract(SOURCE / "grpo_fast.py", ["_compute_tiled_dapo_loss"], ns, methods=True)


def tiled_gradient(lengths, corrected=False):
    total = sum(lengths)
    rank_grads = []
    reference = torch.nn.Linear(1, 2, bias=False)
    reference.weight.data.zero_()
    ref_losses = []
    for rank, n in enumerate(lengths):
        head = torch.nn.Linear(1, 2, bias=False)
        head.weight.data.zero_()
        hidden = torch.full((1, n, 1), float(rank + 1), requires_grad=True)
        mask = torch.ones(1, n, dtype=torch.bool)
        args = SimpleNamespace(
            lm_head_fp32=True,
            world_size=len(lengths),
            sequence_parallel_size=1,
            beta=0.0,
            load_ref_policy=False,
            clip_lower=0.2,
            clip_higher=0.28,
            liger_grpo_loss_chunk_size=2,
            loss_fn="dapo",
            dppo_divergence_type="tv",
            dppo_divergence_threshold=0.1,
            tvpo_truncation_cap=20.0,
        )

        def tiled(rank_count=n, **kwargs):
            if corrected:
                kwargs["loss_scale"] = torch.tensor(rank_count * len(lengths) / total)
            return ns["tiled_grpo_lm_head_loss"](**kwargs)

        grpo = SimpleNamespace(
            tiled_grpo_lm_head_loss=tiled,
            forward_for_liger_hidden_states=lambda *a, hidden=hidden, **kw: hidden,
            get_causal_lm_backbone_and_lm_head=lambda *a, head=head: (None, head),
        )
        ns["grpo_utils"] = grpo
        ns["dist"] = SimpleNamespace(
            all_reduce=lambda value, **kw: value.fill_(total), ReduceOp=SimpleNamespace(SUM=0)
        )
        learner = SimpleNamespace(
            model=head, args=args, _sp_group=None, streaming_config=SimpleNamespace(temperature=1.0)
        )
        with patch("torch.cuda.synchronize"):
            loss, _ = ns["_compute_tiled_dapo_loss"](
                learner,
                query_responses=torch.zeros(1, n + 1, dtype=torch.long),
                position_ids=torch.arange(n + 1)[None],
                response_mask=mask,
                advantages=torch.ones(1, n),
                old_logprobs=torch.full((1, n), -torch.log(torch.tensor(2.0)).item()),
                ref_logprobs=None,
                policy_mask=None,
                policy_freeze_mask=None,
                loss_denominator=total,
                loss_denominator_mode="token",
                rollout_sample_ids=None,
                cp_context=None,
            )
            loss.backward()
        rank_grads.append(head.weight.grad.clone())
        logp = torch.log_softmax(reference(hidden.detach()), dim=-1)[..., 0]
        ref_losses.append(-torch.exp(logp + torch.log(torch.tensor(2.0))).sum())
    (sum(ref_losses) / total).backward()
    measured = torch.stack(rank_grads).mean(0)
    return {
        "lengths": lengths,
        "oi_dp_averaged_head_gradient": measured.flatten().tolist(),
        "global_token_mean_gradient": reference.weight.grad.flatten().tolist(),
        "relative_difference": float(
            torch.linalg.vector_norm(measured - reference.weight.grad)
            / torch.linalg.vector_norm(reference.weight.grad)
        ),
    }


normal = tiled_gradient([1, 2, 3, 4])
balanced = tiled_gradient([4, 4, 4, 4])
assert normal["relative_difference"] > 0.16
assert balanced["relative_difference"] < 1e-6


def adam_step(prime):
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=1e-6, weight_decay=0, betas=(0.9, 0.999), eps=1e-8)
    if prime:
        p.grad = torch.zeros_like(p)
        opt.step()
        opt.zero_grad(set_to_none=True)
    p.grad = torch.ones_like(p)
    opt.step()
    return {"delta": p.item(), "state_step": opt.state[p]["step"].item()}


a, b = adam_step(False), adam_step(True)
result = {
    "scope": "Pinned OI tiled implementation executed on CPU; count all-reduce and DP averaging simulated, no distributed CUDA or full-model equivalence claim.",
    "source_sha256": {
        name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in ("grpo_fast.py", "grpo_utils.py")
    },
    "torch_version": torch.__version__,
    "unequal_counts": normal,
    "equal_counts": balanced,
    "corrected_scaling_control": tiled_gradient([1, 2, 3, 4], corrected=True),
    "adam_fresh": a,
    "adam_after_zero_step": b,
    "first_nonzero_update_ratio": b["delta"] / a["delta"],
}
assert result["corrected_scaling_control"]["relative_difference"] < 1e-6
print(json.dumps(result, indent=2))
