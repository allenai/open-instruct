"""Diff step-1 dumps from grpo.py (oc) vs grpo_fast.py.

Loads rank0 dumps from both runs, summarizes structural and numeric
differences (inputs, per-parameter grad/weight norms).
"""

import argparse
import math

import torch

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def _summarize_inputs(name: str, payload: dict) -> None:
    inputs = payload.get("inputs", {})
    logger.info(f"=== {name} inputs keys: {sorted(inputs.keys())}")
    for k, v in sorted(inputs.items()):
        if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            shapes = [tuple(t.shape) for t in v[:4]]
            dtypes = {str(t.dtype) for t in v}
            logger.info(f"  {k}: list[Tensor] len={len(v)} sample_shapes={shapes} dtypes={dtypes}")
        else:
            logger.info(f"  {k}: {type(v).__name__} value={v if not isinstance(v, torch.Tensor) else tuple(v.shape)}")


def _diff_grad_summary(a: dict, b: dict) -> None:
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    common = keys_a & keys_b
    logger.info(f"grad-summary: |oc|={len(keys_a)} |fast|={len(keys_b)} common={len(common)} only_oc={len(only_a)} only_fast={len(only_b)}")
    if only_a:
        logger.info(f"  only in oc (first 10): {sorted(only_a)[:10]}")
    if only_b:
        logger.info(f"  only in fast (first 10): {sorted(only_b)[:10]}")

    norms_a = []
    norms_b = []
    abs_means_a = []
    abs_means_b = []
    for k in sorted(common):
        ga = a[k]["grad_local_norm"]
        gb = b[k]["grad_local_norm"]
        if math.isfinite(ga) and math.isfinite(gb):
            norms_a.append(ga)
            norms_b.append(gb)
            abs_means_a.append(a[k]["grad_local_abs_mean"])
            abs_means_b.append(b[k]["grad_local_abs_mean"])

    def _stats(label: str, xs: list) -> None:
        if not xs:
            return
        t = torch.tensor(xs)
        logger.info(f"  {label}: n={len(xs)} mean={t.mean():.6e} median={t.median():.6e} max={t.max():.6e} min={t.min():.6e}")

    _stats("oc grad_local_norm", norms_a)
    _stats("fast grad_local_norm", norms_b)
    _stats("oc grad_local_abs_mean", abs_means_a)
    _stats("fast grad_local_abs_mean", abs_means_b)

    ratios = []
    for ka, kb in zip(sorted(common), sorted(common)):
        ga = a[ka]["grad_local_norm"]
        gb = b[kb]["grad_local_norm"]
        if gb > 0 and math.isfinite(ga) and math.isfinite(gb):
            ratios.append((ka, ga / gb, ga, gb))

    ratios.sort(key=lambda r: -abs(math.log(max(r[1], 1e-30))))
    logger.info("top-20 grad_norm ratio outliers (oc / fast):")
    for k, r, ga, gb in ratios[:20]:
        logger.info(f"  {k}: ratio={r:.4e}  oc={ga:.4e}  fast={gb:.4e}")


def _diff_weights(a: dict, b: dict) -> None:
    common = set(a.keys()) & set(b.keys())
    diffs = []
    for k in sorted(common):
        wa = a[k]["weight_local_norm"]
        wb = b[k]["weight_local_norm"]
        if wb > 0 and math.isfinite(wa) and math.isfinite(wb):
            diffs.append((k, abs(wa - wb) / max(wb, 1e-30), wa, wb))
    diffs.sort(key=lambda r: -r[1])
    logger.info("top-20 weight_local_norm rel-diff (|oc-fast|/fast):")
    for k, d, wa, wb in diffs[:20]:
        logger.info(f"  {k}: rel={d:.4e}  oc={wa:.4e}  fast={wb:.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oc", required=True, help="path to oc rank0 .pt")
    ap.add_argument("--fast", required=True, help="path to fast rank0 .pt")
    args = ap.parse_args()

    oc = torch.load(args.oc, map_location="cpu", weights_only=False)
    fa = torch.load(args.fast, map_location="cpu", weights_only=False)

    logger.info(f"oc top-level keys: {sorted(oc.keys())}")
    logger.info(f"fast top-level keys: {sorted(fa.keys())}")

    _summarize_inputs("oc", oc)
    _summarize_inputs("fast", fa)

    if "param_grads" in oc and "param_grads" in fa:
        _diff_grad_summary(oc["param_grads"], fa["param_grads"])
        _diff_weights(oc["param_grads"], fa["param_grads"])
    else:
        logger.info(f"missing param_grads — oc has: {sorted(oc.keys())}; fast has: {sorted(fa.keys())}")


if __name__ == "__main__":
    main()
