# H083 1-step gradient comparison: old image vs the big-slice image

Ledger-manager's ruling (2026-10-23), fixed before the result. With drop logging off, the drop-count image still gives step 2–7
CE different from the old image on sci-s0, while step 1 is bitwise equal. The suspect is the big-slice layer `bs3ec936c3`: its
`with_logical_constraint` pins in the KDA chunked delta rule (maxtext `c5f1251d`, shipped in that layer and never re-tested
against the old image under `olmo35-ep-as-cp`). This measures whether the step-1 gradient differs by more than reduction order.

| | image | digest |
|---|---|---|
| old | `maxtext-posttrain:51f8cc4b-p907d86f3-rfb46e215e` | `sha256:7d7f4938f98b4a70e918cc768d2109f6d1080d5a8944b829a5bceda49b24cabc` |
| bs | `maxtext-posttrain:51f8cc4b-p907d86f3-rfb46e215e-bs3ec936c3` | `sha256:c858123fd5ead4f3fef672fda705f70167217d678c213fb6714ea8c557672ccf` |

- **Setup:** `../gradcmp-old` and `../gradcmp-bs`, identical except the image. Both are R1's overlay (`h083-sci-s0-ep8ref`):
  same weights (`olmo35-small-4t-emo-step5961`), same data and batch 1, `olmo35-ep-as-cp` fsdp16 x ep8, factor 2.5. The changes
  are `STEPS=1`, plain SGD with no clipping and LR 1e8, and a checkpoint after step 1.
- **Why this gives the gradient:** the step-1 gradient doesn't depend on the optimizer. With SGD,
  `p1 = p0 - lr1 * g` in fp32, so `|g_bs - g_old| / |g_old| = |p1_bs - p1_old| / |p0 - p1_old|`. LR 1e8 makes `lr1 * |g|` large
  next to `|p0|`, so fp32 rounding of `p1` (the `floor` column, about `2**-24 * |p1| / |p0 - p1|`) sits far below 1e-5.
- **No code change:** only config knobs, and the same knobs in both runs. The launcher copy reads `opt_type` and
  `gradient_clipping_threshold` from the environment.
- **Rule:** whole-model relative difference ≤ 1e-5 means class B (reduction order). Otherwise stop and tell ledger-manager.
  The worst per-tensor value is reported too.
- **Run:** `run.sh` on Hammond. It posts `GRAD-CMP ...` to the x16prod event log.
- **Receipt:** the Hammond copy of this directory, `/weka/oe-eval-default/abhishekr/watch/tpu/gradcmp/`, holding:
  - `result.json`: whole-model and per-tensor values, top 10 worst;
  - `checks.txt`: batch-1 hashes and step-1 loss parts, equal across the two runs;
  - `run.log`;
  - `logs/`.
