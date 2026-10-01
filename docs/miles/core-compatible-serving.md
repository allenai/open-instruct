# Core-compatible serving modes

Fused rounding is the default for the qualified hybrid MoE checkpoint family when
using an image built from the current runtime lock. The adapter selects it from
the checkpoint configuration and resolved serving settings; no environment flag
is required. Other checkpoint geometries and serving setups retain ordinary
SGLang execution. The slow full reference remains opt-in.

## Choose a mode

Set overrides in the run's `[launch.env]` table. Unset means `auto`.

| `OLMO_SGLANG_CORE_COMPAT` | Behavior | When to use |
| --- | --- | --- |
| unset or `auto` | Fused rounding for the qualified profile below; ordinary serving otherwise | Normal operation |
| `0` or `off` | Ordinary SGLang arithmetic, including its original expert rounding/reduction | Reproduce the old baseline or opt out |
| `rounding` | Core-style BF16 boundaries and FP32 expert reduction/norms; retains SGLang GEMMs, attention, KDA and decode graphs | Explicit compatible arithmetic, or qualification of another supported shape |
| `1` or `full` | Core layouts/grouped GEMMs, eager norms, SDPA and Core-style KDA prefill; no graphs | Numerical investigation; substantial performance cost |

Within `rounding`, `OLMO_SGLANG_ROUNDING_KERNELS` defaults to `fused`.
Set it to `torch` for the original tensor control, or `moe` / `norms` to fuse
only that component for diagnosis. This second flag alone does not enable
rounding. `OLMO_HF_MOE_CORE_REFERENCE` is a separate Transformers flag and has
no effect on SGLang.

The automatic profile is the 12.5B hybrid MoE geometry used by the measured base,
EMO SFT and non-EMO SFT checkpoints: hidden size 1024, 16 layers with full attention
at layers 7 and 15 and KDA elsewhere, 512 experts/top-16, latent width 512,
expert/shared width 1024, dense width 8192 and the measured norm/gating settings.
The serving implementation's [`_ROUNDING_PROFILE`](https://github.com/allenai/olmo-sglang/blob/e0b0849d09509224879ed02aecfca41eff11e50d/src/olmo_sglang/core_compat.py)
is the exact field contract.
Selection requires unquantized BF16 (including `dtype=auto` when resolved to BF16
by the model loader), TP1/EP1, the `auto`/`triton` MoE backend,
full or disabled decode graphs, disabled prefill graphs, no speculation and no
`torch.compile`. It is independent of checkpoint path and EMO ancestry.
Unsupported settings automatically keep the old path; explicit `rounding` and
`full` retain their runtime validation. The model logs the resolved mode at
construction. This does not change graph settings or the trainer backend.

For the measured fast configuration, retain:

```toml
[inference]
rollout_tensor_parallel_size = 1
sglang_cuda_graph_backend_decode = "full"
sglang_cuda_graph_backend_prefill = "disabled"
```

For an explicit opt-out, add:

```toml
[launch.env]
OLMO_SGLANG_CORE_COMPAT = "0"
```

## Optimized serving versus a strict parity check

**Use automatic fused rounding for normal RL rollouts on the qualified hybrid MoE
profile.** It restores the intended rounding and normalization while retaining
fast kernels and decode graphs. Read the model's startup log to confirm that
`auto` resolved to `rounding`; an unset flag alone does not establish which path
ran. Keep Core as the training/scoring policy and monitor probability differences
on the actual rollout tokens.

A strict parity investigation has a different execution contract:

| Path | What to compare | What agreement establishes |
| --- | --- | --- |
| Core versus HF conversion reference | Identical exported weights and token IDs; BF16 forward weights, FP32 router math; `OLMO_HF_MOE_CORE_REFERENCE=1`, matched Torch/SDPA attention (math SDPA for the strict converter), no cache, same sequence and batch shapes | A controlled conversion/forward oracle; the earlier short controls were exact |
| Core versus SGLang `full` | Same full prefixes, Core Torch attention, TP1/EP1, disabled serving graphs | Closest available serving reference; earlier long-prefix selected scores differed by only about 3.5e-7 mean / 1.9e-6 maximum |
| Core versus SGLang automatic `rounding` | Actual cached rollout scores and token choices, rescored by Core on those same prefixes | The practical training/serving discrepancy with the recommended fast implementation |

The **pure parity path is the first row**, with every operator/backend and shape
held fixed and the outputs checked explicitly. Neither flag alone promises
bitwise equality on every workload. Use the
[Core/HF fidelity diagnostic](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/scripts/miles/hero_core_fidelity.py) and its
[recorded source/attention controls](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/hero-core-fidelity-20260923.md)
when validating an export. `OLMO_HF_MOE_CORE_REFERENCE=1` does not alter SGLang,
and enabling SGLang `full` does not turn cached generation into the same
computation as a full-prefix Core forward.

Use SGLang `full` to isolate a numerical discrepancy, not as the default RL
recipe. On the earlier batch-four workload it cost approximately 6.5 times the
generation time and only modestly improved cached-generation probability errors.
The optimized mode recovered the arithmetic changes without the tensor-control
slowdown. No measured learning-quality benefit yet justifies the full mode's
cost.

## Interpreting token and probability agreement

Compare **Core versus SGLang within each checkpoint**. EMO and non-EMO have
different weights and are not expected to produce identical answers. Their
shared architecture does not imply the same sensitivity to arithmetic changes;
the EMO-specific excess error has not been causally isolated.

The existing 0.05 mean absolute log-probability gate is a bounded mechanics
criterion, not a requirement that every token be close or that greedy choices
match. The [paired four-update smoke](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/hero-rl-smoke-20260924.md)
passed that mean criterion for both checkpoints, but its worst individual
probability gaps were 0.734 nats (EMO) and 1.199 nats (non-EMO). It did not capture
Core's argmax choices. A small average alone does not make those tails harmless.

A greedy disagreement means the two systems prefer different next tokens on an
identical prefix. Inspect the top-two margin: a near tie can flip with a tiny
probability change, whereas a large margin calls for closer investigation.
Once independently generated responses diverge, later prefixes differ too;
response equality is a different question from same-prefix forward parity.
Conversely, matching argmax tokens does not imply matching policy probabilities
or identical stochastic samples. Do not infer full-distribution KL from selected
token probabilities or just the top two alternatives.

Core full-sequence teacher forcing and Core prefix-at-a-time forwards also use
different matrix shapes. Retain both when diagnosing serving differences:
causal prefixes match semantically, while finite-precision execution can still
differ. A cached serving/full-sequence scoring discrepancy is not, by itself,
evidence of a cache implementation bug.

## Qualification evidence

The [archived serving record](https://github.com/allenai/open-instruct/blob/7a477910405a65d914b48096f83c13a6c61a60ad/docs/miles/core-compatible-serving.md)
contains the same-prefix probes, throughput tables, source settings and Beaker
experiments. Results apply to their recorded checkpoints, hardware and modes.
Use the [runtime lock](architecture.md#runtime-sources-and-images) for new builds
and repeat the relevant checks when changing that environment.
