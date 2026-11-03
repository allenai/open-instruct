# Core-compatible serving modes

Fused rounding is the default for the hybrid MoE checkpoint family when
using an image built from the current runtime lock. The adapter selects it from
the checkpoint configuration and resolved serving settings; no environment flag
is required. Other checkpoint geometries and serving setups retain ordinary
SGLang execution. The slow full reference remains opt-in.

## Choose a mode

Set overrides in the run's `[launch.env]` table. Unset means `auto`.

| `OLMO_SGLANG_CORE_COMPAT` | Behavior | When to use |
| --- | --- | --- |
| unset or `auto` | Fused rounding for the profile below; ordinary serving otherwise | Normal operation |
| `0` or `off` | Ordinary SGLang arithmetic, including its own expert rounding/reduction | Opt out or compare against ordinary serving |
| `rounding` | Core-style BF16 boundaries and FP32 expert reduction/norms; retains SGLang GEMMs, attention, KDA and decode graphs | Explicit compatible arithmetic, or testing another supported shape |
| `1` or `full` | Core layouts/grouped GEMMs, eager norms, SDPA and Core-style KDA prefill; no graphs | Numerical investigation; substantial performance cost |

Within `rounding`, `OLMO_SGLANG_ROUNDING_KERNELS` defaults to `fused`.
Set it to `torch` for the unfused tensor implementation, or `moe` / `norms` to fuse
only that component for diagnosis. This second flag alone does not enable
rounding. `OLMO_HF_MOE_CORE_REFERENCE` is a separate Transformers flag and has
no effect on SGLang.

The automatic profile is the 12.5B hybrid MoE geometry: hidden size 1024, 16
layers with full attention at layers 7 and 15 and KDA elsewhere, 512
experts/top-16, latent width 512, expert/shared width 1024, dense width 8192 and
the family's norm/gating settings. `_ROUNDING_PROFILE` in the serving
implementation's `olmo_sglang/core_compat.py` is the exact field contract.
Selection requires unquantized BF16 (including `dtype=auto` when resolved to BF16
by the model loader), TP1/EP1, the `auto`/`triton` MoE backend,
full or disabled decode graphs, disabled prefill graphs, no speculation and no
`torch.compile`. It is independent of checkpoint path.
Unsupported settings automatically keep ordinary serving; explicit `rounding` and
`full` retain their runtime validation. The model logs the resolved mode at
construction. This does not change graph settings or the trainer backend.

For the fast configuration, retain:

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

**Use automatic fused rounding for normal RL rollouts on the hybrid MoE
profile.** It restores the intended rounding and normalization while retaining
fast kernels and decode graphs. Read the model's startup log to confirm that
`auto` resolved to `rounding`; an unset flag alone does not establish which path
ran. Keep Core as the training/scoring policy and monitor probability differences
on the actual rollout tokens.

A strict parity investigation has a different execution contract:

| Path | What to compare | What agreement establishes |
| --- | --- | --- |
| Core versus HF conversion reference | Identical exported weights and token IDs; BF16 forward weights, FP32 router math; `OLMO_HF_MOE_CORE_REFERENCE=1`, matched Torch/SDPA attention (math SDPA for the strict converter), no cache, same sequence and batch shapes | A controlled conversion/forward oracle |
| Core versus SGLang `full` | Same full prefixes, Core Torch attention, TP1/EP1, disabled serving graphs | Closest available serving reference |
| Core versus SGLang automatic `rounding` | Actual cached rollout scores and token choices, rescored by Core on those same prefixes | The practical training/serving discrepancy with the recommended fast implementation |

The **pure parity path is the first row**, with every operator/backend and shape
held fixed and the outputs checked explicitly. Neither flag alone promises
bitwise equality on every workload. `OLMO_HF_MOE_CORE_REFERENCE=1` does not alter SGLang,
and enabling SGLang `full` does not turn cached generation into the same
computation as a full-prefix Core forward.

Use SGLang `full` to isolate a numerical discrepancy, not as the default RL
recipe. It is several times slower than fused rounding and only modestly reduces
cached-generation probability errors, while fused rounding already applies the
Core-style rounding and normalization at full speed.

## Interpreting token and probability agreement

Compare **Core versus SGLang within each checkpoint**. Checkpoints with
different weights are not expected to produce identical answers, and a shared
architecture does not imply the same sensitivity to arithmetic changes.

A mean absolute log-probability gate (for example 0.05) is a bounded mechanics
criterion, not a requirement that every token be close or that greedy choices
match. A run can pass the mean criterion while individual tokens differ by
around a nat, so also inspect the worst per-token gaps; a small average alone
does not make the tails harmless.

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

## Changing the serving environment

Agreement depends on the checkpoint, hardware, serving mode and runtime build.
Use the [runtime lock](architecture.md#runtime-sources-and-images) for new builds
and repeat the same-prefix token and probability checks when changing any of them.
