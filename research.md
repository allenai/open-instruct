# Research log

Tracks the *questions* being pursued and what came out of them. Each entry
links to the relevant launch section(s) in [`experiment.md`](experiment.md)
(the raw run log: configs, launch commands, Beaker links) rather than
repeating that detail here.

Status values: `ACTIVE`, `PAUSED`, `DONE`, `BLOCKED`.

---

## [ACTIVE] DAPO n×k allocation: how to split n×k=128 between prompts and samples/prompt

**Question:** at fixed completions/step (n × k = 128), is it better to sample
many prompts with few completions each (n=16,k=8) or few prompts with many
completions each (n=2,k=64)?

**Runs:**
- [Baseline sweep](experiment.md#baseline-sweep-n--k--128-completionsstep-held-constant) — n∈{16,8,4,2}, k=128/n, seed 1
- [Replication seeds 2 & 3](experiment.md#replication-runs-seeds-2--3) — all 4 baselines
- [Additional gradnorm=1.0 seeds](experiment.md#additional-seeds-with-grad-norm-10) — n∈{8,4,2}, one more seed each
- [Best-step held-out evals](experiment.md#best-step-held-out-evals-brumo--hmmt--aime-2025) — cross-config comparison on BRUMO/HMMT/AIME

**Findings:** n4_k32 has a failure mode under grad_norm=1.0 — see the async_steps
finding below. Otherwise no quantitative verdict recorded yet (pull final
pass@1 numbers from the wandb group `deepscaler_eval_best` / per-config eval
Beaker links above).

---

## [ACTIVE] Never-give-up (NGU) exploration bonus: does it beat baseline DAPO, and what's the best `p`?

**Question:** does adding `--never_give_up` (revisiting prompts the policy
hasn't solved yet) improve over plain DAPO, and where's the sweet spot for
the mixing coefficient `p`? All at n=8, k=16 (picked as one of the two
n×k configs to test NGU on, alongside n=16,k=8).

**Runs:**
- [NGU sweep, p ∈ {0.5, 0.9}](experiment.md#ngu-sweep-same-base-add---never_give_up) — first pass, both n8k16 and n16k8
- [p sweep, p ∈ {0.6, 0.75}](experiment.md#ngu-p-sweep-between-05-and-09-n8-k16) — bracketing p=0.5 (best-so-far) and p=0.9
- [p=0.75 replicate seeds + p=0.875 sweep](experiment.md#p075-replicate-seeds-and-p0875-sweep-n8-k16-16)
- [Seed 4 for p=0.5 and p=0.75](experiment.md#seed-4-for-p05-and-p075-n8-k16)
- [gradnorm=1.0 NGU seeds at p ∈ {0.5, 0.75, 0.875}](experiment.md#additional-seeds-with-grad-norm-10)
- [Best-step held-out evals](experiment.md#best-step-held-out-evals-brumo--hmmt--aime-2025) — ngu05/ngu075/ngu0875 vs baseline_dapo_n8_k16

**Findings:** p=0.5 was the leading candidate after the first sweep (drove
the decision to replicate it over 4 seeds and bracket it with p=0.6/0.75).
No final numeric verdict recorded yet across p ∈ {0.5, 0.6, 0.75, 0.875, 0.9}
— pull from the eval Beaker links / wandb group `deepscaler_eval_best`.

**Blocking infra bug (found + fixed):** NGU relies on per-prompt bookkeeping
in the `DataPreparationActor`. All 4 FSDP ranks were independently
snapshotting/restoring this actor's state on checkpoint save/resume, so a
resume could load an inconsistent state (whichever rank's `set_state` ran
last). Fixed in `7c8919d2f` to save/restore from global rank 0 only; see
[NGU relaunch with rank-0 state-saving fix](experiment.md#ngu-relaunch-with-rank-0-state-saving-fix--async_steps-2).
Any NGU conclusions drawn from runs *before* this fix are suspect.

---

## [DONE] max_grad_norm: 5.0 vs 1.0

**Question:** does a tighter grad-norm clip (1.0 vs the original 5.0) change
stability or final performance.

**Runs:** [Additional seeds with grad norm 1.0](experiment.md#additional-seeds-with-grad-norm-10)
(one extra seed per baseline config except n16k8, plus one NGU seed at each
of p=0.5/0.75/0.875).

**Findings:** grad_norm=1.0 didn't itself cause the n4_k32 collapse — see the
async_steps finding below — but it's the config where that failure mode
first surfaced. No clean stability/performance comparison against grad_norm=5.0
recorded yet.

---

## [DONE] n4_k32 collapse: root cause + fix

**Question:** `2k_baseline_dapo_n4_k32_gradnorm1_seed1` entered an
all-zero-reward filtering spin around step ~547 and crawled (13 steps/9h) —
why, and is it config-specific or a general async off-policy risk?

**Runs:** [n4_k32 gradnorm1 stall → rerun with async_steps 2](experiment.md#n4_k32-gradnorm1-stall--rerun-with-async_steps-2)

**Finding (confirmed via log + wandb analysis):** completion lengths drifted
up to the 8192 cap over the run; `val/rho_weight` declined 1.0→0.9 over steps
~535–546 as long generations slowed down async collection (growing
off-policyness, negative advantage dominating), then snapped back to 1.0
post-collapse (steps became rare ⇒ on-policy again) — so trainer and vLLM
weights agreed post-collapse, ruling out weight-sync/optimizer corruption.
Mechanism: the completion-length distribution crossed the 8192 truncation
cliff, so ~60% of completions came back unfinished ⇒ nearly all groups had
all-zero reward ⇒ active-sampling filtered everything out, and the loop
starved. **Fix:** reduce `--async_steps` 4→2 to bound the off-policy feedback
loop. Applied to the n4_k32 rerun and (preventatively) to all later NGU
gradnorm1 relaunches.

**Implication:** any future run with growing completion lengths near the
truncation cap + `--active_sampling` is at risk of this collapse; watch
`val/rho_weight` and `batch/filtered_prompts_solved` as leading indicators.

---

## [ACTIVE] Holmes (B300) cluster compatibility

**Question:** does the existing CUDA 12.8 image work on `ai2/holmes`
(Blackwell Ultra / sm_103 nodes), or does it need a rebuild.

**Runs:** [Holmes (B300) cluster test](experiment.md#holmes-b300-cluster-test-ngu-075-gradnorm1-async2)

**Findings:** precompiled-kernel compatibility itself was fine —
vLLM/flash-attn/torch sm_100 SASS runs on sm_103, and
`detect_attn_implementation` correctly auto-selects FA4 (JIT CuTe DSL) on
compute capability 10.x, so no attention backend changes were needed. Three
infra gaps found and fixed:
1. `ai2/holmes` wasn't in `WEKA_CLUSTERS` in `open_instruct/launch_utils.py`
   (weka wouldn't get mounted) — added.
2. `get_device_name` in `open_instruct/utils.py` didn't recognize the
   `NVIDIA B300 SXM6 AC` device string and raised — added a `b300` entry to
   `GPU_SPECS` (288 GB HBM3e, 8 TB/s, ~2.25 PFLOPS dense BF16, same as B200,
   commit `843258932`).
3. torch's *runtime-compiled* (jiterator) ops — first hit: `erfinv_` in
   weight init — failed with `nvrtc: error: invalid value for
   --gpu-architecture`: torch cu128 pins NVRTC 12.8, which can't target
   compute_103. Overrode `nvidia-cuda-nvrtc-cu12` to 12.9.86 (same
   `libnvrtc.so.12` soname, linux/x86_64 only) in pyproject (commit
   `0b69301e7`). Triton/`torch.compile` needed no fix — it already selects
   its bundled 12.9 `ptxas-blackwell` for arch >= 100.

Third launch on the rebuilt image pending; result TBD (see the `TBD` row in
experiment.md — check Beaker before treating this as closed).

---

## [ACTIVE] Held-out eval methodology: best-checkpoint pass@1 on BRUMO/HMMT/AIME

**Question:** in-training AIME eval only tracks one competition; to compare
configs fairly, eval every seed's *best* in-training-AIME checkpoint on 4
held-out competitions (BRUMO 2025, HMMT Feb/Nov 2025, AIME 2025) instead of
just the training-time metric.

**Runs:** [Best-step held-out evals](experiment.md#best-step-held-out-evals-brumo--hmmt--aime-2025)

**Findings:** methodology validated (bit-exact HF conversion check passed);
infra gotcha found — Beaker's Ray head port is hardcoded on host networking
in `ray_node_setup.sh`, so packing two 4-GPU eval jobs onto one node makes
the second job join the first job's Ray cluster and die. Worked around by
requesting full-node (`NUM_GPUS=8`) for the affected jobs. Actual cross-config
pass@1 comparison not yet summarized here — numbers live in the per-config
`eval/pass_at_1/<label>` wandb metrics linked from the table.

---

## [BACKLOG] Difficulty-quartile behavior

**Question:** the dataset splits into difficulty quartiles
(`math_deepscaler_quartile{0,1,2,3}`) with per-quartile batch metrics
(`batch/nonzero_prompts/<quartile>`, `batch/filtered_prompts_solved/<quartile>`,
etc.) logged throughout every run above — not yet analyzed on its own. Worth
a pass to see whether NGU/gradnorm/async_steps changes affect quartiles
differently (e.g. does NGU specifically help the hardest quartile get
non-zero reward more often?).

**Runs:** every run above logs this data; no dedicated analysis yet.
