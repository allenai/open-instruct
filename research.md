# Research log

Tracks the *questions* being pursued and what came out of them. Each entry
links to the relevant launch section(s) in [`experiment.md`](experiment.md)
(the raw run log: configs, launch commands, Beaker links) rather than
repeating that detail here.

Status values: `ACTIVE`, `PAUSED`, `DONE`, `BLOCKED`.

## Current focus

**Main goal:** get `--never_give_up` (NGU) to reliably beat baseline DAPO on
the DeepScaleR dataset via `scripts/train/qwen/qwen3_4b_deepscaler_math.sh`
— not just at a cherry-picked checkpoint, and especially on the hardest
problems.

**Where things stand (2026-07-11):**
- At `max_grad_norm=5.0`, all configs "overfit" — AIME eval performance
  degrades after ~step 1000. Picking the best checkpoint per seed (early
  stopping) and comparing NGU vs. baseline: NGU wins, but by a small margin,
  and *not* on the subset that matters most — the hardest AIME problems
  (initial pass@64=0) are improved slightly *more* by the plain `N=2,K=64`
  baseline than by any NGU `p`. NGU's edge shows up on the medium-difficulty
  subset instead. See the figure in the NGU idea below.
- Working hypothesis: the grad_norm=5.0 overfitting is confounding the
  comparison, so we're rerunning the full sweep (baselines + NGU) at
  `max_grad_norm=1.0` to remove it.
- Hit a second problem along the way: the `N=4,K=32` gradnorm=1.0 baseline
  collapsed mid-run (`val/rho_weight` crashed 1.0→0.9 over ~20 steps,
  triggered by completion-length growth pushing off-policyness up). Possibly
  the same failure mode threatens `N=2,K=64` given the shared
  "few-prompts-many-samples" shape — watch for it there too. Mitigated, not
  fully solved, by dropping `--async_steps` 4→2.
- **Next:** finish the grad_norm=1.0 sweep (baselines + NGU p-sweep), rerun
  the difficulty-stratified best-checkpoint comparison, and check whether the
  hard-subset gap closes.

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
- [2 more seeds (n8_k16, n2_k64) + 3 more seeds (n4_k32), async4](experiment.md#2026-07-13-2-more-seeds-n8_k16-n2_k64--3-more-seeds-n4_k32-all-async_steps4-default) — n4_k32 gets extra seeds specifically to check whether its async4 collapse reproduces
- [2026-07-16: n4_k32 baseline seed5](experiment.md#2026-07-16-n4_k32-baseline-seed5-and-first-ngu-075-kl-penalty-beta001-run) — 5th seed for the n4_k32 gradnorm1 baseline

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
- [OC=false NGU parity check](experiment.md#ocfalse-grpo_fastpy-ngu-parity-check) — confirmed `grpo_fast.py` (DeepSpeed) already supports NGU + dataset logging with no porting, since it lives in shared modules; investigating moving sweeps off `grpo.py`/OLMo-core (has been unstable — FSDP stalls, B300 issues)
- [NGU gradnorm=1.0 p-sweep, OC=false](experiment.md#ngu-gradnorm10-p-sweep-ocfalse-grpo_fastpy) — replicates the p∈{0.5,0.75,0.875} sweep on the DeepSpeed backend
- [NGU 0.875, 12k response length](experiment.md#ngu-0875-12k-response-length-octrue) — does more room to finish long completions change the hard-tier picture
- [2 more NGU seeds per p, async4, on ai2/titan](experiment.md#2026-07-13-2-more-ngu-seeds-per-p-async_steps4-default-on-ai2titan) — post-rank-0-fix async4 data points (seed1 async4 predates the fix) to compare against the async2 seed1 runs
- [2026-07-14: NGU 0.5 seed 3 written off, new seed 4 (async2, jupiter, urgent) + holmes retry](experiment.md#2026-07-14-ngu-05-seed-3-written-off-new-seed-4-async2-jupiter-urgent--holmes-retry)
- [2026-07-14: NGU 0.875 seed 2 continued on same wandb run, moved to olmo-instruct/urgent](experiment.md#2026-07-14-ngu-0875-seed-2-continued-on-same-wandb-run-moved-to-olmo-instructurgent)
- [2026-07-14: baseline n8_k16 seed4, NGU 0.75/0.875 async2 seed2, first n16_k8 gradnorm1 NGU (p=0.75) seed1](experiment.md#2026-07-14-baseline-n8_k16-seed4-ngu-0750875-async2-seed2-and-first-n16_k8-gradnorm1-ngu-p075-seed1) — n16_k8 gets its first max_grad_norm=1.0 NGU run (previously only tested at grad_norm=5.0)
- [2026-07-14: n16_k8 gradnorm1 NGU p=0.825 seed1](experiment.md#2026-07-14-n16_k8-gradnorm1-ngu-p0825-seed1) — second n16_k8 gradnorm1 data point, bracketing 0.75/0.875
- [2026-07-15: NGU 0.875 async2 seed3, on ai2/titan](experiment.md#2026-07-15-ngu-0875-async2-seed3-on-ai2titan) — third async2 seed for p=0.875
- [2026-07-16: first NGU 0.75 KL-penalty run (beta=0.01, async2)](experiment.md#2026-07-16-n4_k32-baseline-seed5-and-first-ngu-075-kl-penalty-beta001-run) — new sub-config, seed1; tests whether a nonzero KL-to-ref-policy penalty changes the p=0.75 picture
- [2026-07-16: n8_k16 baseline seed5, on ai2/titan](experiment.md#2026-07-16-n8_k16-baseline-seed5-on-ai2titan) — 5th seed for the n8_k16 gradnorm1 baseline
- [2026-07-16: NGU 0.5 async2 seed5](experiment.md#2026-07-16-ngu-05-async2-seed5) — 5th p=0.5 seed (seed3 was written off as bad)
- [2026-07-16: NGU 0.75 seed3 (gz2ux8w0) resumed from checkpoint after gloo comms crash](experiment.md#2026-07-16-ngu-075-seed3-gz2ux8w0-resumed-from-checkpoint-after-gloo-comms-crash) — transient distributed-comms fault at step 948/2000, resumed (not restarted) via the same `checkpoint_state_dir` and same wandb run id
- [2026-07-16: NGU 0.75 seed2 (wf6ttda7) resumed after repeated preemption](experiment.md#2026-07-16-ngu-075-seed2-wf6ttda7-resumed-after-repeated-preemption) — 76% done (step 1521/2000) before being preempted 7x on titan/oe-adapt-code/high; resumed on jupiter/open-instruct-dev/urgent, same wandb run
- [2026-07-16: refreshed best_step across all runs, swapped NGU 0.5 seed3 to zg0thiuz](experiment.md#2026-07-16-refreshed-best_step-across-all-registered-runs-swapped-ngu-05-seed3-to-zg0thiuz) — only gz2ux8w0's best_step actually moved (post-resume progress); NGU 0.5 seed3 slot now zg0thiuz (still running, best_step=800, needs another refresh later)
- [2026-07-17: NGU 0.75 seed4 (async2) + second KL-penalty attempt (beta=0.001)](experiment.md#2026-07-17-ngu-075-seed4-async2--first-kl-penalty-attempt-at-beta0001) — 4th plain async2 seed for p=0.75; second KL sub-config, an order of magnitude below the first beta=0.01 attempt, own seed1
- [2026-07-17: NGU 0.75 seed2 (wf6ttda7) finished cleanly](experiment.md#2026-07-17-ngu-075-seed2-wf6ttda7-finished-cleanly--crash-signature-alert-was-teardown-noise-plus-a-stale-cache-bug-found-in-the-process) — background-monitor "crash" alert was benign post-completion teardown noise (exit code 0, 2000/2000); also surfaced and fixed a stale-`.wandb_cache` bug that can affect any other crash-then-resume run
- [2026-07-17: NGU 0.5 async2 seed6](experiment.md#2026-07-17-ngu-05-async2-seed6) — 6th p=0.5 seed
- [2026-07-17: NGU 0.75 seed3 (gz2ux8w0) also finished cleanly](experiment.md#2026-07-17-ngu-075-seed3-gz2ux8w0-also-finished-cleanly--same-false-alarm--stale-cache-pattern-as-seed2) — same benign-teardown-noise + stale-cache pattern as seed2, two-for-two now
- [2026-07-17: NGU 0.5 seed3 (zg0thiuz) finished cleanly, best_step peak shifted](experiment.md#2026-07-17-ngu-05-seed3-zg0thiuz-finished-cleanly-best_step-peak-shifted) — finished 2000/2000; true peak moved from step 1000 to step 1700 (combined AIME+BRUMO 0.2542 vs 0.2490) now that the full run is in
- [2026-07-17: NGU 0.75 seed3 swapped gz2ux8w0 -> cjr9kfxa (better on hard subset)](experiment.md#2026-07-17-ngu-075-seed3-swapped-gz2ux8w0---cjr9kfxa-better-on-hard-subset) — cjr9kfxa (KL beta=0.001 variant) beats gz2ux8w0, the worst-on-hard registered p=0.75 seed (0.0435 vs 0.0156); still early (1100/2000), best_step will need refreshing
- [2026-07-23: K ablation, n16_k8 p=0.875 + n32_k4 p=0.9375](experiment.md#2026-07-23-k-ablation-holding-n1-p-and-nk-fixed-n16_k8-p0875-n32_k4-p09375) — new sub-question below (`K` ablation holding `N*(1-p)` and `N*K` fixed), just launched, no results yet
- [2026-07-23: K ablation seeds 2 & 3](experiment.md#2026-07-23-k-ablation-seeds-2--3-n16_k8-p0875-n32_k4-p09375) — 2 more seeds each, both configs now at 3 seeds total

**Findings:** p=0.5 was the leading candidate after the first sweep (drove
the decision to replicate it over 4 seeds and bracket it with p=0.6/0.75).

**grad_norm=5.0 result (early-stopping best-checkpoint, held-out AIME/BRUMO/HMMT):**
all configs overfit past ~step 1000 at this grad_norm, so comparison uses
each seed's best in-training-AIME checkpoint (table in
[Best-step held-out evals](experiment.md#best-step-held-out-evals-brumo--hmmt--aime-2025)).
Breaking the improvement-over-`initial_eval` pass@1 down by problem
difficulty (analysis in `notebooks/deepscaler_ngu_dapo.ipynb`):

![NGU p sweep vs. baselines, pass@1 improvement by difficulty](notebooks/figures/deepscaler_ngu_pass_at_1_improvement_by_difficulty.png)

- **Hard (n=17, initial pass@64=0):** NGU does *not* help here — if anything
  the `N=2,K=64` baseline is slightly ahead of every NGU `p`. This is the
  subset we most want to move and currently aren't.
- **Medium (n=6):** NGU clearly wins — all three `p` values beat all three
  baselines by a solid margin (~19–22% vs ~14–19%).
- **Easy (n=7):** all configs converge to roughly the same ~37–40%
  improvement; NGU doesn't hurt here but doesn't help either.

Net effect: NGU's overall edge over baseline is real but small, and it's
coming entirely from the medium tier, not the hard tier — the opposite of
what we want given the goal is to move problems the model currently can't
solve at all. Open question this is meant to inform: does the grad_norm=1.0
rerun (see below) change this picture, or is the medium-only effect a real
property of NGU as configured?

**Blocking infra bug (found + fixed):** NGU relies on per-prompt bookkeeping
in the `DataPreparationActor`. All 4 FSDP ranks were independently
snapshotting/restoring this actor's state on checkpoint save/resume, so a
resume could load an inconsistent state (whichever rank's `set_state` ran
last). Fixed in `7c8919d2f` to save/restore from global rank 0 only; see
[NGU relaunch with rank-0 state-saving fix](experiment.md#ngu-relaunch-with-rank-0-state-saving-fix--async_steps-2).
Any NGU conclusions drawn from runs *before* this fix are suspect.

**Sub-question ([[ngu-k-ablation]]): does NGU's edge hold as `K` shrinks?**
All NGU results above are basically one point on a curve: `n8_k16, p=0.75`
gives revisited-prompt fraction `N*(1-p) = 8*0.25 = 2`. Ablating `K` while
holding both `N*K=128` (the usual completions/step budget) and `N*(1-p)=2`
fixed asks whether the effect is robust to group size, or an artifact of
`k=16`. Two new points launched 2026-07-23: `n16_k8, p=0.875`
(`16*0.125=2`) and `n32_k4, p=0.9375` (`32*0.0625=2`) — see
[K ablation runs](experiment.md#2026-07-23-k-ablation-holding-n1-p-and-nk-fixed-n16_k8-p0875-n32_k4-p09375).
`n32_k4` is a brand-new `n×k` config with no baseline yet (only the NGU run
was launched — see that experiment.md entry for why). No results yet.

---

## [ACTIVE] NGU sequence continuation (`ngu_seq_multiplier`): resume unfinished completions instead of discarding them

**Question:** NGU currently treats "unfinished" (hit the `response_length`
cap) the same as "wrong" — truncated completions score 0 and get thrown away
on a retry. New `--ngu_seq_multiplier M` resumes those completions on the NGU
retry (partial response re-fed as prompt, tokens/masks/logprobs carried
through), granting `response_length` more tokens per retry up to
`response_length * M` total. M=2 at 8k is like a 16k budget with a halfway
NGU check-in. Does that beat just training at 16k outright — same max
sequence length, but cheaper rollouts on prompts that get solved (or dropped)
early?

**Implementation** (commit `84b617078`): `ContinuationPrefix` payloads ride
the `PromptRequest`; the vLLM actor seeds resumed sub-requests with the
prefix; `maybe_filter_group` continues only `finish_reason == "length"`
completions below the cap, requeues fresh samples for finished-wrong ones,
and keeps continued completions out of the NGU pending buffer/baseline so
they aren't double-counted when the stitched version returns. vLLM
`max_model_len`, trainer `max_sequence_length`, and the `pack_length` check
all scale by the multiplier.

**Runs:** [NGU sequence-continuation: 8k×2 vs plain 16k](experiment.md#ngu-sequence-continuation---ngu_seq_multiplier-8k2-vs-plain-16k)
— both arms n=8, k=16, NGU p=0.75, gradnorm 1.0, async_steps 2, seed 1,
16k eval budget. Watch the continuation arm's `val/stop_rate` and
`batch/filtered_prompts_*`: the rho_weight-collapse risk (see below) is
driven by truncation-heavy batches, and continuations should *reduce*
effective truncation — but also produce very long merged completions.

**Status (2026-07-20):** both arms launched and running past step 32/2000
on Beaker (`2k_ngu075_mult2x8k...` at
[01KY05JXM1NEJWD9T9FJHW96J7](https://beaker.org/ex/01KY05JXM1NEJWD9T9FJHW96J7),
`2k_ngu075_seq16k...` at
[01KXZ3J985XXE89MTGG2BQSTXF](https://beaker.org/ex/01KXZ3J985XXE89MTGG2BQSTXF)).
Getting the continuation arm running required two follow-up fixes beyond the
initial implementation — both are utilization-metrics (wandb MFU/MBU
logging) bugs with no effect on training correctness, but both were crash
bugs that killed the run: NGU continuations can make an accepted merge
round finalize fewer than `num_samples_per_prompt_rollout` responses (the
rest get deferred to a later round), breaking an implicit "sample_count is
always a multiple of samples_per_prompt" assumption baked into the
FLOPs/MFU accounting. Fixed once in `open_instruct/utils.py`
(commit `62baca0c5`), then again in `grpo_callbacks.py` (commit
`49a043644`) once the live Beaker run revealed the OLMo-core backend
(`OC=true`, what these experiment arms actually use) has a *separate*
`calculate_utilization_metrics` call site that the first fix missed. Full
root-cause writeup in the experiment.md smoke-test section. Training
results TBD — check Beaker/wandb before treating this as resolved.

A third arm (`2k_baseline_dapo_n8_k16_gradnorm1_async2_seed1_16k`, no-NGU,
same 16k ceiling) was added to isolate whether NGU helps at all vs. plain
16k. It hit a genuine CUDA OOM at step 290/2000 (`--activation_memory_budget`
0.5 too tight for `fsdp_shard_degree 4` at `pack_length 18432`) — fixed by
lowering the budget to 0.25 (commit-free CLI override, no code change
needed); confirmed fixed on relaunch
[01KY0T45SY5EF75X6PWC617340](https://beaker.org/ex/01KY0T45SY5EF75X6PWC617340).
Full root-cause writeup in the experiment.md "Arm 3 OOM root-cause note".
Same relaunch then showed the exact stall signature this entry already
flags as a risk (line above): step throughput collapsed ~1.9→0.16 steps/min
right around step 290-305, with eval `sequence_lengths` mean jumping
1575→7005 tokens and `stop_rate` dropping 0.97→0.81 — matching the
[rho_weight collapse](#rho_weight-collapse-under-grad_norm10-n4_k32-watch-n2_k64-too-root-cause--partial-fix)
pattern. Not yet confirmed as the same collapse (no direct `val/rho_weight`
in stdout logs), so treat as a watch item, not a resolved finding.

**Findings:** TBD (smoke test + both arms launching as of 2026-07-20).

---

## [ACTIVE] max_grad_norm: 5.0 vs 1.0 — fixing the overfitting problem

**Motivation:** at `max_grad_norm=5.0`, AIME eval performance degrades after
~step 1000 across configs ("overfitting"). This confounds the NGU-vs-baseline
comparison above — early-stopping to the best checkpoint recovers a small,
hard-tier-unfavorable NGU edge, but it's not clear how much of that shape is
real vs. an artifact of training past the point of instability. Switching to
`max_grad_norm=1.0` to see if it removes the degradation and gives a cleaner
comparison, especially on the hard tier.

**Runs:** [Additional seeds with grad norm 1.0](experiment.md#additional-seeds-with-grad-norm-10)
(one extra seed per baseline config except n16k8, plus one NGU seed at each
of p=0.5/0.75/0.875).

**Findings so far:** grad_norm=1.0 didn't itself cause the n4_k32 collapse
(see the rho_weight-collapse entry below) — but it's the config where that
failure mode first surfaced, and it may recur on `N=2,K=64`. Clean
overfitting/AIME-degradation comparison against grad_norm=5.0, and the
difficulty-stratified breakdown at grad_norm=1.0, not finished yet — that's
the open question this sweep is for.

---

## [ACTIVE] rho_weight collapse under grad_norm=1.0 (n4_k32, watch n2_k64 too): root cause + partial fix

**Question:** `2k_baseline_dapo_n4_k32_gradnorm1_seed1` entered an
all-zero-reward filtering spin around step ~547 and crawled (13 steps/9h) —
why, and is it config-specific or a general async off-policy risk? Same
`val/rho_weight` 1.0→0.9-over-~20-steps signature is the thing to watch for
on the `N=2,K=64` baseline too, given both configs share the
"few-prompts-many-samples-each" shape that drives long completions.

**Runs:** [n4_k32 gradnorm1 stall → rerun with async_steps 2](experiment.md#n4_k32-gradnorm1-stall--rerun-with-async_steps-2), [3 more n4_k32 seeds at async4 to test reproducibility](experiment.md#2026-07-13-2-more-seeds-n8_k16-n2_k64--3-more-seeds-n4_k32-all-async_steps4-default) (plus 2 more seeds each at n8_k16/n2_k64, which ran clean at async4, as a control)

**Finding (confirmed via log + wandb analysis):** completion lengths drifted
up to the 8192 cap over the run; `val/rho_weight` declined 1.0→0.9 over steps
~535–546 as long generations slowed down async collection (growing
off-policyness, negative advantage dominating), then snapped back to 1.0
post-collapse (steps became rare ⇒ on-policy again) — so trainer and vLLM
weights agreed post-collapse, ruling out weight-sync/optimizer corruption.
Mechanism: the completion-length distribution crossed the 8192 truncation
cliff, so ~60% of completions came back unfinished ⇒ nearly all groups had
all-zero reward ⇒ active-sampling filtered everything out, and the loop
starved. **Fix (partial):** reduce `--async_steps` 4→2 to bound the
off-policy feedback loop. Applied to the n4_k32 rerun and (preventatively)
to all later NGU gradnorm1 relaunches. This mitigates the loop but isn't
considered a full fix — it bounds staleness rather than addressing why
completion lengths drift toward the cap in the first place, so recurrence
(e.g. on `N=2,K=64`) is still possible.

**Implication:** any future run with growing completion lengths near the
truncation cap + `--active_sampling` is at risk of this collapse; watch
`val/rho_weight` and `batch/filtered_prompts_solved` as leading indicators.

**2026-07-13:** this is also diagnosed as the cause of the OC=true NGU runs
going off the rails more broadly (not just the one n4_k32 stall) — completion
length exploding toward the response-length cap. Two things launched to
attack it from different angles: the [OC=false NGU
sweep](experiment.md#ngu-gradnorm10-p-sweep-ocfalse-grpo_fastpy) (same
config, different backend — DeepSpeed instead of FSDP/OLMo-core, on the
hypothesis that `grpo_fast.py` doesn't have whatever's driving the length
growth on the OC=true path), and the [12k response-length
run](experiment.md#ngu-0875-12k-response-length-octrue) (same OC=true
backend, more headroom before hitting the truncation cliff). Watch completion
length trajectories and `val/rho_weight` on both to see which one (if either)
actually avoids the collapse — that will also say whether the root cause is
backend-specific or purely a function of response-length headroom.

---

## [PAUSED] Holmes (B300) cluster compatibility

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

**Status (2026-07-12): giving up on this for now.** The three fixes above are
committed and kept (they're harmless elsewhere), but we're not pursuing the
third launch or further B300 validation — back to running on the usual weka
clusters. Reopen from the `TBD` row in experiment.md if holmes capacity
becomes worth it again.

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
requesting full-node (`NUM_GPUS=8`) for the affected jobs. Concrete output so
far: the difficulty-stratified pass@1-improvement comparison in the NGU entry
above (grad_norm=5.0, best checkpoint per seed). Full cross-config numeric
table not yet written up here — numbers live in the per-config
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

---

## [ACTIVE] Difficulty-bucket eval noise: does 128 samples/prompt change the hard/medium/easy split vs. 64?

**Question:** the hard/medium/easy difficulty buckets used throughout the NGU
analysis (see [Never-give-up exploration bonus](#active-never-give-up-ngu-exploration-bonus-does-it-beat-baseline-dapo-and-whats-the-best-p))
come from a single initial-model solve-rate eval at 64 samples/prompt
(`w47m67sf`). At 64 samples, a "hard" (solve_rate==0) label is only accurate
to within 1/64 ≈ 1.6%; some borderline prompts could be misbucketed. Does
doubling to 128 samples/prompt change which prompts land in which bucket, and
does that shift any of the per-difficulty findings above?

**Runs:** [New initial-model difficulty eval, n=128 samples/prompt](experiment.md#2026-07-16-new-initial-model-difficulty-eval-n128-samplesprompt-replaces-w47m67sf) — replaces `w47m67sf` as `notebooks/deepscaler_ngu_plots.ipynb`'s `DIFFICULTY_RUN_ID`

**Findings:** yes, the split moved. Bucket counts (hard/medium/easy):

| | 64-sample (`w47m67sf`) | 128-sample (`79ol8lss`) |
| --- | --- | --- |
| AIME | 17 / 6 / 7 | 15 / 7 / 8 |
| BRUMO | 16 / 7 / 7 | 13 / 8 / 9 |
| Combined | 33 / 13 / 14 | 28 / 15 / 17 |

5 combined prompts moved out of "hard" (solve_rate==0 at 64 samples but >0 at
128) into medium/easy — confirms the 64-sample noise concern: some
zero-solve-rate labels were false negatives from too few samples, not truly
unsolved. `notebooks/deepscaler_ngu_plots.ipynb` now uses the 128-sample
split (`DIFFICULTY_RUN_ID = "79ol8lss"`); the difficulty-bucket plots/tables
should be treated as updated from any earlier screenshots/numbers taken
against the old 64-sample buckets.

Also worth noting for future eval-only launches: `w47m67sf` was launched
months ago on `grpo_fast.py`'s (now-removed) `--eval_only` support and used
`--async_steps 1`/`--eval_temperature 0.7`, neither of which work on the
current codebase — see the launch log for the two parse-time fixes needed
(`--eval_only` now lives only in `open_instruct/grpo.py`/OC=true;
`--eval_temperature` was removed, use `--temperature` instead).

---

## [ACTIVE] Reinforce-Ada-Seq baseline: NGU=1.0 + async_steps=1 (no active_sampling) vs. the rest of the NGU sweep

**Question:** how does the "always retry, minimal pipelining" extreme
(`--never_give_up 1.0`, `--async_steps 1`, `--active_sampling` off) compare
to the rest of the `p`-sweep above (`p` ∈ {0.5, 0.6, 0.75, 0.875, 0.9375},
all with `active_sampling` on and `async_steps` 2–4)? `active_sampling`
can't stay on at `async_steps=1` — `data_loader.py`'s config asserts
`async_steps > 1` whenever `active_sampling` is set, so this variant
necessarily drops active_sampling's within-batch resampling and relies
purely on NGU's cross-step retry to keep hard prompts in the mix.

**Runs:** [Reinforce-Ada-Seq baseline — NGU=1.0, async_steps=1, no active_sampling (n8_k16, 3 seeds)](experiment.md#2026-07-24-reinforce-ada-seq-baseline--ngu10-async_steps1-no-active_sampling-n8_k16-3-seeds)

**Status (2026-07-24):** 3 seeds just launched, no results yet.

## [ACTIVE] `reinforce_ada_est`: adaptive completions-per-prompt from pre-computed pass_count

**Question:** instead of a fixed `num_samples_per_prompt_rollout` for every
prompt, can we spend the sampling budget where it matters by sampling more
completions for prompts the base model rarely solves and fewer for prompts
it solves often? Uses the pre-computed `pass_count` column (correct-out-of-32
from a prior base-model rollout) already present in
`mnoukhov/deepscaler-10k-qwen3-4b-base-32samples-quartiles`: pass_count >= 8
-> 4 samples, >= 4 -> 8, >= 2 -> 16, else (0 or 1) -> 32. The per-prompt count
is a static property of the prompt (from `pass_count`, which is itself never
updated during training) — no retry/requeue machinery like NGU.

**Implementation:** new `--reinforce_ada_est` bool on `StreamingDataLoaderConfig`
(`open_instruct/data_loader.py`). vLLM/`PromptRequest` already carry `n` per
request and GRPO's grouped-advantage code already groups by a per-prompt
`sample_count` list, so the only real gap was request construction:
`add_prompt_to_generator` now looks up `pass_count` on the dataset row and
overrides that request's `generation_config.n` via `dataclasses.replace`
(bucketing logic in `compute_reinforce_ada_est_samples`,
`open_instruct/data_loader_utils.py`); `process_group`'s response-count
assert checks against the bucketed count instead of the global config `n`
when the flag is set. Requires `batch_by="prompts"` (accumulation just waits
for `num_unique_prompts_rollout` finished groups regardless of each group's
size) and is mutually exclusive with `never_give_up` (untested combination,
rejected at config validation). Batch-size/pool/episode-count estimates
elsewhere in the pipeline still use the nominal `num_samples_per_prompt_rollout`
as an approximate average for sizing purposes only (same pre-existing
approximation NGU's variable group sizes already rely on) — not exact
accounting, but not needed for correctness either.

**Runs:** [reinforce_ada_est implementation + launch](experiment.md#2026-07-24-reinforce_ada_est-implementation--3-seed-launch-grpopy-oc) — 2-GPU smoke test then 3 seeds on `open_instruct/grpo.py` (OC=true only, per explicit request — not tested on `grpo_fast.py`).

**Findings:** Implementation confirmed working end-to-end on `open_instruct/grpo.py`
(OC=true): a local 2-GPU smoke test completed 2/2 training steps with
`pass_count`-driven per-prompt `n` flowing correctly through
`add_prompt_to_generator` -> `process_group` -> `accumulate_inference_batches`
(including under `active_sampling`'s zero-variance filtering), no assert
failures. Along the way, found and fixed an unrelated latent bug in the debug
script: single-learner-GPU configs need `--single_gpu_mode True` or the model
never gets cast to bf16 (FlashAttention dtype crash) — doesn't affect the
production launch, which uses 4 learner GPUs. `test_grpo_fast.py` (22 passed,
1 skipped) shows no regression. 3-seed production launch is now running on
`ai2/jupiter` (links in experiment.md); training-outcome/convergence findings
still TBD pending run progress.

---

## [ACTIVE] DeepCoder-1.5B: reproduce + K/NGU sweep on code RLVR

**Question:** does the K-ablation / NGU sweep methodology developed on
DeepScaleR (math) generalize to a code RLVR domain? Reproduces DeepCoder's
training setup (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` on
`agentica-org/DeepCoder-Preview-Dataset`, converted to open-instruct's
`code_stdio` RLVR format) as the base config, then runs the same single-seed
K∈{16,32,64} baseline + NGU p∈{0.5,0.75,0.875} sweep structure used on
deepscaler.

**Runs:** [DeepCoder-1.5B data pipeline + K/NGU sweep launch](experiment.md#2026-07-25-deepcoder-15b-data-pipeline--kngu-sweep-launch-grpopy-oc)

**Status (2026-07-25):** All 6 sweep jobs launched and confirmed training
cleanly. Getting here required fixing three real infrastructure bugs
surfaced by the sanity-check launch (all in shared `open_instruct` code, not
DeepCoder-specific):

1. `deepcoder_1_5b.sh`'s `fsdp_shard_degree` didn't match its
   `num_learners_per_node` (copy-paste from a template with different
   learner count) — `grpo_utils.py` requires the product to equal total
   learner GPUs.
2. **`open_instruct/grpo.py`'s OLMo-core backend had no support at all for
   Qwen2-architecture checkpoints** (only OLMo-2/3 and Qwen3 families were
   registered in `olmo_core_utils.OLMO_MODEL_CONFIG_MAP`), so
   `DeepSeek-R1-Distill-Qwen-1.5B` (HF `model_type="qwen2"`) couldn't be
   used with `grpo.py` at all before this session. Added a full Qwen2 preset
   plus HF↔olmo-core conversion mappings in both directions (verified
   bit-exact end-to-end). This is a reusable, general capability — any future
   Qwen2/2.5-family checkpoint (not just this one) can now run on
   `grpo.py`/OLMo-core.
3. The separate vLLM weight-sync path (used for pushing trained weights to
   the inference engines) had its own independent name-mapping table with
   the same Qwen2 gap (missing bias entries), fixed similarly.

**Data pipeline finding:** `create_deepcoder_data.py`'s initial conversion
crashed on Arrow storage limits and would have also failed at the
code-execution API's payload limit (a handful of LiveCodeBench-v5 stress
tests run multi-MB each). Fixed by capping to the largest ~15 tests per
problem within a 500KB budget — this also happens to match DeepCoder's own
published recipe of sampling "the 15 most challenging tests" per problem, so
the fix is a correctness improvement, not just a workaround.

**Status (2026-07-25, update):** Added `--eval_temperature` (mirrors
`--eval_top_p`) to decouple eval sampling temperature from training
temperature; set to 0.6 for this sweep. Launched 2 more seeds (2, 3) per
config — all 6 arms now have 3 seeds each (18 runs total) — on
`ai2/oe-adapt-code` at `high` priority (moved off `ai2/olmo-instruct`). See
[the seed2/3 launch subsection](experiment.md#--eval_temperature--2-more-seeds-per-config).
All 12 new jobs confirmed scheduled/started with no exit codes at launch
time.

**Next:** let the sweep run, then repeat the difficulty-stratified
best-checkpoint comparison methodology from the DeepScaleR NGU work (see
the K-ablation and NGU entries above) on code eval sets (LCB-v5 test,
Codeforces test) once these runs have made meaningful progress.
