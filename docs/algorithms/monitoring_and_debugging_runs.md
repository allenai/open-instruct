# Monitoring and debugging ML/RL training runs

A practical, opinionated field guide to watching a training run in Weights & Biases (wandb), knowing whether it's healthy, and deciding what to change. It is written for someone who knows the theory of GRPO/RL but hasn't spent much time staring at live dashboards, so it spends time on *why* each metric exists and *what shape* it should have.

It is grounded in the metrics this repo actually logs (the exact wandb keys appear in [Part 9](#part-9-this-codebases-exact-metric-glossary)), with a focus on **GRPO** and on **RL where the reward comes from an executable environment / agent** (tool use, sandboxes, test execution). It cross-references:

- [terminal_rl_trajectory_analysis.md](terminal_rl_trajectory_analysis.md) — the hands-on recipe + ready-to-run scripts for analyzing a Terminal-RL run (wandb metrics + truncation-vs-genuine failure analysis)
- [grpo_pipeline_overview.md](grpo_pipeline_overview.md) — plain-language pipeline
- [grpo_fast_internals.md](grpo_fast_internals.md) — Ray actors, async pipeline, loss/KL/advantage math
- [rollout_loop_internals.md](rollout_loop_internals.md) — token-level multi-turn rollout detail
- [terminal_rl_walkthrough.md](terminal_rl_walkthrough.md) — end-to-end story of one Terminal-RL step (who computes which logprobs, where mismatch comes from, how async is layered on)
- [rl_with_environments.md](rl_with_environments.md), [tool_training.md](tool_training.md) — environments & tools
- [tmax_4b_script_reference.md](tmax_4b_script_reference.md) — an annotated agentic-RL script

> **Last audited 2026-09-18** against the metric keys emitted by `grpo_fast.py` / `data_loader.py` / `grpo_utils.py` on `omni_agent` and against a live 9B DPPO Terminal run (`oe-general-agents/g55rr33t`: `loss_fn=dppo`, `async_steps=4`, `use_vllm_logprobs`, `beta=0`, liger loss, SP=4). Keys marked **(conditional)** in [Part 9](#part-9-this-codebases-exact-metric-glossary) only appear under the stated flag; if you cannot find a key on your dashboard, check there first.

---

## Part 0 — The mental model: what you are actually watching

### What "training" is doing, mechanically

Every training step is the same three-beat loop:

1. **Produce data.** In SFT this is a fixed dataset batch. In RL it's *generated on the fly*: the current policy rolls out responses (for agents, multi-turn tool-using trajectories), and an environment/verifier assigns each a **reward**.
2. **Compute a loss** — a single scalar that says "how wrong was the model on this batch," defined so that *lowering it* moves the model in the direction you want.
3. **Backprop + optimizer step** — compute the gradient of the loss w.r.t. every weight, then nudge the weights a tiny amount (scaled by the learning rate) in the descent direction.

Repeat millions of times. **Everything you monitor is a measurement of either the inputs to this loop (data/reward), the internals (loss/gradient/policy stats), or the outputs (model behavior over time).**

### The three questions every metric answers

When you look at any panel, you're trying to answer one of three things. It helps to consciously label which:

1. **Is it learning?** — Is the thing I actually care about (reward, solve rate, eval score) improving? *Lagging* indicator: tells you the outcome, slowly.
2. **Is it healthy / stable?** — Will it keep learning, or is it about to diverge, collapse, or NaN? (gradient norm, KL, entropy, loss scale, ratio/clip). *Leading* indicators: tell you about trouble **before** reward craters.
3. **Is it efficient?** — Am I wasting GPU-hours? (tokens/sec, MFU, step time, idle time, dropped/stale rollouts). Doesn't affect correctness, affects your wall-clock and budget.

A run can be "learning" and "unhealthy" at the same time (reward climbing while entropy quietly collapses → it'll plateau and you won't know why unless you watched the leading indicator). The whole point of monitoring is to catch the health problem *before* the outcome metric tells you.

### Leading vs lagging — the single most useful framing

- **Reward / eval score is lagging.** By the time it's visibly bad, the damage happened many steps ago.
- **Entropy, KL, gradient norm, clip fraction, ratio are leading.** They wobble first. Train yourself to read these as an early-warning system, not the reward curve alone.

---

## Part 1 — Refresher on the machinery (so the metrics make sense)

You said you're rusty on fundamentals, so here's the minimum mechanical vocabulary. Each item reappears as a metric later.

### Loss

A scalar. Lower = "the model fit this batch better, *by the definition of the loss*." Crucial caveat for RL: **the loss value itself is meaningless in policy-gradient methods** — it's an engineering quantity whose *gradient* is what matters, not its level (more in [Part 4](#policy-loss-why-its-value-is-meaningless)). In SFT, loss level *is* meaningful (it's the negative log-likelihood; lower genuinely = better fit).

### Gradient and gradient norm

The gradient is the vector of partial derivatives of the loss w.r.t. every parameter — it points "uphill," so we step the opposite way. The **gradient norm** is the length of that vector (`sqrt(sum of squares of all gradient elements)`). It's a single number summarizing "how big a step are we about to take."

- **Why you log it:** It's the best single leading indicator of instability. A healthy run has a roughly stable grad norm (maybe slowly drifting). A **spike** means the model hit a batch that produced a huge update — often the precursor to divergence/NaN.
- **Gradient clipping:** to prevent one bad batch from blowing up the model, frameworks clip the gradient norm to a max value (e.g. 1.0). If the raw norm exceeds the cap, the whole gradient is scaled down. If your logged grad norm is pinned exactly at the clip value constantly, clipping is firing every step (the model "wants" bigger steps than you allow) — usually a sign the LR is too high or the data is too noisy.

### Learning rate (LR) and schedules

The LR is the scalar multiplier on each step: `new_weights = old_weights - lr * gradient`. Too high → unstable / divergent / it overshoots minima. Too low → painfully slow or stuck.

- **Warmup:** start LR near 0 and ramp up over the first N steps. Why: at the very start, the optimizer's internal statistics (Adam's running averages) are garbage and the model is far from anything good, so big steps are dangerous. Warmup avoids early blow-ups.
- **Schedule:** `constant`, `linear` decay, `cosine` decay, etc. Decaying the LR late in training lets the model settle into a minimum instead of bouncing around it.
- **Why you log it:** to confirm the schedule is doing what you set (e.g. warmup actually ramping), and because a tiny printed `lr` (like `1e-6`) can *look* like `0.0` in a rounded display and panic you needlessly.

### Batch size, gradient accumulation, global batch

- **Per-device batch size:** how many sequences one GPU processes before… nothing yet.
- **Gradient accumulation:** run several micro-batches, sum their gradients, *then* step. Lets you simulate a big batch on limited memory.
- **Global / effective batch size** = per_device × num_devices × grad_accum. This is the number that actually matters statistically. Bigger batch = lower-variance gradient = you can usually afford a higher LR, but each step costs more.
- **In RL specifically** the "batch" is built from *prompts × samples-per-prompt*. In GRPO you generate K samples per prompt and the **group** of K is what advantages are computed over (see [Part 3](#grpo-in-one-paragraph)).

### Tokens, sequence length, packing

LLMs process *tokens*. Two sequences of different lengths waste compute if you pad the short one to match the long one. **Packing** concatenates multiple sequences into one fixed-length block (e.g. 35840 tokens) to eliminate padding waste. Why you care when monitoring: `packed_ratio`, `sequence_lengths`, and throughput all interact — a run that suddenly generates much longer responses will pack fewer sequences per block and slow down.

### Gradient checkpointing (you flagged this one)

Normally, to compute gradients, the framework must keep every intermediate activation from the forward pass in memory (you need them for the backward pass). For a big model with long sequences this is enormous. **Gradient checkpointing** trades compute for memory: it *throws away* most activations during the forward pass and *recomputes* them on demand during the backward pass. Result: ~30% slower per step, but you can fit much bigger models / longer sequences / bigger batches. When you see `--gradient_checkpointing` in a script and wonder why throughput is lower than you'd expect — this is why, and it's usually a deliberate, correct trade.

### Mixed precision, and DeepSpeed / ZeRO stages

- **Mixed precision (bf16/fp16):** do math in 16-bit to halve memory and roughly double throughput. bf16 (brain-float) is preferred over fp16 because it has the same exponent range as fp32, so it rarely overflows/underflows — fewer NaN headaches.
- **ZeRO / DeepSpeed stages (1/2/3):** ways to shard the optimizer state, gradients, and finally the parameters themselves across GPUs so a model too big for one GPU fits across many. **Stage 3** (used in your scripts) shards *everything*, including parameters — maximum memory savings, more communication overhead. Relevant to monitoring because stage-3 runs are communication-heavy: a chunk of "step time" is GPUs talking to each other, and that shows up as lower MFU even when nothing is wrong. See [olmo_core_sharding.md](olmo_core_sharding.md).

### MFU (Model FLOPs Utilization)

The fraction of your GPUs' theoretical peak FLOPs you're actually using for useful model math. 100% is unreachable; large-model training often lands 30–55%. **Why log it:** it's your "is the hardware busy or twiddling thumbs" gauge. A *drop* in MFU mid-run means something started stalling (data starvation, comms, a slow node).

---

## Part 2 — The universal ML dashboard (applies to SFT, DPO, RL alike)

These are the panels you check on *any* run regardless of algorithm.

### Loss curve

- **SFT/DPO:** should trend down and flatten. A train loss that keeps dropping while validation loss rises = **overfitting**. A loss that's flat from step 0 = nothing is learning (bad LR, frozen weights, broken data, label mismatch).
- **NaN/Inf loss:** hard failure. Almost always (a) LR too high, (b) fp16 overflow (switch to bf16), (c) a bad batch (e.g. a sequence of all-padding), or (d) a division-by-zero in a custom reward/advantage. NaN is contagious — once weights go NaN, every subsequent step is NaN.
- **Read the shape, not the number.** Smooth descent = good. Sawtooth that trends down = fine (noise). Sudden cliff up = instability event — line it up against the grad-norm panel.

### Learning rate

Confirm warmup ramp and decay match your config. This is a "did my config take effect" check more than a diagnostic.

### Gradient norm

Your primary stability gauge (see Part 1). Watch for **spikes** and for **a slow upward creep** (often precedes divergence). Correlate spikes with loss spikes and with specific data.

### Throughput & efficiency

- **tokens/sec** (overall and per-step), **step time**, **MFU.**
- These should be roughly flat. A **downward trend or periodic dips** means stalls. In distributed runs the usual culprits are: one slow/preempted node, data loader starvation, checkpoint saving (look for a periodic spike every `save_freq` steps — that's expected), or network contention.
- **Why you care even if learning is fine:** at, say, 8 min/step, the difference between 35% and 50% MFU is days of wall-clock and real money.

### Memory / OOM

OOM (out-of-memory) kills the run. If you OOM intermittently it's usually because *sequence lengths grew* (RL models learn to generate longer → bigger activations) and you're near the edge. Levers: gradient checkpointing, smaller per-device batch, shorter `pack_length`/`response_length`, more sharding.

### Determinism / reproducibility

Log the seed, git commit, and full config (this repo logs `GIT_COMMIT` and the arg set). When a run behaves weirdly, the first question is always "what changed since the last good run." If you can't answer that from the logged config, you're debugging blind.

---

## Part 3 — RL: why it's harder to monitor, and the conceptual layer

### Why RL monitoring is fundamentally harder than SFT

1. **The data is non-stationary.** The model generates its own training data. As the policy changes, the distribution of rollouts changes, which changes the rewards, which changes the gradients… It's a feedback loop, so it can spiral (good or bad) in ways SFT can't.
2. **The signal is sparse and noisy.** A reward of "did the tests pass" is one bit per *entire multi-thousand-token trajectory*. Estimating a good gradient from that is statistically hard; your metrics are correspondingly noisy.
3. **Reward ≠ what you want.** The model optimizes the reward *exactly as written*, including loopholes. "Reward went up" can mean "the model found a bug in your reward." (See [reward hacking](#55-reward-hacking-the-thing-that-looks-like-success).)
4. **There's an exploration/exploitation tension.** The model must keep trying varied things to discover better strategies, while also exploiting what already works. **Entropy** is the measurement of "how varied": it's a number for *how spread-out the model's next-token probability distribution is*. High entropy = the model spreads probability across many possible continuations, so sampling produces diverse rollouts (it's exploring). Low entropy = it piles almost all probability on one continuation, so every rollout looks the same (it's committed/greedy). The failure mode is **entropy collapse**: entropy falls toward zero early, the model stops exploring, and it locks into whatever mediocre strategy it had found — after which reward plateaus and *no* amount of further training helps, because there's no variety left to learn from. This is why entropy is a first-class RL metric and irrelevant in SFT (where the data is fixed and the model isn't generating its own).

### GRPO in one paragraph

For each prompt, sample a **group** of K responses. Score each. The **advantage** of a response = how much better its reward was than the group's average (normalized by the group's spread). Responses better than their group's average get their probability pushed up; worse-than-average get pushed down. There is no separate value network — the group average *is* the baseline. Then optionally clip the update (PPO-style) so no single step moves the policy too far, and optionally add a KL penalty to a reference model so it doesn't wander off into gibberish. See [grpo_fast_internals.md](grpo_fast_internals.md) for the exact math as implemented here.

Two consequences for monitoring fall straight out of this:

- **If every response in a group gets the same reward, the advantage is exactly zero → that group contributes no gradient.** This is why "group performance," "zero-std filtering," and "fraction of useful groups" are things you track. An all-solved or all-failed prompt teaches the model nothing.
- **The reward's *spread within a group* matters more than its absolute level.** A prompt where the model solves it 50% of the time is pure gold (max learning signal). A prompt it always solves or never solves is dead weight.

### The five vital signs of any policy-gradient RL run

Memorize these five. Almost every RL diagnosis is some combination:

| Vital sign | What it measures | Failure mode it catches |
|---|---|---|
| **Reward / solve rate** | Are we getting better at the task? | Not learning (lagging) |
| **KL divergence** (to reference and to previous policy) | How far the policy has moved | Collapse (wandered into garbage) / frozen (not moving) |
| **Entropy** | How much randomness/exploration remains | Premature convergence (the #1 silent RL killer) |
| **Advantage spread (std)** | Is there a usable gradient signal at all | Dead signal (all-zero advantages) |
| **Policy ratio + clip fraction** | How aggressive each update is | Instability / off-policy drift |

> **Note on this repo:** entropy **is** available as `policy/entropy_avg`, but only with `--record_entropy` (default off), and it is **incompatible with `--use_liger_grpo_loss`** (the config rejects the combination, because the tiled loss never materializes the full logits the entropy needs). The 9B/27B Terminal runs use liger, so they do not have it. Without it, watch the *symptoms*: collapsing `val/avg_group_performance_pre_filter`, `unsolved_batch_size_ratio` drifting to 0, solved/unsolved sequence-length histograms converging, `debug/dppo_mask_frac_kept` dropping (DPPO), and reward plateauing. If you can afford the non-liger path on a small run, turn `--record_entropy` on: it is the cleanest leading indicator there is.

---

## Part 4 — The RL vital-signs dashboard, in depth

For each metric: *what it is → why log it → healthy shape → failure modes → lever.*

### Reward / score (`scores`, `objective/*reward`)

- **What:** the mean reward over the batch. In this repo `scores` is `raw_scores.mean()` — the actual environment reward before any length shaping.
- **Why:** it's the objective. Everything else is in service of moving this.
- **Healthy shape:** noisy but upward, *then* a plateau. Plateaus are normal and not necessarily bad — the model may have saturated the easy gains. The slope matters more than any single point; fit a trend over a window, don't eyeball two points.
- **Failure modes:**
  - **Flat from the start** → no signal reaching the model. Check: are advantages non-zero? Is the reward function actually returning varied values? Is the LR non-zero? Is the format such that the model can ever succeed?
  - **Climbs then collapses** → instability (often KL/ratio blew up) or reward hacking got patched, or a data/infra change. Line it up against KL, grad norm, clip fraction.
  - **Climbs but eval doesn't** → overfitting to the train prompts or reward hacking.
- **Critical subtlety — read the distribution, not the mean.** A mean reward of 0.5 can be "every rollout scores ~0.5" (continuous partial credit) or "half score 1.0, half score 0.0" (bimodal). These need *completely different* interventions. Always look at the reward **histogram** (`val/solve_rate_hist`, `val/advantages_hist`) alongside the mean.

### Solve rate / group performance (`val/avg_group_performance_pre_filter` / `_post_filter`)

- **What:** the fraction of prompts effectively solved, computed over groups. `pre_filter` is before active-sampling/zero-std filtering; `post_filter` is after.
- **Why:** it's the cleaner, lower-variance view of "are we learning" than the raw per-token reward mean, because it aggregates at the group level. In the run we looked at, *this* was the metric that clearly trended up (0.30 → 0.51) even while raw `scores` looked noisy and flat — **prefer it as your primary learning signal here.**
- **Healthy shape:** monotone-ish climb.
- **Failure mode:** `pre_filter` flat while you *think* it should move → genuinely not learning, or all your signal is in a few prompts the filter is dropping.
- **Watch pre vs post together:** if they diverge a lot, filtering is reshaping your effective training distribution (could be fine, could mean you're throwing away most of the batch — see `real_batch_size_ratio`).
- **Why this, and not the trained-batch solve rate, is the learning signal:** with `--active_sampling` the *trained* batch is filtered down to only the "contested" (mixed-outcome) prompts, which pins the post-filter/`scores` solve rate near ~50% no matter how much the model improves — like an adaptive tutor that keeps handing you problems you're getting half-right. Improvement shows up as prompts "graduating" out of that set into all-solved, which only the **pre_filter** number sees. Full walk-through with the analogy: [terminal_rl_trajectory_analysis.md → "Why the trace solve rate is flat"](terminal_rl_trajectory_analysis.md#why-the-trace-solve-rate-is-flat-and-why-that-is-not-its-not-learning).

### Advantage statistics (`val/advantages_mean/min/max`, `val/advantages_hist`)

- **What:** the per-token/-response advantage after group normalization.
- **Why log it:** this is your **"is there a gradient signal at all"** gauge. After group-centering, the **mean is ~0 by construction** — do *not* be alarmed that `advantages_mean ≈ 0`; that's correct, not a bug. What you care about is the **spread**: `min`/`max` and the histogram width.
- **Healthy shape:** a symmetric spread around zero with real width (e.g. ±0.5 to ±1.0). In the example run, advantages spanned ±0.875 every step — healthy.
- **Failure mode:** `min ≈ max ≈ 0` → advantages collapsed → **no learning signal** → reward will be flat. Causes: every group is all-solved or all-failed (task too easy/hard for the current policy), or zero-std filtering removed everything with signal, or normalization is misconfigured.

### KL divergence (`objective/kl0/1/2/3_avg`, `loss/kl_avg`)

**What KL divergence is, plainly:** a single number that measures *how different two probability distributions are*. Here the two distributions are the model's next-token probabilities now vs. some reference (the starting model, or the previous step's policy). KL = 0 means identical; the larger it grows, the more the model is putting its probability on *different* tokens than the reference did. (It's not a symmetric "distance" in the strict mathematical sense, but for monitoring just read it as "how far has the policy moved from where it started.") A model can wander arbitrarily far during RL — KL is the odometer.

**Two distinct uses, don't confuse them:**

1. **KL to the *reference/initial* model** — "how far have we wandered from the starting model." This is what the optional KL *penalty* (controlled by `beta`) pulls back on. Purpose: stop the policy from exploiting the reward by degenerating into fluent nonsense or repetitive hacks that no longer resemble a real model.
2. **KL to the *previous* policy / behavior policy** — "how big was this one update." This is the **trust region**: the idea that an update should only be trusted to move the policy a *small* amount, because the advantages you computed are only valid near the policy that generated the data — step too far in one update and you're optimizing against stale information. PPO/GRPO clipping (below) is the mechanism that enforces this limit.

This repo logs **four estimators** of KL (all functions of `Δ = new_logprobs − ref_logprobs`). Why four? You can't compute the true KL exactly from a handful of sampled tokens — you *estimate* it, and different formulas trade **bias** (a systematic offset — consistently reads a bit high or low) against **variance** (run-to-run noise — jumps around even when nothing changed). You want low-bias *and* low-variance, but you usually can't have both, so the repo logs several and you pick ([model_utils.py:`estimate_kl`](../../open_instruct/model_utils.py)):

- **`kl0`** — linear, `Δ = logp_new − logp_ref`. The raw mean log-ratio. **Can be negative** (it's not a true KL, it's the inside of one). Useful as a signed "which direction did we move."
- **`kl1`** — quadratic, `Δ²/2`. Always ≥ 0, grows as the policy diverges. In the example run this rose 0 → 0.09, cleanly showing "the policy is steadily moving away from base." Good drift tracker.
- **`kl2`** — the numerically-stable **k3 estimator**, `expm1(−Δ) + Δ`. Always ≥ 0, low variance, the repo's **preferred default**. Use this as your headline KL number.
- **`kl3`** — importance-weighted, `ratio · Δ`.

- **`loss/kl_avg`** is the KL *penalty term actually added to the loss* = `kl_estimator_value × beta`. **With `beta = 0.0` (your tmax/swerl scripts) this is identically 0** and KL is *not* constraining training at all — it's monitor-only. That's a deliberate choice (let the policy move freely, rely on clipping + low LR for stability), but it means **you alone are the KL safety check** — if KL runs away, nothing in the loss stops it. Watch it.

- **All five KL keys exist only because `load_ref_policy` defaults to `True`.** The ref forward is what produces them. If you pass `--load_ref_policy false` to save the ref's memory and the per-step ref forward (legitimate when `beta=0`), `objective/kl{0..3}_avg` and `loss/kl_avg` **disappear from the dashboard** and you lose your drift odometer. Decide which you value more before flipping it. Under liger only `kl2` enters the loss, but all four are still logged.

- **Healthy shape:** slow, steady growth (the policy *should* move away from base — that's the point). 
- **Failure modes:**
  - **KL explodes** (sharp upward, often with a grad-norm spike) → the policy is diverging; reward usually collapses right after. Lever: lower LR, turn on/raise `beta`, tighten the clip range.
  - **KL pinned at ~0 and reward flat** → policy isn't moving → effectively not training (LR too low, advantages dead, or updates being clipped/dropped).

### Policy ratio & clip fraction (`val/ratio`, `val/ratio_var`, `policy/clipfrac_avg`)

- **Ratio** = `exp(new_logprob − old_logprob)` per token = "how much more/less likely is this token under the policy being trained vs the policy that generated it." In PPO/DAPO the ratio is **clipped** to `[1−ε_l, 1+ε_h]` so a single update can't move any token's probability too far.
- **What `old_logprob` is depends on one flag.** With `--use_vllm_logprobs true` (every Terminal script) it is the logprob vLLM recorded while *sampling* the token, so the ratio measures the current trainer policy against the actual **behavior** policy — which includes both weight staleness from the async pipeline and the vLLM-vs-HF numerical mismatch. With it false and one mini-batch, `old_logprob` is the trainer's own detached logprob on the first pass, so the ratio is **identically 1** and the clip can never fire; the flag is not a nicety, it is what makes clipping/DPPO do anything at all.
- **`val/ratio`** is the token-weighted mean of the per-row mean ratio. ≈ 1.0 means on-policy on average (the sample run sits at 1.0000 to four decimals). **`val/ratio_var`** is the variance *across packed rows* of those per-row means, not the per-token variance — so it is tiny (1e-8 on the sample run) even when individual tokens have ratios of 0.5 or 2. A jump in `ratio_var` means some rows are systematically off-policy: a weight-sync bug, or a batch mixing very stale and very fresh rollouts. For per-token mismatch use the `debug/vllm_vs_local_*` metrics below.
- **`policy/clipfrac_avg`** = fraction of response tokens where the clipped surrogate was the active one (`pg_losses2 > pg_losses`). Under **DAPO** a few % is normal and healthy; >20–30% means the update wants to move far beyond the trust region → LR too high or data too stale.
- **Under DPPO, CISPO and TVPO this key is identically 0.** Those losses set `pg_losses2 = pg_losses` (no symmetric clip), so the comparison is never true. It is not "no clipping happened", it is "this metric does not apply". The DPPO equivalent is `debug/dppo_mask_frac_kept` (next section).

### Truncated importance sampling and trust-region masks (`val/tis_ratio`, `val/tis_clipfrac`, `debug/tis_mask_frac_kept`)

When rollouts are generated by vLLM but trained by the learner, the two computations of token probabilities differ slightly (different kernels, precision) and the weights may be a few steps stale. Several optional mechanisms compensate; each has its own metric, and each metric sits at 0 (or is absent) when the mechanism is off — **expected, not broken**:

| flag | mechanism | metric |
|---|---|---|
| `--truncated_importance_sampling_ratio_cap C` (0 = off) | multiply the loss by `min(π_old/π_vllm, C)`; only meaningful when `old ≠ vllm`, i.e. `use_vllm_logprobs=false` | `val/tis_ratio` (mean clamped weight), `val/tis_clipfrac` (fraction capped) |
| `--tis_mask_lower / --tis_mask_upper` (0 = off) | zero the gradient on tokens whose ratio leaves `[1−l, 1+u]` | `debug/tis_mask_frac_kept` |
| `--sequence_tis_mask_log_ratio_threshold` (0 = off) | zero whole rollouts whose mean log-ratio exceeds the threshold | folded into `debug/tis_mask_frac_kept` |

All the Terminal scripts leave these at 0 and rely on `use_vllm_logprobs` + the loss's own trust region instead.

### Train/inference mismatch (`debug/vllm_vs_local_logprob_diff_mean/max/std`, `debug/vllm_local_reverse_kl`)

- **What:** on every packed row the trainer compares its own logprob of each sampled token against the one vLLM recorded at sampling time. `_mean`/`_max`/`_std` are over `|log π_trainer − log π_vllm|` on response tokens; `vllm_local_reverse_kl` is `Σ π_vllm · (log π_vllm − log π_trainer)`. Under the liger loss these come from an extra no-grad forward, so they cost a little.
- **Why:** this is the *direct* measurement of the two mismatch sources the ratio has to absorb: kernel/precision differences and weight staleness. Everything in the ratio/clip/DPPO machinery is downstream of this number.
- **Healthy shape (sample 9B run, `async_steps=4`):** `_mean` ≈ 0.005–0.011 nats/token, `_max` ≈ 0.27–0.49 (a handful of tokens per 64k-token row disagree a lot; that is normal), `reverse_kl` ≈ 1e-4–6e-4. These should be **flat over training**. The fp32 LM-head patches on both sides (`patch_vllm_qwen3_5_lm_head_fp32`, `--lm_head_fp32`) exist to keep them there.
- **Failure modes:** a **step change** in `_mean` = something structural changed (a weight-sync silently failed, a vLLM backend flag differs from the trainer, prefix-cache corruption after an in-flight update). A **slow climb** that tracks `stale_results_dropped` or the `model_step` gap = staleness, not numerics. `_max` alone spiking on isolated steps is usually one degenerate row and can be ignored.
- **Related knob:** `--save_trainer_logprobs` (with `--save_traces`) dumps per-token trainer and vLLM logprobs to `rollouts_save_path` so you can find *which* tokens disagree offline.

### Loss-function-specific metrics (DAPO / DPPO / TVPO / CISPO)

`--loss_fn` changes which trust-region metric is meaningful. Read the row for your loss and ignore the others:

| `loss_fn` | trust region | metric to watch | dead metrics |
|---|---|---|---|
| `dapo` (default) | symmetric clip `[1−ε_l, 1+ε_h]` | `policy/clipfrac_avg` (few % healthy) | `debug/dppo_mask_frac_kept`, `debug/tvpo_mask_frac_kept`, `actor/ppo_tv` absent |
| `dppo` (Terminal 9B/27B) | per-token **mask**: zero the update on tokens where the Bernoulli TV/KL between `π_vllm` and `π_trainer` exceeds `dppo_divergence_threshold` *and* the update would push further away | `debug/dppo_mask_frac_kept` — fraction of response tokens that survived. Sample run: **0.9999**, i.e. the mask almost never fires at this staleness. A drop toward 0.99 or below is the DPPO early-warning that the policy is moving faster than the rollouts can follow; correlate with `kl1` and `vllm_vs_local_logprob_diff_mean`. | `policy/clipfrac_avg` ≡ 0 |
| `tvpo` | prompt-level TV bound; freeze whole prompts over threshold | `debug/tvpo_mask_frac_kept`, `actor/ppo_tv` (mean prompt-level TV) | `policy/clipfrac_avg` ≡ 0 |
| `cispo` | one-sided cap on the detached ratio | nothing dedicated; watch `val/ratio` and the mismatch metrics | `policy/clipfrac_avg` ≡ 0 |

DPPO requires `use_vllm_logprobs=true` (validated in config) so that the mask's anchor `μ` is the actual rollout policy. Its mask is computed twice on the liger path — once inside the tiled loss for the gradient and once outside for the logged fraction — so the metric is exact, not an approximation.

### Policy loss — why its value is meaningless

`loss/policy_avg` in policy gradient is **not** "how wrong the model is." It's `−(advantage × logprob)` summed up — a quantity engineered so its *gradient* equals the policy-gradient estimator. Its **level** drifts with the advantage scale and tells you almost nothing; do **not** try to read it like an SFT loss ("it's not going down, training is broken!"). What you actually watch is reward, KL, entropy, grad norm. The loss is a means, not a measurement. (`loss/total_avg` = policy loss + `beta`×KL; with beta=0 it's just the policy loss.)

### Gradient norm in RL (`optim/grad_norm`)

Same meaning as Part 1, but **noisier** in RL because the data is noisy. Healthy = stable with occasional bumps. **Spikes** here are more dangerous than in SFT because the feedback loop can amplify one bad update into a collapse. In the example run grad norm sat ~0.02 (very small) and stable — consistent with a small LR (1e-6) and a deliberately gentle, slow-but-safe regime. Small+stable = safe but slow; if you want faster learning and grad norms are tiny and stable, that's headroom to raise the LR.

### Response/sequence length dynamics (`val/sequence_lengths`, `_solved`, `_unsolved`, hists)

- **Why:** length is a behavioral fingerprint and a budget constraint. RL famously discovers "think longer → higher reward" and response lengths *grow* over training. That's often good (more reasoning) but has three failure modes:
  - **Length hacking:** the model pads with filler to game a length-correlated reward without solving anything. Tell-tale: length climbs but solve rate doesn't.
  - **Truncation wall:** responses grow into the `response_length` cap and get cut off → those rollouts can never succeed (see next metric).
  - **Collapse to too-short:** lengths shrink and the model stops reasoning / stops calling tools — usually entropy collapse.
- **Read solved vs unsolved separately.** In the example run, `sequence_lengths_unsolved ≈ 23k` was pressing the 32k cap while `solved ≈ 13k` — a strong hint that many failures are *running out of budget*, not being genuinely wrong.

---

## Part 5 — RL on executable environments / agents (the part that's special)

This is where agentic RL diverges sharply from "RLHF on a chat model," and where most of your debugging time will go. The rollout is no longer "model emits text"; it's a **multi-turn trajectory**: the model thinks, emits a tool call, an **environment executes it** (runs bash in a sandbox, runs tests, resets state), returns an observation, and this repeats until the model submits or runs out of budget. Reward comes from the environment (e.g. a verifier reading `/logs/verifier/reward.txt`). See [rollout_loop_internals.md](rollout_loop_internals.md) and [rl_with_environments.md](rl_with_environments.md).

This adds **four new categories of things that can go wrong**, none of which exist in plain RLHF:

### 5.1 — Where does the reward even come from? (and why `objective/verifiable_reward` can read 0.0)

There are (at least) two reward code paths in this repo:

1. The **verifier-function path** (`apply_verifiable_reward` in [ground_truth_utils.py](../../open_instruct/ground_truth_utils.py)) — runs ground-truth checkers and populates `objective/verifiable_reward`, `objective/verifiable_correct_rate`, etc.
2. The **environment path** — the environment's `StepResult.reward` (e.g. [swerl_vanillux_sandbox.py](../../open_instruct/environments/swerl_vanillux_sandbox.py) parsing a `[0,1]` reward from a file). This flows straight into `scores`/`raw_scores`.

**If your reward comes from the environment, `objective/verifiable_reward` and friends can be a flat 0.0 for the entire run — and that is NOT a bug.** It just means that code path isn't the one producing your reward. This is a real trap: it looks like "reward is zero, nothing is learning," but the live signal is in `scores` and `val/avg_group_performance_*`. **First thing to do on any new env: confirm which key carries your real reward, and ignore the dead ones.**

### 5.2 — Tool / environment behavior metrics

| Metric (this repo) | What it tells you |
|---|---|
| `tools/aggregate/avg_calls_per_rollout`, `tools/bash/avg_calls_per_rollout` | How many tool calls per trajectory. Tracks the agent's *strategy*. A healthy agent often *reduces* calls over training (gets more efficient) — but a sudden drop to ~0 means it stopped using tools (collapse / learned to give up). A blow-up to the max means it's flailing/looping. |
| `tools/aggregate/avg_runtime`, `tools/bash/avg_runtime` | Wall-clock per tool call. Spikes = the sandbox/environment is slow or overloaded — an *infra* signal, not a learning one, but it dominates your throughput. |
| `tools/aggregate/failure_rate`, `tools/bash/failure_rate` | Fraction of tool calls that errored **or timed out** (a per-call timeout is recorded as `success=False`, so it lands here; there is no separate timeout-rate key). **Distinguish two kinds:** (a) the *model* wrote a broken command (a non-zero exit code is *not* a failure — only a timeout or an infra error is), vs (b) the *infra* failed (container died, OOM, gateway 503). A creeping infra failure rate silently poisons training because failed executions usually mean zero reward through no fault of the policy. Sample run: 0.07–2%. |
| `tools/env_reset/failure_rate` **(conditional)** | Appears **only** when a sandbox reset exhausted its retries under `SWERL_RESET_FAILURE_ZERO_REWARD=1`: the rollout loop records a synthetic `env_reset` failure, ends the episode with reward 0, and that zero **enters the group's advantage computation**. If this key exists at all on your run, your environment provisioning is failing and some of your "negative examples" are infra noise. Successful resets are not counted anywhere, so there is no `tools/env_reset/avg_runtime`. |
| `tools/tool_call_format_error/*` **(conditional)** | Appears only with `tool_call_format_error_feedback=true` in the env config (off by default for the sandbox): counts turns where the model emitted no / a malformed tool call and was nudged instead of ended. |
| `env/<env_name>/<metric>` (e.g. `env/swerl_vanillux_sandbox/step_count`) | Whatever the environment's `get_metrics()` returns, averaged over the batch. The sandbox currently reports only `step_count` (turns used per episode), which for a bash-only env equals `tools/bash/avg_calls_per_rollout`. Sample run: ~27–33 of `max_steps=64`. Rising toward `max_steps` with flat reward = the agent is flailing / not submitting; add per-env metrics here if you want richer behaviour signals (Part 11). |

> **Why these matter so much:** in agentic RL, **infra failures masquerade as learning signal.** If 10% of sandboxes silently OOM and return reward 0, the model is being told "those trajectories were bad" when actually your cluster hiccuped. The model can't learn the task if the reward is dominated by infra noise. **Watch failure rates as obsessively as you watch reward.**

### 5.3 — Format / protocol adherence and "did the agent finish properly"

The model has to emit *syntactically valid tool calls* and *actually submit an answer*. Several metrics track this:

| Metric | Meaning & why you watch it |
|---|---|
| `val/stop_rate` | Fraction of completions that ended with a proper stop token (the model chose to finish) vs got cut off. Low/falling stop rate = model is rambling into the length cap instead of concluding. |
| `val/non_submitting_completion_fraction` | Fraction of trajectories that **never called submit / never produced a final answer**. These usually get zero reward. Early in training this is high (model hasn't learned the protocol); it should **fall** as the model learns to actually submit. If it stays high, your format/parser may be too strict, the system prompt unclear, or the budget too small to ever reach submission. |
| `val/truncated_completion_fraction`, `_count`, `_length_mean`, `_correct_count` | Fraction of rollouts cut off by the `response_length` cap. **This is one of the most important agentic metrics.** A truncated trajectory can't succeed. If 20–30% of rollouts truncate (as in the example run) and your *unsolved* sequence lengths are near the cap, you are **budget-bound**: a big chunk of "failures" are really "ran out of tokens." `_correct_count` tells you how many truncated ones still somehow got reward. **Two very different causes produce high truncation — don't conflate them (see the note below).** |
| `tool_parser_type` (config) | Which parser extracts tool calls from model output. A mismatch between the parser and the model's actual format = the model's perfectly good tool calls get dropped → it gets punished for correct behavior. A classic silent killer on a new model/template. |

**The interpretation move:** when reward is low, *decompose the failures*. Is the model failing because it's wrong, because it never submitted, or because it got truncated? `non_submitting`, `truncated`, and `stop_rate` let you split "genuinely wrong" from "didn't get a fair shot." The fix is completely different in each case (better reasoning vs clearer protocol vs bigger budget).

> **Truncation has two causes with opposite fixes — read the trend, not just the level:**
> - **Budget-bound** (the common case): truncation is *steady* at 20–30% while reward is flat/rising and `unsolved` lengths hug the cap → the task genuinely needs more tokens. **Fix: raise `response_length`/`per_turn_max_tokens`/`pack_length`, or curriculum toward shorter tasks.**
> - **Divergence-driven**: truncation **spikes** (e.g. to ~40%) *coincident with* reward dropping, `kl1` climbing, and `stop_rate` falling → the policy is collapsing and **rambling into the cap instead of concluding**. Here truncation is a *symptom of divergence*, not a budget problem — raising the budget won't help. **Fix: arrest the divergence (raise `beta`, lower LR, or roll back to a pre-divergence checkpoint).** With `beta=0` this is a useful **early-warning**: a truncation-fraction jump past ~0.35 that lines up with a `kl1` rise often precedes the visible reward collapse by a few steps. (Observed on the 9B swerl run: truncation held ~0.10–0.22 for ~300 steps, then jumped 0.20→0.40 over ~step 300–380 as grpPerf fell 0.72→0.36 and `kl1` drifted 0.10→0.20.)

### 5.4 — The async pipeline: staleness, dropped rollouts, weight sync

Agentic RL is slow to generate (a rollout might be minutes of sandbox execution), so this repo runs generation and training **asynchronously and pipelined** (`async_steps`, `inflight_updates`) — the generators are always working a few steps ahead of the trainer using slightly older weights. See [terminal_rl_walkthrough.md §7](terminal_rl_walkthrough.md) for how the pipeline is layered on the synchronous loop and [grpo_fast_internals.md](grpo_fast_internals.md) for the actors. This buys throughput but introduces **off-policyness**, and several metrics police it. Two mechanics to keep in mind while reading them: (1) DataPrep refuses to prepare step `s` until step `s − async_steps` has been consumed, and drops any group older than that, so staleness is *bounded*, not merely discouraged; (2) with `inflight_updates=true` a trajectory can straddle a weight update mid-episode, and its `model_step` stamp will not show it.

| Metric | Meaning & healthy reading |
|---|---|
| `model_step_min/max/mean` vs `training_step` | The weight version that generated each *group* in the batch (stamped when the prompt was dequeued, so a trajectory that straddles an in-flight weight update still reports the older step). `training_step − model_step_min` is the staleness. **It cannot grow without bound:** DataPrep drops any group older than `async_steps` before it reaches the trainer, so the gap is bounded at `async_steps` (+1, because DataPrep's step counter is 0-based and the logged `training_step` is 1-based). Sample run with `async_steps=4`: gap 0–5, median 2.5. A gap pinned at the cap together with non-zero `stale_results_dropped` means generation is slower than training and the pipeline is throwing rollouts away to stay in-spec. |
| `stale_results_dropped` | Groups discarded for exceeding the staleness cap (each one also enqueues a replacement prompt). Steady-state should be 0. **Bursts** almost always coincide with **resume/restart events** (preemptible jobs): after a resume the engines regenerate the whole in-flight buffer and the first few batches carry stale stamps. Occasional bursts = wasted compute, not corrupted training. *Constant* drops = generation cannot keep up with training at this `async_steps`; either add engines or raise `async_steps` (accepting more staleness). |
| `real_batch_size_ratio` | Actual vs expected batch size. <1.0 means filtering/dropping shrank the batch — your effective batch (and gradient quality) is smaller than you think. |
| `unsolved_batch_size_ratio` | Fraction of the kept batch that's unsolved. Drifting toward 0 = tasks getting too easy (curriculum exhausted, little left to learn). Stuck near 1 = tasks too hard (no positive examples to learn from). The sweet spot is in between — that's where group advantages have spread. |
| `packed_ratio` | Sequences per packed block — efficiency, interacts with length growth. |
| `batch/total_prompts`, `batch/filtered_prompts`, `batch/filtered_prompts_zero / _solved / _nonzero`, `batch/no_resampled_prompts` | **Active-sampling health.** `total_prompts` is the number of groups that made it into the trained batch (= `num_unique_prompts_rollout` when `active_sampling` is on). `filtered_prompts` is how many *additional* groups were generated and thrown away as zero-std to get there, split into all-failed (`_zero`), all-solved (`_solved`) and same-non-zero-score (`_nonzero`). Sample run: 8 kept, 5–18 filtered per step → **1.6–3.2× generation overhead** per useful group. `_solved` rising over training is the curriculum graduating (good); `_zero` rising is the task mix getting too hard. `no_resampled_prompts` counts groups excluded for good via `no_resampling_pass_rate`. |
| `batch/percent_solved_mean`, `batch/percent_solved_hist`, `batch/prompt_lengths`, `batch/response_lengths` | Per-group solve rates and raw length histograms for the *kept* batch. `percent_solved_hist` is the per-prompt solve-rate distribution over the trained groups; with active sampling it is bimodal by construction (only mixed groups survive). |
| `time/total`, `time/training`, `time/getting_response`, `time/weight_sync*`, `time/saving`, `time/health_check` | Where wall-clock goes. `time/training` is the trainer `step()` including its wait for data; `time/total` adds sync and checkpointing. **`time/getting_response` is not "generation cost per step":** it is the longest (enqueue → reward) latency among the batch's groups, and under async that latency overlaps several training steps, so it is normal for it to exceed `time/total` (sample: ~900–2400 s vs ~400–500 s). `weight_sync` is the cost of pushing fresh weights to the vLLM engines (sample: 7–10 s for 9B with `gather_whole_model`). |
| `time/trainer_idle_waiting_for_inference`, `time/generation_idle_waiting_for_trainer` | **The two-sided bottleneck detector.** The trainer's wait inside `get_data` vs DataPrep's wait for the trainer to consume a step so it may run ahead again. Sample run: trainer idle ≈ 2 s, generation idle ≈ 0 → balanced, with a 940 s trainer-idle spike on the first step after a resume (the engines were refilling the buffer). Trainer-idle chronically high = generation-bound: add engines, shrink `response_length`, or raise `async_steps`. Generation-idle chronically high = training-bound: `async_steps` is only buying staleness, lower it. |
| `learner_mfu`, `actor_mfu`, `actor_mbu` | Percent utilization. `learner_mfu` = training FLOPs / (`time/training` × learner peak); sample ≈ 3–4%, low because `time/training` includes the data wait and SP=4/ZeRO-3 communication. `actor_mfu` / `actor_mbu` = generation FLOPs / memory bytes over `time/getting_response` × engine peak; since that denominator is a latency, not busy time, these **under**-report (sample: MFU 0.2–0.4%, MBU 0.4–0.8%). Use them for *relative* comparison between runs, not as absolute efficiency. |
| `learner_tokens_per_second_step / _overall`, `val/actor_tokens_per_second`, `val/num_step_tokens`, `val/num_total_tokens` | Throughput in tokens. `val/num_step_tokens` is prompt + response tokens in the trained batch; `_overall` divides the running total by total training time and is the number to use for wall-clock projections. |

**Why all this exists:** the async pipeline is the thing that makes agentic RL tractable on a cluster, and it's also the thing most likely to silently degrade your run (stale data, dropped batches, a desynced engine). These metrics are the instrumentation for "is my distributed system actually feeding the optimizer good, fresh data."

### 5.5 — Reward hacking: the thing that looks like success

The model optimizes the literal reward. In executable environments the loopholes are creative and common:

- Writing tests that always pass, or deleting the failing test, instead of fixing the bug.
- Printing the expected output / hard-coding the answer it scraped from the environment.
- Exiting `0` without doing the work, if the reward keys off exit code.
- Padding with reasoning tokens if reward correlates with length.

**Signature in the dashboard:** reward (or solve rate) climbs *smoothly and fast*, but held-out eval doesn't move or gets worse; or reward climbs while the *behavior* metrics go weird (lengths explode, tool calls drop to near zero, or jump to a fixed exploit pattern). **The only reliable detector is reading actual rollouts** (`--save_traces`, `rollouts_save_path`). No scalar will tell you "the model is cheating" — you have to look at what it's doing. Make reading 5–10 real trajectories part of your ritual (Part 7).

---

## Part 6 — Diagnostic playbook (symptom → cause → check → lever)

The fast-lookup table. "Check" = the panels to correlate; "Lever" = what to change (one at a time).

| Symptom | Likely causes | Check these together | Lever |
|---|---|---|---|
| **Reward flat from step 0** | No signal: dead advantages, LR too low, reward fn broken, format unsatisfiable, wrong reward key | `advantages_min/max` (collapsed?), `lr` (nonzero?), reward histogram, `non_submitting_fraction` (can it ever submit?), parser type | Confirm reward key is live; verify advantages have spread; raise LR; loosen format/parser; sanity-check reward fn on a known-good trajectory |
| **Reward climbs then collapses** | Instability (KL/ratio runaway), or a reward exploit got patched, or infra change | `kl1` (steady climb?), `kl2`, `optim/grad_norm` (spike?), `clipfrac` (jumped?), **`truncated_fraction` (spiking?) + `stop_rate` (falling?)** — the collapsing policy rambles into the cap, `tools/*/failure_rate` (infra broke?) | Lower LR; enable/raise `beta`; roll back to a pre-divergence checkpoint; tighten clip ε; check for a node/infra change at the collapse step |
| **Reward up, eval flat/worse** | Reward hacking or overfitting to train prompts | Read traces; behavior metrics (lengths, tool calls); train vs eval gap | Fix the reward loophole; add held-out eval; diversify prompts; add KL penalty |
| **Loss = NaN/Inf** | LR too high; fp16 overflow; bad batch; div-by-zero in reward/advantage | `grad_norm` just before; precision (bf16?); the offending batch | bf16; lower LR; add grad clipping; guard the reward/advantage math |
| **Grad norm spiking/creeping up** | LR too high; noisy outlier batches; impending divergence | loss, KL, the data at the spike step | Lower LR; stronger grad clip; investigate outlier prompts |
| **Entropy/diversity collapsing** (proxy: lengths converge, `unsolved_ratio`→0, reward plateaus early) | Premature exploitation; LR too high early; too little KL freedom or too much | length hists, `unsolved_batch_size_ratio`, group performance plateau | Add entropy bonus if available; lower LR; raise sampling temperature; ensure prompt difficulty spread |
| **KL explodes** | Policy diverging; updates too big; stale data | `clipfrac`, `ratio_var`, `grad_norm`, `model_step` gap | Lower LR; raise `beta`; reduce `async_steps`/staleness; tighten clip |
| **KL stuck ~0, reward flat** | Policy not moving | `lr`, `advantages` spread, `clipfrac`, `stale_results_dropped` | Raise LR; confirm advantages non-zero; confirm updates aren't all being dropped |
| **Most rollouts never submit** | Protocol too hard / unclear; budget too small; parser mismatch | `non_submitting_fraction`, `truncated_fraction`, `stop_rate`, parser type, system prompt | Clarify system prompt; raise `response_length`/turn budget; fix/loosen parser; add small format reward |
| **High truncation; unsolved lengths near cap** | Token budget too small for the task | `truncated_fraction`, `sequence_lengths_unsolved` vs `response_length` | Raise `response_length`/`per_turn_max_tokens`/`pack_length`; or curriculum toward shorter tasks first |
| **Tool failure rate creeping up** | Infra (sandbox OOM/timeout/container death), not the model | `tools/*/failure_rate` vs `tools/*/avg_runtime`, idle times, OOM logs | Fix sandbox infra (memory, concurrency, timeouts); this is not a learning problem |
| **Throughput dropping over time** | Length growth, slow/preempted node, comms, checkpoint stalls | `tokens/sec`, MFU, `time/*` idle, `sequence_lengths`, `stale_results_dropped` | Expected if lengths grew; else find the slow node / reduce save freq / check network |
| **`stale_results_dropped` bursts** | Preemption/restart of a preemptible job | `time/*_idle` spikes at same step, `model_step` gap | Usually benign; if constant, retune `async_steps` or get less-preemptible nodes |
| **`debug/dppo_mask_frac_kept` falling** (DPPO) | Policy moving faster than the rollouts can follow: LR too high, staleness up, or a divergence starting | `kl1`, `debug/vllm_vs_local_logprob_diff_mean`, `model_step` gap, `optim/grad_norm` | Lower LR; lower `async_steps`; if it keeps falling with reward, roll back |
| **`debug/vllm_vs_local_logprob_diff_mean` step-changes** | A weight sync silently failed or a vLLM/trainer kernel or precision setting diverged | `time/weight_sync` at that step, engine logs, `val/ratio_var` | Check the sync thread logs; confirm `lm_head_fp32` / GDN backend match; restart engines |
| **`batch/filtered_prompts` ≫ `batch/total_prompts`** | Task mix too easy or too hard for the current policy; you are paying 3–5× generation per useful group | `_solved` vs `_zero` split, `val/avg_group_performance_pre_filter` | Rebalance the task mix; raise `no_resampling_pass_rate` use; raise `num_samples_per_prompt_rollout` |
| **`tools/env_reset/failure_rate` appears** | Sandbox provisioning failing (registry mirror, podman host, gateway) | `tools/bash/avg_runtime`, Beaker logs, `time/getting_response` | Fix the infra; these zeros are not learning signal |
| **Reward histogram bimodal, mean misleading** | Task is all-or-nothing | `solve_rate_hist`, `advantages_hist` | Consider partial-credit reward shaping; curriculum; per-difficulty analysis |

---

## Part 7 — How to actually look at a run (the ritual)

Scalars alone will mislead you. Here's a concrete routine.

### First 5–15 minutes (the smoke test at step 0–10)

This is the highest-leverage habit: **most broken runs are broken at the start**, and catching it now saves days of GPU time.

1. **Is the reward key live?** Find the metric that actually carries reward (`scores`, the right `objective/*`, or env reward) and confirm it's non-zero and varied. (Recall Part 5.1 — the obvious key may be a dead path.)
2. **Are advantages non-degenerate?** `advantages_min/max` should have real spread, not ≈0.
3. **Is the LR what you set?** (and not displaying-as-0 due to rounding).
4. **Is grad norm finite and sane?** No NaN, not astronomically large.
5. **Can the model satisfy the protocol at all?** `non_submitting_fraction` < 1.0, `stop_rate` > 0, tool failure rate not ~100%.
6. **Is throughput in the right ballpark?** (so you can estimate wall-clock — see below).
7. **Read 2–3 actual rollouts.** Does the trajectory look like a sane agent attempt? Are tool calls parsed? Is the reward sensible for what the model did?

If any of these is off, **stop and fix it now.** Don't let a misconfigured run burn a weekend.

### Estimate wall-clock immediately

`total_episodes / num_unique_prompts_rollout = planned steps`. Multiply by observed step time. (For the example: 128000/32 ≈ 4000 steps × ~8 min ≈ weeks.) Knowing this upfront prevents the "why isn't it done" surprise and tells you whether your plateau is "converged" or "barely started."

### Daily / periodic monitoring

- **Look at curves over a window, not two points.** RL is noisy; trends emerge over tens of steps. Fit a line or use wandb smoothing.
- **Correlate panels, don't read them solo.** A reward dip *with* a grad-norm spike *with* a KL jump is a stability event. A reward dip *with* a tool-failure-rate spike is an infra event. Same symptom, opposite fix. Put reward, KL, grad norm, clipfrac, and your env-health metrics on one view.
- **Read histograms, not just means.** `solve_rate_hist`, `advantages_hist`, `sequence_lengths_*_hist`. The mean hides bimodality, which is where the real story often is.
- **Read fresh rollouts every session.** 5–10 trajectories. This is the only way to catch reward hacking, format drift, and "the model found a degenerate strategy." Scalars cannot show you this.
- **Note every intervention in the run notes / a log.** "Step 200: raised LR to 2e-6." Future-you debugging a collapse needs to know what changed and when.

### Comparisons

- Always keep a **baseline run** on the same plot. "Is 0.51 good?" is unanswerable in isolation; "0.51 vs the baseline's 0.42 at the same step" is an answer.
- When ablating, **change one thing at a time** and keep everything else (seed, data, config) identical, or you can't attribute the difference.

---

## Part 8 — Deciding what to change (intervention discipline)

The hard-won meta-lessons:

1. **One change at a time.** RL is a feedback loop with long latency; two simultaneous changes make the result uninterpretable.
2. **Wait long enough to judge.** RL is noisy. A 20-step "improvement" is often noise. Give a change tens of steps before concluding, unless it's an obvious instability (then kill fast).
3. **Know each lever's job:**
   - **LR** — overall speed/stability. Tiny stable grad norms = room to raise; spikes/instability = lower.
   - **`beta` (KL coefficient)** — leash to the reference model. Raise if the policy degenerates/reward-hacks into nonsense; lower (or 0) if it's frozen and you want faster movement (then stability rests on LR + clipping).
   - **Clip ε** — per-update aggressiveness/trust region. Tighten if updates are too violent (high clipfrac + instability).
   - **Sampling temperature** — exploration at generation time. Raise to fight premature convergence; lower for more exploitation once converging.
   - **Length budget** (`response_length`, `per_turn_max_tokens`, `pack_length`) — raise when truncation is capping success; the lever for "budget-bound" failures.
   - **Batch size / samples-per-prompt** — gradient quality/variance. More samples per prompt = better group baseline = less noisy advantages, at higher cost.
   - **Reward shaping / curriculum** — change *what* you optimize. The most powerful and most dangerous lever (reward hacking lives here). Change last, change carefully, and always re-check eval + read traces after.
4. **Prefer the cheapest diagnosis.** Before retraining, ask: can I answer this by reading 10 rollouts or looking at one histogram? Usually yes.
5. **Distinguish "not learning" from "learning slowly."** A small-but-positive `avg_group_performance` slope with tiny stable grad norms isn't broken — it's under-driven. The fix (raise LR) is different from a truly dead run (fix the signal).

---

## Part 9 — This codebase's exact metric glossary

Every wandb key `grpo_fast.py` can emit, grouped by which process produces it. **(conditional)** = only present under the stated flag; **(loss-specific)** = only meaningful for that `--loss_fn`. Verified against the code on `omni_agent` and against run `g55rr33t` (2026-09-18).

### Emitted by the trainer ranks (token-weighted average across ranks)

| key | meaning | notes |
|---|---|---|
| `loss/policy_avg` | mean policy-gradient surrogate over response tokens | level not meaningful (Part 4) |
| `loss/total_avg` | the scalar actually back-propagated (policy + `beta`·KL, after the global-token denominator) | |
| `loss/kl_avg` | `kl{kl_estimator} × beta` | 0 when `beta=0`; **(conditional on `load_ref_policy`)** |
| `objective/kl0_avg` … `kl3_avg` | four KL-to-reference estimators (linear / quadratic / k3 / importance-weighted) | `kl2` is the headline; **(conditional on `load_ref_policy`)** |
| `policy/clipfrac_avg` | fraction of tokens where the clipped surrogate was active | **≡ 0 under `dppo` / `cispo` / `tvpo`** |
| `policy/entropy_avg` | mean next-token entropy on response tokens | **(conditional on `--record_entropy`; incompatible with liger)** |
| `val/ratio`, `val/ratio_var` | mean per-row ratio and its variance across rows | not per-token (Part 4) |
| `val/tis_ratio`, `val/tis_clipfrac` | truncated-IS weight mean and cap fraction | 0 when `truncated_importance_sampling_ratio_cap=0` |
| `debug/tis_mask_frac_kept` | fraction of tokens kept by the TIS / sequence-TIS masks | **(conditional on those masks being on)** |
| `debug/dppo_mask_frac_kept` | fraction of tokens kept by the DPPO trust-region mask | **(loss-specific: dppo)** |
| `debug/tvpo_mask_frac_kept`, `actor/ppo_tv` | TVPO kept fraction and mean prompt-level TV | **(loss-specific: tvpo)** |
| `debug/vllm_vs_local_logprob_diff_mean/max/std`, `debug/vllm_local_reverse_kl` | trainer-vs-vLLM per-token logprob disagreement | always on; the direct mismatch gauge |
| `optim/grad_norm` | global grad norm after the accumulation group (mean if several) | may be NaN/inf on a bad step |
| `lr` | scheduler LR | small values display as 0 |
| `loss/value_avg`, `value/clipfrac_avg`, `value/grad_norm`, `value/returns_mean/std`, `value/predictions_mean/std`, `value/explained_variance` | value head diagnostics | **(conditional on `--use_value_model`)** |

### Emitted by the DataPreparationActor (per step, attached to the batch)

| key | meaning | notes |
|---|---|---|
| `scores` | mean raw environment reward over the trained batch | pre length-shaping |
| `val/avg_group_performance_pre_filter` / `_post_filter` | group-level solve rate including / excluding the zero-std-filtered groups | **primary learning signal** with active sampling |
| `val/solve_rate_hist`, `batch/percent_solved_hist`, `batch/percent_solved_mean` | per-prompt solve-rate distribution of the kept groups | |
| `val/advantages_mean/min/max/hist` | centered advantages | mean ≈ 0 by construction |
| `val/total_reward_groups` | kept groups (= trained rollouts / `num_samples_per_prompt_rollout`) | |
| `real_batch_size_ratio` | trained rollouts / expected | 1.0 with active sampling |
| `unsolved_batch_size_ratio` | fraction of trained rollouts with score < max | |
| `packed_ratio` | packed rows / rollouts | efficiency; drops as responses lengthen |
| `batch/total_prompts`, `batch/filtered_prompts`, `batch/filtered_prompts_zero/_solved/_nonzero`, `batch/no_resampled_prompts` | active-sampling accounting (Part 5.4) | |
| `batch/prompt_lengths`, `batch/response_lengths` | raw length histograms | also feed the MFU calculation |
| `val/sequence_lengths` (+ `_min/_max/_solved/_unsolved`, `_solved_hist/_unsolved_hist`) | response lengths, split by outcome | |
| `val/stop_rate` | fraction whose *last turn* ended with `stop` | |
| `val/truncated_completion_count/_fraction/_correct_count/_length_mean/_length_max` | budget-wall stats (`finish_reason != stop` **or** `len ≥ response_length`) | computed pre-masking |
| `val/non_submitting_completion_count/_fraction`, `_unmasked_count/_unmasked_fraction` | episodes that never reached `done` | `_unmasked_*` only non-zero with `mask_non_submitting_completions_percent` |
| `model_step_min/max/mean` | weight version of the batch's groups | staleness (Part 5.4) |
| `stale_results_dropped` | groups dropped for exceeding `async_steps` | |
| `objective/verifiable_reward`, `objective/verifiable_correct_rate`, `objective/<verifier>_reward`, `objective/<verifier>_correct_rate` | verifier-function path | **flat 0 on env-reward datasets** (`<verifier>` = `passthrough`); not a bug |
| `val/format_scores` | R1-style format reward | **(conditional on `apply_r1_style_format_reward`)** |
| `tools/aggregate/avg_calls_per_rollout/failure_rate/avg_runtime`, `tools/<tool>/…` | per-tool call stats | `tools/env_reset/*`, `tools/tool_call_format_error/*` **(conditional)**, see 5.2 |
| `env/<env_name>/<metric>` | env `get_metrics()` averaged | sandbox: `step_count` only |
| `concave_length_penalty/x_*`, `penalty_*`, `shaped_score_mean`, `raw_score_mean`, `group_success_penalty_gap_*`, `groups_with_multi_success` | length-penalty shaping diagnostics | **(conditional on `add_concave_length_penalty`)** |
| `time/getting_response`, `val/actor_tokens_per_second`, `time/generation_idle_waiting_for_trainer` | generation latency / throughput / backpressure wait | Part 5.4 |

### Emitted by the main thread / sync thread

| key | meaning | notes |
|---|---|---|
| `training_step`, `episode`, `global_step` (= episode), `epoch` | step counters; `epoch = episode / n / len(train_dataset)` | `episode` counts *sampled* rollouts including filtered ones |
| `val/num_step_tokens`, `val/num_total_tokens` | prompt + response tokens this step / cumulative | |
| `learner_tokens_per_second_step/_overall`, `learner_mfu`, `actor_mfu`, `actor_mbu` | throughput and utilization (percent) | Part 5.4 for caveats |
| `time/total`, `time/training`, `time/saving`, `time/health_check`, `time/trainer_idle_waiting_for_inference` | main-thread timers | |
| `time/weight_sync`, `time/weight_sync_mean/min/max/median` | sync wall-clock and per-trainer-rank spread | arrives one step late (queue) |

### Emitted only when an eval set is configured (`--dataset_mixer_eval_list`, every `local_eval_every` steps)

`eval/scores`, `eval/pass_at_1`, `eval/pass_at_<eval_pass_at_k>`, `eval/pass_at_<2^j>_unbiased`, `eval/sequence_lengths` (+ `_min/_max`), `eval/stop_rate`, `eval/actor_tokens_per_second`, `eval/<reward metric>` for every key the reward function returns, and a `sample_completions` table. The sample run has **none of these**: its eval list is empty, so `local_eval_every=10` does nothing. If you want held-out evals you have to give it a dataset (or use the checkpoint-boundary terminal-bench evals on the `terminal_val_eval` branch).

## Part 10 — Stack-specific footguns (learned the hard way)

- **The "reward is zero" panic that isn't.** `objective/verifiable_reward=0` with a healthy `scores` just means env-path reward. Always identify the live reward key first. (Part 5.1)
- **`advantages_mean≈0` is by design** (group centering), not a dead signal — check `min/max` spread instead.
- **`loss/policy_avg` not decreasing is not a bug** — RL loss level is meaningless. Watch reward/KL/entropy.
- **`lr` showing `0.0`** is often just rounding of a small value (e.g. 1e-6) in the display.
- **`beta=0.0` means KL does not constrain training** — you are the only KL safety check; watch `kl2` yourself.
- **Tool-parser / chat-template mismatch on a new model** silently drops valid tool calls → the model gets punished for correct behavior; spikes `non_submitting`/tool-failure. Verify the parser matches the model's emitted format on day one.
- **Infra failures look like negative reward.** A flaky sandbox (OOM, timeout, container death) returns reward 0 for trajectories the policy didn't actually fail. Watch `tools/*/failure_rate` and runtimes; don't tune the policy to fix an infra bug.
- **Preemptible jobs** produce `stale_results_dropped` bursts and idle spikes at restarts — usually benign, but they make curves jagged; don't mistake a restart artifact for a learning event.
- **Truncation is invisible if you only watch reward.** A 25% truncation rate means a quarter of your rollouts never had a chance; check `truncated_fraction` and unsolved lengths vs the cap before concluding the model is "bad at the task."
- **Eval is the source of truth, reward is the proxy.** Always keep a held-out eval (`local_eval_every` *and* a non-empty `--dataset_mixer_eval_list`; the frequency flag alone logs nothing) — it's your reward-hacking and overfitting alarm.
- **`policy/clipfrac_avg` is exactly 0 under DPPO/CISPO/TVPO.** It is not "no clipping needed"; the metric does not apply. Read `debug/dppo_mask_frac_kept` instead.
- **Passing `--load_ref_policy false` deletes every KL panel.** Fine if you consciously trade the drift odometer for memory; surprising if you did not.
- **`time/getting_response` > `time/total` is normal under async.** It is a latency that overlaps several steps, not a per-step cost; `actor_mfu`/`actor_mbu` inherit this and under-report.
- **The staleness gap is capped, so a "runaway" is impossible.** If you think data is too stale, look at `stale_results_dropped` (how much is being *thrown away* to keep the cap) rather than the gap.
- **`tools/env_reset/*` only appears on failure.** Its presence is the alarm; its absence is not proof resets are fast.
- **`env/<env>/step_count` == `tools/bash/avg_calls_per_rollout`** for a bash-only sandbox; they are the same number from two code paths.

---

## Part 11 — What is *not* logged yet (and where you would add it)

Things a Terminal-RL run would benefit from that the current code does not emit. Each names the cheapest insertion point; most are a few lines in `data_loader.py`'s `_data_preparation_loop` (which already has every per-rollout field in hand) or in `populate_sample_loss_stats` on the trainer.

| gap | why it matters | where the data already is |
|---|---|---|
| **Entropy on the liger path** | the cleanest collapse indicator is unavailable on every production run | needs a chunked entropy inside `TiledGRPOLMHeadLoss` (logsumexp and the logits are already computed per tile) |
| **Ratio / mismatch broken down by rollout age** | separates staleness from numerics: `E[|log π − log μ|]` for `model_step = k−1` vs `k−4` tells you whether raising `async_steps` is safe | `data_BT.model_steps` is a per-token tensor already on the trainer; bucket `debug/vllm_vs_local_logprob_diff` by it |
| **Tool-output vs model-token fraction** | how much of each 32k trajectory is the model's own text vs pasted shell output; drives both context pressure and what `mask_tool_use` is hiding | `sum(mask)` vs `len(response)`; computed today only inside the concave-length-penalty branch |
| **Tool-call timeout rate, separate from errors** | timeouts are the model's own long-running commands; errors are infra. Today both land in `failure_rate` | `request_info.timeouts` and `tool_errors` are collected per rollout and never aggregated |
| **Reset attempt count / reset latency** | resets that succeed on retry 3 are invisible; they are the leading indicator of a dying mirror or gateway | `_do_reset` already records phase timings under `SWERL_SANDBOX_TIMING_LOGS`; surface them via `get_metrics()` → `env/…` |
| **Submit rate by outcome** | "submitted and failed tests" vs "never submitted" vs "truncated" is the failure decomposition Part 5.3 asks for, but only the last two are logged | `rollout_states[i]["done"]` + `raw_scores[i]` |
| **Per-turn histogram** | `env/…/step_count` is a mean; a bimodal (3 turns vs 64 turns) distribution is a different diagnosis | `step_count` per rollout is in `rollout_states`; pass an array instead of `np.mean` in `_aggregate_env_metrics` |
| **Solve rate by task family / difficulty** | the task mix saturating (the settled explanation for the 9B plateau) is only visible per task | `batch.indices` + a dataset column; log a small `wandb.Table` every N steps |
| **Reward for truncated vs non-truncated, per turn budget** | is `per_turn_max_tokens` binding? how often does a single turn hit 8k? | vLLM `finish_reason` per *turn* is available inside `process_request` but only the last is kept |
| **Queue depths and engine occupancy over time** | `prompt_Q` / `inference_results_Q` sizes and active tasks per engine are the real async health signals | already polled by `ActorManager` for the HTTP dashboard (`--enable_queue_dashboard`, default on); they are just not pushed to wandb |
| **Per-DP-rank token counts / loss-scale sanity** | uneven packing across ranks was the root cause of the liger loss-normalization bug | `_token_count` is computed per rank and discarded after weighting |

---

### TL;DR ritual

1. At step ~5: confirm reward key is live, advantages have spread, LR/grad-norm sane, model can submit, estimate wall-clock. **Read 3 rollouts.**
2. Daily: trends over windows; correlate reward + KL(`kl2`) + grad_norm + clipfrac + env-health on one view; read histograms; read fresh rollouts; watch tool/truncation/staleness metrics. **Read more rollouts.**
3. To intervene: one change, wait tens of steps, keep a baseline, log what you changed.
4. Outcome (reward/eval) is lagging; KL/entropy/grad-norm/clip are leading — trust the leading indicators to catch trouble early.
