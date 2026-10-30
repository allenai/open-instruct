# Research

Questions/ideas being pursued. One section per idea, each linking to its section(s) in `experiment.md`.

## Olmo Hybrid release checkpoint inputs

**Status:** GPU GATE PASSED — native/HF parity, update, export/reload validated; RL queued.

[HYBRID_CHECKPOINTS.md](HYBRID_CHECKPOINTS.md) records Yasha's primary and science
release layouts, the authoritative Weka maps, final primary steps, and the pinned
transformers/vLLM plugins and evaluation settings. October 21 inspection confirmed
the primary mirror counts but found that the science handoff inventory had drifted:
the map declares 1,523 checkpoints across 60 arms, the HF mirror has 939 directories,
and the stated native science mirror is absent. Directory counts are not load tests.

October 22 integration pins the release plugins plus stock transformers/vLLM,
registers both architecture names, supports paired release-tree native configs
and an explicit config override, and reuses the verified hybrid tensor mapping
for initialization, publication, and export. All 192 real 275m HF tensor names and
shapes match HF/native meta models. The explicit base prompt is
`Question: <question>\nAnswer:`. Earlier SFT thinking prompts are not assumed.

The proposed input is primary 275m midtrain final `step0038147`. The one-GPU
actual-weight parity/update/export test passed at the production learning rate.
The user then requested a two-GPU, 1,000-update run with native checkpoints every
100 updates and automatic resume. That run is queued on Saturn. Zero-variance
filtering is disabled; zero reward can still mean zero learning.
See [the preparation/validation log](experiment.md#olmo-hybrid-release-275m-rl-preparation-2026-10-22).

## MILES migration for Olmo Hybrid release models

**Status:** ASSESSED — keep the current backend for the first 275m baseline.

The MILES integration checkout at `2fd9ca6bededc1d424e501bf0f56c9e8ac7fe806`
pins MILES `4772a6767f1079eb3548d680c59fe98e481d6ff5`, OLMo-core
`7e97bc43ff25d080ae4fd0eab1050a79caf80e55`, and olmo-sglang
`92b35cdf2771353a9e2c527a5b96b29cbb3d38c8`. Inspection of those pins, rather
than only the newer local MILES worktree, found:

- The dense HF factory covers llama/qwen3/olmo2/olmo3 and rejects mainline_ladder.
  An explicit native model config bypasses construction but not the generic HF
  import/export converter. The hybrid mapping must be added to initialization,
  streamed publication, export, and worker registration.
- The pinned olmo-sglang extension implements Olmo3Moe/KDA. The pinned stock
  SGLang model list has no MainlineLadder or OlmoHybrid executor. The vLLM plugin
  cannot be used directly by SGLang. A serving port needs GDN/negative-eigenvalue
  recurrence, convolution/decode state, NoPE gated attention, peri-norm blocks,
  and embedding normalization, followed by HF/Core prefill/decode/logprob parity
  and post-publication checks. Shared recurrent-attention vocabulary does not
  make KDA and this GDN checkpoint interchangeable.
- Structured config, math/GSM8K data/rewards, save/resume, producer backfill,
  bounded staleness, and refresh already exist. Those are reusable infrastructure,
  not model support. The prepared MILES GSM8K verifier normalizes numeric answer
  formatting, whereas the legacy verifier compares extracted strings; match that
  behavior deliberately in any learning comparison.

Recommendation: first qualify this branch's short 275m smoke, then consider a
bounded MainlineLadder port if many hybrid runs, NGU, or fully async experiments
justify it. The serving/numerical-validation work dominates the migration; this
is a model-integration project, not a TOML-only switch. No speedup for 275m is
established by the existing Qwen/Olmo MoE MILES experiments.

Pinned sources: [MILES dense factory and interchange](https://github.com/allenai/miles-olmo-core/blob/4772a6767f1079eb3548d680c59fe98e481d6ff5/miles/backends/core_utils/standard_models.py),
[olmo-sglang supported architecture](https://github.com/allenai/olmo-sglang/tree/92b35cdf2771353a9e2c527a5b96b29cbb3d38c8),
and [the pinned SGLang model implementations](https://github.com/sgl-project/sglang/tree/3145136dcd1238754e0ea2b2ffd546532119c71c/python/sglang/srt/models).

## Hammond checkout comparison

**Status:** RESOLVED

At the initial comparison, the checkouts differed. The cluster checkout is named `nit-open-instruct` at
`/weka/oe-adapt-default/michaeln/nit-open-instruct`, on `hybrid-small-suite` at
`e2d831a5ca7ad8d62ff3e554460753ec17ae62bb`. Local HEAD is
`9584735ef421640cc282b19fccaf0f9223466e36` on the same branch. Both working trees
were clean before the local comparison documentation was added.

The cluster has 76 commits after `8fc89d9584913b93d56f46cf6b566b7a316140ff`,
including GRPO/GDN weight-sync and activation-checkpointing changes, chat-template
handling, Dolci scoring scripts, and extensive run findings. Local has a separate
evaluation-launch fix touching `scripts/eval/oe-eval.sh` and
`scripts/submit_eval_jobs.py`. The subsequent synchronization merged both histories, preserving
the evaluation-launch fix, cluster work, and local attachment documentation.
The merge passed `make style`, `make quality`, and ten focused CPU tests covering
name mapping, checkpoint wrappers, and GRPO resource planning.

The old session died October 17; the approved replacement is
`01M5FZAJ0GKWGRBYFRCHCW3CCG`. [BEAKER_ATTACH.md](BEAKER_ATTACH.md) records how to
attach and the stale `--host-networking` option in `beakernew`.

See [experiment.md#hammond-checkout-comparison-2026-10-21](experiment.md#hammond-checkout-comparison-2026-10-21).

## OLMo-hybrid 275M small suite: GRPO on GSM8K

**Status:** ACTIVE

Does GRPO post-training improve GSM8K accuracy for the OLMo-hybrid small suite's
275M SFT checkpoint (`hybrid-small-sft-think-275M-lr2e-4`)? This is the first
GRPO run for the small suite (see `scripts/train/debug/olmo_hybrid_275m_4gpu_gsm8k.sh`),
intended as a debug/smoke run to confirm the `olmo_hybrid_small` architecture
trains end-to-end under `grpo.py` (OLMo-core) before scaling up.

Run 1 (`response_length=2048`) had a very low stop rate (~12%), killed at step 183/1000.
Leading hypothesis is that this "-think-" SFT checkpoint's reasoning traces don't fit in
2048 tokens, so rollouts are truncated before `eos_token` rather than genuinely failing
to terminate. Testing with `response_length=8192` in run 2.

**Correction (2026-08-07):** the original chat-template audit for run 1 was incomplete —
it compared `olmo_thinker` only against sibling debug *scripts* (which use the same
override), not against the checkpoint's actual baked-in `tokenizer_config.json`
template. A direct re-audit found the generation-prompt tail (`<|im_start|>assistant\n<think>`)
matches, but the *default system prompt* does not: `olmo_thinker` injects
`"You are a helpful AI assistant."` when no system message is present, while the
checkpoint's real template injects a function-calling-flavored default
(`"You are a helpful function-calling AI assistant. You do not currently have access
to any functions...."`). Since `ai2-adapt-dev/rlvr_gsm8k_zs` has no system messages,
every rollout sees an out-of-distribution system prefix relative to SFT. This is a
plausible secondary contributor to the low stop rate, alongside truncation. Details in
[experiment.md#2026-08-07-chat-template-re-audit](experiment.md#2026-08-07-chat-template-re-audit).
Fix candidate: switch to the `olmo123`-style pattern used by the 7B hybrid instruct
scripts (a `--chat_template_name` not present in `CHAT_TEMPLATES`, so the checkpoint's
own template is used unmodified) instead of the hard-coded `olmo_thinker`.

See [experiment.md#olmo-hybrid-275m-grpo-gsm8k](experiment.md#olmo-hybrid-275m-grpo-gsm8k).

## OLMo-hybrid small suite: think vs open-close-think GSM8K GRPO

**Status:** ACTIVE

Extends the 275M/GSM8K idea above to the 450M size, and adds a second axis: does
forcing the "-think-" SFT checkpoints to skip reasoning (pre-filling a closed
`<think></think>` in the generation prompt — the standard way to turn a thinking model
into instruct-only, cf. Qwen3's `enable_thinking=False`) change GSM8K performance
relative to normal `<think>`-open reasoning, both before and after GRPO?

Also applies the chat-template fix identified in the 2026-08-07 correction above: new
`olmo_hybrid_small_think` / `olmo_hybrid_small_open_close_think` templates
(`open_instruct/dataset_transformation.py`) use these checkpoints' actual default
system prompt (function-calling-flavored, read from `tokenizer_config.json`) instead of
`olmo_thinker`'s `"You are a helpful AI assistant."`, since `rlvr_gsm8k_zs` has no
system messages and both checkpoints' baked-in default differs from `olmo_thinker`.

Script moved out of debug to `scripts/train/olmo-hybrid-small/gsm8k_grpo.sh`
(parameterized by `MODE={think,open_close_think}` and `EVAL_ONLY={true,false}`), with
`response_length` raised to 16384 and `pack_length` to 32768.

Plan: an eval-only baseline (single training step, `learning_rate=0`,
`--eval_on_step_0` so a GSM8K eval is still collected) for each of {275M, 450M} x
{think, open_close_think}, then full GRPO training runs for the same 4 combinations.

**Eval-only baseline results (2026-08-10):** all 4 confirm the `response_length=16384`
+ corrected-system-prompt fix worked — stop rates are now 0.79-0.99 (vs. ~12% in the
original run 1), so truncation is no longer the dominant failure mode. GSM8K
`correct_rate`:

| | think | open_close_think |
|---|---|---|
| 275M | **0.12** | 0.06 |
| 450M | 0.05 | **0.14** |

275M behaves as expected (reasoning helps a small model: 0.12 > 0.06). **450M inverts
this** (0.05 < 0.14) — forcing reasoning makes the 450M checkpoint noticeably worse than
both forcing no-reasoning and the smaller 275M-think baseline, despite a healthy 0.97
stop rate (so it's not truncation). Leading hypothesis: the 450M checkpoint's own
baked-in `tokenizer_config.json` chat template does *not* end its generation prompt
with `<think>` (unlike 275M's, which does) — see the 2026-08-07 chat template re-audit
in experiment.md. If 450M's SFT data trained it to *decide* whether to open a `<think>`
block rather than always expecting one to be forced open, then forcing `<think>` open
(as `olmo_hybrid_small_think` does, matching `olmo_thinker`'s behavior) may push it
off-distribution in a way it doesn't recover from within a single unconditioned
response — worth inspecting a handful of raw 450M-think eval transcripts (wandb run
linked in experiment.md) to confirm before drawing conclusions from the GRPO training
runs. GRPO training runs for all 4 combinations are in progress; will check whether RL
narrows or preserves this gap.

**Crash (2026-08-10, ~07:29-07:30 UTC):** all 4 GRPO training runs died within the same
~2-minute wall-clock window, roughly 65-70 minutes after each one's own start time,
despite being at very different training steps (275M think ~step 40, 450M think ~step
99, 450M open_close_think ~step 161, 275M open_close_think ~step 300+). Root cause:
Ray's GCS process got killed by the Linux OOM killer (host RAM, not GPU memory —
`ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on
memory`, raylet "marked dead", then `Failed to connect to GCS within 60 seconds`).
Since crash timing tracked wall-clock time rather than step count across 4 very
different progress rates, this looks like a memory leak proportional to elapsed time
(e.g. per weight-sync round) rather than to training steps. Suspicious correlated
signal in the logs: repeated `WARNING [layerwise.py:230] <Module>: Failed to load
weights` for OLMo-hybrid-specific modules (`OlmoHybridMLP`, `OlmoHybridSmallAttention`,
`OlmoHybridSmallDecoderLayer`, `SiluAndMul`, ...) during vLLM in-flight weight sync
(`--inflight_updates`), recurring on every sync round — a plausible leak source, though
not confirmed as the cause. This is a pre-existing infra issue in the `olmo_hybrid_small`
vLLM weight-sync path, not something introduced by this task's script/template changes
(the crash is independent of MODE and model size). Worth a follow-up investigation into
`layerwise.py`'s weight-loading path for the OLMo-hybrid architecture under
`--inflight_updates`, and/or adding host-memory monitoring or lower weight-sync
frequency as a workaround if it recurs. Only one run (275M open_close_think) had a
save at a step-100 boundary (`step200` in its `checkpoint_state_dir`) to resume from;
the other 3 only ever reached `step0`. Relaunched all 4 (275M open_close_think resumed
from its step200 checkpoint via `--checkpoint_state_dir`, the rest restarted fresh);
confirmed `--checkpoint_state_dir` resume works cleanly as a crash-recovery pattern for
this script (see the 2026-08-10 relaunch note in experiment.md).

**Crash did not recur (2026-08-10, checked at ~10:02 UTC, ~65-70 min past the relaunch,
same critical window as the first crash):** all 4 relaunched runs were still healthy
with no errors — the OOM was **not deterministic/reliably reproducible on a second
attempt**, so it's more likely transient (e.g. host memory pressure from a co-located
job on that particular node/moment) than a hard bug that will reliably recur every run.
Downgrading this from "avoid re-running until root-caused" to "known occasional
failure mode, worth an occasional health check + resume via `--checkpoint_state_dir`
if it happens again." Training is progressing well past the crash point with real
accuracy gains: 275M open_close_think 0.14→0.18→0.21 (steps 299/399/499, continuing its
pre-crash trend), 450M open_close_think 0.14→0.17 (steps 0/99), 450M think 0.06→0.15
(steps 0/99-100) — notably, 450M think is now improving substantially with RL despite
its poor (0.05) eval-only baseline, 275M think still only has its step-0 eval (0.11,
consistent with the earlier baseline) since think-mode rollouts remain the slowest.

**Update (2026-08-10, ~11:05 UTC):** both 450M runs jumped sharply between their last
two eval points — 450M think 0.15→**0.38** and 450M open_close_think 0.17→**0.33**
(both steps ~100→200) — while 275M open_close_think continues its slower steady climb
(0.21→0.22, steps 499→599) and 275M think is still stuck at its step-0 eval (0.11; long
think-mode rollouts mean it's only completed ~1/5 as many steps as the open_close_think
runs in the same wall-clock time). This resolves the earlier "450M inverts the
think/open_close_think pattern" baseline puzzle: at the *eval-only* baseline (no
training), forcing `<think>` open hurt the 450M checkpoint (0.05 vs. 0.14) — but a
couple hundred steps of GRPO fixes this, and 450M-think ends up *higher* (0.38) than
450M-open_close_think (0.33) at the same step. Consistent with the hypothesis that the
450M SFT checkpoint just wasn't well-calibrated to always opening `<think>` out of the
box, and RL quickly corrects that miscalibration rather than reasoning being
fundamentally harmful for this checkpoint.

**Update (2026-08-10, ~12:07 UTC):** 275M think finally reached its second eval point —
0.11→0.15 (steps 0→99) — so all 4 configs now have a real trajectory, not just a
baseline. Both 450M runs keep climbing steeply: 450M think 0.38→**0.47**, 450M
open_close_think 0.33→**0.45** (both steps 200→300) — both now ~4x their step-0 eval.
275M open_close_think is growing more gradually at this point: 0.22→0.23 (steps
599→699). Current ranking at comparable step counts: 450M think > 450M open_close_think
> 275M open_close_think > 275M think, i.e. bigger model wins regardless of mode, and
within a model size the `think` mode is now pulling ahead for 450M specifically (the
open question is whether the same holds for 275M once its `think` run has more
steps — currently only 275M's mode comparison at step ~99 area is available: 275M think
0.15 @ step99 vs. 275M open_close_think 0.09 @ step199, so early signal is `think` >
`open_close_think` for 275M too, matching 450M's pattern once RL kicks in — differs
from the *eval-only baseline* where 275M favored think but 450M favored
open_close_think).

**Update (2026-08-10, ~13:09 UTC) — growth leveling off:** 275M open_close_think has
essentially plateaued: 0.21→0.22→**0.23→0.22** (steps 499→599→699→799), i.e. flat/noisy
around 0.22-0.23 for the last ~300 steps after its earlier steady climb. Both 450M runs
are still climbing but with clearly diminishing returns compared to their earlier
sharp jump: 450M think 0.47→**0.49** and 450M open_close_think 0.45→**0.46** (both
steps 300→400, vs. the ~2x jump seen from step 100→200). If this pattern holds, 450M
may be converging somewhere in the high-0.4s/low-0.5s range on this GSM8K eval set
within the next few hundred steps, well short of `total_episodes=512000`'s full ~2000
planned steps — worth revisiting whether to early-stop these runs once the plateau is
confirmed, rather than running all the way to 2000 steps for no further gain. 275M
think still only has its step0/step99 pair (no new eval since last check).

**Update (2026-08-10, ~14:11 UTC) — plateau confirmed durable for 3/4 runs:**
275M open_close_think has now been flat for 400+ steps: 0.23→0.22→0.22→**0.24→0.23**
(steps 699/799/899/999/1099) — noisy around 0.22-0.24, not a temporary dip, genuinely
converged. 450M open_close_think is flat too: 0.46→**0.46** (steps 400→500). 450M think
has nearly converged with only marginal residual growth: 0.49→**0.50** (steps 400→500).
**Recommendation for whoever picks this up next:** given `total_episodes=512000` implies
~2000 planned steps but 3/4 runs converged by step ~500-1100, these are very likely done
extracting signal from GSM8K's difficulty ceiling for these model sizes — continuing to
2000 steps is probably not worth the compute unless the goal is specifically to confirm
long-run stability. Not stopping the jobs myself since that's a call best left to a
human, but flagging it clearly. 275M think, by contrast, is *not* plateaued — it's still
climbing steadily (0.11→0.15→**0.18**, steps 0/99/199) since its long think-mode
rollouts mean it's only ~1/5 as far along in step count as its open_close_think
counterpart at the same wall-clock time; worth letting it run longer to see where it
lands before drawing final conclusions about 275M think vs. open_close_think.

**Update (2026-08-10, ~16:14 UTC):** 275M think got another point and keeps climbing:
0.18→**0.20** (steps 199→299). More interesting: 450M think **dipped slightly**
(0.52→0.50, steps 600→700) — its first non-monotonic move — while 450M open_close_think
kept climbing and caught up (0.48→**0.52**, steps 600→700). At matching step counts the
two 450M modes are now statistically indistinguishable (~0.50-0.52 each), which reframes
the earlier "think pulls ahead once RL kicks in" read from the ~13:09 update: that gap
looks like it was transient (think converged faster, not necessarily higher) rather than
a persistent mode advantage. Small-N eval noise (single eval-set pass) means a 2-point
dip isn't itself alarming, but worth watching whether 450M think continues to fall
behind or recovers. 275M open_close_think unchanged, still flat at 0.22-0.24.

**Update (2026-08-10, ~17:22 UTC) — dip resolved as noise:** 450M think recovered and
hit a new high, 0.50→**0.53** (steps 700→800); 450M open_close_think held essentially
flat, 0.52→0.51 (steps 700→800). Confirms the earlier dip was eval-noise, not a real
regression, and both 450M modes are now settled around **0.51-0.53** — genuinely
converged to the same plateau, no lasting mode advantage at this model size. 275M
open_close_think remains extremely stable: 0.23/0.23/0.24/0.23 across steps
1199/1299/1399/1499 (800+ steps flat) and is now approaching the full ~2000-step run
with total_episodes=512000. 275M think has no new eval point (still step 299, 0.20) —
expected, given its rollout-length bottleneck.

**Update (2026-08-10, ~19:26 UTC) — 275M think closing the gap:** 275M think reached
0.20→**0.22** (steps 299→399), now nearly matching 275M open_close_think's plateau
(0.23-0.24, currently at step 1599, still flat). If 275M think's climb continues even a
little further it will likely match or exceed open_close_think's ceiling — mirroring the
450M pattern (both modes converge to the same plateau, `think` if anything reaching
it slightly higher/faster). This is the clearest evidence yet against a persistent
`open_close_think` advantage for GSM8K at either model size: at both 275M and 450M, GRPO
appears to erase whatever gap existed at the eval-only baseline, regardless of which
direction that baseline gap originally favored. Both 450M runs continue their slow
convergent climb: 450M think 0.54→**0.55** (steps 900→1000), 450M open_close_think
0.51→**0.53** (steps 800→900).

**Extension to 810M & 1.4B (2026-08-10, ~20:30 UTC):** launched the same eval-only
baseline (think + open_close_think) for the two remaining small-suite sizes, to
complete the eval-only comparison table across all 4 sizes before deciding whether to
also run GRPO training at 810M/1.4B. Both checkpoints turned out to be under `saumyam`
on disk, not `yashasbls` as originally assumed when requesting this batch — see the
2026-08-10 note in experiment.md.

**Finding: `think` mode GPU-OOMs at 810M+ under current settings (2026-08-10, ~20:47
UTC).** Both `think`-mode eval-only runs (810M and 1.4B) crashed with a genuine CUDA
`OutOfMemoryError` during the backward pass (e.g. "GPU 0 has a total capacity of 79.25
GiB ... 75.60 GiB memory in use" for 810M) — a different failure mode from the Ray-GCS
host-RAM OOM seen in the 275M/450M GRPO training runs above. 1.4B `open_close_think`
subsequently crashed the same way too (only 810M `open_close_think` actually succeeded);
so 3 of the 4 new launches OOM'd.

**Root cause (2026-08-10): `--gradient_checkpointing` was a dead flag on this whole
script all along.** It only wires up in the older DeepSpeed-based trainer path;
`grpo.py`'s OLMo-core actor never reads it. The real control,
`--activation_checkpointing_mode`, defaults to `"budget"` with
`activation_memory_budget=1.0` — per `olmo_core_utils.build_ac_config`, that combination
applies **zero** activation checkpointing, at any model size, in every run of this
script so far (275M/450M just happened to fit in 80GB without it). Separately, `"budget"`
mode can't checkpoint through this architecture's opaque GDN kernels regardless —
`"selected_modules"` mode (wraps individual block submodules including the GDN mixer)
is the one that's actually safe and effective here. Fixed in `gsm8k_grpo.sh`:
swapped `--gradient_checkpointing` for `--activation_checkpointing_mode
selected_modules`. Per explicit instruction, the 4/4 GPU (learner/vLLM) split was left
unchanged — this is a correctness fix (turning on checkpointing that was silently never
applied), not a parallelism change. Relaunched 810M think, 1.4B think, and 1.4B
open_close_think with the fix; results pending.

810M open_close_think baseline (from the run that succeeded pre-fix): `correct_rate`
**0.32**, the highest of any size/mode combo so far (vs. 0.06-0.14 at 275M/450M) —
consistent with GSM8K accuracy scaling with model size once `think` isn't a confound.

**`selected_modules` checkpointing itself broken (2026-08-10, ~21:15 UTC): reverted.**
All 3 relaunches (810M think, 1.4B think, 1.4B open_close_think) crashed again — this
time in ~3 min, with an identical, deterministic error across all 3:
`ValueError: There is no module or parameter named 'blocks' in
OlmoHybridSmallForCausalLM`. Cause: wrapping submodules for checkpointing changes their
qualified parameter names, which breaks `olmo_core_to_hf_name`'s regex/dict-based
OLMo-core-to-HF weight renaming (`grpo_callbacks.py`) used to sync weights into vLLM —
the lookup silently misses the wrapped name and falls through to returning the raw,
untranslated name, which vLLM's HF-style loader can't resolve. This is a real bug in the
weight-sync path, not fixable by retrying, and not something addressed here (would need
either patching `olmo_core_to_hf_name` to strip the checkpoint-wrapper's inserted name
segment, or an upstream olmo-core change). **Reverted** the flag — back to no real
activation checkpointing on this script, same as before. **810M think, 1.4B think, and
1.4B open_close_think remain unresolved**, blocked on this weight-sync bug (or the
original GPU OOM, whichever gets fixed first); 810M open_close_think's 0.32 baseline is
the only usable 810M/1.4B result so far. Deprioritizing further attempts at these three
absent a real code fix, rather than continuing to spend GPU-hours on the same failure.

**Collateral OOM in the ongoing 275M/450M GRPO training runs (2026-08-10, ~21:00 UTC):**
while these 3 relaunches (and the earlier round of 4) were running, 3 of the 4
already-stable GRPO training jobs (275M think, 275M open_close_think, 450M
open_close_think) hit the familiar Ray-GCS host-OOM within the same ~20:42-21:08 UTC
window, after 12+ hours of stable running — the tight time correlation with launching 3
more 8-GPU jobs on the same shared clusters is suspicious (possible resource
contention), though not confirmed. Only 450M think survived uninterrupted. Resumed the
other 3 from their latest saved checkpoints (275M think: step400, 275M open_close_think:
step1800, 450M open_close_think: step1000) — see experiment.md for links. Given this is
the second time concurrent large launches have coincided with a training-job crash
wave, worth keeping in mind: avoid launching many new large jobs on `saturn`/`jupiter`/
`ceres` while the training runs are still going, if avoidable.

**First completed run (2026-08-11, ~00:39 UTC): 275M open_close_think finished its full
2000-step training run cleanly** (resumed from step1800, ran to step 2000/2000, "Training
complete", exited 0, final model saved). This is the first genuinely-completed run of
all 8. **Final eval:** `correct_rate` **0.23** (consistent with its long-standing
0.22-0.24 plateau — confirms the plateau held all the way to the end, no late
improvement or regression in accuracy). Two things worth flagging in the final eval,
though, both notably different from its plateau-era numbers: `stop_rate` dropped to
**0.79** (down from ~0.98-0.99 throughout most of training) and average
`sequence_lengths` jumped to **3730** tokens (up from ~800-1000) — i.e. the *final*
policy is generating substantially longer responses and finishing at `eos_token` less
often than it was mid-training, even though final-answer accuracy didn't change. Only
one eval batch, so could be noise, but also could indicate some late-training drift
(e.g. drifting away from the closed `<think></think>` prefix's intended brevity) worth a
closer look at the actual generations before treating this checkpoint as final.

**More OOM tricks tried for 810M/1.4B (2026-08-11), one confirms a real capacity limit:**
added `PYTORCH_ALLOC_CONF=expandable_segments:True` (all launches) plus halved
`response_length`/`pack_length` (8192/16384, `think`-mode retries only). **1.4B
open_close_think still OOMs** — and the diagnostic is now unambiguous: allocator
fragmentation dropped sharply (expandable_segments worked as intended) but *allocated*
memory still climbed to 73-74GB and total usage still hit ~79GB. This means 1.4B's
model + optimizer state + reference policy, sharded across only 4 FSDP-degree learner
GPUs, doesn't fit in an 80GB A100 regardless of sequence length or allocator tuning —
a genuine capacity ceiling, not a memory-management inefficiency. The only remaining
lever is more FSDP shards (more learner GPUs, i.e. changing the 4/4 learner/vLLM
split) or cutting per-GPU memory some other way. Not attempted without explicit
sign-off, since changing the GPU split was previously specifically declined. 810M think
and 1.4B think are still running as of this note, past the point where they previously
failed both times — a good sign, but not yet confirmed successful.

**810M/1.4B think both succeeded (2026-08-11, ~03:30 UTC), and real activation
checkpointing fixed via allenai/open-instruct#1813:** 810M think finished with
`correct_rate` **0.34**; 1.4B think finished with `correct_rate` **0.74** — by far the
highest of any size/mode combo in this whole study (previous max was 810M
open_close_think's 0.32). Both used the halved `response_length=8192`/`pack_length=16384`
workaround, so aren't strictly apples-to-apples with the 16384/32768 everything else
used — if anything this makes 0.74 a *lower bound* on 1.4B think's true capability.

Separately, the user opened and I cherry-picked
[allenai/open-instruct#1813](https://github.com/allenai/open-instruct/pull/1813), which
fixes the actual weight-sync bug diagnosed above: PyTorch's checkpoint wrapper inserts a
`_checkpoint_wrapped_module` path segment into parameter names, and the old
`olmo_core_to_hf_name` didn't strip it before its OLMo-core-to-HF lookup — the PR does,
plus adds regression tests confirming the exact case. With this fix, `--activation_checkpointing_mode
selected_modules` is real and usable again — re-added to `gsm8k_grpo.sh`. Relaunched
1.4B `open_close_think` (the one config that OOM'd for a fundamental capacity reason,
not sequence length) with the real fix and normal-length settings, no more need for the
`response_length` workaround; result pending. If this succeeds, all 8 eval-only
size/mode combos will finally have a real baseline, and it may be worth redoing 810M/
1.4B `think` at full `response_length=16384` too, now that checkpointing works, for a
cleaner cross-size comparison.

**Eval-only baseline `correct_rate` summary across all sizes (2026-08-11):**

| | think | open_close_think |
|---|---|---|
| 275M | 0.12 | 0.06 |
| 450M | 0.05 | 0.14 |
| 810M | 0.34* | 0.32 |
| 1.4B | **0.74**\* | pending |

\*810M/1.4B `think` used `response_length=8192` (half of everything else's 16384),
so treat as a lower bound, not directly comparable.

**PR #1813 fixes one bug but exposes a second, deeper one (2026-08-11, ~03:45 UTC):**
the 1.4B `open_close_think` relaunch with the properly-cherry-picked fix crashed with
the *identical* error as before, at the very first weight sync (before any training
step — checkpoint wrapping applies at model construction, not lazily during backward).
This is **not** the bug #1813 fixed — it's a second, independent gap in the same
name-mapping table. This architecture's GDN mixer is natively named `attention` in
OLMo-core (confirmed by `olmo_core_utils.py`'s own docstring), so `selected_modules`
checkpointing's default module list wraps GDN layers exactly like real attention
layers. After #1813's wrapper-name stripping, a GDN parameter's cleaned name looks like
`blocks.N.attention.in_proj_qkvg.weight` — syntactically valid, but
`_OLMO_CORE_TO_HF_LAYER_MAPPINGS` (`grpo_name_mapping.py`) only has entries for
standard-attention param names (`w_q`/`w_k`/`w_v`/`w_out`/`q_norm`/`k_norm`), nothing
for GDN-specific ones (`in_proj_qkvg`, `A_log`, `dt_bias`, `conv1d`, `o_norm`,
`a_proj`, `b_proj`, `o_proj` — all confirmed present in this model's actual vLLM-side
`linear_attn.*` parameters). So it falls through to the unmapped name again, same
end symptom, different root cause. Also spotted a likely-unrelated third mismatch:
this model's HF/vLLM side uses a **fused** `self_attn.qkv_proj.weight` (+
`attn_gate.weight`), not the separate `q_proj`/`k_proj`/`v_proj` the mapping dict
assumes — not confirmed to matter yet since no standard-attention layer has failed.

**Since this whole architecture is GDN-based, `selected_modules` checkpointing would
break every future checkpointed run regardless of model size** — reverted the script
flag again. **1.4B `open_close_think` remains unresolved.** Real options left: (1)
extend `_OLMO_CORE_TO_HF_LAYER_MAPPINGS` with GDN parameter mappings (a further PR /
code change, not attempted here), or (2) the GPU-split lever (more learner GPUs),
previously declined. Given two independent name-mapping gaps have now surfaced from
the same underlying fix attempt, checkpointing for this architecture likely needs a
more systematic pass over the mapping table (all param name variants across both
attention and GDN layers) rather than another one-off patch.

**First 810M/1.4B GRPO training runs (2026-08-11, ~14:48 UTC), `think` mode only:**
launched using the same `response_length=8192`/`pack_length=16384` override that
already worked for both sizes' `think` eval-only baseline (0.34 and 0.74). Not
`open_close_think` at these sizes — 810M open_close_think already has a clean baseline
without needing training yet, and 1.4B open_close_think is still blocked on the
checkpointing/GDN-mapping gap above, so wasn't launched.

**450M think and 450M open_close_think both completed (2026-08-11):** final
`correct_rate` **0.61** (think) and **0.60** (open_close_think) — essentially identical,
confirming the mode-convergence pattern held all the way to the end at 450M. Both had
healthy stop_rate (0.96) at completion, unlike 275M open_close_think's anomalous final
eval (stop_rate dropped to 0.79) — that earlier anomaly looks specific to that run, not
a general late-training pattern. Also: the ~13:09 UTC "plateau" call for 450M (~0.5
range around step 800-1000) turned out to be a slowdown, not a ceiling — both kept
climbing another ~0.10 to their final ~0.60-0.61.

**Completed-run summary so far:**

| | think | open_close_think |
|---|---|---|
| 275M | in progress (plateauing 0.15-0.18, step 1568) | 0.23 (final) |
| 450M | **0.61** (final) | **0.60** (final) |
| 810M | in progress — jumped 0.33→**0.62** (steps 0→99) | 0.32 (baseline only) |
| 1.4B | in progress (still step 0, 0.74 baseline) | blocked (weight-sync/OOM) |

810M think's early jump (nearly doubling in its first 99 steps) is the steepest
early-training climb seen yet, consistent with 450M's pattern of rapid gains in the
first few hundred steps before slowing down. 275M think, by contrast, still shows no
sign of matching 275M open_close_think's 0.23 — plateauing lower (0.15-0.18) even
approaching its full run length (step 1568/2000), unlike the other sizes where think
and open_close_think converged to the same endpoint. Worth flagging as a possible
genuine model-size-dependent difference (275M may be too small to make think-mode's
extra reasoning pay off) rather than assuming think always catches up.

1.4B think's first post-baseline point: 0.74→0.75 (steps 0→99) — essentially flat,
unlike 810M's steep early jump, consistent with a ceiling effect since 1.4B already
started so high. 275M think still at 0.15 (step 1499 eval; training itself has
progressed to ~step 1600, next eval pending).

**Real root cause of the checkpointing weight-sync bug found (2026-08-11):** PR #1813's
fix was correct but for a mapper that isn't actually used here — `grpo_olmo_core_actor.py`
unconditionally overwrites the generic mapper with a **dedicated**
`make_hybrid_small_name_mapper` (local to this branch) whenever the checkpoint is
detected as `olmo_hybrid_small`, which is every checkpoint in this suite. That
dedicated mapper had the *same class of bug* as #1813 fixed, independently: its
GDN-layer detection and its HF-name lookup both required exact unwrapped submodule
names, so `selected_modules` checkpointing's inserted `_checkpoint_wrapped_module`
segment made it detect **zero** GDN layers and fail every lookup — same symptom, a
second, separate bug. Fixed with the identical wrapper-stripping approach, added to
`open_instruct/olmo_core_utils.py`; 6/6 regression tests pass (3 new, for the
hybrid-small mapper's unwrapped/wrapped-GDN/wrapped-full-attention cases, plus the 3
existing ones for the generic mapper). This fix is local to the hybrid-small-suite
branch, not something to upstream via #1813. Re-enabled `--activation_checkpointing_mode
selected_modules` as the script default.

**1.4B open_close_think finally resolved (2026-08-11, ~17:38 UTC):** third attempt,
with the real fix, succeeded cleanly at the normal `response_length=16384`/
`pack_length=32768` settings — `correct_rate` **0.61**, stop_rate 0.99. No more
workarounds needed for this config. This completes the eval-only baseline across all 4
sizes × 2 modes:

| | think | open_close_think |
|---|---|---|
| 275M | 0.12 | 0.06 |
| 450M | 0.05 | 0.14 |
| 810M | 0.34* | 0.32 |
| 1.4B | 0.74* | **0.61** |

\*810M/1.4B `think` still used the halved-length workaround (launched before the fix),
so treat those two as lower bounds, not directly comparable to the rest. Pattern across
sizes: 275M and 450M favor `open_close_think` or are mixed at baseline; 810M and 1.4B
clearly favor `think` at baseline (0.34>0.32, 0.74>0.61) — though recall from the GRPO
training runs that this baseline gap tends to *close* under RL (275M/450M converged to
near-identical endpoints regardless of starting direction), so whether that holds at
810M/1.4B is still open pending their training runs finishing.

With this config now working, a full 1.4B `open_close_think` GRPO training run could be
launched next (mirroring the `think` training runs already in progress for 810M/1.4B) —
not launched automatically here since it's a further multi-hour compute commitment;
flagging as the natural next step.

**810M think crash root-caused as host-RAM OOM, not GPU (2026-08-11):** died at step
513/2000. No CUDA/GPU OOM anywhere in the log — Ray's own host-memory monitor killed a
worker, and the job hung for ~15 min on the dead worker before an RPC keepalive
timeout finally killed it. Same failure category as the earlier 275M/450M crashes
(all traced to Ray/raylet host-memory pressure, never GPU memory), now confirmed a
third time across different sizes and physical nodes — looks like a systemic host-memory
issue in the pipeline rather than bad luck, though not root-caused to a specific leak
source here. User resumed from the `step400` checkpoint themselves.

**New checkpoint variant with a corrected built-in chat template (2026-08-11):**
`saumyam/hybrid-small-sft-think-450m-lr8e-4/step24834-hf-fixed-chat-template` bakes the
corrected system prompt directly into `tokenizer_config.json`, rather than needing
`gsm8k_grpo.sh`'s `think`/`open_close_think` override templates. Added a `builtin` mode
to the script (omits `--chat_template_name` so the checkpoint's own template is used
unmodified) and launched an eval-only baseline; result pending.

**1.4B think GRPO training: exceptional result (2026-08-12, ~17:00 UTC).** Now at step
1899/2000 (essentially done), `correct_rate` has climbed to **0.90-0.92**, up from its
0.74/0.75 baseline — by a wide margin the best result anywhere in this study (next
highest was 810M think's ~0.62-ish trajectory and the 450M runs' ~0.60-0.61 finals).
Progression: 0.90 (step 1399) → 0.91 (1499-1799, briefly flat) → 0.92 (1899). Given
`response_length=8192` was used for this run (the length-reduction workaround, not the
full 16384), this is an even more striking result — 1.4B think is solving GSM8K at
>90% with half the reasoning budget the smaller sizes get.

**Finished (2026-08-12, ~18:53 UTC): final `correct_rate` 0.92 at step 2000/2000**,
exitCode 0, clean completion — confirms the 0.90-0.92 plateau held all the way through,
no late regression. Completed-run summary across all sizes now:

| | think (final) | open_close_think (final) |
|---|---|---|
| 275M | in progress | 0.23 |
| 450M | 0.61 | 0.60 |
| 1.4B | **0.92** | not yet trained |

1.4B think is the standout result of the whole study — nearly 3x 450M's ceiling, and
achieved with *less* reasoning budget (8192 vs 16384 tokens). 810M think GRPO training
is still in progress (was resumed by the user after its own OOM crash); no completed
number for it yet.

**275M think GRPO finished (2026-08-13, ~03:39 UTC): final `correct_rate` 0.13**,
step 1999/2000, exitCode 0, "Training complete". This is the last of the 8 size×mode
training runs to finish (excluding 810M think, still tracked by the user directly).
Unlike every other size, 275M's `think` mode plateaus **well below**
`open_close_think` (0.13 vs 0.23) rather than catching up to or exceeding it — `stop_rate`
also stayed low throughout the back half of training (0.57-0.64 across the last 4 evals,
never approaching the ~0.9+ seen at other sizes/modes), so 275M `think` likely never
learned to reliably terminate its `<think>` reasoning within the 16384-token budget,
capping its effective score. Final completed-run summary across the whole study
(all 8 size×mode combos, 810M think excluded — still training, user-tracked):

| | think (final) | open_close_think (final) |
|---|---|---|
| 275M | **0.13** | 0.23 |
| 450M | 0.61 | 0.60 |
| 810M | in progress (user-tracked) | 0.32 (baseline only, never trained) |
| 1.4B | **0.92** | 0.61 (baseline only, never trained) |

Takeaways: (1) `think` mode's benefit scales strongly with model size — it's a clear
liability at 275M (worse than open_close_think), roughly on par at 450M, and the
runaway best result at 1.4B (0.92, using half the reasoning budget). (2) Only 275M and
450M got both modes fully trained; 810M and 1.4B only have `open_close_think` and
`think` baselines respectively for the other mode (810M `open_close_think` and 1.4B
`open_close_think` were never launched for full GRPO training in this study — the
baseline eval numbers shown are the only data for those cells).

**275M think's 4th host-RAM-OOM occurrence, resumed near the finish (2026-08-12):**
same Ray host-memory-monitor signature as every prior crash in this session (never GPU
memory) — now 4 occurrences total across 275M/450M/810M. Recovered from `step1600`
(only ~400 steps short of the full run) after two failed resume attempts hit a
transient, unrelated `uv sync` HTTP/2 hiccup fetching the `flash-attn` wheel (plain
`curl` to the same URL worked fine, so not a real connectivity issue — just retry if
this recurs).

**450M fixed-chat-template eval confirms the corrected template is accurate
(2026-08-12):** `correct_rate` 0.05, matching the original 450M `think` baseline
exactly — the checkpoint's own baked-in corrected chat template and the earlier
hand-reconstructed `olmo_hybrid_small_think` override produce equivalent behavior.

**Dolci difficulty scoring (275m/450m) launched and progressing (2026-08-12):** both
jobs cleared the GPU queue, `code_api_setup.sh` came up cleanly on both, and dataset
download is complete/in-progress. No errors yet; this is a long-running job (102,014
rows × up to 16 completions each need per-completion verifier calls across rule-based
math/ifeval, the code API, and the LLM judge) — expect it to take a while.

**Caveat found mid-run: LLM judge context overflow silently zeroes some scores.** The
judge server's `--max-model-len 32768` can't fit some `general-quality`/
`general-quality_ref` prompts (input alone >30.7k tokens) plus the requested 2048
output tokens; those calls fail and get a `0.0` default after 3 retries — not a crash,
but means some long-context general-quality rows in the `_scored` output will have
spuriously low passrates that reflect context overflow, not actual judged quality.
Left running rather than restarting (would lose all progress for a partial-data
issue). Whoever uses the `_scored` datasets should treat exact-0.0
general-quality/general-quality_ref passrates with some suspicion.

**Dolci scoring is much slower than expected — ~5-6 day ETA per model (2026-08-13):**
275m scoring's own tqdm bar surfaced in the logs after ~6h: `7%|▋| 7356/102014
[5:56:27<120:32:30, 4.58s/it]`, i.e. ~126 hours (5.25 days) total wall time at the
observed per-sample rate. Root cause is almost certainly the LLM judge server: every
`general-quality`/`general-quality_ref` row that hits it is paying for up to 3 retries
at ~60-180s each (context-overflow failures are fast, but the `litellm.Timeout`s
observed alongside them take a full 181s per attempt under load) before falling back
to 0.0, and with a single Qwen3-32B judge instance serving both the 275m and 450m
scoring jobs concurrently, throughput is bottlenecked on judge-server capacity rather
than CPU/verifier speed. Not a crash, no action taken — flagging so nobody expects
`_scored` datasets imminently; if the wait is unacceptable, the fix would be scaling
out the judge server (more replicas) or splitting `general-quality*` scoring into its
own lower-concurrency pass, not something to change mid-run.

**Throughput improved substantially since the initial ETA estimate (2026-08-13,
~24h in):** 450m scoring's tqdm bar now reads `33%|███▎| 33323/102014
[24:07:26<39:13:17, 2.06s/it]` — per-sample time roughly halved (2.06s/it vs the
original 4.58s/it), dropping the remaining-time estimate from ~120h to ~39h. Likely
explanation: judge-server load evens out over a long run as the mix of
`general-quality`-heavy vs rule-based-verifier-heavy rows varies, or judge-server
contention eased. Still a multi-day job, but meaningfully better than the initial
worst-case estimate; no action needed, just an updated expectation.

**Throughput fluctuates — treat any single ETA as noisy (2026-08-14, ~45.5h in):**
450m scoring is now 60% done (61625/102014) but tqdm's own ETA has regressed to
36-55h remaining (worse than the ~39h estimated at the 33% mark a day earlier),
per-sample rate bouncing between ~2s/it and ~5s/it across checks. Confirms the
judge-server-contention theory (shared with 275m's concurrent job, load varies with
whatever else is hitting the judge at a given moment) rather than a steady-state rate
— don't treat any single tqdm reading as a reliable completion forecast, just a rough
sense of "still multi-day." Job otherwise healthy, no new error types.

**450m scoring at 80%, nearing completion (2026-08-15, ~61h in):** 80944/102014 rows
done, tqdm ETA ~25h remaining. Still healthy, no new error types — should complete
sometime around 2026-08-16.

**Both scoring jobs approaching completion (2026-08-15, ~68.5h in):** 275m is at 96%
(97559/102014, ETA ~2h) — should finish imminently. 450m is at 90% (91534/102014, ETA
~8.5h) — also close. Both still healthy, no new error types.

**275m scoring completed the scoring pass (102014/102014, ~20:15:30 UTC 2026-08-15,
~72h wall time) and moved into the post-scoring `push_to_hub` step**
(`filter_existing_dataset_correctness.py:188` logged "Filtered 0 samples out of
102014" as expected, since `--lower_bound 0 --upper_bound 1` keeps every row). No new
log lines appeared for 40+ minutes after that and the target dataset
(`mnoukhov/dolci_think_rl_7b_messages_hybrid_275m_scored`) doesn't exist on the hub
yet as of this check — the job hasn't exited either way (no crash, no success signal).
Plausible explanation: `push_to_hub` for a 102k-row dataset with long generation-text
columns can take a long time to serialize/upload without incremental per-shard
logging showing up in the captured beaker logs, so this isn't necessarily a hang —
but it's also not confirmed healthy. Continuing to monitor closely; if it's still
silent much longer this is worth someone checking the raw pod logs directly (beaker
logs may just not be capturing HF's own upload progress bar).

**ESCALATION — 275m scoring job likely genuinely stuck in `push_to_hub` (2026-08-15,
~1h50m silent):** as of ~22:04 UTC, still no new log output since the 20:15:33 UTC
"Filtered 0 samples" line, still no `exited` status, and
`mnoukhov/dolci_think_rl_7b_messages_hybrid_275m_scored` still doesn't exist on the
hub (checked via the HF API). By contrast, **450m scoring (running the identical
`push_to_hub` code path, just not there yet) is still progressing completely
normally** — 95% done, actively logging new tqdm lines every few minutes as of the
same check. This makes "large dataset, slow upload with no logging" a much less
likely explanation than it seemed at the 40-minute mark, since there's nothing
architecturally different about 275m's push that should make it dramatically slower
or silent compared to 450m's (same row count target, same columns, same HF library
call). Genuinely looks stuck (network stall, silent exception swallowed somewhere, or
a hung Beaker log stream not actually reflecting a hung process — can't distinguish
these from Beaker's log API alone). **Not killing the job** — no destructive action
taken, this needs a human decision (worth checking Beaker's own dashboard/node view
directly, or considering a restart of just the push step from the already-written
local `/tmp/hybrid_275m_scored.jsonl` if the process is confirmed dead, since all the
expensive scoring compute already completed and shouldn't need to be redone).

**ROOT CAUSE FOUND — host OOM kill, not a hang (2026-08-15, resolved after ~3h11m
silent):** `01KZVN9K504CT8J3AHZAS3NDEY` finally exited at 23:26:10 UTC with
**`exitCode 137`, `message: "out of memory"`** — the process was silently killed by
the host's OOM killer (SIGKILL, so no Python traceback or error log — explains the
total silence), not stuck on a network call. A `multiprocessing.resource_tracker`
"leaked semaphore" warning appeared ~1 minute before the kill, consistent with the
OS reclaiming memory out from under a live process. Root cause is almost certainly
`datasets.Dataset.push_to_hub()`'s in-memory serialization of the full 102,014-row
scored dataset (with long `messages`/generation-text columns) exceeding the
container's memory limit — the scoring pass itself (streaming, incremental) never
came close to this, but building + serializing the final Arrow/Parquet table
in-memory for upload apparently does.

**Consequence: the 275m scored dataset was lost.** The only copy of the fully-scored
data was the local `/tmp/hybrid_275m_scored.jsonl` inside that now-dead container —
it was never pushed to the hub before the kill, and Beaker containers don't persist
local disk after exit. All ~72h of scoring compute for 275m needs to be re-run (or
the push step needs to be restructured to avoid the memory spike — e.g. streaming
push, sharded push, or writing the scored jsonl to persistent /weka storage as a
safety net before attempting push_to_hub) before this model's difficulty scores can
be recovered.

**450m is at serious risk of hitting the identical failure**: as of this same check
it's at 99% (100927/102014) and will enter the same `push_to_hub` step within
minutes. Flagging this urgently — if it OOMs the same way, both scoring jobs will
need a full redo. Nothing to intervene with proactively (can't resize a running
container's memory limit or preemptively change its code), just watching closely and
will flag immediately if it recurs.

**450m SUCCEEDED — narrowly survived the same risky window (2026-08-16, exited
02:36:15 UTC, exitCode 0):** `01KZVNBH8BF3327WV96RB04XR1` finished cleanly.
`mnoukhov/dolci_think_rl_7b_messages_hybrid_450m_scored` is live on the hub
(confirmed via HF API, `lastModified` 02:35:22 UTC). This resolves the OOM question:
**both jobs hit the identical silent gap** (finished scoring → "Filtered 0 samples" →
total silence with no log output) — 275m's gap ended in an OOM kill at ~3h11m, 450m's
ended in success at ~2h28m. The final log lines from 450m show *why* the gap is so
long and silent: the actual `push_to_hub` shard upload (151 shards, ~235MB) only
took **~37 minutes** once it started and *did* log progress — so roughly **~2 hours
of the silent gap is spent building/serializing the in-memory Arrow table before any
upload progress appears**, and that serialization step is where the OOM risk lives
(not the upload itself). 450m's serialization apparently fit in memory (barely,
given how long it took — possibly swapping/thrashing near the limit) where 275m's
didn't; this reads as a **timing/memory-pressure-dependent failure, not fully
deterministic** — re-running 275m's scoring+push under the same code isn't
guaranteed to fail again, but isn't guaranteed to succeed either.

**Recommendation for any future re-run of 275m** (or any similar scoring job): before
attempting `push_to_hub`, first write the scored dataset to persistent `/weka`
storage (not just `/tmp`) so a mid-push OOM doesn't lose the ~72h of scoring compute
again, and/or investigate whether `push_to_hub` can be made to serialize
incrementally/in shards rather than building the full in-memory table upfront (e.g.
`Dataset.to_parquet` in chunks, or `max_shard_size` tuning) to reduce peak memory.

**This closes out the Dolci difficulty-scoring pipeline for this session**: 450m
scoring succeeded end-to-end (`mnoukhov/dolci_think_rl_7b_messages_hybrid_450m_scored`
live); 275m scoring completed all compute but lost its final output to an OOM kill
during push and needs a re-run (with the persistence fix above) to actually produce
`mnoukhov/dolci_think_rl_7b_messages_hybrid_275m_scored`.

See [experiment.md#olmo-hybrid-small-think-vs-open-close-think-gsm8k-grpo](experiment.md#olmo-hybrid-small-think-vs-open-close-think-gsm8k-grpo).


## Olmo Hybrid release 275m GSM8K learning and resume

**Status:** ACTIVE — learner BF16 forward fix committed; corrected 1,000-update Saturn rerun submitted.

Does correctness-only GRPO on the primary 275m midtrain release improve held-out
GSM8K accuracy, and does the run restore policy/optimizer/data state after preemption?
The recipe uses full train/eval splits, four prompts x four samples, and 1,000 updates,
with native checkpoints every 100 updates. All-zero correctness groups are retained
so collection terminates; that can still produce no learning signal.

Resume finding: the old initial vLLM broadcast ran before native Trainer.fit restored
its checkpoint. A pre-train callback now republishes restored weights and model step
before resumed rollouts are requested. Fresh/resumed CPU regressions pass. Actual
preemption recovery is pending; the stored Beaker spec enables autoResume, the Weka
checkpoint directory remains fixed, and W&B uses one stable ID across restarts.

Numerical finding: BF16 eager/FlashAttention pointwise logit comparisons were too
strict near zero. FP32 math-attention interchange passed; at production lr 1e-6,
a full 2560-token BF16 backward/update and HF export/reload passed. After the update,
HF/native BF16 max next-token KL was 0.00044331 and BF16/FP32 max KL 0.00051355.
A synthetic probe at 100x production lr showed much larger precision sensitivity;
that did not reproduce at the production rate. These are short synthetic checks,
not GSM8K learning or full serving/decode qualification.

[GPU gate](https://beaker.org/ex/01M5H2C69J8HBS8F70X3HT2K3C) passed.
[Training](https://beaker.org/ex/01M5HYS6EMRWQ01SDFJW85F1GF) failed on its initial attempt and all three retries before the first update.
The real GRPO learner bypassed OLMo-core's forward autocast context, leaving FlashAttention inputs in FP32 with one learner and no FSDP cast. The standalone numerical gate used its own autocast and did not cover this path. GRPO now sets autocast from `model_dtype` and wraps learner, old-policy, and reference-policy forwards; optimizer parameters remain FP32. CPU regression tests cover all three actual forward paths.
[Corrected rerun](https://beaker.org/ex/01M5ZAPEQFTT3YEMK107ZSVFK4) is submitted with the verified 2-GPU/auto-resume/every-100 checkpoint spec. See [the run log](experiment.md#olmo-hybrid-release-275m-bf16-rerun-2026-10-27). No end-to-end RL result is claimed.
See [the run log](experiment.md#olmo-hybrid-release-275m-gsm8k-1000-updates-2026-10-22).


## Temporary MILES hybrid serving qualification

**Status:** ACTIVE — five-scale BF16 serving qualified; native MILES RL validation pending.

Can a temporary bridge reuse the pinned release Gated DeltaNet implementation
and SGLang scheduling without requiring a full custom inference kernel?
Implementation lives on `codex/hybrid-mainline` in the sibling
`miles-olmo-core-hybrid` and `open-instruct-miles-hybrid` checkouts. The original
Saturn rerun continues on Open Instruct/Core/vLLM.

Finding: compare cached serving with HF cached decode as well as full-sequence
HF evaluation. At 810m and a one-token prefix, HF cached-versus-full KL reaches
0.00176434, exactly reproducing the initial SGLang-versus-full result. An FP32
output head alone only reduces this to 0.00151127. This is a real precision/path
difference relevant to RL logprobs, not sufficient evidence of a serving bug.
The expanded qualification retains both diagnostics and requires separate
selected-token and full-distribution parity with the cached reference.

See [the interactive experiment](experiment.md#interactive-rhea-sglang-qualification-2026-10-27)
and [attachment instructions](BEAKER_ATTACH.md#interactive-rhea-gpu-qualification-2026-10-27).


Serving finding (2026-10-27): all five final-midtrain sizes passed the BF16
reference-backbone fallback on Rhea: 160 functional requests and 40 dense probes,
including long prompts, queued requests, slot reuse, and disk reload. Maximum
selected-token logprob error stayed below 5e-4 and greedy agreement was 100%.
This selects `MILES_HYBRID_REFERENCE_BACKBONE=1` with BF16 logits for the temporary
recipe. It localizes the earlier discrepancy to differences introduced by the
faster packed/RadixAttention path, without isolating one specific operation.
That path remains unqualified across the whole family. The fallback is slower
(roughly 62–67 output tokens/s at 275m and 20–22 at 2.7b for these warm batches).
HF cached-versus-full discrepancies remain, so native RL publication, importance
ratios, restart recovery, and learning evidence are still pending; the overall
MILES integration question remains ACTIVE.


## GSM8K 500-update comparison of Core/vLLM and MILES

Status: ACTIVE — 450m smoke passed; both 500-update jobs scheduled; 275m deferred.

Do the two stacks show useful learning and acceptable throughput on the275m and
450m final-midtrain releases? Four fresh runs use500 updates, eval/checkpoint
every100, BF16, and a hard one-update policy-age limit. Preserve existing reward
and optimizer differences per user instruction, except match PPO upper clipping
to0.272. Compare held-out scores, valid reward groups, wall time/update, serving
throughput, consumed policy age/stale drops, and checkpoint/restart behavior.
The initial serving matrix qualifies the BF16 HF fallback, not the native RL path.

See [the preparation batch](experiment.md#gsm8k-infrastructure-comparison-275m-and-450m-500-updates-2026-10-27).


### 450m-first launch update (2026-10-27, queue refresh 2026-10-28 UTC)

User narrowed the approved comparison to 450m: one two-A6000 MILES smoke,
then two 500-update jobs, one learner plus one inference A6000 each (four peak),
only after the smoke succeeds. All GPU jobs use ai2/rhea, normal/preemptible,
minRuntime 0, budget ai2/oe-other and workspace ai2/open-instruct-dev.
Main jobs retain auto-resume, three retries, 24h per attempt, BF16,
max policy lag 1, eval/checkpoint every 100, and PPO clip lower0.2/upper0.272.
275m configs are prepared but no new275m GPU runs are authorized by this batch.

Smoke attempts:
- [01M5ZFP2Z9D41ZFW6YW4AJR85C](https://beaker.org/ex/01M5ZFP2Z9D41ZFW6YW4AJR85C): stopped before training; hard-coded256GiB host RAM requested a third slot. Fixed configurable128GiB RAM/32GiB shm; later attempts acquired exactly two GPUs.
- [01M5ZFZGNFPMQ8F3PPVXG3EACQ](https://beaker.org/ex/01M5ZFZGNFPMQ8F3PPVXG3EACQ): BF16 attention forward/backward passed; failed before training because Ray manager rendered ServerArgs before registering mainline_ladder. Fixed in pushed miles-olmo-core hybrid branch, commit a5ec0c61.
- [01M5ZGAG9ZB0NSFGN1172QJCNB](https://beaker.org/ex/01M5ZGAG9ZB0NSFGN1172QJCNB): registration passed, but SGLang dtype=auto selected non-BF16; server rejected startup and controller waited until the one-hour timeout. Fixed explicit sglang_dtype=bfloat16 plus a600-second serving-startup deadline in companion commit3f0933565.
- [01M608040GER8BMAC4E8DRQ7Q7](https://beaker.org/ex/01M608040GER8BMAC4E8DRQ7Q7): corrected BF16 smoke, currently queued because Rhea lacks two free slots. Image01M607Z48PYPNN0KHVZ5PGYJEG.

Native MILES learning remains unvalidated. A one-time Hammond controller,
`scripts/miles/hybrid_compare_450m.py` in the companion checkout, waits for this
smoke and checks two completed updates, successful publication/evaluation,
committed native checkpoints and final HF export before submitting the two main
runs through build_image_and_launch.sh. It fails closed on missing evidence,
worker failure or an interrupted submission. Durable controller status and launch
receipts live in `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/runs/compare-450m-controller-20261028`.


2026-10-28 05:04 UTC launch update: corrected BF16 smoke completed both learner
updates with successful publication/evaluation stages, two committed native
checkpoints and final HF export. The retry-policy guard caught a missing Beaker
retry field; its first main submission was stopped before allocation. Fix pushed
in companion commit f22385319 (100 focused tests and lint/type checks passed).

The approved main pair is now submitted and scheduled, each with exactly two
A6000s on Rhea. Verified autoResume=true, minRuntime=0s, allowedTaskRetries=3,
500 updates, eval/checkpoint100, BF16, max policy lag1. No275m comparison submitted.
- MILES: [01M60E822XVYFNHWJ97N6WDBY2](https://beaker.org/ex/01M60E822XVYFNHWJ97N6WDBY2), image01M607Z48PYPNN0KHVZ5PGYJEG.
- Core/vLLM: [01M60E9H8YVK2VJNWPE4BHBFM0](https://beaker.org/ex/01M60E9H8YVK2VJNWPE4BHBFM0), image01M6087KP6KYPHQ9MMZTEX9SBJ.
Controller status: both_submitted, under runs/compare-450m-controller-20261028-retry
on Weka. Main-run optimizer progress and learning quality are not yet verified.

Separate Saturn275m/1k run01M5ZAPEQFTT3YEMK107ZSVFK4 has suffered six scheduler
preemptions by urgent allocated workloads. Current job01M604JS88XG6RFSGNF2Z0FWB1
has waited almost three hours; scheduler reports no slots on any of27 eligible
nodes. Cluster snapshot:216slots, zero available,161queued jobs. Run unchanged.


## Fresh 450m Core/vLLM restart on four Rhea GPUs (2026-10-29)

Status: ACTIVE — image built; replacement GPU smoke queued after scheduler preemption.

User requested a fresh RL run on four GPUs on Rhea after the previous run
stalled. Interpret this as the 450m Core/vLLM run under discussion, two learner
A6000s (FSDP shard degree2) and two independent TP1 vLLM engines. Keep final
midtrain step0038147, BF16/FA2,500 updates, four prompts x four samples,
learning rate1e-6, lower/upper PPO clip0.2/0.272, maximum policy lag1,
eval/native checkpoint/HF save every100, normal/preemptible, minRuntime0,
auto-resume, three retries, timeout24h per attempt, workspace ai2/open-instruct-dev,
budget ai2/oe-other. Fresh output root, no old optimizer/checkpoint restoration.
First validate two updates on the same four-GPU split with a one-hour cap and
zero retries; start the500-update run only after actual updates and publication.

Stopped old Core/vLLM experiment01M60E9H8YVK2VJNWPE4BHBFM0 to replace it.
The275m Saturn experiment remains canceled. MILES step200 checkpoint is preserved.

Root cause in the non-inflight publication path: broadcast_weights_to_vllm put
vLLM's scheduler to sleep, then update_weights waited for active HTTP completions
to finish. Requests queued before or during that boundary could never finish
until wake_up, which itself waited for publication. Fix by atomically pausing
local request admission, draining active completions while the engine remains
awake, then sleeping and publishing. Reopen admission after wake succeeds;
raise a clear timeout after600seconds of unsuccessful draining. Preserve the
existing intentionally-inflight mode. Regression tests cover ordering, bounded
failure, and a request dequeued concurrently with publication.

Parameterize the existing release launcher for learner/engine counts, preserving
its one-plus-one defaults. Add --existing-image-only to the standard image
wrapper so local submission cannot accidentally fall through to a Docker build.
The replacement image is an overlay of qualified Core/vLLM image
01M6087KP6KYPHQ9MMZTEX9SBJ and must be built/published on Beaker.

Stall monitoring is now ACTIVE in this chat, every10minutes. It compares actual
optimizer/generation/eval progress, flags20minutes without progress, and reports
new failures, stalls, recoveries and completions without repeated unchanged pings.
It does not stop or allocate jobs. W&B internal_step is not optimizer progress.

See [the launch batch](experiment.md#fresh-450m-corevllm-restart-on-four-rhea-gpus-2026-10-29).


2026-10-29 19:07 UTC launch update:
- User explicitly confirmed Core/vLLM alone, two learner plus two inference GPUs.
- Source fix: `652c880ebc1ae13206e642629c3f1226037b7452` on pushed
  `hybrid-small-suite`; 22 focused CPU tests and `make quality` passed.
- Built on Beaker Rhea (one slot, released before the smoke):
  [builder](https://beaker.org/ex/01M64F9F8NRCCJVAWWE14ZCP9F), exit0;
  immutable image `01M64G5M526S9HNF960NP3QRVB`, overlaying qualified image
  `01M6087KP6KYPHQ9MMZTEX9SBJ`. No laptop Docker build.
- [First four-GPU smoke](https://beaker.org/ex/01M64G7EY7YN20B455DW3QDWJK)
  started18:56 UTC, then was scheduler-preempted19:05:28 UTC by an urgent
  allocated workload. Both learners loaded weights; both inference engines
  completed initial weight sync; startup backward kernel compilation completed.
  No optimizer update verified. This is an interrupted smoke, not a pass.
- [Replacement smoke](https://beaker.org/ex/01M64GW3PMAGVNFRQ74GX7WFQ2)
  submitted19:06 UTC, queued for four Rhea slots as of19:07 UTC. Run name
  `hybrid-450m-core-vllm-rhea4-smoke2-20261029`, Weka output under
  `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/runs/`.
  Two updates; eval eight examples and native/HF checkpoints every update;
  timeout1h/attempt, zero application failure retries, autoResume=true,
  minRuntime0. Auto-resume was enabled after the scheduler eviction.
- Main500-step run is NOT submitted yet. A one-time local controller waits for
  smoke exit0 plus actual step2 checkpoint, publication, and final cleanup before
  submitting `hybrid-450m-core-vllm-rhea4-20261029` using the standard wrapper:
  `MODEL_SIZE=450m NUM_LEARNERS=2 NUM_INFERENCE_ENGINES=2 CLUSTER=ai2/rhea
  RUN_NAME=hybrid-450m-core-vllm-rhea4-20261029 AUTO_RESUME=true MAX_RETRIES=3
  JOB_TIMEOUT=24h ./scripts/train/build_image_and_launch.sh --existing-image-only
  scripts/train/olmo-hybrid-small/gsm8k_infra_compare_500.sh`.
  Main settings remain BF16,500 updates, max policy age1, eval/save/native
  checkpoint100, normal/preemptible, minRuntime0, autoResume, three failure
  retries,24h/attempt. No old RL optimizer state is loaded.
- Controller `/tmp/hybrid-rhea4-resume.py` runs from an isolated clean source
  snapshot at `/tmp/hybrid-rhea4-launch-652c880eb` so recording launch receipts
  cannot invalidate the source/image guard. State: `/tmp/hybrid-rhea4-state.json`;
  log: `/tmp/hybrid-rhea4-resume.log`. It is local and depends on this host staying
  alive. Reconcile state/Beaker before manually restarting to avoid duplicates.
- Heartbeat `hybrid-training-progress-and-stalls` checks every10minutes and flags
  twenty minutes without actual optimizer/generation/eval progress; it also
  reports controller failure. Monitoring is observational; it cannot resubmit.


## Fresh 450m MILES run in personal W&B project (2026-10-30)

Status: ACTIVE — prepared for a fresh500-update run.

The earlier comparison used different W&B entities: Core/vLLM logged to
`mnoukhov/open_instruct_internal`, MILES to `ai2-llm/open_instruct_internal`.
The Core/vLLM reference is now complete at500updates/17.287% accuracy; previous
MILES reached durable200updates/10.670% before its restart cap. User requested
fresh MILES with the same training recipe and personal W&B entity. Keep all
training/resource parameters and qualified runtime; change only fresh run/output
identity and tracking entity/group. New checkpoints begin from the original
450m final-midtrain weights, with a fresh optimizer and data cursor.

See [launch configuration and receipts](experiment.md#fresh-450m-miles-run-in-personal-wb-project-2026-10-30).


2026-10-30 correction: user selected four GPUs, split2 learners+2 inference.
The initial two-GPU submission was canceled while queued, with no training.
The replacement retains the same objective and personal W&B namespace, increases
producer concurrency4->8 to fill two inference engines, and uses a fresh run/output
identity. Status ACTIVE — prepared for the four-GPU submission; see the launch
batch above for the old canceled experiment and validation limits.
