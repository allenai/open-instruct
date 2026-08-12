# Research

Questions/ideas being pursued. One section per idea, each linking to its section(s) in `experiment.md`.

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

**275M think's 4th host-RAM-OOM occurrence, resumed near the finish (2026-08-12):**
same Ray host-memory-monitor signature as every prior crash in this session (never GPU
memory) — now 4 occurrences total across 275M/450M/810M. Recovered from `step1600`
(only ~400 steps short of the full run) after two failed resume attempts hit a
transient, unrelated `uv sync` HTTP/2 hiccup fetching the `flash-attn` wheel (plain
`curl` to the same URL worked fine, so not a real connectivity issue — just retry if
this recurs).

See [experiment.md#olmo-hybrid-small-think-vs-open-close-think-gsm8k-grpo](experiment.md#olmo-hybrid-small-think-vs-open-close-think-gsm8k-grpo).
