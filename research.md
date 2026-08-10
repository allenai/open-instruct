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

See [experiment.md#olmo-hybrid-small-think-vs-open-close-think-gsm8k-grpo](experiment.md#olmo-hybrid-small-think-vs-open-close-think-gsm8k-grpo).
