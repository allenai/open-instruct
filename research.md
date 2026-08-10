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

See [experiment.md#olmo-hybrid-small-think-vs-open-close-think-gsm8k-grpo](experiment.md#olmo-hybrid-small-think-vs-open-close-think-gsm8k-grpo).
