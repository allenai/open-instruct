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
Chat template (`olmo_thinker`) was audited and looks correct/consistent with the rest of
the codebase and the sibling SFT/DPO scripts for this checkpoint. Leading hypothesis is
that this "-think-" SFT checkpoint's reasoning traces don't fit in 2048 tokens, so
rollouts are truncated before `eos_token` rather than genuinely failing to terminate.
Testing with `response_length=8192` in run 2.

See [experiment.md#olmo-hybrid-275m-grpo-gsm8k](experiment.md#olmo-hybrid-275m-grpo-gsm8k).
