# Research

Questions/ideas being pursued. One section per idea, each linking to its section(s) in `experiment.md`.

## Hammond checkout comparison

**Status:** RESOLVED

The checkouts differ. The cluster checkout is named `nit-open-instruct` at
`/weka/oe-adapt-default/michaeln/nit-open-instruct`, on `hybrid-small-suite` at
`e2d831a5ca7ad8d62ff3e554460753ec17ae62bb`. Local HEAD is
`9584735ef421640cc282b19fccaf0f9223466e36` on the same branch. Both working trees
were clean before the local comparison documentation was added.

The cluster has 76 commits after `8fc89d9584913b93d56f46cf6b566b7a316140ff`,
including GRPO/GDN weight-sync and activation-checkpointing changes, chat-template
handling, Dolci scoring scripts, and extensive run findings. Local has a separate
evaluation-launch fix touching `scripts/eval/oe-eval.sh` and
`scripts/submit_eval_jobs.py`. These changes need reconciliation before treating
the two checkouts as interchangeable. No branch or working files were synced.

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
Chat template (`olmo_thinker`) was audited and looks correct/consistent with the rest of
the codebase and the sibling SFT/DPO scripts for this checkpoint. Leading hypothesis is
that this "-think-" SFT checkpoint's reasoning traces don't fit in 2048 tokens, so
rollouts are truncated before `eos_token` rather than genuinely failing to terminate.
Testing with `response_length=8192` in run 2.

See [experiment.md#olmo-hybrid-275m-grpo-gsm8k](experiment.md#olmo-hybrid-275m-grpo-gsm8k).
