# Experiment Log

Raw run log — configs, launch commands, Beaker links. Appended per launch batch.
See `research.md` for the questions/ideas each entry is in service of.

## OLMo-hybrid 275M GRPO GSM8K

Related: [research.md#olmo-hybrid-275m-small-suite-grpo-on-gsm8k](research.md#olmo-hybrid-275m-small-suite-grpo-on-gsm8k)

### 2026-07-20 run 1

- Script: `scripts/train/debug/olmo_hybrid_275m_4gpu_gsm8k.sh`
- Launch: `./scripts/train/build_image_and_launch.sh scripts/train/debug/olmo_hybrid_275m_4gpu_gsm8k.sh`
- Config: 4 GPUs (2 training FSDP shard_degree=2, 2 generation vLLM engines), model
  `hybrid-small-sft-think-275M-lr2e-4/step23206-hf`, dataset `ai2-adapt-dev/rlvr_gsm8k_zs`,
  eval `mnoukhov/gsm8k-platinum-openinstruct`, beta=0.0, lr=1e-6, response_length=2048, pack_length=4096.
- Beaker: [01KY0GWKBZFE1AV5TATNHJRKVT](https://beaker.org/ex/01KY0GWKBZFE1AV5TATNHJRKVT)
- Result: Killed at step 183/1000 (18.3%) — stop rate only ~12%. Hypothesis: this is a
  "-think-" SFT checkpoint whose `<think>...</think>` + answer traces don't fit in
  `response_length=2048`, so most rollouts get truncated before emitting `eos_token`
  rather than genuinely failing to stop. Chat template (`olmo_thinker`) checked and
  looks correct: opens generation with `<|im_start|>assistant\n<think>` matching this
  checkpoint, uses the same `eos_token`-terminated pattern as every other OLMo template
  in `dataset_transformation.py`, and matches the template used by the sibling SFT/DPO
  debug scripts for this same checkpoint. Re-launching with `response_length=8192`,
  `pack_length=16384` (must satisfy `pack_length >= max_prompt_token_length + response_length`,
  see `data_loader.py:501`) to test the truncation hypothesis before looking further at
  the template/tokenizer.

### 2026-07-20 run 2

- Script: `scripts/train/debug/olmo_hybrid_275m_4gpu_gsm8k.sh` (response_length 2048→8192, pack_length 4096→16384)
- Launch: `./scripts/train/build_image_and_launch.sh scripts/train/debug/olmo_hybrid_275m_4gpu_gsm8k.sh`
- Beaker: [01KY2MJSAHCXYFG0ZABBDQDHAW](https://beaker.org/ex/01KY2MJSAHCXYFG0ZABBDQDHAW)
- Result: TBD (run in progress)

## Hammond checkout comparison 2026-10-21

Related: [research.md#hammond-checkout-comparison](research.md#hammond-checkout-comparison).

- Control session only, not a training run or GPU test.
- Beaker: [01M5FZAJ0GKWGRBYFRCHCW3CCG](https://beaker.org/job/01M5FZAJ0GKWGRBYFRCHCW3CCG).
- Launch: the `beaker session create` command in [BEAKER_ATTACH.md](BEAKER_ATTACH.md#replace-an-ended-session), matching `beakernew` except the unsupported `--host-networking` flag.
- Config: `ai2/hammond`, workspace `ai2/michaeln`, budget `ai2/oe-other`, priority
  `low`, image `ai2/cuda12.8-dev-ubuntu22.04-notorch`, detached/bare, no explicit
  CPU/memory/GPU request or timeout. Existing Weka/secret mounts retained.
- User explicitly approved creating the replacement session. Previous session
  `01M46TWNTH609VQJD2KAM06MAT` was canceled October 17 when its node became unavailable.
- Result: session ready on `hammond-cs-aus-452.reviz.ai2.in`. The persistent
  checkout is `/weka/oe-adapt-default/michaeln/nit-open-instruct`, branch
  `hybrid-small-suite`, clean working tree, HEAD
  `e2d831a5ca7ad8d62ff3e554460753ec17ae62bb`. Local HEAD before documentation edits
  was `9584735ef421640cc282b19fccaf0f9223466e36`, also clean.
- Cluster changes since local's parent `8fc89d9584913b93d56f46cf6b566b7a316140ff`:
  76 commits, 27 files, 2,103 insertions and 85 deletions. Includes GRPO/GDN
  mapping and checkpointing changes, template handling, Dolci scoring scripts,
  a general `gsm8k_grpo.sh`, and expanded experiment/research logs.
- Local's additional commit changes two evaluation launch files:
  `scripts/eval/oe-eval.sh` and `scripts/submit_eval_jobs.py` (717 insertions,
  212 deletions). Neither is in the cluster's changed-file list from that parent.
- Exact comparison: 1 local-only commit and 76 cluster-only commits, merge base
  `8fc89d9584913b93d56f46cf6b566b7a316140ff`. Local HEAD to cluster HEAD changes
  29 files with 2,315 insertions and 802 deletions.
- A cluster GitHub fetch stalled and was stopped; a temporary small Git bundle
  was transferred to compare local history without changing the remote branch
  or working tree. No training or GPU tests were launched.
- Local validation: Ruff lint/format checks, `ty check`, four focused CPU tests
  in `open_instruct/test_grpo_fast_resource_plan.py`, and MkDocs build passed
  (MkDocs emitted existing link/deprecation warnings).
