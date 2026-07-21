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
- Beaker: TBD
- Result: TBD
