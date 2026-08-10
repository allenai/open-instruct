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

### 2026-08-07 chat template re-audit

The run-1 note above ("chat template checked and looks correct... matches the template
used by the sibling SFT/DPO debug scripts") was **incomplete** — it only compared
`olmo_thinker` (code, `dataset_transformation.py:347-386`) against sibling *scripts*,
which also pass `--chat_template_name olmo_thinker`, not against the checkpoint's own
baked-in template. Read `tokenizer_config.json` directly from
`/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr2e-4/step23206-hf/`
(no separate `chat_template.jinja` present, so this field is authoritative) and diffed
it against `olmo_thinker`:

- **Matches:** `<|im_start|>`/`<|im_end|>` role framing, user/assistant/environment role
  shape, `eos_token`-terminated last turn, and — critically for the truncation
  hypothesis — the generation-prompt tail `<|im_start|>assistant\n<think>`.
- **Diverges:** the default system prompt used when no system message is present in the
  data (true for `ai2-adapt-dev/rlvr_gsm8k_zs`, a zero-shot dataset). `olmo_thinker`
  emits `"You are a helpful AI assistant.<|im_end|>\n"`. The checkpoint's real template
  emits `"You are a helpful function-calling AI assistant. You do not currently have
  access to any functions. <functions></functions><|im_end|>\n"`. The checkpoint's
  template also supports OpenAI-style `tool_calls` and a `tool` role that `olmo_thinker`
  has no equivalent for (not exercised by GSM8K data, so lower-priority).
- Since GSM8K prompts carry no system message, every single training/rollout sequence
  gets a system-prompt prefix the model never saw during SFT. This is a plausible
  secondary contributor to the low stop rate in run 1, on top of the truncation
  hypothesis.
- The "final" 7B hybrid instruct scripts (`scripts/train/olmo-hybrid/7b_instruct_dpo.sh`
  etc.) avoid this by passing `--chat_template_name olmo123` — a name that satisfies the
  `"olmo" in chat_template_name` assertion in `dataset_transformation.py` but isn't a key
  in `CHAT_TEMPLATES`, so the tokenizer's own baked-in template is used unmodified. Worth
  trying the same pattern (drop `--chat_template_name olmo_thinker`, or pass a name not
  in the dict) for the small-suite debug scripts once run 2's truncation-hypothesis
  result is in.

## OLMo-hybrid small suite: think vs open-close-think GSM8K GRPO

Related: [research.md#olmo-hybrid-small-suite-think-vs-open-close-think-gsm8k-grpo](research.md#olmo-hybrid-small-suite-think-vs-open-close-think-gsm8k-grpo)

Script: `scripts/train/olmo-hybrid-small/gsm8k_grpo.sh` (moved out of
`scripts/train/debug/`; supersedes `olmo_hybrid_gsm8k.sh`). Two new chat templates
added to `open_instruct/dataset_transformation.py`'s `CHAT_TEMPLATES`:
`olmo_hybrid_small_think` (generation prompt opens `<think>`) and
`olmo_hybrid_small_open_close_think` (generation prompt pre-fills a closed
`<think>\n\n</think>\n\n`), both using the checkpoints' real default system prompt
instead of `olmo_thinker`'s. `response_length` 8192→16384, `pack_length` 16384→32768.

Models:
- 275M: `/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr2e-4/step23206-hf`
- 450M: `/weka/oe-training-default/ai2-llm/checkpoints/saumyam/hybrid-small-sft-think-450m-lr8e-4/step24834-hf`

### 2026-08-09 eval-only baselines (275M & 450M, think & open_close_think)

- Launch (each): `./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid-small/gsm8k_grpo.sh MODEL_PATH EXP_NAME MODE true`
- 275M think: `EXP_NAME=olmo_hybrid_275m_gsm8k_think_eval MODE=think` — Beaker: [01KZN4DKSWS2PHY407YG2ZBT11](https://beaker.org/ex/01KZN4DKSWS2PHY407YG2ZBT11)
- 275M open_close_think: `EXP_NAME=olmo_hybrid_275m_gsm8k_open_close_think_eval MODE=open_close_think` — Beaker: [01KZN4G055K2QCTC0G6ED8HGQD](https://beaker.org/ex/01KZN4G055K2QCTC0G6ED8HGQD)
- 450M think: `EXP_NAME=olmo_hybrid_450m_gsm8k_think_eval MODE=think` — Beaker: [01KZN4H87DZYJR973AJMZPTPQT](https://beaker.org/ex/01KZN4H87DZYJR973AJMZPTPQT)
- 450M open_close_think: `EXP_NAME=olmo_hybrid_450m_gsm8k_open_close_think_eval MODE=open_close_think` — Beaker: [01KZN4J9F782RFYYDWV9ZHK45Q](https://beaker.org/ex/01KZN4J9F782RFYYDWV9ZHK45Q)
- Result: TBD (all 4 launched and confirmed past model-init/kernel-compile with no errors; 1 training step + eval, discarded afterward since learning_rate=0)

### 2026-08-09 GRPO training runs (275M & 450M, think & open_close_think)

- Launch (each): `./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid-small/gsm8k_grpo.sh MODEL_PATH EXP_NAME MODE false`
- 275M think: `EXP_NAME=olmo_hybrid_275m_gsm8k_think_grpo MODE=think` — Beaker: [01KZN4YBE9D3RKV28H76B98TQR](https://beaker.org/ex/01KZN4YBE9D3RKV28H76B98TQR)
- 275M open_close_think: `EXP_NAME=olmo_hybrid_275m_gsm8k_open_close_think_grpo MODE=open_close_think` — Beaker: [01KZN511RWBVNBK4S11BHX8Y71](https://beaker.org/ex/01KZN511RWBVNBK4S11BHX8Y71)
- 450M think: `EXP_NAME=olmo_hybrid_450m_gsm8k_think_grpo MODE=think` — Beaker: [01KZN52HSKZH7DFK46MEFA5790](https://beaker.org/ex/01KZN52HSKZH7DFK46MEFA5790)
- 450M open_close_think: `EXP_NAME=olmo_hybrid_450m_gsm8k_open_close_think_grpo MODE=open_close_think` — Beaker: [01KZN54NHSTX7J9KRC9368E3PG](https://beaker.org/ex/01KZN54NHSTX7J9KRC9368E3PG)
- Result: TBD (all 4 launched)

Note: the first eval-only launch's Docker build failed once transiently (a `git
status` call inside a cached vllm git checkout timed out after 40s during the
nltk-downloader build step) — unrelated to any code change here. Retried and
succeeded; the resulting image was reused (cache hit) for all 7 subsequent launches.
