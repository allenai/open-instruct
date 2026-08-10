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
- Result: all 4 exited cleanly (exitCode 0). `gsm8k-platinum-openinstruct` eval at
  step 0 (`objective/gsm8k_correct_rate`, `stop_rate`, avg `sequence_lengths`):
  - 275M think: correct_rate **0.12**, stop_rate 0.79, seq_len 9492 — [wandb](https://wandb.ai/mnoukhov/open_instruct_internal/runs/tktv3usv)
  - 275M open_close_think: correct_rate 0.06, stop_rate 0.98, seq_len 765 — [wandb](https://wandb.ai/mnoukhov/open_instruct_internal/runs/iv64fs5x)
  - 450M think: correct_rate 0.05, stop_rate 0.97, seq_len 1905 — [wandb](https://wandb.ai/mnoukhov/open_instruct_internal/runs/zuvh4s9e)
  - 450M open_close_think: correct_rate **0.14**, stop_rate 0.99, seq_len 1027 — [wandb](https://wandb.ai/mnoukhov/open_instruct_internal/runs/61fc3sf2)
  - Stop rates (0.79-0.99) confirm the `response_length=16384` + corrected default
    system prompt fix worked — run 1's ~12% stop rate is resolved. Surprising result:
    275M follows the expected think > open_close_think pattern, but 450M inverts it
    (open_close_think > think) despite a healthy stop rate — see the
    2026-08-10 note in research.md for the leading hypothesis (450M's own baked-in
    chat template doesn't force `<think>` open, unlike 275M's).

### 2026-08-09 GRPO training runs (275M & 450M, think & open_close_think)

- Launch (each): `./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid-small/gsm8k_grpo.sh MODEL_PATH EXP_NAME MODE false`
- 275M think: `EXP_NAME=olmo_hybrid_275m_gsm8k_think_grpo MODE=think` — Beaker: [01KZN4YBE9D3RKV28H76B98TQR](https://beaker.org/ex/01KZN4YBE9D3RKV28H76B98TQR)
- 275M open_close_think: `EXP_NAME=olmo_hybrid_275m_gsm8k_open_close_think_grpo MODE=open_close_think` — Beaker: [01KZN511RWBVNBK4S11BHX8Y71](https://beaker.org/ex/01KZN511RWBVNBK4S11BHX8Y71)
- 450M think: `EXP_NAME=olmo_hybrid_450m_gsm8k_think_grpo MODE=think` — Beaker: [01KZN52HSKZH7DFK46MEFA5790](https://beaker.org/ex/01KZN52HSKZH7DFK46MEFA5790)
- 450M open_close_think: `EXP_NAME=olmo_hybrid_450m_gsm8k_open_close_think_grpo MODE=open_close_think` — Beaker: [01KZN54NHSTX7J9KRC9368E3PG](https://beaker.org/ex/01KZN54NHSTX7J9KRC9368E3PG)
- Result: all 4 progressed with a healthy upward `objective/gsm8k_correct_rate` trend
  (275M open_close_think 0.04→0.06→0.09→0.13 over steps 0/99/199/299; 450M
  open_close_think 0.15→0.17 over steps 0/99; 275M think and 450M think only reached
  their step-0 eval before crashing, since "think" rollouts are ~10x longer and
  generation-bottlenecked), then **all 4 crashed within the same ~07:29-07:30 UTC
  window** (see the 2026-08-10 crash note in research.md) — a Ray GCS host-RAM OOM,
  independent of step count/MODE/model size, most likely tied to repeated
  `[layerwise.py:230] <Module>: Failed to load weights` warnings for OLMo-hybrid's
  custom modules during `--inflight_updates` vLLM weight sync. Not caused by this
  task's script/template changes. Only 275M open_close_think had a checkpoint at a
  `save_freq` boundary (`step200`); the other 3 only had `step0`.

### 2026-08-10 GRPO training relaunch after OOM crash

- 275M think (fresh restart): `EXP_NAME=olmo_hybrid_275m_gsm8k_think_grpo` — Beaker: [01KZNDY5W488WAGDR4CJT4PRW6](https://beaker.org/ex/01KZNDY5W488WAGDR4CJT4PRW6)
- 275M open_close_think (resumed from `step200` via `--checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/michaeln/1786342568_674077`): `EXP_NAME=olmo_hybrid_275m_gsm8k_open_close_think_grpo` — Beaker: [01KZNDYSBJ2Z269R7A77TZ17XS](https://beaker.org/ex/01KZNDYSBJ2Z269R7A77TZ17XS)
- 450M think (fresh restart): `EXP_NAME=olmo_hybrid_450m_gsm8k_think_grpo` — Beaker: [01KZNDZEXB349J3VP3QFD3VPSW](https://beaker.org/ex/01KZNDZEXB349J3VP3QFD3VPSW)
- 450M open_close_think (fresh restart): `EXP_NAME=olmo_hybrid_450m_gsm8k_open_close_think_grpo` — Beaker: [01KZNE01XAA65XJXPCH9SHVQPN](https://beaker.org/ex/01KZNE01XAA65XJXPCH9SHVQPN)
- Result: TBD (all 4 launched; will confirm the 275M open_close_think resume actually
  picked up from step200 rather than restarting from scratch)

Note: the first eval-only launch's Docker build failed once transiently (a `git
status` call inside a cached vllm git checkout timed out after 40s during the
nltk-downloader build step) — unrelated to any code change here. Retried and
succeeded; the resulting image was reused (cache hit) for all 7 subsequent launches.
