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
  task's script/template changes. Only 275M open_close_think had a checkpoint past
  `step0` (`step200`) — note `--save_freq` is a no-op on the olmo-core path; full
  training-state checkpoints are actually controlled by `--checkpoint_state_freq`
  (default 200, per a `grpo_utils.py:258` warning seen in the relaunch logs), which is
  why only the run that got past step 200 had anything to resume from.

### 2026-08-10 GRPO training relaunch after OOM crash

- 275M think (fresh restart): `EXP_NAME=olmo_hybrid_275m_gsm8k_think_grpo` — Beaker: [01KZNDY5W488WAGDR4CJT4PRW6](https://beaker.org/ex/01KZNDY5W488WAGDR4CJT4PRW6)
- 275M open_close_think (resumed from `step200` via `--checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/michaeln/1786342568_674077`): `EXP_NAME=olmo_hybrid_275m_gsm8k_open_close_think_grpo` — Beaker: [01KZNDYSBJ2Z269R7A77TZ17XS](https://beaker.org/ex/01KZNDYSBJ2Z269R7A77TZ17XS)
- 450M think (fresh restart): `EXP_NAME=olmo_hybrid_450m_gsm8k_think_grpo` — Beaker: [01KZNDZEXB349J3VP3QFD3VPSW](https://beaker.org/ex/01KZNDZEXB349J3VP3QFD3VPSW)
- 450M open_close_think (fresh restart): `EXP_NAME=olmo_hybrid_450m_gsm8k_open_close_think_grpo` — Beaker: [01KZNE01XAA65XJXPCH9SHVQPN](https://beaker.org/ex/01KZNE01XAA65XJXPCH9SHVQPN)
- Result: all 4 launched and running with no errors. **Confirmed the resume worked**:
  01KZNDYSBJ2Z269R7A77TZ17XS's logs show `Loading checkpoint from
  '.../1786342568_674077/step200'...` and `Will resume training from step 200, epoch 1`
  — passing an existing `--checkpoint_state_dir` to `gsm8k_grpo.sh`'s trailing args is a
  valid crash-recovery pattern for this script (mason.py only auto-generates a fresh
  `--checkpoint_state_dir` when one isn't already explicitly passed).

Note: the first eval-only launch's Docker build failed once transiently (a `git
status` call inside a cached vllm git checkout timed out after 40s during the
nltk-downloader build step) — unrelated to any code change here. Retried and
succeeded; the resulting image was reused (cache hit) for all 7 subsequent launches.

### 2026-08-10 eval-only baselines (810M & 1.4B, think & open_close_think)

Extends the eval-only baseline comparison to the two larger sizes in the small suite.
Both checkpoints are under `saumyam`, not `yashasbls` as originally assumed when this
batch was requested (`yashasbls`'s 810M lr-sweep only goes up to `lr4e-4`, no `lr8e-4`;
`yashasbls` does also have an identically-named `hybrid-small-sft-think-1.4b-lr4e-4`,
but `saumyam`'s was used per the explicit "from saumya" request).

Models:
- 810M: `/weka/oe-training-default/ai2-llm/checkpoints/saumyam/hybrid-small-sft-think-810m-lr8e-4/step24834-hf`
- 1.4B: `/weka/oe-training-default/ai2-llm/checkpoints/saumyam/hybrid-small-sft-think-1.4b-lr4e-4/step24834-hf`

- Launch (each): `./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid-small/gsm8k_grpo.sh MODEL_PATH EXP_NAME MODE true`
- 810M think: `EXP_NAME=olmo_hybrid_810m_gsm8k_think_eval MODE=think` — Beaker: [01KZPNKJRWQRHAHHTJK3FRRXEQ](https://beaker.org/ex/01KZPNKJRWQRHAHHTJK3FRRXEQ)
- 810M open_close_think: `EXP_NAME=olmo_hybrid_810m_gsm8k_open_close_think_eval MODE=open_close_think` — Beaker: [01KZPNNEA485AP4K6SJVBQDRTM](https://beaker.org/ex/01KZPNNEA485AP4K6SJVBQDRTM)
- 1.4B think: `EXP_NAME=olmo_hybrid_1.4b_gsm8k_think_eval MODE=think` — Beaker: [01KZPNRMF4RH023EE15BPSYQD8](https://beaker.org/ex/01KZPNRMF4RH023EE15BPSYQD8)
- 1.4B open_close_think: `EXP_NAME=olmo_hybrid_1.4b_gsm8k_open_close_think_eval MODE=open_close_think` — Beaker: [01KZPNVJDBBFW49AJ4TXVD6W65](https://beaker.org/ex/01KZPNVJDBBFW49AJ4TXVD6W65)
- Result: **3 of 4 crashed with a genuine CUDA GPU OOM** (distinct from the Ray-GCS
  host-RAM OOM seen in the 275M/450M training runs): both `think`-mode runs (810M,
  1.4B) and, once it finished queueing, 1.4B `open_close_think` too — only 810M
  `open_close_think` succeeded. `torch.OutOfMemoryError` during `loss.backward()`,
  e.g. "Tried to allocate 6.07 GiB. GPU 0 has a total capacity of 79.25 GiB ... 75.60
  GiB memory in use" for 810M think, and ~78-79 GiB in use for 1.4B open_close_think.
  No eval score was recorded for any of the 3 crashed runs (crash occurred
  before/during the step, not after eval collection).
  - 810M open_close_think: `correct_rate` **0.32**, stop_rate 0.99, seq_len 1335 — highest
    eval-only baseline of any size/mode combo so far (cf. 275M 0.06-0.12, 450M 0.05-0.14).

**Root cause found:** `--gradient_checkpointing` (used in every launch of this script
until now) is a no-op on the OLMo-core GRPO path — it only wires up in the older
DeepSpeed-based trainer (`model_utils.ModelConfig.gradient_checkpointing`, read by
`grpo_fast.py`, never read by `grpo_olmo_core_actor.py`). The actual control,
`--activation_checkpointing_mode`, defaults to `"budget"` with
`activation_memory_budget=1.0`; per `olmo_core_utils.build_ac_config`, that combination
returns `None` (i.e. **no activation checkpointing was ever applied**, at any model
size, in any run of this script so far — the 275M/450M runs merely fit in 80GB without
it). Additionally, `"budget"` mode can't checkpoint through this architecture's opaque
GDN (`fla`) kernels regardless of the budget value — only `"selected_modules"` mode
(wraps individual block submodules, including the GDN mixer) is safe for this
architecture. Fixed in `gsm8k_grpo.sh`:
`--gradient_checkpointing` → `--activation_checkpointing_mode selected_modules`.

### 2026-08-10 810M/1.4B relaunch with real activation checkpointing

GPU/FSDP split intentionally left unchanged (still `GPUS=8`, 4 learners / 4 vLLM
engines) per explicit instruction — the fix is enabling real activation checkpointing,
not changing parallelism.

- 810M think: Beaker [01KZPQGDQBSRZBBY2A2C92ZKH9](https://beaker.org/ex/01KZPQGDQBSRZBBY2A2C92ZKH9)
- 1.4B think: Beaker [01KZPQJJBYEV6C6838KQDZXQAP](https://beaker.org/ex/01KZPQJJBYEV6C6838KQDZXQAP)
- 1.4B open_close_think: Beaker [01KZPQNDWJMTVTCJRP86B0WHVW](https://beaker.org/ex/01KZPQNDWJMTVTCJRP86B0WHVW)
- Result: **all 3 crashed again, in ~3 minutes, with a different and deterministic error**:
  `ValueError: There is no module or parameter named 'blocks' in
  OlmoHybridSmallForCausalLM.` — identical across all 3. Root cause:
  `activation_checkpointing_mode=selected_modules` wraps individual block submodules,
  which changes their qualified parameter names in the state dict (PyTorch's checkpoint
  wrapper inserts an extra name segment). `olmo_core_to_hf_name`
  (`open_instruct/grpo_callbacks.py:48`) does regex-based renaming from OLMo-core-native
  names (`blocks.N.<submodule>`) to HF/vLLM names (`model.layers.N.<submodule>`) via a
  small lookup dict (`_OLMO_CORE_TO_HF_LAYER_MAPPINGS`) keyed on exact submodule-name
  strings (e.g. `"attention.w_q.weight"`). The wrapped names no longer match those exact
  strings, so the lookup silently misses and the function falls through to
  `return name` (the raw, untranslated OLMo-core-native name) — which vLLM's HF-style
  weight loader then can't find, since it has no `blocks` attribute at all. This breaks
  the weight sync from trainer to vLLM inference engine entirely, for any sequence
  length/model size — not something worth blindly retrying, since it reproduced
  identically 3/3 times. **Reverted** the `--activation_checkpointing_mode
  selected_modules` change (removed the flag entirely, since the underlying
  `--gradient_checkpointing` it replaced was already a proven no-op anyway — see the
  root-cause note above). Fixing this properly would require patching
  `olmo_core_to_hf_name` to strip the checkpoint-wrapper's inserted name segment before
  the lookup (exact wrapper name not yet confirmed — would need to inspect the live
  wrapped module hierarchy, not just static code reading) or an upstream olmo-core fix.
  **Net status: 810M think, 1.4B think, and 1.4B open_close_think remain unresolved** —
  blocked on either this weight-sync bug or the original GPU OOM, whichever is fixed
  first. 810M open_close_think's baseline (`correct_rate` 0.32) stands as the only
  usable result from this size range so far.

### 2026-08-10 collateral OOM in the 4 ongoing GRPO training runs

While the above 3 eval-only relaunches were running (8 more GPUs each on the shared
`saturn`/`jupiter`/`ceres` clusters, on top of the 4 already-running training jobs), 3 of
the 4 training jobs hit the familiar Ray-GCS host-RAM OOM (see the 2026-08-10 crash note
under the main GRPO training section above) within the same ~20:42-21:08 UTC window —
plausibly triggered or worsened by the extra concurrent load, though not confirmed.
Only 450M think (`01KZNDZEXB349J3VP3QFD3VPSW`) survived. Resumed the other 3 from their
latest saved checkpoints:

- 275M think: resumed from `step400` (`.../deletable_checkpoint_states/michaeln/1786351916_614241`) — Beaker [01KZPS2TF1NN97HPH5YCH9RQ95](https://beaker.org/ex/01KZPS2TF1NN97HPH5YCH9RQ95)
- 275M open_close_think: resumed from `step1800` (`.../deletable_checkpoint_states/michaeln/1786342568_674077`) — Beaker [01KZPS5EAYP94HFPCCYXAWFWHN](https://beaker.org/ex/01KZPS5EAYP94HFPCCYXAWFWHN)
- 450M open_close_think: resumed from `step1000` (`.../deletable_checkpoint_states/michaeln/1786351978_851104`) — Beaker [01KZPS7CWBAC1RPC0EGXCBDDGY](https://beaker.org/ex/01KZPS7CWBAC1RPC0EGXCBDDGY)
- Result: **275M open_close_think completed its full run** (2026-08-11 ~00:39 UTC, ran
  step1800→2000/2000, exited 0, "Training complete", final model saved to
  `/weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/olmo_hybrid_275m_gsm8k_open_close_think_grpo__1__1786398974`)
  — first fully-completed run of all 8. Final eval: `correct_rate` **0.23** (matches its
  long-standing plateau — no late change in accuracy), but `stop_rate` dropped to
  **0.79** (from ~0.98-0.99) and average `sequence_lengths` jumped to **3730** tokens
  (from ~800-1000) in this final eval batch — worth inspecting raw generations before
  treating this checkpoint as final, though could just be single-batch noise. 275M
  think and 450M open_close_think still running as of this note.

### 2026-08-11 additional OOM-avoidance tricks for 810M/1.4B (no code checkpointing changes)

Added `--env PYTORCH_ALLOC_CONF=expandable_segments:True` to `gsm8k_grpo.sh` (all
launches, low-risk). For the two `think`-mode retries specifically, also overrode
`--response_length 8192 --pack_length 16384` (half the script's 16384/32768 default)
via trailing args, since `think` mode's long rollouts were the driver of that OOM.
1.4B `open_close_think` got the env var only (no sequence-length change), since its
short rollouts meant sequence length wasn't the issue.

- 1.4B open_close_think: Beaker [01KZQCKGDRTP2HVQE3WGE1K956](https://beaker.org/ex/01KZQCKGDRTP2HVQE3WGE1K956)
  — **still OOMs** (exitCode 1). Diagnostic: `expandable_segments` clearly reduced
  fragmentation (the "reserved by PyTorch but unallocated" figure dropped from ~3GB to
  380-695MB across the 3 failing processes), but PyTorch's actual *allocated* memory
  grew to 73-74GB (up from 70-72GB pre-fix) and the process still hit ~78.6-79.2GB
  total, i.e. this is a genuine capacity shortfall, not fragmentation — 1.4B's
  base memory footprint (model + optimizer state + reference policy, sharded 4-way via
  FSDP) doesn't fit in 79GB regardless of allocator tuning or sequence length.
  `response_length` reduction wouldn't help this one even if applied, since its
  rollouts were already short. The only remaining lever is more FSDP shards (more
  learner GPUs) or reducing per-GPU memory some other way (e.g. dropping the reference
  policy, though `--load_ref_policy True` is presumably needed for KL). Not attempted —
  would mean changing the 4/4 learner/vLLM split, which needs explicit sign-off.
- 810M think: Beaker [01KZQCM8YQ8HR2PPXNNFNP9PXS](https://beaker.org/ex/01KZQCM8YQ8HR2PPXNNFNP9PXS)
  — **succeeded** (exitCode 0). `correct_rate` **0.34**, stop_rate 0.88, seq_len 2583
  (max hit the reduced 8192 cap). First usable "think" result at 810M+.
- 1.4B think: Beaker [01KZQCN2NHABMQ18XPT4YY97HA](https://beaker.org/ex/01KZQCN2NHABMQ18XPT4YY97HA)
  — **succeeded** (exitCode 0). `correct_rate` **0.74**, stop_rate 0.88, seq_len 3179
  (max hit the reduced 8192 cap) — by far the highest score in this whole study (next
  highest was 810M open_close_think at 0.32). Both of these used the halved
  `response_length=8192`/`pack_length=16384`, so are not directly apples-to-apples with
  the 16384/32768 used at 275M/450M/810M-open_close_think/1.4B-open_close_think — a
  shorter reasoning budget could understate `think` mode further, so if anything this
  0.74 is a lower bound.

### 2026-08-11 real fix: cherry-picked allenai/open-instruct#1813

The user opened a PR fixing the actual root cause of the weight-sync break:
`grpo_name_mapping.olmo_core_to_hf_name` (moved out of `grpo_callbacks.py`) now strips
PyTorch's `_checkpoint_wrapped_module` path segment — the exact wrapper name inserted by
`selected_modules` checkpointing — before doing the OLMo-core-to-HF name lookup, which
is exactly the bug diagnosed above. Also adds a legacy-flag shim (`grpo.py`: setting
`--gradient_checkpointing` now auto-enables `selected_modules` checkpointing if native
checkpointing wasn't otherwise configured) and two regression tests
(`tests/test_grpo_callbacks.py`) confirming the exact `blocks.N.attention._checkpoint_wrapped_module.*`
and `blocks.N.attention_norm._checkpoint_wrapped_module.*` cases. Cherry-picked commit
`62985feda1485e556640d31f126daba79f1f93d1` cleanly (`git cherry-pick -x`); both new
tests pass; `make style && make quality` clean (ruff caught one leftover unused import
in `grpo_callbacks.py` from the PR, removed). Re-added
`--activation_checkpointing_mode selected_modules` to `gsm8k_grpo.sh`.

- 1.4B open_close_think relaunch (real checkpointing, normal `response_length=16384`/
  `pack_length=32768` — no need for the length-reduction workaround now): Beaker
  [01KZQEE8KGWQ9KF6PRRMBKS84H](https://beaker.org/ex/01KZQEE8KGWQ9KF6PRRMBKS84H) —
  **crashed again with the identical error** (`ValueError: There is no module or
  parameter named 'blocks' in OlmoHybridSmallForCausalLM`), this time at
  `run_initial_weight_sync` (the very first weight sync, before any training step —
  PyTorch's checkpoint wrapper applies structurally at model-construction time, not
  just during backward passes). **Root cause is different from what PR #1813 fixed,
  and is a second, pre-existing gap in the same weight-sync code**: per
  `olmo_core_utils.py`'s own docstring, this architecture's GDN mixer is natively named
  `attention` in OLMo-core (same submodule name as real self-attention layers), so
  `selected_modules` checkpointing's default `activation_checkpointing_modules` list
  (`blocks.*.attention`, etc.) wraps GDN layers too. After PR #1813's fix strips the
  `_checkpoint_wrapped_module` segment, the resulting name for a GDN layer looks like
  `blocks.N.attention.in_proj_qkvg.weight` (or `.A_log`, `.dt_bias`, `.conv1d.weight`,
  `.o_norm.weight`, `.a_proj.weight`, `.b_proj.weight`, `.o_proj.weight` — all visible
  in this run's "available parameters" error dump under `linear_attn.*` on the vLLM/HF
  side) — none of which appear in `_OLMO_CORE_TO_HF_LAYER_MAPPINGS`
  (`grpo_name_mapping.py`), which only has entries for standard-attention parameter
  names (`w_q`/`w_k`/`w_v`/`w_out`/`q_norm`/`k_norm`). So the lookup still falls through
  to the raw, untranslated OLMo-core name, and vLLM still can't resolve it — same
  symptom, different cause. Also noticed in the same error dump: this model's HF/vLLM
  side uses a **fused** `self_attn.qkv_proj.weight` (plus `self_attn.attn_gate.weight`),
  not separate `q_proj`/`k_proj`/`v_proj` as the mapping dict assumes — a third,
  possibly-unrelated mismatch, not confirmed to matter here since standard-attention
  layers weren't the ones failing in this run.
  **Since this architecture is entirely GDN-based, `selected_modules` checkpointing
  would break every future checkpointed run of this script regardless of model size —
  reverted `--activation_checkpointing_mode selected_modules` from `gsm8k_grpo.sh`
  again.** 1.4B `open_close_think` remains unresolved; would need
  `_OLMO_CORE_TO_HF_LAYER_MAPPINGS` extended with GDN (`linear_attn.*`) entries (and
  possibly the fused-qkv naming) before checkpointing is viable here, or the GPU-split
  lever (more learner GPUs) as a fallback.

### 2026-08-11 first GRPO training runs at 810M/1.4B (think only)

First full training runs (not eval-only) at these sizes, using the same
`response_length=8192`/`pack_length=16384` override that already got both `think`
eval-only baselines working (0.34 and 0.74 respectively) — no activation checkpointing
(currently broken for this GDN architecture, see above), same 4/4 learner/vLLM split as
every other run.

- 810M think: `EXP_NAME=olmo_hybrid_810m_gsm8k_think_grpo` — Beaker [01KZRMQ4KNAJMG09AEEZY4PV8V](https://beaker.org/ex/01KZRMQ4KNAJMG09AEEZY4PV8V)
- 1.4B think: `EXP_NAME=olmo_hybrid_1.4b_gsm8k_think_grpo` — Beaker [01KZRMQV47PHK15WJVN36JDZ90](https://beaker.org/ex/01KZRMQV47PHK15WJVN36JDZ90)
- Result: both healthy as of 2026-08-11 ~15:05 UTC, no crash yet, results TBD

### 2026-08-11 450M think and 450M open_close_think GRPO training completed

Both finished cleanly at step 2000/2000 ("Training complete", exitCode 0):

- **450M think** (`01KZNDZEXB349J3VP3QFD3VPSW`, completed ~07:44 UTC): final
  `correct_rate` **0.61**, stop_rate 0.96, seq_len 7385 (up from baseline 0.05).
- **450M open_close_think** (`01KZPS7CWBAC1RPC0EGXCBDDGY`, completed ~14:50 UTC): final
  `correct_rate` **0.60**, stop_rate 0.96, seq_len 7713 (up from baseline 0.14).

Both modes converge to essentially the same final performance (0.60 vs 0.61), matching
the pattern already seen mid-training. Unlike 275M open_close_think's anomalous final
eval (stop_rate dropped to 0.79), both of these kept a healthy 0.94-0.96 stop_rate
through to the end — no late-training degradation signal here. Also note: growth
continued well past what was earlier called a "plateau" (~0.5 range around step
800-1000) up to ~0.60-0.61 by step 2000 — the mid-training slowdown was real but not a
hard ceiling.

### 2026-08-11 real root cause found: wrong mapper was being debugged

PR #1813's fix targets `grpo_name_mapping.olmo_core_to_hf_name`, but that generic
mapper turns out **not to be what's actually used for our checkpoints**. In
`grpo_olmo_core_actor.py`'s `setup_model()`, `self.name_mapper` gets set twice: once
(early) to the generic mapper (with or without PR #1813's fix), then — a few dozen
lines later, inside the same method, unconditionally overwriting the first assignment
— to `olmo_core_utils.make_hybrid_small_name_mapper(...)` whenever
`is_hybrid_small_checkpoint(self.model_name_or_path)` is true, which it is for every
checkpoint used in this suite. This dedicated, more specific mapper (local to this
branch, not part of #1813 upstream) is the one actually exercised at weight-sync time.

Its bug was structurally identical to the one #1813 fixed, but independent: its
GDN-layer detection (`match.group(2) == "attention.A_log"`) and its HF-name lookup both
matched on exact, unwrapped submodule names. Under `selected_modules` checkpointing,
every parameter name gets a `_checkpoint_wrapped_module` segment inserted, so **zero**
layers were ever detected as GDN (the detection pass silently found nothing), and every
lookup missed too — same end symptom (`no module or parameter named 'blocks'`), same
fix shape (strip the wrapper segment first), different function
(`open_instruct/olmo_core_utils.py::make_hybrid_small_name_mapper`, not
`grpo_name_mapping.py`). Its underlying naming scheme (unfused `w_q`/`w_k`/`w_v`, split
`q_conv1d`/`k_conv1d`/`v_conv1d`) was otherwise correct — vLLM fuses them during its own
loading — so this was purely the missing wrapper strip, not a naming-scheme mismatch.

Added `_strip_checkpoint_wrapper` + applied it before both the GDN-detection regex
match and the lookup; 3 new regression tests
(`tests/test_olmo_core_utils.py`) cover unwrapped/wrapped-GDN/wrapped-full-attention
cases, alongside the 3 existing tests for the generic mapper — 6/6 pass. Re-added
`--activation_checkpointing_mode selected_modules` to `gsm8k_grpo.sh` as the default
(now genuinely working for both mapper code paths).

- 1.4B open_close_think eval-only relaunch (third attempt, real fix): Beaker
  [01KZRXVBBDNK9KP6SMQYDNTZQN](https://beaker.org/ex/01KZRXVBBDNK9KP6SMQYDNTZQN) —
  **succeeded** (exitCode 0). `correct_rate` **0.61**, stop_rate 0.99, seq_len 1284, at
  the *normal* `response_length=16384`/`pack_length=32768` (no reduction workaround
  needed — real activation checkpointing gave it enough headroom). This completes the
  eval-only baseline for all 8 size × mode combinations:

  | | think | open_close_think |
  |---|---|---|
  | 275M | 0.12 | 0.06 |
  | 450M | 0.05 | 0.14 |
  | 810M | 0.34* | 0.32 |
  | 1.4B | 0.74* | **0.61** |

  \*810M/1.4B `think` still used the halved `response_length=8192` workaround (launched
  before this fix landed), so not directly comparable to the others.

### 2026-08-11 810M think GRPO training crashed (host-RAM OOM, not GPU), resumed by user

`01KZRMQ4KNAJMG09AEEZY4PV8V` died at step 513/2000 (exitCode 1, ~19:43 UTC). Confirmed
**not** a CUDA/GPU OOM — no `torch.OutOfMemoryError` anywhere in the log. Root cause:
Ray's own host-memory monitor killed a worker (`(raylet) node_manager.cc:3256: 1
Workers (tasks / actors) killed due to memory pressure (OOM)`), after which the rest of
the job hung for ~15 minutes waiting on the dead worker (`wait_count=1000`+, gRPC
`keepalive_ping_end` errors) before the RPC keepalive watchdog finally timed out and
cascaded into `ray.exceptions.ActorUnavailableError`. Same underlying failure mode as
the earlier 275M/450M host-RAM OOM crashes, just a different (slower, RPC-timeout-based)
downstream symptom this time instead of an immediate Python `OutOfMemoryError`. Given
this has now recurred across multiple sizes/GPU configs/physical nodes over the course
of this session, it looks systemic (a host-memory leak somewhere in the CPU-side
rollout/data-prep pipeline that grows with wall-clock time) rather than one-off bad
luck — not root-caused further here; would need live host-memory profiling during a run
to pin down. User resumed the job themselves (checkpoint was available at `step400`).

### 2026-08-11 450M "fixed-chat-template" checkpoint eval, using the checkpoint's own template

New checkpoint variant with a corrected chat template baked directly into
`tokenizer_config.json` (rather than needing `--chat_template_name` to override it):
`saumyam/hybrid-small-sft-think-450m-lr8e-4/step24834-hf-fixed-chat-template`. Added a
`builtin` `MODE` to `gsm8k_grpo.sh` that omits `--chat_template_name` entirely
(previously the script always passed one of the two hardcoded `olmo_hybrid_small_*`
templates), so `get_tokenizer_tulu_v2_2` falls through to the checkpoint's own
`tokenizer_config.json` `chat_template` unmodified. Confirmed via the printed launch
command that no `--chat_template_name` flag is present.

- Launch: `./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid-small/gsm8k_grpo.sh MODEL_PATH EXP_NAME builtin true`
- `EXP_NAME=olmo_hybrid_450m_gsm8k_fixed_chat_template_eval` — Beaker [01KZVG1PZJBHXR5K3GRR16EXV3](https://beaker.org/ex/01KZVG1PZJBHXR5K3GRR16EXV3)
- Result: **crashed in ~40s** (exitCode 1) with `ValueError: Model/config
  '.../step24834-hf-fixed-chat-template' not found`. Root cause unrelated to the
  `builtin` MODE change: `olmo_core_utils._load_native_transformer_config` requires
  `model_name_or_path` to literally end in the suffix `-hf` to find its sibling
  olmo-core-native checkpoint (`path.with_name(path.name[:-len("-hf")])` — deserializes
  that sibling's `config.json` for the exact `TransformerConfig`, since there's no
  named `olmo_hybrid_small` preset). `step24834-hf-fixed-chat-template` doesn't end in
  `-hf`, so the check fails immediately and falls through to the "no matching preset"
  error. **Fix**: rather than touch this shared detection helper or the checkpoint
  owner's (`saumyam`) directory, created a small symlink pair in my own weka space —
  `/weka/oe-adapt-default/michaeln/checkpoint_symlinks/hybrid-small-450m-fixed-chat-template/`
  with `step24834-hf` → the real `-fixed-chat-template` dir and `step24834` → the
  original native checkpoint (identical architecture, only `tokenizer_config.json`
  differs) — satisfying the naming convention without modifying either. Relaunched
  pointed at the symlink: Beaker [01KZVM5VB3C9TTQSWX5MA9XYP2](https://beaker.org/ex/01KZVM5VB3C9TTQSWX5MA9XYP2).

### 2026-08-12 1.4B think GRPO training nearing completion with an exceptional result

`01KZRMQV47PHK15WJVN36JDZ90`, at step 1899/2000 as of ~17:00 UTC:
`objective/gsm8k_correct_rate` progression 0.90 (step 1399) → 0.91 (1499-1799) → 0.92
(1899) — up from its 0.74/0.75 baseline, and by far the best result anywhere in this
study, achieved with the halved `response_length=8192` workaround (not even the full
16384 budget).

**Completed 2026-08-12 ~18:53 UTC**: final `correct_rate` **0.92** at step 2000/2000,
exitCode 0, clean finish ("Training complete"). Best result in this whole study.

### 2026-08-12 Dolci difficulty scoring pipeline launched

Completes the pipeline from the 2026-08-06 generation runs (see the
`dolci-difficulty-estimation-pipeline` memory) — the raw-completions datasets
(`mnoukhov/dolci_think_rl_7b_messages_hybrid_275m` / `_hybrid_450m`, 102,014 rows each,
confirmed pushed successfully) had never been scored for pass-rate. Added
`scripts/data/rlvr/score_dolci_difficulty.sh` (downloads the hub dataset, converts to
local jsonl, scores via `filter_existing_dataset_correctness.py` using RLVR verifiers +
a self-hosted code API + the LLM judge, pushes the annotated
`<repo>_scored` dataset with `total_rollouts`/`total_correct_rollouts`/`passrate`
columns) and a small `download_hub_dataset_as_jsonl.py` helper.

Relaunched the Qwen3-32B judge server (both prior instances from 08-06 had since died)
as `01KZVK7C2AA0VE1H7547TQ8DBQ`; took ~20 min to schedule (heavy GPU contention this
session), confirmed working at `http://ceres-cs-aus-447.reviz.ai2.in:8001/v1`
(`/v1/models` returns `Qwen/Qwen3-32B`).

Hit and fixed a positional-arg bug in the new script (`BEAKER_IMAGE="${1:-default}"` is
ambiguous when called with exactly 2 args — same class of issue `gsm8k_grpo.sh`'s own
comments warn about; fixed by making `BEAKER_IMAGE` an env var override instead of a
positional arg).

- 275m scoring: `EXP_NAME=dolci_difficulty_scoring_hybrid_275m` — Beaker [01KZVN9K504CT8J3AHZAS3NDEY](https://beaker.org/ex/01KZVN9K504CT8J3AHZAS3NDEY)
- 450m scoring: `EXP_NAME=dolci_difficulty_scoring_hybrid_450m` — Beaker [01KZVNBH8BF3327WV96RB04XR1](https://beaker.org/ex/01KZVNBH8BF3327WV96RB04XR1)
- Result: both still queued (only `created`, no `scheduled`) as of ~19:20 UTC, ~25 min
  after launch — GPU contention remains heavy this session.

### 2026-08-12 275M think GRPO crashed (host-RAM OOM, 4th occurrence), resumed near the finish

`01KZPS2TF1NN97HPH5YCH9RQ95` crashed at 00:56 UTC (missed in earlier checks — this
session's monitoring cadence had shifted focus to other jobs). Same signature as every
prior occurrence: `ray.exceptions.OutOfMemoryError: Task was killed due to the node
running low on memory` — Ray's host-memory monitor, not GPU. This is now the 4th time
this exact failure mode has hit a training run in this session (275M think ×2, 450M
think, 810M think), always on `saturn`/`jupiter`/`ceres`, never GPU memory — reinforces
that this looks like a systemic host-memory issue in the pipeline rather than
per-job bad luck, still not root-caused to a specific leak source.

Found a checkpoint at `step1600` (much further along than the first crash's `step400`)
under the same `checkpoint_state_dir`
(`/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/michaeln/1786351916_614241`)
— only ~400 steps from the full 2000-step run. Resume hit a **separate, transient local
issue** on the first two attempts: `uv sync` (inside `build_image_and_launch.sh`, and
again when the target script's own `uv run python mason.py` triggered a re-sync)
failed with an HTTP/2 "stream error received: refused stream" fetching the `flash-attn`
prebuilt wheel from GitHub releases — confirmed the URL itself is reachable (plain
`curl` succeeded), so this looks like a `uv`-specific transient HTTP/2 hiccup, not a
real network/connectivity problem. Third attempt succeeded. Resumed: Beaker
[01KZVPTCJW89KJPRAHQPVZYQDK](https://beaker.org/ex/01KZVPTCJW89KJPRAHQPVZYQDK).

### 2026-08-12 450M fixed-chat-template eval result

`01KZVM5VB3C9TTQSWX5MA9XYP2` succeeded (exitCode 0) after the symlink fix.
`correct_rate` **0.05**, stop_rate 0.98, seq_len 1872 — essentially identical to the
original 450M `think` baseline (also 0.05, using the hand-rolled
`olmo_hybrid_small_think` override template). Good sanity check: the checkpoint's own
corrected built-in template and the manually-reconstructed override template produce
matching behavior, validating the earlier template-correction work.

### 2026-08-12 Dolci difficulty scoring progressing

Both jobs started (~19:25-19:26 UTC) after clearing the GPU queue. `code_api_setup.sh`
came up cleanly on both (128 uvicorn workers + nginx load balancer, health checks
passing). 275m: dataset download from the hub completed (102,014/102,014 rows, ~5 min)
and moved into the actual scoring pass. 450m: still downloading as of this check. No
errors on either.

**Known non-fatal issue found in 275m scoring (~20:11 UTC):** the LLM judge
(`--llm_judge_max_tokens 2048` against the judge server's `--max-model-len 32768`)
fails with `litellm.ContextWindowExceededError` for any `general-quality`/
`general-quality_ref` prompt whose input alone exceeds ~30.7k tokens (input + the
requested 2048 output tokens > 32768). `filter_existing_dataset_correctness.py`
catches this after 3 retries and falls back to a default score of `0.0` for that
sample — the job does **not** crash, but affected long-context general-quality rows
will get a spuriously low (0.0) passrate rather than a real judged score. Not
intervening mid-run (would lose all progress so far for a partial-data-quality issue,
not a systemic failure) — flagging for whoever consumes `_scored`'s
`general-quality`/`general-quality_ref` rows: treat exact-0.0 passrates in those two
domains with suspicion, they may be context-overflow artifacts rather than genuine
low-quality judgments. A future re-run could avoid this with a larger
`--llm_judge_max_context_length`-compatible judge server config or a shorter
`--llm_judge_max_tokens`.

### 2026-08-13 Dolci scoring throughput measured: ~5 day ETA

275m scoring's tqdm bar surfaced in the logs (~02:10 UTC, ~6.75h since launch):
`Scoring: 7%|▋| 7356/102014 [5:56:27<120:32:30, 4.58s/it]` — at this rate the full
102,014-row pass is ~126 hours (~5.25 days), far slower than anticipated for what's
mostly rule-based verification. Both 275m and 450m scoring jobs share one Qwen3-32B
judge instance; `general-quality`/`general-quality_ref` rows there are paying for up
to 3 retries at up to 181s each (`litellm.Timeout`, judge server under concurrent
load from both jobs) on top of the already-known context-overflow failures — judge
server capacity, not verifier/CPU throughput, is the bottleneck. Job is healthy, just
slow; not intervening (see research.md for the write-up and possible future fixes:
more judge replicas, or splitting `general-quality*` into a separate lower-concurrency
pass).

### 2026-08-13 Dolci scoring throughput update: faster than initially measured

~24h after launch, 450m scoring's tqdm bar reads `33%|███▎| 33323/102014
[24:07:26<39:13:17, 2.06s/it]` — per-sample rate nearly doubled vs the ~6h mark
(2.06s/it vs 4.58s/it), bringing the ETA down from ~126h to ~39h remaining. Still
running, no crashes, no new error types (same context-overflow/timeout/504-gateway/
aiohttp-unclosed-session set as before). Updated expectation only, no action taken.

### 2026-08-14 Dolci scoring throughput fluctuates, not monotonically improving

~45.5h after launch, 450m scoring is at 60% (61625/102014) but tqdm's own ETA has
regressed back to 36-55h remaining (worse than the ~39h read a day earlier at 33%) —
per-sample rate bounced between ~2s/it and ~5s/it across the two checks. Confirms
throughput is noisy (judge-server contention varies with load, not steadily
improving), so any single tqdm reading is a rough sense-of-scale, not a reliable
forecast. No crashes, no new error types.

### 2026-08-13 275M think GRPO training completed — last of the 8 core runs

`01KZVPTCJW89KJPRAHQPVZYQDK` (resumed from `step1600` after its 4th host-RAM-OOM crash)
finished cleanly at step 1999/2000 (~03:39 UTC, exitCode 0, "Training complete"). Final
`objective/gsm8k_correct_rate` **0.13**, `stop_rate` 0.62, seq_len 6610 — well below its
`open_close_think` counterpart's final 0.23, and the only size in the study where
`think` mode ends up *worse* than `open_close_think` rather than matching or exceeding
it (450M: 0.61 vs 0.60; 1.4B: 0.92 vs 0.61-baseline-only). `stop_rate` stayed in the
0.57-0.64 range across the last 4 eval checkpoints (steps 1699/1799/1899/1999), well
below the ~0.9+ typically seen elsewhere, suggesting 275M `think` never reliably learned
to close its `<think>` block within the 16384-token budget. This completes all 8
size×mode eval-only baselines and 6 of 8 full GRPO training runs (810M `think` still
training, resumed and tracked by the user directly; 810M/1.4B `open_close_think` were
never launched for full training in this study, only eval-only baselines).

### 2026-08-15 275m Dolci scoring OOM-killed in push_to_hub — scored data lost

`01KZVN9K504CT8J3AHZAS3NDEY` completed its full scoring pass (102014/102014 rows,
~72h wall time) at 20:15:30 UTC, logged `filter_existing_dataset_correctness.py`'s
"Filtered 0 samples out of 102014" line, then went completely silent. Initially
suspected as a hung `push_to_hub()` network call (no new log output, no `exited`
status, target dataset not appearing on the hub for 3+ hours, while 450m running the
identical code path continued logging normally). At 23:26:10 UTC the job finally
exited with **`exitCode 137`, `message: "out of memory"`** — a host-level OOM kill
(SIGKILL), not a hang. A `multiprocessing.resource_tracker` "leaked semaphore"
warning appeared ~1 min before the kill (expected side effect of the OS reclaiming
memory from a live process). Root cause: almost certainly `datasets.Dataset.
push_to_hub()`'s in-memory Arrow/Parquet serialization of the full 102k-row dataset
(long `messages`/generation-text columns) exceeding the container's memory limit —
the streaming/incremental scoring pass itself never approached this.

**The scored dataset was never pushed and is now lost** — the only copy was the
local `/tmp/hybrid_275m_scored.jsonl` inside the now-terminated container, and Beaker
doesn't persist local disk after a job exits. Recovering 275m's difficulty scores
requires either re-running the full ~72h scoring pass, or (better) fixing
`score_dolci_difficulty.sh`/`filter_existing_dataset_correctness.py` to write the
scored jsonl to persistent `/weka` storage before attempting `push_to_hub`, and/or
switching to a lower-memory push strategy (sharded/streaming push) so a future re-run
doesn't lose all its work to the same failure mode.

450m (`01KZVNBH8BF3327WV96RB04XR1`) was at 99% (100927/102014) at the time this was
diagnosed and is expected to hit the same `push_to_hub` step within minutes — at
serious risk of an identical OOM kill. Flagged to the user; no proactive fix possible
without modifying/relaunching the running job.

### 2026-08-16 450m Dolci scoring succeeded — survived the same risky window

`01KZVNBH8BF3327WV96RB04XR1` exited cleanly at 02:36:15 UTC (exitCode 0).
`mnoukhov/dolci_think_rl_7b_messages_hybrid_450m_scored` is live on the hub
(`lastModified` 02:35:22 UTC, confirmed via HF API). Reached the same
"finished scoring → silent" state as 275m at 00:07:51 UTC and stayed silent for
~2h28m before exiting successfully — comparable to but shorter than 275m's ~3h11m
before its OOM kill. The final logs explain the silent gap: 450m's actual
`push_to_hub` shard upload (151 shards, ~235MB) only took ~37 minutes once it started
and logged visible progress throughout — meaning roughly ~2h of the ~2.5h gap is
spent building/serializing the in-memory dataset table *before* any upload progress
appears, and that's where the OOM risk lives, not the upload transfer itself. 450m's
serialization fit in memory (barely, given the unusually long silent stretch — likely
near the container's memory ceiling); 275m's didn't. Reads as a **timing/
memory-pressure-dependent failure, not fully deterministic**.

**Net outcome for the Dolci difficulty-scoring pipeline this session:** 450m
succeeded end-to-end, scored dataset live. 275m completed all ~72h of scoring compute
but lost the final output to the OOM kill during push and needs a re-run to produce
`mnoukhov/dolci_think_rl_7b_messages_hybrid_275m_scored` — ideally after fixing the
push step to persist to `/weka` before attempting `push_to_hub` (or reducing peak
memory via a sharded/incremental serialization approach), so a repeat OOM doesn't
lose the work again.

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

### Synchronization

- Preserved the local evaluation-launch fix and attachment/comparison documentation,
  then merged cluster HEAD `e2d831a5ca7ad8d62ff3e554460753ec17ae62bb` into
  `hybrid-small-suite`. The only merge conflict was overlapping appended content
  in this log; both histories were retained. No code conflict occurred.
- Validation: `UV_NO_SYNC=1 make style` and `UV_NO_SYNC=1 make quality` passed;
  ten CPU tests passed in `tests/test_grpo_callbacks.py`,
  `tests/test_olmo_core_utils.py`, and `open_instruct/test_grpo_fast_resource_plan.py`.
  The evaluation shell script passed `bash -n`, and the Python evaluation launcher
  passed compilation. These are static/CPU checks, not GPU training validation.

## Olmo Hybrid release 275m RL preparation 2026-10-22

Related: [research.md#olmo-hybrid-release-checkpoint-inputs](research.md#olmo-hybrid-release-checkpoint-inputs).

- Preparation only; no GPU allocation or training experiment submitted.
- Input: `/weka/oe-training-default/yashasbls/hybridish-local/hf/v1/275m/midtrain/step0038147`;
  native config is in the matching `olmo-core/v1` tree. Only metadata/tokenizer
  files and the safetensors header were copied to local `/tmp` for CPU checks;
  checkpoint weights stayed on Weka.
- Runtime: transformers 5.15.1, stock vLLM 0.19.1, both release plugins pinned to
  `0fb485ce38733103f7da73973cdf0e74b0fc75db`, native OLMo-core pinned to
  `098b7db005e2a77fd4fa1a490e2cda3406b4ec1a`. Legacy architecture registration retained.
- Validation: 50 distinct focused CPU regressions passed across the combined
  regression run and subsequent two classifier cases; the CUDA-only test skipped.
  All 192 released tensor names/shapes match both meta models. Ruff format/lint,
  compilation, ty, shell syntax, and MkDocs build passed. The existing tokenizer
  LFS fixture was downloaded with its SHA256 checked for the tokenizer tests,
  then its original pointer restored. New locked dependencies were tested in
  `.venv-release`, leaving the previous shared `.venv` untouched.
- Launcher arguments were captured without invoking Mason, then parsed through
  the actual GRPO dataclass parser: 20 updates, 4 prompts × 4 completions, 320
  total episodes, base `Question: ...\nAnswer:` prompt, 512/2048 prompt/response
  tokens, packing budget 4096, 1 learner/1 rollout engine, BF16 compute, lr 1e-6,
  beta 0, no reference policy, no zero-std filtering/refill, one async batch.
- Planned smoke: 2 A100 80GB GPUs on Saturn, workspace `ai2/open-instruct-dev`,
  budget `ai2/oe-other`, normal priority, preemptible, no minimum-runtime promise,
  two-hour timeout, no automatic retries/resume, native save every ten steps and
  final HF export. Output: `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/runs/<run-name>`.
- Build without allocating GPUs: `./scripts/train/build_image_and_launch.sh --build-only`.
  Private plugin fetches use an ephemeral BuildKit GitHub-token secret.
- Built-container validation caught uv's default managed environment being stored
  under the BuildKit cache mount, leaving a dangling `/stage/.venv` after build.
  The Dockerfile now explicitly places the environment at `/stage/.venv` and
  Python at `/opt/python`, and asserts the environment is not a symlink.
  GPU jobs must use the rebuilt, container-validated image rather than the first build.
- GPU gate (separate from training):
  `GPU_TEST_CLUSTER=ai2/saturn GPU_TEST_TIMEOUT=1h GPU_TEST_BUDGET=ai2/oe-other ./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh open_instruct/test_hybrid_release_gpu.py`.
  One A100 80GB, urgent/preemptible, no guaranteed minimum runtime, zero retries.
  Tests actual-weight HF/Core parity, a synthetic full-length gradient/optimizer
  update, and HF export/reload. GPU gate and end-to-end RL validation remain pending.
- Smoke command after the gate passes:
  `./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid-small/gsm8k_release_275m_smoke.sh`.


## Olmo Hybrid release 275m GSM8K 1000 updates 2026-10-22

Related: [research.md#olmo-hybrid-release-275m-gsm8k-learning-and-resume](research.md#olmo-hybrid-release-275m-gsm8k-learning-and-resume).

- User authorized the one-A100 gate, then 1,000 updates on two A100 80GB GPUs if it passes.
  Workspace `ai2/open-instruct-dev`, cluster `ai2/saturn`, budget `ai2/oe-other`.
- Launcher: `scripts/train/olmo-hybrid-small/gsm8k_release_275m.sh`. Full GSM8K train
  and platinum test splits; same base prompt, correctness-only GRPO (beta 0), lr 1e-6,
  four prompts x four completions, 16,000 episodes = 1,000 learner updates.
- One learner / one vLLM engine; 512 prompt / 2048 response tokens; no zero-variance
  filtering; preemptible normal priority; three automatic retries; 24h per-attempt cap.
- Native model/optimizer/trainer/data-preparation checkpoint every 100 updates,
  loaded automatically with `LoadStrategy.if_available`; latest checkpoints remain
  under the launch's stable `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/runs/<run-name>/native`.
  W&B ID is stable across Beaker restarts. Restarts roll back to the latest fully completed checkpoint; a pending
  asynchronous save can increase rollback beyond the nominal 100-update interval.
- Resume audit found the initial serving-weight sync preceded Trainer.fit's checkpoint
  load. Fixed VLLMWeightSyncCallback.pre_train to republish restored weights/step before
  requesting resumed training data. CPU regression covers fresh and restored startup.
- Previous image `01M5H054GQTHEWEXE1ZT4M81H3` passed offline container registration
  and ten CPU release tests; CUDA test skipped on CPU. Rebuild required for resume fix.
- Launch IDs/results will be appended after submission. Gate remains a prerequisite.

### GPU gate attempt 1

- [Beaker](https://beaker.org/ex/01M5H14REXKB7J3FJ5EY6AM8Y6), image
  `01M5H14C0C0C53G6XV16JXHM94`, code `426992b92`: **FAILED**, exit 1.
- Actual CUDA test ran (not skipped). It loaded all 192 tensors but stopped before
  the optimizer/export checks at BF16 HF-eager vs native-FlashAttention pointwise
  parity: max absolute logit difference 0.140625, 18,288 / 1,605,632 elements (1.1%)
  exceeded rtol 0.02 / atol 0.05. No training launch follows a failed gate.
- Diagnostic revision adds FP16 interchange parity with stricter rtol 0.005 /
  atol 0.01, BF16 next-token KL < 1e-3 per position, and reports each model's
  BF16-vs-FP16 KL. Full-length optimizer update remains BF16. This separates
  reduced-precision kernel drift from architecture/weight interchange errors;
  the revised GPU result is still pending.

### GPU gate attempt 2 and precision diagnosis

- [Beaker](https://beaker.org/ex/01M5H1JP4RYQ0DVMKBD2TH9XAK), code
  `3370c0ee3`: **FAILED** the tighter FP16 pointwise check before update/export.
- FP16 max absolute difference 0.019531 (versus BF16 0.140625); only 40 /
  1,605,632 values exceeded rtol 0.005 / atol 0.01, max failing absolute 0.015625.
  This precision scaling supports rounding drift but is not by itself a pass.
- Next gate checks structural parity with native Torch math attention and HF eager
  in FP32 (rtol 5e-4 / atol 1e-3), reports FlashAttention FP16 differences, and
  independently gates FlashAttention BF16 next-token KL < 1e-3 per position.
  The full-length optimizer test and export/reload remain required; training is pending.

### GPU gate attempt 3

- [Beaker](https://beaker.org/ex/01M5H1TK0Q7VAZG446WG6BC6ZT), code
  `6802233b0`: **PASSED**, 1 actual CUDA test, exit 0, 307.21s.
- Before update: FP32 math max absolute error 0.00067067; FP16 FlashAttention
  max error 0.019531; BF16 cross-backend max KL 0.00055496. BF16-vs-FP16
  reference drift within HF 0.00043708 and native 0.00034995.
- Full 2560-token BF16 CE loss 13.774601; finite backward, AdamW parameter update,
  HF export and reload passed. After update: FP32 math max error 0.00140429
  (within relative+absolute bound); BF16 cross-backend max KL 0.00053793.
- Follow-up concern: after the synthetic update, HF BF16-vs-FP16 max KL rose to
  0.18166320 (native 0.19600232), despite both backends agreeing at each precision.
  This probe used lr 1e-4, 100x the production 1e-6. Before starting the long run,
  repeat at the actual production learning rate and report both BF16 and FP16
  against FP32 to identify relevant precision sensitivity. No production math changed.

### Production-rate GPU gate and 1000-update launch

- [GPU test](https://beaker.org/ex/01M5H2C69J8HBS8F70X3HT2K3C), code
  `fe8ae0ba7`, image `01M5H2BXPNCMGW951BABKEPJGF`: **PASSED**, exit 0,
  1 actual CUDA test, 383.41s. No skip.
- Before the update: FP32 HF/native math max absolute logit difference 0.00067067;
  BF16 HF/native max next-token KL 0.00055496; BF16 vs FP32 max KL 0.00047473.
- Full 2560-token BF16 CE loss 13.773473, finite backward, AdamW at production
  lr 1e-6 changed parameters; HF export/reload passed. After update/reload:
  FP32 HF/native max error 0.00114083 (within relative+absolute bound), BF16
  HF/native max KL 0.00044331, BF16 vs FP32 max KL 0.00051355, FP16 vs FP32
  max KL 0.00006948. The earlier 100x-larger synthetic update's large precision
  sensitivity did not reproduce at the production rate. This qualifies native/HF
  interchange and a synthetic update, not task learning or a preemption restart.
- [Training](https://beaker.org/ex/01M5HYS6EMRWQ01SDFJW85F1GF), submitted after
  the gate passed, code `fe8ae0ba7`, same validated image.
- Command: `RUN_NAME=hybrid-release-275m-gsm8k-1k-20261022-140323 ./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid-small/gsm8k_release_275m.sh`.
- Verified stored experiment spec: 2 GPUs, Saturn, normal priority, workspace
  `ai2/open-instruct-dev`, budget `ai2/oe-other`, `minRuntime: 0s`, `autoResume: true`,
  3 retries, 24h per-attempt cap. Exact 16,000 / (4 x 4) = 1,000 updates; native
  checkpoints every 100; `WANDB_RESUME=allow`, fixed W&B ID `yibm1xmx`.
- Durable checkpoint path: `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/runs/hybrid-release-275m-gsm8k-1k-20261022-140323/native`.
- Initial status: **QUEUED**. Scheduler reports all 27 matching Saturn nodes lack
  sufficient slots for two GPUs. End-to-end RL and actual preemption recovery
  remain unvalidated until the run executes.


## Olmo Hybrid release 275m BF16 rerun (2026-10-27)

- User authorized rerunning the previous Saturn experiment with its existing recipe.
- Prior experiment `01M5HYS6EMRWQ01SDFJW85F1GF` failed on all four attempts
  during `Trainer._dry_run_batch`: FlashAttention received FP32 tensors. No RL
  update completed. The standalone gate did not exercise GRPOTrainModule.
- Root cause/fix: GRPO's overridden train_batch bypassed the base module's
  forward autocast context. Set autocast_precision from model_dtype and enter
  that context for reference, old-policy, and differentiable learner forwards.
  FP32 optimizer parameters are preserved. Three CPU regressions exercise these
  actual paths, including the dry-run entry; both resume regressions also pass.
- Recipe unchanged: primary 275m midtrain step0038147, full GSM8K training and
  platinum evaluation; lr 1e-6; 4 prompts x 4 samples; 16,000 episodes = 1,000
  updates; prompt/response limits 512/2048; pack length 4096; beta 0.
- Resources: ai2/saturn, 2 A100 80GB (one learner / one vLLM engine), normal
  preemptible priority, minRuntime 0, auto-resume, three retries, 24h timeout;
  workspace ai2/open-instruct-dev; budget ai2/oe-other. Weka native checkpoint
  and HF save every 100 updates; evaluation every 100 plus step zero.
- Image built and published on the existing Beaker session (no local Docker build):
  `01M5ZA477GD4BSRBTTSP5692QB`, source `c8b5da548`, layered on the validated
  release image `01M5H2BXPNCMGW951BABKEPJGF`; dependency versions unchanged.
- [Rerun](https://beaker.org/ex/01M5ZAPEQFTT3YEMK107ZSVFK4), submitted via
  `RUN_NAME=hybrid-release-275m-gsm8k-1k-bf16-20261027 ./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid-small/gsm8k_release_275m.sh`.
- Verified stored spec: 2 GPUs on Saturn, minRuntime 0s, autoResume true,
  3 retries, timeout 24h. Stable W&B ID `bmauwnfk`, WANDB_RESUME=allow.
  Native state directory:
  `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/runs/hybrid-release-275m-gsm8k-1k-bf16-20261027/native`.
- Cluster preparation required unsetting the session's malformed BEAKER_TOKEN,
  temporary GitHub credentials for the pinned private plugin, and
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPEN_INSTRUCT=0.0.1 to avoid a slow Weka
  git-describe timeout. These affected the launch environment, not the image.
- Validation: 34 focused precision/resume/GRPO/release CPU tests passed; formatting,
  lint, compilation, and type checks passed. GPU execution status at submission:
  QUEUED. Neither a successful GRPO update nor preemption recovery is claimed.


## Interactive Rhea SGLang qualification (2026-10-27)

- User explicitly requested direct experimentation in an interactive Rhea session.
  Session `01M5ZB6KJDNFG7KYKWHN8X948W`, node `rhea-cs-aus-473.reviz.ai2.in`,
  one RTX A6000 48GB, CPU12, host memory96GiB, shm16GiB, normal priority,
  minRuntime0, timeout90m; workspace ai2/open-instruct-dev, budget ai2/oe-other.
  Image `01M5ZA7AP531QVWGSAGSC8M5B8`, Torch2.13/CUDA13, Transformers5.12.1,
  SGLang0.5.19.dev49+g3145136, FLA0.5.2, scaling-ladders plugin `0fb485ce`.
- Live Weka source `/weka/oe-adapt-default/michaeln/miles-olmo-core-hybrid`,
  branch `codex/hybrid-mainline`; tmux `hybrid-tests`; preserve the image's
  PYTHONPATH after prepending the checkout. No local Docker builds.
- All five final midtrain releases: 275m/450m step0038147, 810m step0023842,
  1.4b/2.7b step0011921. BF16, TP1/PP1, no CUDA graphs, prefix cache, chunked
  prefill, or quantization. Four running requests, memory fraction0.35, context4096.
- Initial batch `01M5ZA7ZXSFH1T18AQFMX8P9RQ` was stopped after exposing a
  harness bug: SGLang's returned prompt suffix includes an initial None; dropping
  it shifted token alignment. Corrected harness regression tests cover this.
- Expanded source `d023770c`: 275m/1.4b/2.7b passed selected-token/full-sequence
  checks before and after disk reload; 450m/810m exceeded the 0.08 pointwise
  logprob bound. All sizes loaded and decoded. Reports:
  `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/qualification/rhea-interactive-d023770c`.
- Follow-up whole-vocabulary checks exposed 810m cached-versus-full-sequence
  KL 0.00176434 at prompt length1. Independent HF cached-versus-full evaluation
  reproduced that exact KL and TV0.0244253. FP32 logits alone: KL 0.00151127.
  No tolerance relaxed; the reference now records cached and full-sequence paths
  separately. This distinction does not validate learner/rollout importance ratios.

### Completed BF16 reference-backbone qualification

All five final-midtrain sizes PASSED on the same A6000 and harness/adapter
revision `8bb6da4a`, with BF16 model computation and BF16 logits (no FP32-head
override). Each passed 16 functional requests before plus 16 after disk reload,
and four dense-distribution probes before plus four after reload: 160 functional
requests and 40 dense probes total. Greedy agreement was 1.0 in every case.

| Size | Result | Max selected logprob error | Warm output tokens/s |
| --- | --- | ---: | ---: |
| 275m | PASSED | 1.23978e-05 | 62.1–67.1 |
| 450m | PASSED | 3.00407e-05 | 42.2–45.0 |
| 810m | PASSED | 0.000198603 | 39.2–45.0 |
| 1.4b | PASSED | 0.000464261 | 28.9–32.0 |
| 2.7b | PASSED | 9.9659e-05 | 19.8–22.4 |

Throughput ranges are complete warm batches after reload, including prefill and
request overhead; these are short synthetic serving measurements, not RL throughput.
Maximum measured cached-reference KL was0 after nonnegative clamping at float32
resolution; the small nonzero logprob errors above mean this is not exact equality.
HF cached-versus-full discrepancies remain (450m up to 0.00240536; 810m0.00176434),
so training/rollout importance-ratio calibration remains a separate GPU requirement.

Reports: `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/qualification/rhea-hf-bf16-8bb6da4a`.
Harness SHA256: `e0f9d51f60e463149cffd72ed6f7bec92e0d013ce12eec82830a49387c37307c`.
Adapter SHA256: `73e345ec258aded60a37bea5e93d70a5bc5f753199bc41b004c809a6ebf4df62`.
No checkpoint weights were modified. This qualifies the HF-backbone fallback;
the packed/RadixAttention path does not pass the same matrix at every scale.
Native MILES learner updates, live publication, restart recovery, and learning
quality remain pending. CPU validation:19 runtime tests and148 companion tests
passed; scoped Ruff and companion formatting/lint/compile/type checks passed.

Saturn rerun status refresh (2026-10-27): initial job
`01M5ZAPEVK9K2S43GCB1JPT57B` was preempted by an urgent allocated workload
while starting at 19:34 UTC. autoResume created replacement
`01M5ZDP3XP4YA4V7WASNNGC0RT`, scheduled at 19:41 UTC. No learner logs or
completed update yet. This verifies scheduler requeue, not policy/optimizer
checkpoint restoration. The experiment and durable output/W&B identity are unchanged.


## GSM8K infrastructure comparison: 275m and 450m, 500 updates (2026-10-27)

Status: 450m smoke PASSED; both 500-update main jobs submitted and scheduled on Rhea.

Four fresh runs: Core/vLLM and MILES at each of275m and450m, final general-tree
midtrain step0038147. Each uses BF16, four prompts x four completions (batch16),
500 optimizer updates (8,000 completions), full GSM8K training and platinum
evaluation, prompt512/response2048, packed length4096, lr1e-6, seed1, no KL or
entropy penalty, and retains zero-variance groups. Eval at0/100/200/300/400/500;
native checkpoints and HF saves every100. Fresh output roots per run, auto-resume,
normal preemptible priority, minRuntime0, three retries, timeout24h per attempt.

User requested a hard maximum of one optimizer update off-policy. MILES uses
fully_async=true, max_weight_staleness=1 (Core max_policy_lag=1), barrier publication,
one group/four samples in flight and one batch of completed buffering. The old
stack uses async_steps=1, inflight_updates=false, and max_result_age_steps=1.
The hard guards reject older groups; queue sizing is not the age guarantee.

User chose to retain the stacks' existing reward and optimizer differences, then
explicitly requested matching the upper PPO clip: lower0.2/upper0.272 in both.
MILES keeps numeric GSM8K answer comparison, zero warmup, grad clip1 and eval
top_p0.95; Core/vLLM keeps its exact-string GSM8K answer comparison, existing
warmup/gradient behavior and eval top_p1. This is an end-to-end stack comparison,
not an isolated causal test of the inference backend. Sampling is not bit-identical.

Recommended resources: four jobs x two Rhea RTX A6000 48GB, one learner plus
one inference GPU per job, TP1, all on the same hardware. Eight GPUs peak.
Rhea had26 free slots at preparation, including two empty eight-GPU nodes;
Saturn A100 80GB is the alternative if Rhea availability changes. No extra
learner GPUs recommended for these small models. Budget ai2/oe-other, workspace
ai2/open-instruct-dev; mounts oe-adapt-default and oe-training-default.

MILES retains the qualified HF-backbone SGLang fallback, no graphs/prefix cache/
chunked prefill, max_running_requests4 and memory fraction0.35. Core/vLLM keeps
its existing eager/FLASH_ATTN setup with FP32 SSM cache and memory fraction0.6.
The new MILES native learner/publication path needs a short two-update lifecycle
smoke for each size before launching the four longer jobs. Proposed smoke cap:
two GPUs sequentially, at most1h per size, zero retries, small held-out eval slice,
checkpoint every update; then the full runs only if both smokes pass.

Launch files: MILES `configs/miles/benchmarks/hybrid_{275m,450m}_gsm8k_compare_500.toml`;
Core/vLLM `scripts/train/olmo-hybrid-small/gsm8k_infra_compare_500.sh` with
MODEL_SIZE=275m or450m. Run-file schema planning passed; images/data are prepared
on the existing Hammond CPU session, with no local Docker build.


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

Status: SMOKE QUEUED; cluster image built; GPU learning validation interrupted by preemption.

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

Status: PREPARED (not submitted yet).

User requested a new MILES run from scratch logging to `mnoukhov`.
Reuse qualified image `01M607Z48PYPNN0KHVZ5PGYJEG` and the previous450m
MILES recipe. Change only run/output identity and W&B entity/group:
`hybrid-450m-miles-mnoukhov-20261030`,
`mnoukhov/open_instruct_internal`, group matching the new run name.
No old RL checkpoint/optimizer or attempt ledger is reused. Model initialization
remains final450m midtrain `step0038147`.

Resource/recipe: two A6000s on ai2/rhea (one native OLMo-core learner, one SGLang
inference), BF16/FA2,500 updates,4 prompts x4 samples, lr1e-6, PPO clip0.2/0.272,
max policy lag1/TIS, eval/native checkpoint100, final HF export. Normal/preemptible,
minRuntime0, autoResume=true, three retries,24h per attempt,128GiB host RAM,
32GiB shared memory, workspace ai2/open-instruct-dev, budget ai2/oe-other.
Preemption can consume the MILES restart ledger even after productive attempts;
preserve the existing three-retry bound. W&B entity is corrected; existing runtime
still creates separate W&B runs on restart. No resume-identity source change.

Config: `/home/mnoukhov/open-instruct-miles-hybrid/runs/hybrid-450m-miles-mnoukhov-20261030.toml`
(ignored individual run file, frozen into submitted Beaker payload/receipt).
SHA256: `166e7b7eee31781ac1c93df4fe8272a80d45ec8fb93b68d27482330d4c947667`.
Submission source: companion branch `codex/hybrid-mainline`, commit `07ad70fd3`.
Runtime image source `3f0933565`; only later runtime-path difference is the local
launcher retry-field fix, used when rendering this submission. Training code and
runtime are unchanged. Plan/validate passed;96 focused recipe/launch tests passed.

Launch through the committed wrapper in the companion checkout:
`MILES_EXISTING_IMAGE=01M607Z48PYPNN0KHVZ5PGYJEG ./scripts/train/build_image_and_launch.sh --miles runs/hybrid-450m-miles-mnoukhov-20261030.toml`.

The prior four-GPU Core/vLLM reference completed500 updates with17.287% final
GSM8K Platinum accuracy vs3.970% initially, checkpoint500 and final export saved,
exit0. Its one preemption restored nativecheckpoint100 and data cursor correctly.
Reference: https://beaker.org/ex/01M64J9FRG4PYSB6E7WJ739A7F.
Previous MILES run reached durable200updates and10.670% eval before exhausting
its restart ledger; this new run targets a complete500-update result.
