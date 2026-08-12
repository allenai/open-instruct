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
