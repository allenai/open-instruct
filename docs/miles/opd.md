# On-policy distillation with Miles

Use `python -m open_instruct.miles` with Open Instruct data preparation and Beaker
launching. The learner backend is explicit: OLMo-core for our registered Olmo MoE
and dense Olmo 3 models, or native Miles/Megatron for the Qwen3.5 prototype.
Core OPD reuses the existing Core training loop. See
[Core learners and independent teachers](#olmo-core-learners-and-independent-teachers)
for the cross-tokenizer Qwen-to-Olmo path.

The initial exercise is two updates of **Qwen3.5-4B from Qwen3.5-9B** using
student-generated responses and sampled-token teacher log probabilities. The
[tiny run passed](measurements/qwen35-opd-20260914/README.md): teacher scoring,
two optimizer updates, checkpoint/export auditing and fresh-process export reload.
The broader [prototype plan](plans/qwen35-opd-prototype-20260914.md) includes
additional checks before research-scale use.

## Qwen configuration and launch

Start from [the tiny run file](https://github.com/allenai/open-instruct/blob/robertb/miles-qwen35-opd/configs/miles/opd/qwen35-4b-tiny.toml).
Change `name`, `output.root`, and `output.assets` for your own workspace.
The run directory must be fresh. Assets may be shared across runs; immutable
model revisions and tokenizer identities are recorded when preparing them.

```bash
python -m open_instruct.miles plan configs/miles/opd/qwen35-4b-tiny.toml
python -m open_instruct.miles validate configs/miles/opd/qwen35-4b-tiny.toml
```

The image adds `ISEEKYAN/mbridge` at
`89eb10887887bc74853f89a4de258c0702932a1c`, matching the pinned Miles converter.
This package is distinct from NVIDIA Megatron Bridge. Existing Megatron patches
are retained. The candidate pins a complete cuDNN wheel and sets `CUDNN_HOME` /
`CUDNN_PATH` consistently for PyTorch and Transformer Engine. Qwen uses packed
sequences with explicit FlashAttention; the native CP1 padding metadata is
corrected so FlashAttention 4 can handle its head dimension on B300.
Commit changes before building and launching; `run` invokes the
repository's required `build_image_and_launch.sh --miles` wrapper.

```bash
export MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks
python -m open_instruct.miles run configs/miles/opd/qwen35-4b-tiny.toml \
  --set 'training.phase="prepare"' \
  --set 'launch.cluster="ai2/saturn"' \
  --set 'name="qwen35-opd-prepare"' \
  --set 'output.root="/weka/oe-training-default/YOUR_USERNAME/opd/preparation"'
```

Wait for preparation to succeed, then launch the unchanged training config:

```bash
python -m open_instruct.miles run configs/miles/opd/qwen35-4b-tiny.toml
python -m open_instruct.miles status configs/miles/opd/qwen35-4b-tiny.toml
```

CPU-only preparation with WEKA is restricted to Saturn. Training requests one
four-GPU allocation: two Megatron trainer GPUs (TP2), one learner SGLang GPU, and
one teacher SGLang GPU. Ray sees only the first three GPUs. The teacher loads the
pinned 9B checkpoint and scores complete learner token sequences on a local
endpoint; startup, timeout and failure checks are owned by the launcher.

## Repeat the exercised image

The current shared image contains runtime source
`54c09c020e2b0946eab10889df324353a308eca4` and supports the Core and native Qwen
paths. The [integration qualification](measurements/core-opd-20260914/README.md)
records exact images and tests, including the Qwen regression on its immediate
predecessor. From a clean, committed checkout, use fresh run and asset paths:

```bash
MILES_EXISTING_IMAGE=01M2HNJ96AQMCE69SK5ZA0EJQT \
python -m open_instruct.miles run configs/miles/opd/qwen35-4b-tiny.toml \
  --set 'name="qwen35-4b-opd-repeat"' \
  --set 'output.root="/weka/oe-training-default/YOUR_USERNAME/opd/runs/tiny-01"' \
  --set 'output.assets="/weka/oe-training-default/YOUR_USERNAME/opd/assets"'
```

This still uses `build_image_and_launch.sh --miles`, reusing the explicit image.
Preparation runs automatically if assets are absent; the separate Saturn step
above avoids staging downloads on the GPU allocation. For runtime code changes,
leave `MILES_EXISTING_IMAGE` unset and build with `MILES_BASE_IMAGE`.

When the image cannot be rebuilt locally but only wrapper source changed (no new
dependencies), add `MILES_CODE_OVERLAY=1`: the job fetches the exact committed
HEAD from `origin` (push it first) and copies `open_instruct/`, `scripts/miles/`
and `configs/miles/` over `/opt/core-rl` before `open_instruct.miles train`
starts. The launch receipt records `code_overlay: true` and the `revision` that
ran; the job log prints `code overlay: <revision>`.

## Semantics and evidence

Training uses pure sampled-token OPD: task rewards are zero, and each response
position receives `kl_coef * (teacher_logp - student_logp)` as its advantage.
The reward hook returns that zero as `sample.reward` and keeps the teacher's
scoring response in `sample.metadata` until `post_process` turns it into
`teacher_log_probs`; Miles's rollout metrics round and group by the reward, so
with one sample per prompt every group reports under `zero_std/count_0.0`
(expected noise, not a signal). Missing, nonfinite or misaligned teacher scores
fail the run. GSM8K correctness
is measured separately for evaluation. Thinking is disabled in the prepared
chat templates. This prototype does not train the vision tower.

The run retains native Megatron checkpoints and HF exports on WEKA. The final
HF export includes the trained language weights and, for Qwen3.5, the unchanged
base vision/MTP weights. An audit checks teacher-derived advantages, finite nonzero
gradients, weight changes, and (for Qwen3.5) FP32 `A_log` tensors; a plain language
model such as Qwen3 has neither frozen extras nor `A_log` tensors, and every base
tensor must appear in its export. The exported model is loaded in a fresh
SGLang process and used to generate a short response before marking completion.

Inspect `result.json`, `audit.json`, `teacher-preflight.json`,
`teacher-scores.jsonl`, `export-reload.json`, `training.log`, and `workflow.json`.
Small JSON/log artifacts are copied to Beaker results; tensor checkpoints and
training dumps remain on WEKA. A successful two-update run establishes mechanics,
not an improvement in task accuracy.

## EOPD: entropy-gated forward KL over the teacher top-k

`[distillation] eopd = true` adds the EOPD term of arXiv 2603.07079 (Eq. 9-10) on top of the
sampled-token OPD update above: `L = L_OPD + eopd_alpha * 1[H_t > eopd_tau] * FKL_t`, where
`FKL_t = sum_{j in top-k} q~_t(j) (log q~_t(j) - log p_theta,t(j))` runs over the teacher's
`eopd_top_k` tokens at each response position, `q~_t` is the teacher renormalised over those
tokens and `p_theta` is the student's full-vocabulary probability. `H_t` is the entropy of
`q~_t`, the paper's proxy for the teacher entropy (an SGLang teacher only returns its top-k).
Defaults follow the paper: alpha 1.0, tau 0.8, k 16. It works with either student side
(`distillation.use_rollout_logprobs`); only the rollout-log-prob path has run on GPUs so far
(the tiny EOPD smoke), so a trainer-log-prob EOPD arm wants a tiny smoke first.

Mechanics: `opd_hooks.reward` asks the teacher for `top_logprobs_num = k` alongside the
sampled-token scores and checks every position returned exactly k finite entries;
`opd_hooks.post_process` stores the ids and log-probs in `sample.train_metadata`, which Miles
ships to the trainer as the batch's `metadata` list (the runtime patch adds `metadata` to the
Megatron train-step keys). The learner runs with `--loss-type custom_loss
--custom-loss-function-path open_instruct.miles.eopd_loss.policy_loss`: upstream
`policy_loss_function` unchanged plus the gated FKL through the same per-sample reducer, with the
student log-probs at the k ids computed on each tensor-parallel vocabulary shard
(`eopd_math.student_log_probs_at`). Settings reach the hooks and the loss as
`OI_OPD_EOPD_TOP_K/ALPHA/TAU`. Training logs `train/eopd_fkl_loss`, `train/eopd_fkl`,
`train/eopd_gate_frac`, `train/eopd_teacher_proxy_entropy` and `train/eopd_teacher_topk_mass`;
`teacher-scores.jsonl` gains per-token `eopd_gate`, `eopd_proxy_entropy` and `eopd_topk_mass`;
`audit.json` gains an `eopd` block that re-derives the gate from the dumped top-k and checks the
logged metrics. [The EOPD tiny run](https://github.com/allenai/open-instruct/blob/robertb/miles-qwen35-opd/configs/miles/opd/eopd-eopd-qwen3-tiny.toml)
is the OPD tiny run with the term switched on.

## Qwen configuration reference

`plan` prints the resolved defaults. Beyond the tiny run file, the Megatron OPD
schema accepts:

- `model.source` / `teacher.source`: a Hugging Face repository with an immutable
  40-character `revision` (the pinned Qwen3.5-2B, 4B and 9B revisions, and the
  Qwen3-1.7B-Base, Qwen3-4B-Base and Qwen3-8B revisions used by the EOPD
  replication, are filled in when omitted) or a local checkpoint directory
  (`/weka/...`, `./relative`), which must not set `revision`.
  `model.architecture` names the Megatron profile (`qwen3.5-2B`, `qwen3.5-4B`,
  `qwen3.5-9B`, `qwen3-1.7B`, `qwen3-4B`, `qwen3-8B`); it is inferred for the
  pinned repositories and required for local learners. Profiles under
  `open_instruct/miles/model_profiles/` shadow the Miles copies; the 2B profile
  lives there and has not been exercised on GPUs.
- `[data]`: one registered task with `eval_count` (`gsm8k`, `math`, ...), or
  pre-rendered prompts through `prompt_data`, `eval_prompt_data` name/path pairs
  and the `reward_config` verifier registry. Held-out samples are scored by the
  verifiers named in their `metadata.verifiers` (the Core route's registered
  reward), so evaluation follows the data rather than a fixed GSM8K scorer.
  `scripts/miles/prepare_qwen35_math_prompts.py` renders the Open Instruct
  Qwen3.5 math campaign data (fixed DAPO split, AIME 2025, BRUMO 2025, MATH-500)
  with the `qwen_instruct_user_boxed_math` template and the `math` verifier in
  that layout; it drops the three DAPO training prompts that repeat holdout
  problems, which Miles would otherwise reject.
  `scripts/miles/prepare_eopd_math_prompts.py` renders the EOPD paper's
  (arXiv 2603.07079) MATH / DAPO-Math-14k training prompts and its six
  evaluation sets through the Qwen3 chat template in non-thinking mode for the
  `configs/miles/opd/eopd-opd-*.toml` replication specs. `eopd-math-v1` on Weka carries
  the paper's App. C suffix on every set (evaluation); `eopd-math-v2-train` re-renders only
  the two training sets with the suffix the authors' verl preprocessing uses ("Let's think
  step by step and output the final answer within \boxed{}."), keeping all 14,116 DAPO
  prompts as the paper did (no eval-overlap drop, since the rendered strings differ).
- `training.num_rollouts`, `training.save_interval` (HF export cadence) and
  `training.eval_interval` (`0` evaluates before the first update and after the
  last one, as the prototype did). `training.optimizer_steps_per_rollout`
  (default 1) splits each rollout into that many PPO mini-batches: Miles'
  `--global-batch-size` becomes `rollout_batch_size * samples_per_prompt`
  divided by it, so `128 x 1` prompts with `4` steps is the paper-style
  "batch 128, mini-batch 32" schedule; the audit expects
  `num_rollouts * optimizer_steps_per_rollout` optimizer steps.
  `training.loss_aggregation` (`response`, the Miles default per-response mean, or `token`,
  verl's `token-mean` over the mini-batch via `--calculate-per-token-loss`; the EOPD
  replication specs use `token`).
- `optimizer.learning_rate`, `optimizer.lr_decay_style` (`constant`, `cosine`,
  `linear`; anything but `constant` sets `--lr-decay-iters` to the total number
  of optimizer steps), `optimizer.lr_warmup_iters` and `optimizer.min_lr`;
  `optimizer.weight_decay`, `optimizer.adam_beta1`, `optimizer.adam_beta2` (defaults 0.0,
  0.9, 0.98 as the Qwen3.5 runs used; the EOPD replication specs set verl's 0.01, 0.9, 0.999).
- `inference.max_response_length`, `inference.max_context_length`,
  `inference.max_running_requests` (learner SGLang concurrency; the KV budget is
  `max_context_length` times this), `inference.top_p` (rollout nucleus
  sampling; must stay `1.0` while the rollout log-probs are the student side),
  `inference.eval_temperature`, `inference.eval_top_p`,
  `inference.eval_samples_per_prompt` (the logged `eval/<set>` reward is the mean
  over every sample, i.e. Avg@k) and `inference.eval_max_response_length` (`0`
  reuses `max_response_length`; a longer evaluation budget must still fit in
  `max_context_length`).
- Topology: `trainer.gpus` (a multiple of `trainer.tensor_parallel_size`),
  `inference.gpus` (a multiple of `inference.tensor_parallel_size`) and
  `teacher.gpus` (the teacher's SGLang tensor parallelism). The task requests
  their sum, at most one node; roles occupy consecutive devices in that order.
  Only 2/1/1 has been exercised; `plan` warns on any other topology.
- Local teacher checkpoints saved by transformers 5 as a text-only
  `Qwen3_5ForCausalLM` keep the repository's `model.language_model.*` tensor
  names. SGLang's text-only loader skips every such tensor, serves random
  weights and the sampler hits a CUDA device-side assert on NaN probabilities
  (Beaker 01M2K5Y9JT4CD9WN1HRD12GGA9, 01M2K86VPW6HB549MPCNVEEDB2). `prepare`
  detects this layout and stages a renamed copy of the shards (`model.*`,
  vision and MTP tensors dropped) instead of symlinking them.
- `distillation.use_rollout_logprobs`: score the student side of the reverse KL
  with the rollout engine's log-probs instead of the trainer's pre-update forward
  pass. Open Instruct's `--use_vllm_logprobs` OPD runs behave like `true`; the
  exercised prototype used `false`.
- `tracking.wandb_mode` (`offline`, `online`, `disabled`), `tracking.wandb_project`
  and `tracking.wandb_entity`; `online` requires `launch.secrets.WANDB_API_KEY`.
  `model.align_eos_with_teacher = true` (set in the `eopd-opd-qwen3-*` specs) gives the
  Qwen3-Base learner the teacher's `<|im_end|>` as its eos during preparation (the prepared
  asset gets an `-eos` suffix and its `generation_config.json` keeps `<|endoftext|>` as a
  second stop id), so rollouts stop where the teacher's answers end instead of running on
  after `<|im_end|>` until `<|endoftext|>`, and the special-token identity check passes.
  The learner still ends most answers with its own `<|endoftext|>`, which the chat-trained
  teacher never emits after an answer (its log-prob there is around -21 nats, so pure OPD
  would punish stopping and drive every response to the length cap; arm 2 attempt 2 hit
  100% truncation by rollout 20 this way). The launcher therefore reads the prepared
  learner's `generation_config.json` and exports `OI_OPD_TEACHER_EOS_REMAP`
  (`learner_stop_id:teacher_eos_id`), and the rollout hook scores a *terminal* learner stop
  id as the teacher's eos: the teacher is asked about `<|im_end|>` at that position, the
  sample keeps its real tokens, and the `teacher-scores.jsonl` record carries
  `eos_remapped`. A learner stop id inside a response and truncated responses are untouched.
- Qwen3 learners use the repository profiles `open_instruct/miles/model_profiles/qwen3-1.7B.py`
  and `qwen3-4B.py`, which pin `--padded-vocab-size 151936`. Without it Megatron pads the
  vocabulary to 152064 under TP2 while mbridge 0.15.1 scatters the unpadded HF embedding, and
  `convert_hf_to_torch_dist.py` fails with `ProcessGroupNCCL::scatter: invalid tensor size`.
- `[miles]`: native passthrough for anything the schema does not model. Keys are the
  pinned parser's option names with underscores (`use_tis = true`, `tis_clip = 2.0`,
  `eps_clip_high = 0.28`, `clip_grad = 0.5`, `sglang_mem_fraction_static = 0.7`,
  `rollout_shuffle = false`), the same spelling and validation as the structured format's
  `[miles]` table (see [native options](native-options.md); `options.json` is the snapshot).
  An option the wrapper hard-codes is replaced in place, a switch set to `false` disappears,
  and everything else is appended, so the run file stays the single source of truth. An option
  the schema or the launcher already owns (`weight_decay`, `num_rollout`, `save`, `opd_type`,
  `calculate_per_token_loss`, ... the `OWNED_NATIVE_OPTIONS` table in `opd_config.py`) is
  rejected with the control that sets it. `plan` lists the passed-through names as a warning
  because nothing here is qualified by the wrapper; `--set miles.use_tis=true` works from the
  command line.

## Qwen prototype limits

- Resume: set `training.resume = true` and `launch.auto_resume = true`. Beaker then
  re-queues a preempted job, the workflow re-enters the same `output.root`, and the
  learner passes `--load output.root/checkpoints` whenever
  `latest_checkpointed_iteration.txt` exists there (weights, optimizer, RNG and the
  data cursor from `checkpoints/rollout`; training continues at the next rollout).
  Beaker caps `launch.min_runtime` at 8h, so runs longer than that need this.
  W&B starts a new run per attempt inside the same group.
- The student rollout router runs with `--router-disable-circuit-breaker`. Its default
  breaker opens for 60s after ten failed requests within two minutes (the aborts at the
  end of a rollout count), returning `503 no_available_workers` to every request, and
  Miles retries a request at most 60 times one second apart. The 2B math runs logged
  thousands of these retries per run and reached attempt 42 of 60 before the fix.
- Only top-k zero and pure OPD are exposed; only GSM8K data, the 4B learner, the
  default 2/1/1 topology, saving every update and offline tracking have been
  exercised on GPUs. The general Core configuration reference does not describe
  this schema.
- Before a colleague scales up, review independent teacher/student probability
  agreement, publication correctness, resume behavior, longer contexts and task
  quality. The original plan describes those broader qualification gates.

## OLMo-core learners and independent teachers

The Core OPD extension uses the existing Core actor, optimizer, expert parallelism,
publication, evaluation and checkpoint loop. It selects the learner architecture
from the prepared HF `model_type`: our registered `olmo3moe` and dense `olmo3`
paths are the initial targets. The teacher is an independent SGLang service and
need not have the learner architecture. No additional CLI command is needed.

Copy `configs/miles/opd/olmo-moe-tiny.toml` or `olmo3-tiny.toml`, choose a fresh
`output.root`, then use the usual `plan`, `validate`, and `run` commands. These
examples specify the checkpoint, two Core trainer GPUs, one learner rollout GPU,
and one Qwen3.5-9B teacher GPU. A different teacher can use another local HF
checkpoint or a remote repository with an immutable revision; set its GPU count
and tensor parallelism together. This does not automatically implement new
learner architectures or unsupported SGLang teacher architectures.

```toml
[training]
algorithm = "opd"

[trainer]
backend = "olmo_core"

[model]
source = "/weka/path/to/learner-hf"
format = "hf"

[teacher]
source = "Qwen/Qwen3.5-9B"
revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
gpus = 1
tensor_parallel_size = 1
max_context_length = 8192
chat_template_kwargs = { enable_thinking = false }

[distillation]
alignment = "exact_text_spans"
kl_coef = 1.0
```

`shared_token_ids` requires identical tokenizers. `exact_text_spans` renders the
original conversation with the teacher's template, scores the learner response
under the teacher tokenizer, and supervises only identical token spans. This is
a partial objective inspired by [SimpleOPD](https://arxiv.org/html/2608.14277v1),
not full cross-vocabulary KL. Unmatched tokens, special tokens and `</think>`
termination spans receive zero OPD contribution. The ordinary loss mask and
normalization are preserved. Prepared inputs must retain `metadata.opd_messages`;
the built-in task and manifest preparation paths now retain those messages.

The first implementation requires synchronous barrier publication on one node.
It forces pre-update Core scoring, uses zero task reward, disables advantage
whitening, and leaves optional reference KL in `optimizer.kl_loss_coef`.
The MoE example retains its existing router regularizers. Alignment coverage and
an exact advantage check are recorded in `checkpoints/training_contract_rank*.jsonl`,
alongside gradient and weight-change diagnostics. Teacher identity is retained in
`cluster/<attempt>/teacher.json`; tokenizer fingerprints travel with each sample.
An enabled final HF export is reloaded through SGLang and recorded in
`export-reload.json` before the supervisor reports success.

The [Core qualification report](measurements/core-opd-20260914/README.md) records
successful two-update MoE and dense runs, including teacher-signal audits,
checkpoints and fresh export reloads. It also records the passing native Qwen OPD
and Core GRPO compatibility checks. The image retains Megatron/mbridge for Qwen;
Core learners use the registered OLMo-core trainer.

To reproduce the MoE exercise on the final image:

```bash
MILES_EXISTING_IMAGE=01M2HNJ96AQMCE69SK5ZA0EJQT \
python -m open_instruct.miles run configs/miles/opd/olmo-moe-tiny.toml \
  --set 'name="core-opd-moe-repeat"' \
  --set 'output.root="/weka/oe-training-default/YOUR_USERNAME/opd/runs/moe-tiny-01"'
```

Use `olmo3-tiny.toml` for the dense learner. These files deliberately keep the
image outside the run schema: `MILES_EXISTING_IMAGE` selects an immutable runtime;
unset it and use `MILES_BASE_IMAGE` to build committed runtime changes. Always use
a fresh output root. The two-update exercises establish mechanics; response
truncation and partial alignment need attention before a learning comparison.
