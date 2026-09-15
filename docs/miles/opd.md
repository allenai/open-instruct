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

## Semantics and evidence

Training uses pure sampled-token OPD: task rewards are zero, and each response
position receives `kl_coef * (teacher_logp - student_logp)` as its advantage.
Missing, nonfinite or misaligned teacher scores fail the run. GSM8K correctness
is measured separately for evaluation. Thinking is disabled in the prepared
chat templates. This prototype does not train the vision tower.

The run retains native Megatron checkpoints and HF exports on WEKA. The final
HF export includes the trained language weights and unchanged base vision/MTP
weights. An audit checks teacher-derived advantages, finite nonzero gradients,
weight changes, and FP32 `A_log` tensors. The exported model is loaded in a fresh
SGLang process and used to generate a short response before marking completion.

Inspect `result.json`, `audit.json`, `teacher-preflight.json`,
`teacher-scores.jsonl`, `export-reload.json`, `training.log`, and `workflow.json`.
Small JSON/log artifacts are copied to Beaker results; tensor checkpoints and
training dumps remain on WEKA. A successful two-update run establishes mechanics,
not an improvement in task accuracy.

## Qwen configuration reference

`plan` prints the resolved defaults. Beyond the tiny run file, the Megatron OPD
schema accepts:

- `model.source` / `teacher.source`: a Hugging Face repository with an immutable
  40-character `revision` (the pinned Qwen3.5-2B, 4B and 9B revisions are filled
  in when omitted) or a local checkpoint directory (`/weka/...`, `./relative`),
  which must not set `revision`. `model.architecture` names the Megatron profile
  (`qwen3.5-2B`, `qwen3.5-4B`, `qwen3.5-9B`); it is inferred for the pinned
  repositories and required for local learners. Profiles under
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
- `training.num_rollouts`, `training.save_interval` (HF export cadence) and
  `training.eval_interval` (`0` evaluates before the first update and after the
  last one, as the prototype did).
- `inference.max_response_length`, `inference.max_context_length`,
  `inference.max_running_requests` (learner SGLang concurrency; the KV budget is
  `max_context_length` times this), `inference.eval_temperature` and
  `inference.eval_samples_per_prompt`.
- Topology: `trainer.gpus` (a multiple of `trainer.tensor_parallel_size`),
  `inference.gpus` (a multiple of `inference.tensor_parallel_size`) and
  `teacher.gpus` (the teacher's SGLang tensor parallelism). The task requests
  their sum, at most one node; roles occupy consecutive devices in that order.
  Only 2/1/1 has been exercised; `plan` warns on any other topology. Keep
  `teacher.gpus = 1` for Qwen3.5 hybrid GDN teachers: with SGLang tensor
  parallelism 2 the first `/generate` probe returned NaN logits and the sampler
  hit a CUDA device-side assert (Beaker 01M2K5Y9JT4CD9WN1HRD12GGA9).
- `distillation.use_rollout_logprobs`: score the student side of the reverse KL
  with the rollout engine's log-probs instead of the trainer's pre-update forward
  pass. Open Instruct's `--use_vllm_logprobs` OPD runs behave like `true`; the
  exercised prototype used `false`.
- `tracking.wandb_mode` (`offline`, `online`, `disabled`), `tracking.wandb_project`
  and `tracking.wandb_entity`; `online` requires `launch.secrets.WANDB_API_KEY`.

## Qwen prototype limits

- Automatic restart and resume are rejected until native checkpoint restoration
  and the data cursor have been exercised together.
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
