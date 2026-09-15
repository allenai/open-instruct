# Experimental Qwen3.5 OPD

This prototype uses the public `python -m open_instruct.miles` entry point,
Open Instruct data preparation and Beaker launch plumbing, and the native Miles
Megatron trainer. It is a separate route from the OLMo-core GRPO adapter.

The initial exercise is two updates of **Qwen3.5-4B from Qwen3.5-9B** using
student-generated responses and sampled-token teacher log probabilities. The
[tiny run passed](measurements/qwen35-opd-20260914/README.md): teacher scoring,
two optimizer updates, checkpoint/export auditing and fresh-process export reload.
The broader [prototype plan](plans/qwen35-opd-prototype-20260914.md) includes
additional checks before research-scale use.

## Configuration and launch

Start from [the tiny run file](../../configs/miles/opd/qwen35-4b-tiny.toml).
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

The tested source is `298254e50a1ece28d6291863f67f13942537c3e1`; the immutable
image below contains that source and the complete candidate runtime. From a clean,
committed checkout, use fresh run and asset paths in your workspace:

```bash
MILES_EXISTING_IMAGE=01M2H301PJ0WFB19NCGY5QBP51 \
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

## Current limits

- Only the 4B learner and 9B teacher are accepted. The requested 2B learner needs
  its own architecture profile and exercise.
- Automatic restart and resume are rejected until native checkpoint restoration
  and the data cursor have been exercised together.
- Only GSM8K with held-out evaluation, saving every update, the fixed topology,
  top-k zero, pure OPD and offline tracking are exposed.
  The general Core configuration reference does not describe this prototype's
  closed schema; `plan` shows its resolved defaults.
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

Build the committed integration using the existing `MILES_BASE_IMAGE` procedure;
reusing the earlier Qwen image does not include Core OPD. The image retains
Megatron/mbridge for the original Qwen OPD example and the existing Core GRPO
path. Qualification results for this extension must identify the new immutable
image and source revision; the original Qwen result above does not qualify it.
