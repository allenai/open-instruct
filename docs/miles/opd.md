# Experimental Qwen3.5 OPD

This prototype uses the public `python -m open_instruct.miles` entry point,
Open Instruct data preparation and Beaker launch plumbing, and the native Miles
Megatron trainer. It is a separate route from the OLMo-core GRPO adapter.

The initial exercise is two updates of **Qwen3.5-4B from Qwen3.5-9B** using
student-generated responses and sampled-token teacher log probabilities. The
first GPU run is in progress; this page does not yet establish qualification.
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
are retained. Commit changes before building and launching; `run` invokes the
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
- Only the fixed topology, top-k zero, pure OPD and offline tracking are exposed.
  The general Core configuration reference does not describe this prototype's
  closed schema; `plan` shows its resolved defaults.
- Before a colleague scales up, review independent teacher/student probability
  agreement, publication correctness, resume behavior, longer contexts and task
  quality. The original plan describes those broader qualification gates.
