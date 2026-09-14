# Qwen3.5 OPD prototype exercise — September 14, 2026

Status: **passed**. [The tiny run](https://beaker.org/ex/01M2H33R16RBRTBCZHCH5BYCW9)
finalized with exit code 0 after two learner updates, an artifact audit and a
fresh-process HF export reload. The [machine-readable summary](summary.json)
records the evidence. This qualifies the bounded 4B mechanics exercise.

## Successful run evidence

- Source: `298254e50a1ece28d6291863f67f13942537c3e1` on
  `robertb/miles-qwen35-opd`.
- Immutable Beaker image: `01M2H301PJ0WFB19NCGY5QBP51`.
- Result dataset: `01M2H33R1CDW9VDBK93TQ15SQ2`.
- Runtime: September 14, 23:14:21–23:31:05 UTC, about 16 minutes 44 seconds.
- Teacher scores: 16 learner responses, 3,947 response tokens; finite scores with
  checked token IDs and response alignment.
- Updates: steps 0 and 1, gradient norms 11.2436 and 12.7093. Both saved batches
  exactly match `advantage = teacher_logp - learner_logp` (maximum error zero).
- Export: 359 tensors differ from the base and 334 differ from the preceding
  update. All 24 `A_log` tensors remain FP32. The export restores 312 unchanged
  base tensors omitted by the native language-only export.
- Lifecycle: native checkpoints and HF exports saved at both updates; final native
  checkpoint marker is `1` (zero-based rollout ID). A fresh SGLang process loaded
  `hf-1` and generated 32 tokens with finite output log probabilities.
- Checks: 200 focused CPU tests, eight GPU runtime preflight checks, Ruff and
  OPD-module type checks passed. Documentation builds successfully. These are
  prototype tests, not a run of `scripts/test/run_gpu_pytest.sh`.

Held-out accuracy was 4/8 before training and 3/8 after, with 5/8 responses
truncated in each evaluation. This tiny, length-capped exercise demonstrates no
quality improvement. The first train phase took 238 seconds including compilation;
the second took 11 seconds. Saving and exporting at every update took roughly
one to two minutes per update, so this run is unsuitable for throughput claims.

Artifacts remain under
`/weka/oe-training-default/robertb/open-instruct/opd/runs/tiny-20260914-f`.
The final export is `hf-1`; small logs, audit and result JSON are also in Beaker
results. Tensor training dumps and native checkpoints remain on WEKA.

## Scope and integration review

The prototype is Qwen3.5-4B learning from a frozen Qwen3.5-9B teacher. It uses
16 prepared GSM8K training prompts and eight held-out prompts, seed 17, with
thinking disabled. Each of two updates consumes four prompts with two learner
responses each. Responses are capped at 256 tokens and context at 2048; this
is a mechanics check, not a quality experiment.

Open Instruct owns configuration, prompt preparation, Beaker submission,
managed-teacher startup/shutdown, token-score checks and result collection.
The pinned Miles driver owns Megatron training, learner rollout generation,
weight publication and native checkpoint/export writes. OPD dispatch is
separate from the existing OLMo-core GRPO route.

The allocation is four NVIDIA B300 GPUs on Holmes: two trainer ranks with TP2,
one learner SGLang engine, and one teacher SGLang engine. Ray sees only the
first three devices. CPU-only asset preparation was submitted to Saturn.

The image adds `ISEEKYAN/mbridge` at
`89eb10887887bc74853f89a4de258c0702932a1c`, separately from NVIDIA Megatron
Bridge. Existing Megatron patches remain in place. The candidate also pins
`nvidia-cudnn-cu13==9.23.2.1` and explicitly points Transformer Engine at that
wheel using `CUDNN_HOME` and `CUDNN_PATH`; the inherited system installation
otherwise loaded cuDNN 9.14 while PyTorch loaded the pip installation.

Qwen's linear-attention adapter requires packed sequences. The working
candidate therefore keeps `qkv_format=thd` and explicitly selects FlashAttention.
For OPD with CP1, a narrow Miles patch marks the batch as having no gaps between
sequences: native `get_batch` already represents end padding as a separate dummy
sequence. This enables FlashAttention 4 for the dense attention layers without
changing the linear-attention layout. The GPU preflight uses actual Miles batch
construction, compares real-token attention against a math reference, runs
backward and verifies zero gradients on the dummy sequence.

Other fixes are a text-prompt guard in Miles' multimodal dataset path and an
evaluation hook that restores the training reward hooks on the same cached
argument object. Pure OPD delegates its loss and teacher-score extraction to
Miles; GSM8K evaluation uses Open Instruct's verifier.

## Attempts and findings

1. [Asset preparation](https://beaker.org/ex/01M2GYJ27QY0ED9F1W0KRPAV6V)
   succeeded and recorded pinned models, matching token identities and data.
2. [Attempt A](https://beaker.org/ex/01M2GYRGTPZ5QMESV03B4EWXT9) converted the
   learner and scored a teacher probe, then rejected an obsolete native flag.
3. [Attempt B](https://beaker.org/ex/01M2GZFYA8ZHMNYDC2XG2E30QD) exposed the
   multimodal dataset's assumption that a formatted text prompt was a message list.
4. [Attempt C](https://beaker.org/ex/01M2H046FGM4MTFBV8R33T4GJM) and
   [attempt D](https://beaker.org/ex/01M2H15V3RS2HBFV6EE4P706XB) reached teacher
   scoring and initial weight publication, then failed in cuDNN attention.
   Updating the pip wheel alone did not change Transformer Engine's system-library
   selection. A standalone FlashAttention probe passed, but initially did not
   reproduce native packed padding metadata.
5. [Attempt E](https://beaker.org/ex/01M2H2862YT3K77E2TNQFPGTGF) passed a padded
   FlashAttention probe but showed that Qwen's linear-attention adapter requires
   packed metadata. No optimizer update completed in attempts A–E.
6. [Attempt F](https://beaker.org/ex/01M2H33R16RBRTBCZHCH5BYCW9) exercises the
   corrected native packed path and completed successfully, including all eight
   GPU preflight checks, two updates and the export reload.

## Handoff boundaries

The colleague should first reproduce the exact 4B tiny configuration and image.
The requested 2B learner still needs its own profile, conversion and GPU exercise.
Resume/automatic restart, top-k distillation, other data/evaluators and alternative
topologies are outside this prototype's accepted configuration. Checkpoints are
saved every update; the final export is audited and reloaded before completion.

Before scaling, check independent teacher/student probability agreement,
publication correctness, native resume plus data-cursor restoration, longer
contexts, throughput and held-out task quality. The B300 attention route is the
only hardware route exercised here. The new candidate image has not separately
requalified the OLMo-core GPU path after its cuDNN dependency change.
