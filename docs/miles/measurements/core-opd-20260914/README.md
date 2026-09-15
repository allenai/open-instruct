# Core OPD prototype exercise — September 14, 2026

Status: **passed**. Both Core OPD learners completed two updates, checkpoint
saving, HF export, fresh SGLang reload/generation and clean shutdown. Native Qwen
OPD and Core GRPO compatibility checks also exited successfully. The
[machine-readable summary](summary.json) identifies each run and image.

## Integration

Open Instruct now routes `training.algorithm = "opd"` and
`trainer.backend = "olmo_core"` through the existing registered Core actor.
The existing loop owns optimizer updates, expert parallelism, packed scoring,
rollout weight publication, task evaluation, native checkpoints and HF export.
This adds teacher supervision to that loop; it does not introduce another trainer.

The learner architecture comes from the prepared HF checkpoint's `model_type`.
The two exercised profiles are our custom `olmo3moe` SFT checkpoint and dense
`olmo3`. The independent frozen teacher is Qwen3.5-9B at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Its own tokenizer and chat template
render the original conversation and score the exact learner response text.

Only one-to-one equal text spans receive teacher supervision. Tokenizer boundary
mismatches, special tokens and structural `</think>` spans are masked from OPD.
The existing loss mask and denominator remain intact. Each matched response
position receives `kl_coef * (teacher_logp - preupdate_student_logp)` as its
advantage; task rewards are zero. This partial sampled objective is not full
cross-vocabulary KL. The MoE learner retains its router auxiliary losses.

Open Instruct owns the independent teacher service, readiness and score probes,
GPU reservation outside Ray, alignment validation, diagnostics and final export
reload. The first implementation requires one-node synchronous barrier
publication, top-k zero and trainer-scored pre-update probabilities. Automatic
restart/resume and asynchronous OPD remain outside the accepted pilot.

## Image and source identity

Branch: `robertb/miles-qwen35-opd`, based on `robertb/miles-olmo-core` with its
latest tokenizer/template fixes merged. The tested MoE image is
`01M2HNJ96AQMCE69SK5ZA0EJQT`, containing runtime source
`54c09c020e2b0946eab10889df324353a308eca4`.

Dense OPD and the final compatibility checks use `01M2HKRHZV515XFY44E84JHVEB`,
source `06b43758687e1b7a519d18c257191a16471304fe`. The later image changes only
the Core HF reload command to allow its custom configuration code, plus tests
and documentation. The training code and dependency layers are identical between
those images; the unaffected paths were not repeated after that reload-only fix.
Subsequent source cleanup applies the pinned formatter and rejects a missing
Miles source-module path before native Qwen startup; it does not change the
training behavior exercised by either image.

[Launch receipts](launches.json) retain all resolved configuration fields,
allocation, output paths, image IDs, submission revisions and config hashes.
A receipt's submission revision can be newer than the reused image; the runtime
source revisions above identify the code inside each image.

## Run configurations

- [MoE example](https://github.com/allenai/open-instruct/blob/robertb/miles-qwen35-opd/configs/miles/opd/olmo-moe-tiny.toml): two Core ranks
  with EP2, one learner SGLang GPU and one teacher GPU.
- [Dense example](https://github.com/allenai/open-instruct/blob/robertb/miles-qwen35-opd/configs/miles/opd/olmo3-tiny.toml): two Core ranks
  with EP1, one learner SGLang GPU and one teacher GPU.
- Both use four B300 GPUs on Holmes, 16 prepared GSM8K training prompts and eight
  held-out prompts, two updates of four prompts with two responses each, a 1024
  response-token cap, and learning rate 1e-6. Native checkpoints save at each
  update; the final HF export must load and generate in a fresh SGLang process.
- The learner retains its SFT chat template. The Qwen teacher uses
  `enable_thinking = false`. These settings deliberately preserve each model's
  native conversation format.

## Results

Runs:

1. Dense Core OPD, passed: [Beaker](https://beaker.org/ex/01M2HM19A6K4QT01XSV3XB2V2E).
2. MoE Core OPD, passed: [Beaker](https://beaker.org/ex/01M2HNJJ63684V0FSJ1W3P5HNB).
3. Qwen/Megatron OPD regression, passed: [Beaker](https://beaker.org/ex/01M2HMYEVHYAAVVJ7FR0VDAW2R).
4. Core GRPO compatibility check, passed: [Beaker](https://beaker.org/ex/01M2HN5C71YSD6YBP2AM0GQAXF).

The [Qwen regression audit](qwen-regression-audit.json) records two eight-sample
updates with zero advantage error, gradient norms 10.6805 and 13.2538, 356 tensors
changed from the base and 328 changed since the preceding update. All 24 `A_log`
tensors remain FP32; 312 unchanged base tensors are retained in the export. Fresh
SGLang reload generated 32 tokens with finite output log probabilities.

The [Core GRPO audit](grpo-regression-audit.json) verifies both updates and
nonzero finite gradients/parameter changes on both EP2 ranks. This short run's
128-token responses received mean task reward 0.125 in each training batch,
with one mixed-reward prompt group in each. Zero reported policy loss at the
unchanged policy is expected with centered GRPO advantages and does not imply
zero policy gradient. Router auxiliary losses remain enabled. The earlier full-length
[GRPO regression](https://beaker.org/ex/01M2HK6J19QZK2SYJF3KWQSFD1) also passed
with nonzero task reward and checkpoint saving on the first integration image.

The [dense artifact audit](dense-audit.json) verifies two updates on both ranks,
finite nonzero gradients and sampled parameter changes, exact teacher-derived
advantages, both checkpoint markers, teacher identity/probe, completed supervisor
cleanup, and fresh export generation with finite token log probabilities.

The [MoE artifact audit](moe-audit.json) passes the same checks on both EP2 ranks,
including nonzero dense, expert and router gradients and sampled parameter
updates. It aligned 13,246 of 15,764 response tokens; batch coverage was
80.62% and 87.72%, with maximum advantage error zero. Fresh export reload
generated 32 tokens with finite log probabilities. Held-out reward moved from
2/8 to 5/8, with 7/8 and 6/8 responses truncated respectively. Eight examples and
two updates do not establish a learning improvement.

The dense run aligned 15,092 of 16,289 response tokens. Batch coverage was 91.87%
and 93.44%; the advantage audit had maximum error zero. Its held-out score was
3/8 before and after, with all responses hitting the 1024-token cap. These are
mechanics results, not evidence of improved task quality or production throughput.

## Attempts and fixes

1. [MoE attempt A](https://beaker.org/ex/01M2HK1H4R9THMQAX50FQYBR2P) exposed
   SGLang metadata becoming available before warmup finished. Readiness now
   requires healthy serving, matching checkpoint identity and a real input-logprob
   request before training starts. It completed no optimizer updates.
2. [Attempt B](https://beaker.org/ex/01M2HKMCZTB9ASVKXEQ9AS1TCM) was canceled
   while starting after inspection found native rollout metrics require scalar
   rewards before reward postprocessing. The teacher hook now attaches scores to
   the sample and returns scalar zero immediately.
3. [MoE attempt C](https://beaker.org/ex/01M2HKRVGZPZ9M3NDRY3CXP60Y) completed
   two audited updates and both native checkpoints, then failed the fresh HF
   reload because its custom model configuration needed `--trust-remote-code`.
   The reload command now includes the same permission used for the registered
   learner. This attempt is not counted as an end-to-end success.
4. [MoE attempt D](https://beaker.org/ex/01M2HNJJ63684V0FSJ1W3P5HNB) passed
   with the reload fix. Its runtime was 04:36:59–05:07:41 UTC on September 15
   (September 14 locally). Dense OPD took about 14m34s; the Qwen and short GRPO
   regressions took about 19m12s and 17m09s. Loading, compilation, checkpoint I/O
   and fresh-process warmup dominate these tiny exercises.
5. A [queued duplicate full GRPO check](https://beaker.org/ex/01M2HMYMHP7JFP9Y740HFE1SWX)
   was canceled and replaced with a shorter
   two-update check. The earlier full GRPO regression already covered checkpoint
   saving; the shorter check disables saving/export and caps responses at 128.

## Local and runtime checks

- 161 focused CPU tests passed on the final source: Core OPD configuration,
  alignment, masked gradients, scalar reward ordering, service readiness/reload,
  packing, scoring, native OPD audit, launch contracts and documentation.
- The final MoE image passed 30 GPU runtime preflight tests and the separate
  FlashAttention 4 forward/backward probe. These are runtime preflights, not
  `scripts/test/run_gpu_pytest.sh`.
- The pinned Ruff formatter and linter pass, as do Python compilation, generated
  reference checks, the documentation build and targeted OPD type checks.
- Repository-wide `make quality-check` still reports six type diagnostics in
  existing `models.py`, `moe_models.py` and `standard_models.py`, against the
  local unpatched OLMo-core dependency. No OPD module diagnostics remain. These
  local dependency errors do not replace the GPU evidence from the pinned,
  patched runtime.

## Recommendation and boundaries

Use the matching tiny config and immutable image for the colleague's first run,
with a fresh output root. After reproducing the mechanics, the next experiment
should be a bounded learning comparison against the same SFT baseline. Measure
alignment coverage, answer completion/truncation, held-out task reward and policy
drift; tune response lengths and the teacher's conversation format before drawing
quality conclusions. Optional reference KL remains `optimizer.kl_loss_coef` and
needs its own nonzero-coefficient exercise.

Keep the existing Megatron patches and mbridge in the shared image: native Qwen
OPD still uses them, while Core OPD does not require them for learner training.
The Qwen3.5-2B profile, additional learner registrations, arbitrary teacher
architectures, full vocabulary alignment, resume and larger topologies need
separate qualification. No production-scale run is implied by these tiny tests.
