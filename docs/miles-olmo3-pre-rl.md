# Published Olmo 3 through MILES and Core

Branch: `robertb/miles-olmo3-pre-rl`, rebased onto the primary integration branch
`robertb/miles-hero-support` at `446189de7` (original base `738356e39`). This is preparation for a dense Olmo 3 recipe comparison. Tiny-model
qualification has passed; no full 7B training or learning comparison has run on
this branch.

## Rebase and GSM8K readiness

The rebase completed without conflicts. The inherited runtime lock matches the
primary Core branch at `cfc42934d818036728d63f7ccdcd3b541eab9880`, MILES at
`df24d2ed5da4598264c6682f9dbe49aee64acc95`, and olmo-sglang at
`02ccb5dcf641cbabc9b78a5bc65dacf8690707a7`. No dependency-branch changes were
needed for dense Olmo 3.

The rebased regression pass has 175 CPU and 77 runtime tests passing (10 skips,
16 deselections), plus `make style quality`. The actual Ray/SGLang/Core tiny
Olmo 3 run passed two updates and a fresh-process resume into update three using
the new locked runtime sources; scoring/training-forward checks were bit exact.
A real GSM8K preparation exercise
used the published tokenizer and pinned original RL template: 64 train and
16 held-out rows, no overlap, maximum prompt length 134 tokens, and 160 correct/
incorrect-answer verifier checks passed. These held-out rows are excluded from
our selected training subset of the RLVR source's **train split**, not the
official GSM8K test set. Row identities and preparation hashes are retained in
[the rebase/preflight record](measurements/miles-olmo3-rebase-20260912.json).

The full checkpoint is now staged on WEKA: three shards, 14,596,063,712 bytes,
with pinned revision and the original RL template. The Saturn preparation job
[passed](https://beaker.org/ex/01M2A1Z8HQTKHM500H9SNE20Q1), including all 160
reward canaries. The concrete
[qualification config](../configs/miles/qualification/olmo3-think-gsm8k-robertb-20260912.toml)
was submitted to [Beaker](https://beaker.org/ex/01M2A28EYHS4BYYK5TPDJTQNPT):
Holmes, urgent, minimum runtime one hour, two Core GPUs and one SGLang GPU,
two updates, initial/final held-out evaluation, native saves and final HF export.
Its immutable image is `01M2A1Z1FVWZ4WB1ET7RDSJ5P6`, built from `311840575caf`.
This is still a qualification run, not a completed learning comparison. See
[the launch and preparation record](measurements/miles-olmo3-gsm8k-20260912.json)
for source paths and resolved settings.

## Implementation

The existing standard-model adapter already connects MILES to Core's
`TransformerTrainModule` and `AdamWConfig`, using FSDP for multiple trainer ranks.
It remains separate from the specialized MoE trainer. The MILES rollout loop,
open-instruct data/rewards, and SGLang serving remain shared. Dense Olmo 3 does
not need router replay or expert parallelism.

Three missing details were found and fixed:

1. **Published YaRN configuration.** Translate the HF descriptor into Core's
   existing YaRN configuration, applied only to full-attention layers. Sliding
   layers retain ordinary RoPE. Validate unsupported scaling settings rather
   than silently approximate them. This requires no new Core kernels or trainer.
2. **Native resume comparison.** Per-layer Core overrides have integer keys in
   memory and string keys after JSON serialization. Canonicalize these indices
   when comparing checkpoint architectures. Invalid/duplicate indices and changed
   architecture values still fail validation.
3. **HF export descriptor.** Preserve the released `rope_scaling`/`rope_theta`
   layout for Olmo 3 exports. Transformers 5 serializes `rope_parameters`, which
   triggers premature validation in the pinned SGLang Olmo 3 config reader.
   Weights and their names are unchanged.

The third fix was necessary to start an actual SGLang engine, not merely to pass
an HF conversion test. SGLang internally selects its `Olmo2ForCausalLM`
implementation for Olmo 3; that implementation handles the per-layer attention
and scaling choices.

## Checkpoint and prompt identity

The proposed starting point is the published
[Olmo-3-7B-Think-DPO](https://huggingface.co/allenai/Olmo-3-7B-Think-DPO), revision
`7b18bf927b430ff06376fdfa5610eb3b1b6a5c38`: after SFT and DPO, before RL. This is
not the earlier experimental MoE SFT checkpoint.

The downloaded config is retained in
[`tests/miles/fixtures/olmo3-think-dpo-config.json`](../tests/miles/fixtures/olmo3-think-dpo-config.json).
It describes 32 dense blocks, hidden width 4096, intermediate width 11008,
32 query/KV heads, three sliding layers followed by one full-attention layer,
a 4096 sliding window, and YaRN factor 8 from an original 8192-token context.

The checkpoint's HF chat template differs from the original open-instruct RL
script's `olmo_thinker` template. In particular, the latter defaults to
`You are a helpful AI assistant.` The copied original template is
[`configs/miles/templates/olmo-thinker.jinja`](../configs/miles/templates/olmo-thinker.jinja),
SHA256 `eba6e269f669706e5c788e370160dad953137f3fa7014fa69d03c7b3ad9f0e72`.
Use it in both comparison arms; recording only the checkpoint name is insufficient.

Download the pinned snapshot first, then stage it without modifying its source:

```bash
python scripts/miles/stage_olmo3.py /path/to/pinned-snapshot /path/to/oi-template
python -m open_instruct.miles plan configs/miles/qualification/olmo3-think-gsm8k.toml
```

Replace `YOUR_USERNAME` and the model path in a copied run config. The stager
copies tokenizer/config metadata, references weight files through symlinks, and
records source and template identities. Retain the original snapshot. It does
not download weights or authenticate their HF revision itself.

## Validation completed

An independent reference is essential here. The installed runtime has
Transformers 5.12.1, whose Olmo 3 implementation uses one scaled rotary embedding
across layers. The release-era
[Transformers 4.57.0 implementation](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/olmo3/modeling_olmo3.py)
uses unscaled RoPE on sliding layers and scaled RoPE on full layers, matching
Core and pinned SGLang. Comparing against the installed HF forward would have
validated different semantics. No runtime-wide Transformers downgrade was made.

The reference generator runs separately under Transformers 4.57.0. It saves
synthetic weights and full logits without importing Core. Qualification then
imports those weights into Core under the current runtime. Tests cross the tiny
sliding-window and original-context boundaries, including lengths 15/16/17 and
31/32/33, and extend to 128 tokens.

| Check | Observed result |
| --- | --- |
| Nine BF16 reference lengths | Maximum relative logit L2 error 0.015733; maximum absolute error 0.029297; maximum mean log-probability error 0.003971 |
| HF → Core → HF weights | Bit exact |
| YaRN native checkpoint resume, with and without activation recomputation | Next optimizer step reproduces model/optimizer/scheduler state exactly in the runtime tests |
| Actual Ray + SGLang + Core run | Two RL updates, checkpoint and HF export; fresh-process resume completes update three |
| Serving versus Core response log-probabilities in that run | Maximum absolute gaps 0.007678, 0.008037, 0.008495 across the three updates |
| Separate scoring versus training forward | Exactly equal at checked updates 0 and 2 |
| Reload of updated HF export | All 47 tensors import strictly into Core; SGLang config parsing succeeds |
| Weight publication | SGLang equality check passes; 47 tensors / 1,446,144 BF16 bytes through colocated IPC |

Numerical reference acceptance thresholds are relative L2 ≤ 0.025, absolute
logit error ≤ 0.08, and mean log-probability error ≤ 0.015. They are synthetic
qualification tolerances, not evidence of full-model quality. The Ray smoke uses
synthetic alternating rewards and a 723,072-parameter model on one RTX 4090;
it measures machinery, not learning. Attention uses the local torch backend.

The measured results are retained in
[`measurements/miles-olmo3-yarn-20260912.json`](measurements/miles-olmo3-yarn-20260912.json).
Reference runtime sources were MILES `df24d2ed5da4598264c6682f9dbe49aee64acc95`
and Core `cfc42934d818036728d63f7ccdcd3b541eab9880`. The reference and
two-update/fresh-process-resume exercises were then repeated successfully on
the exact locked Core revision, `779b183d81d290e91fc289a47cd32fd9c0bd7311`.
The reference measurements were identical. The locked runtime regression gate
passed 72 tests (10 MoE-only skips, 14 KDA cases deselected); 175 CPU workflow
and config tests passed. `make style quality` and explicit script/test Ruff
checks passed. Two inherited workflow fixtures were updated for the judge
bindings field already present in the branch base.
Local detailed logs are `/tmp/olmo3-yarn-qualification.log`,
`/tmp/olmo3-yarn-smoke-v2.log`, and `/tmp/olmo3-yarn-smoke-resume.log`;
local run artifacts are under `/tmp/miles-validation/olmo3-yarn-smoke-v2`.
Locked-repeat logs are `/tmp/olmo3-locked-qualification.log`,
`/tmp/olmo3-locked-runtime-tests.log`, `/tmp/olmo3-locked-smoke.log`, and
`/tmp/olmo3-locked-resume.log`; run artifacts are under
`/tmp/miles-validation/olmo3-yarn-smoke-locked`.
These temporary paths are not durable experiment storage.

Reproduction, using separate reference and MILES environments:

```bash
# Separate environment: Transformers 4.57.0, compatible tokenizers/hub, PyTorch.
# Avoid importing optional modern hub-kernels packages into this older environment.
python scripts/miles/olmo3_reference.py /artifacts/reference

# Current MILES/Core runtime, with this checkout and its sibling sources on PYTHONPATH.
python scripts/miles/olmo3_qualification.py /artifacts/reference
python -m pytest -q tests/miles/test_olmo3_yarn.py tests/miles/test_runtime.py -k yarn
PYTHONPATH="$PYTHONPATH:$PWD/tests/miles" python tests/miles/smoke.py /artifacts/smoke --model-type olmo3
PYTHONPATH="$PYTHONPATH:$PWD/tests/miles" python tests/miles/smoke.py /artifacts/smoke --model-type olmo3 --resume
```

## Preparing the recipe comparison

Two configs are included:

- [`qualification/olmo3-think-gsm8k.toml`](../configs/miles/qualification/olmo3-think-gsm8k.toml):
  candidate full-model debug run, two Core ranks plus one SGLang GPU, two updates,
  native saves every update, initial/final held-out evaluation and final HF export.
  Its memory fit, FSDP execution and long-context performance are not yet qualified.
- [`proposals/olmo3-think-dolci-200.toml`](../configs/miles/proposals/olmo3-think-dolci-200.toml):
  future 200-update comparison, 8 Core ranks, 7 serving GPUs and one managed judge
  GPU. It deliberately references future prepared datasets/judge assets, and is
  not ready to submit. Multi-replica auto-resume is disabled.

The original released recipe script is
[`7b_think_rl_no_pipeline.sh`](../scripts/train/olmo3/7b_think_rl_no_pipeline.sh).
It uses **HF/DeepSpeed training and vLLM serving**. Our branch uses **Core/FSDP
training and SGLang serving**. Thus this preserves the dense Olmo 3 architecture
and aims to match its RL recipe, but is not an identical trainer backend.

| Control | Original script / current source audit | Proposed MILES comparison |
| --- | --- | --- |
| Initial weights and template | Think-DPO, `olmo_thinker` | Pinned snapshot and copied template |
| Training data | Full Dolci-Think-RL-7B mix | Same source/proportions; immutable prepared IDs and excluded held-out IDs |
| Collection/update | 64 prompts × 8 responses, one minibatch/epoch | 64 × 8, global batch 512 |
| Optimizer | LR 1e-6 constant; current AdamW defaults β₂=0.999, ε=1e-8 | Explicit same values; weight decay 0, clip grad 1 |
| Objective | Current source: centered advantages, token reduction, clips 0.2/0.272, KL 0 | Explicit same controls; no GRPO std normalization |
| Importance correction / filtering | Current source: trainer/serving ratio clamped to [0,2], zero-variance group filtering | TIS [0,2] and MILES nonzero-reward-std filter; exact semantics still need paired-batch qualification |
| Lengths | Prompt 2048, response 32768, training pack length 35840; separate max-token input setting 10240 | Prompt 2048, response 32768, model context 34816; no claim of identical input admission or packing |
| Sampling / truncation | Temperature 1; no non-stop penalty or truncation masking | Temperature 1; audit retained loss masks and verifier behavior in the full-model smoke |
| Scheduling | 16 learner GPUs, 56 vLLM engines; `async_steps=1`, inflight updates false | 8 Core GPUs, 7 SGLang engines; initially synchronous, lower hardware scale |
| Judge/code rewards | Hosted Qwen3-32B and code service | Managed pinned Qwen3-32B and prepared verifier registry; canaries required |
| Evaluation | In-loop every 50; script selects 8 entries from training split; external task suite | Initial and every 20 on fixed excluded IDs, greedy one sample; separate external evaluation later |
| Checkpoints | HF every 25; resumable state every 100 | Native every 50, final HF export; dense Core saves synchronously |

Some original settings are implicit Python defaults. The current source audit
is not proof of what the historical release job resolved. Pin that job's code
revision and arguments before calling this a reproduction. In particular,
importance correction, group filtering, input admission, reward implementations,
and masked-token reduction warrant paired fixed-batch comparisons.

Use the existing [Dolci preparation proposal](miles-dolci-production-proposal.md)
and [managed judge workflow](miles-managed-judges.md), rebuilding with the dense
checkpoint tokenizer and pinned template. Record task proportions, dropped input
counts, source revisions, reward configs, and train/held-out disjointness. The
existing mixture infrastructure does not establish that all those services have
been exercised together for this recipe.

## Next execution gates

1. Check full 7B serving/trainer logits on identical inputs, including prefill
   and decode beyond the real 4096 sliding window. Qualify the proposed Flash4
   trainer and serving attention backends. SGLang's observed Olmo 3 path disables
   its hybrid sliding-window KV memory optimization; account for full cache
   allocation when sizing long-context serving. The dense 32-layer MHA geometry
   costs approximately 0.5 MiB of BF16 KV per cached token, so the candidate
   caps the cache at 131,072 tokens (about 64 GiB), rather than reusing the MoE
   profile's much larger token cap. Peak concurrency still needs measurement.
2. Exercise two-rank FSDP import, update, native save, resume, and disaggregated
   NCCL weight publication. Repeat next-step equivalence after restore and reload
   an exported HF checkpoint. The single-GPU IPC smoke does not cover these.
3. Run the two-update 7B qualification config, then verifier/judge canaries on
   math, code, instruction following and general-quality samples. Verify prompt
   serialization and truncation masks as well as rewards.
4. Prepare the immutable full mixture and common held-out set. Run a paired early
   comparison at 0/20/40/60/80/100 updates before extending to 200 or a full recipe.

Compare held-out reward by domain, actual generations, response lengths/cap-hit
rates, accepted and filtered sample counts, log-probability gaps, PPO ratios,
gradient norms, and optimizer clocks. Plot against optimizer updates, trained
tokens, and elapsed GPU-hours. Split timing into rollout, scoring, training,
publication, evaluation, and checkpointing. Preserve initial and periodic
samples and stable question IDs in both arms; use additional seeds once the
single-run comparison establishes that behavior is sensible.

Cluster submission remains through the committed
`scripts/train/build_image_and_launch.sh --miles` workflow described in
[miles-workflow.md](miles-workflow.md). GPU candidates use Holmes,
`open-instruct-dev`, urgent priority and a minimum runtime. CPU-only preparation
that accesses WEKA must run on Saturn. No cluster jobs were launched for this
branch's tiny-model qualification.
