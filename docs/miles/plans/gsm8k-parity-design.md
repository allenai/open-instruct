# Matched GSM8K learning comparison

> Historical proposal. For current operating instructions, start at the [MILES guide](../index.md).

Shared preparation passed on Beaker, and both 100-update runs are submitted:
[Core](https://beaker.org/ex/01M26P6XX6SN886DCVZ68WMQK2) and
[Megatron](https://beaker.org/ex/01M26P8G0X2EHG52KYGAXT3BFP).
Their live priorities were updated to urgent before either process started. The purpose is
to compare learning and operation through our customized olmo-miles/Megatron
trainer and open-instruct/MILES/Core under the closest feasible settings. It is
not an assertion of identical numerical execution.

## Shared model, runtime and data

Use the recent SFT source pair under
`/weka/oe-training-default/robertb/olmo-miles/checkpoints/`:

- `olmoe3-kda-1.2b-dolci-think-sft-65536-router-bf16-autocast-v2-hf`
- `olmoe3-kda-1.2b-dolci-think-sft-65536-router-bf16-autocast-v2-megatron`

The HF descriptor must use the pinned RL chat template from the Megatron
bundle's `.olmo-miles/hf/chat_template.jinja`, SHA-256
`f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b`.
This reads tokenizer/template assets from that directory; Core does not train
from Megatron state. Both sides must pass initial serving equality against the
same HF weights. Record config, tensor-header, tokenizer and template hashes.

Both runs should share the already built compiled image
`01M24E7MSDGN2QFW1T8Z31BCKS` as their base. Docker inspection established:

| Component | Image revision |
| --- | --- |
| olmo-miles image source | `acab8b3532dc0d551e7024b4624f82ea009008b5` |
| MILES upstream | `dbbab1566ae438f7202fff653eae938e07b1d4b6` |
| olmo-megatron | `b84044ffb0d52620ec1599eda19d8f3d1de816b4` |
| Megatron Bridge | `db723bae699dae5d29003ec4789c67730a343c32` |
| olmo-sglang | `81a312ee8326e279a4641e03542ef971a5ffb863` |
| SGLang | `3145136dcd1238754e0ea2b2ffd546532119c71c` |
| PyTorch / CUDA | `2.13.0+cu130` / CUDA 13 |

The baseline checkout's ordinary blessed image is older. Use the documented
**candidate gate** for this paired experiment, with the matching candidate pins
already present at `acab8b35`; do not silently substitute the blessed image or
change its promotion status. Core adds its source overlays and native trainer
on this same compiled base. Record final source/patch hashes for each arm.

Prepare one immutable campaign at:

```text
/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1
```

Use `ai2-adapt-dev/rlvr_gsm8k_zs` revision
`93ffaae6cd2acb8f821f6d4712651320a889b1b9`: **400 training prompts from `train`
and 128 held-out prompts from official `test`**. The baseline preparation API
selects deterministic subsets using seed 17 and source names `gsm8k-train` and
`gsm8k-eval`. `shuffle=false` disables final reordering; it does not select the
first 400/128 rows. Freeze actual source IDs and artifact hashes once, then
translate those same rows for Core. Remove reference assistant answers.

The canonical baseline artifact has `messages`, `ground_truth` and verifier
metadata; its `baseline/rl-manifest.json` records paths/hashes and
`olmo_miles.rl.rewards.registered_reward`. Core renders the same messages with
the same template and invokes open-instruct's GSM8K verifier. Prove identical
input token IDs, targets and ordered four-prompt batches across all 100 updates;
differentially test both reward implementations before launch. Never prepare a
second independent subset or render an already rendered prompt again. Audit
train/test prompt overlap; official test membership does not prove the SFT
checkpoint was uncontaminated.

## Matched settings

| Setting | Both arms |
| --- | --- |
| Hardware / placement | One allocation of 3 B300s: 2 trainer ranks at EP2, 1 dedicated TP1 SGLang engine |
| Training | Synchronous, 100 optimizer updates; 4 prompts × 4 samples = global batch 16; one unpadded Core sequence / one Megatron sequence per microbatch |
| Evaluation | Same 128 questions at policy steps 0, 20, 40, 60, 80, 100; one greedy response each |
| Sampling | Train temperature 1, top-p 1, top-k -1; evaluation temperature 0; seed and rollout seed 17 |
| Length | Prompt allowance 2048, response cap 4096, context 6144; same cap for evaluation |
| Optimizer | LR 1e-6 constant, warmup 0, weight decay 0; Adam betas 0.9/0.95, epsilon 1e-8; clip norm 1 |
| Policy objective | GRPO, response averaging, **std normalization disabled**, actor-recomputed old-policy anchor; PPO clip lower 0.2 / upper 0.28; no KL, entropy, TIS or replay |
| Execution | Activation recomputation on; FA4 full attention; no trainer offload; Megatron compatibility backend and padding multiple 1 |
| Serving | Decode graphs through batch 4 (1/2/4), prefill graphs off; radix cache off; client and engine concurrency 4; recurrent slots 8; static fraction 0.6 |
| Publication | Direct HF-named export, 1 GiB buckets, every update; initial serving equality check |
| Outputs | Full response artifacts and diagnostics on WEKA; no optimizer checkpoints or final HF export in this bounded learning comparison |
| Cache / scheduling | Persistent compiler cache disabled for the first pair; explicit urgent priority, positive minimum runtime, no automatic resume |

The original Core SFT trial did **not** disable GRPO std normalization; blindly
reusing its configuration changes the algorithm relative to olmo-miles. Baseline
`config.py` unconditionally emits `--disable-grpo-std-normalization`, which Core
must explicitly match. Set every optimizer field rather than trusting different
parser defaults.

**The primary pair retains auxiliary balancing 0.01 and z-loss 1e-5.** Production
olmo-miles defaults are balancing 0.01 and z-loss 1e-5. New explicit baseline
configuration fields preserve those defaults and allow a 0/0 policy-only arm.
Retaining 0.01/1e-5 compares the intended production recipes but leaves a known
aggregation difference: Megatron BSHD pads to the local collection maximum, and
its PeriMLP path does not pass a padding mask to the router; Core processes true
sequence lengths. Megatron averages per-sequence auxiliary objectives, while
Core's denominator weights by true model tokens. This padding exists even with
compatibility/all-to-all dispatch and padding multiple 1; it is distinct from
DeepEP communication-capacity padding. Quantify pad fraction and weighted
auxiliary contribution before deciding whether a policy-only follow-up is
needed. Rank dumps at `megatron/train_data/{rollout_id}_{rank}.pt` retain
`total_lengths` and `max_seq_lens`; compute padding fraction as
`(sum(max_seq_lens) - sum(total_lengths)) / sum(max_seq_lens)`, aggregating
numerators and denominators across ranks. They contain training tensors, not
model or optimizer checkpoints. Do not silently change the primary objective
based on a padding guess.

## Baseline launch and validation

Use the isolated checkout
`/home/robert/proj/open-instruct/.worktrees/olmo-miles-gsm8k-parity`, branch
`robertb/core-gsm8k-parity`, based on `acab8b35`. The original olmo-miles checkout
has an unrelated untracked document and remains untouched. The proposed config
is `examples/qualification/gsm8k-core-parity-20260910.toml` in that worktree.
Its auxiliary controls explicitly retain the production coefficients.

```bash
# From a clean, committed, pushed baseline worktree; use its CLI/PYTHONPATH.
olmo-miles dev beaker preflight examples/qualification/gsm8k-core-parity-20260910.toml \
  --runtime candidate --image 01M24E7MSDGN2QFW1T8Z31BCKS
olmo-miles dev beaker launch examples/qualification/gsm8k-core-parity-20260910.toml \
  --runtime candidate --image 01M24E7MSDGN2QFW1T8Z31BCKS
```

This is the documented public maintainer launcher, not hand-submitted Beaker
YAML. The ordinary `olmo-miles run` command does not expose candidate/image
flags. The exact-image local preflight has passed with fixture artifacts and
confirmed the compiled std-normalization, clipping and explicit auxiliary
arguments. Real WEKA inputs, CUDA kernels and the full model remain job checks.
The full baseline unit suite and final source commit/push are required before
launch. Do not bypass the source-pin checks.

Use the established olmo-miles scheduling convention: **urgent** priority,
`ai2/open-instruct-dev` workspace, Holmes for these GPU jobs, and a positive
minimum runtime covering expected execution. CPU-only WEKA preparation and
audits stay on Saturn. Zero-minimum backfill is an explicit exception for a
very short job when current placement conditions justify it; it is not the
default for training or work on the launch critical path.

Both arms request four hours minimum runtime; **minimum runtime is
not a timeout**. Use the public scheduler controls and monitor a declared
wall-clock budget rather than inferring termination from that field. Avoid
launching either dependent arm if its shared-artifact or initial-weight check
fails. The baseline launcher does not expose a hard timeout; monitor and stop
a stuck run at the four-hour budget. Record any truncation, retry, hardware or
configuration difference.

## W&B report and interpretation

Use entity `ai2-llm`, project `olmo-rl-comparison`, groups/run names
`gsm8k-core-megatron-20260910-v1-core` and
`gsm8k-core-megatron-20260910-v1-megatron`. MILES sets the name from the
group when random suffixes are disabled; the comparison prefix is shared.
The Beaker secret **name** is `robertb_WANDB_API_KEY`; baseline source bootstrap
uses `robertb_GITHUB_TOKEN`. No secret values belong in code, manifests or logs.
The baseline already sets `--wandb-always-use-train-step` and writes native
rollout/evaluation metrics. Core should log the same logical policy-step axis.
Preserve raw metric names, then add common comparison fields in the report.

The primary panels are held-out accuracy versus optimizer step and allocated
GPU-hours, alongside mean response tokens, cap-hit fraction, training correct
rate, mixed-reward groups, gradient norm, policy loss and elapsed phase times.
Log exact denominators: 128 evaluation questions, 16 train responses/update.
Keep diagnostic same-version republication time visible; it should not be
mistaken for ordinary weight-sync cost. An export round trip validates transfer
consistency, not an independent mapping oracle for changed parameters.

At each evaluation, retain per-question answers/scores. Compare each arm's change
from its own step-zero score and the paired difference in those changes on the
same questions. Include a paired question-bootstrap interval and the raw count
of discordant answers. Report best-step results as exploratory unless selection
was fixed in advance; step 100 is the primary endpoint. One seed, 100 updates and
128 questions give a useful learning-curve screen, not equivalence or a reliable
ranking of trainers. Shared seeds do not guarantee identical generated samples
once model arithmetic or scheduling diverges.

Remaining mismatches to disclose include padding/auxiliary aggregation, native
expert execution and optimizer reduction details, the adapter/source overlays,
serving token-pool limits not expressible through identical public keys, and
possible kernel/sampling-backend defaults. Verify the actual resolved serving
arguments, not merely matching config labels. The fixed-batch numerical
comparison remains a separate, complementary contract gate.
