# Lessons retained from the older integration branches

The remaining historical local branches were retired on September 21, 2026,
leaving `robertb/miles-olmo-core` and `robertb/miles-qwen35-opd` as the two local
working branches. Their histories were archived and independently restored before
removal. This note preserves useful engineering lessons; it does not claim new
benchmark results, qualify a new runtime, or merge the legacy implementations.

The quantitative router conclusions are already in
[Router investigation: findings worth keeping](router-findings-20260921.md).
The private campaign and learning-probe snapshot add no stronger final learning
claim than that consolidated record. Their reusable production controls were
selected separately for primary; their historical helper implementations remain
in the local archive.

## Checkpoint initialization, resume, and optimizer reset are different operations

The old SFT work distinguished starting from pretrained model weights from
restoring a training run. Model initialization excluded both trainer and optimizer
state. A full resume restored the training state; an explicit optimizer reset
needed to avoid loading the optimizer tensors in the first place.

The correction from `387e0b3e6` to `a1fda307a` used the loader's
`load_optim_state` control instead of relying on an optimizer-reset keyword.
The enduring lesson is to test the actual state-loading boundary: weights,
optimizer moments, scheduler, counters and data position each have a meaning.
A run that starts successfully is not proof that its resume semantics are right.
These historical API spellings are not configuration advice for current MILES.

Related current documentation: [workflow](../workflow.md) and
[models and checkpoints](../models-and-checkpoints.md).

## Model conversion must preserve the architecture and initialization lifecycle

The SFT lineage preserved the native fused-attention layout rather than rebuilding
it solely from a serving configuration (`a78936f3b`). Its export fix selected the
registered Olmo MoE implementation instead of unconditionally preferring remote
model code (`6ee43791d`). The legacy GRPO and SFT integrations moved pretrained
weight loading into model initialization before distributed wrapping
(`13327fd75`, `15ead42b4`).

These are examples of a broader contract: check architecture, parameter layout,
model-class selection and weight ownership across load, wrapping and export.
Do not infer correctness from matching tensor counts or a successful save.
Reopening these old implementations would require their dependency versions and
fresh execution checks; they are not drop-in patches for today's Core adapter.

## Packed-record boundaries and replay layer indexing must be explicit

The Qwen SFT record-boundary change (`40cb9af2a`) dealt with two distinct problems:
truncation can remove a record's terminal token, and an EOS used inside a chat
conversation need not mean the end of the serialized training record. Its
opt-in handling preserved the label mask and validated that a configured boundary
marker encoded as exactly one token. Replacing a truncated token with EOS changes
the training target, so this is a deliberate data decision, not a universal fix.

The legacy replay change (`4d1a1bc19`) distinguished an absolute transformer-layer
axis from an axis containing only MoE layers. It selected the routed layers
explicitly and rejected incompatible dimensions. A mixed dense/MoE architecture
can have the right top-k shape but the wrong layer mapping.

The durable requirements are independent of serving engine: verify document
isolation and labels at boundaries, and specify token, layer and expert indexing
at the replay interface. Current MILES uses SGLang; the old vLLM transport is
historical. See [packing](../sequence-packing.md) and
[implementation contracts](../core.md).

## Learning and publication need separate evidence

The early probe work added paired before/after checkpoint comparisons and
between-arm differences in improvement (`978fcbec8`, `e95dbf2f7`), and accounted
for all completed allocations and retries (`b2dab743f`). These are useful methods,
not evidence that every early numerical conclusion survived later controls.

For future comparisons, retain starting checkpoints, prompt identities, tokenizer
and template, verifier, decoding parameters, length cap and evaluation seed.
Report truncation and response length with accuracy. Keep within-prompt sampling
uncertainty separate from variation across independent training runs. Count
failed attempts when reporting compute cost. A learning check on repeated training
prompts is distinct from a held-out quality comparison; the existing
[overfit diagnostic](overfit-20260915.md) makes that scope explicit.

The legacy weight-publication diagnostics added shipped-weight fingerprints
(`d0c9e17d0`). Such fingerprints can help detect a stale or wrong publication,
but they do not replace full tensor parity, model reload, or a matched forward
check. Successful transfer and successful learning are separate contracts.

## What was already preserved elsewhere

The pre-rebase Olmo 3 branch held YaRN and checkpoint/template preparation work
(`68e56fa04`). Its subject is already covered by the
[dense Olmo 3 record](../olmo3-pre-rl.md), including the need to record the chat
template as well as the checkpoint. No additional old recipe is promoted here.

The old SFT/QMoE/router-replay lineages contain useful historical implementations,
not outstanding work required for current MILES RL. Their removal does not mean
all their commits were merged or that SFT support was requalified. For new RL work,
use the [MILES guide](../grpo.md); do not revive deprecated GRPO paths from these
archives by default. OPD remains a separate integration effort.

## Local archive and restoration

`runs/final-branch-archive-20260921/historical-branches.bundle` contains complete
Git history for the eight retired local branch tips. Its adjacent `manifest.json`
records full SHAs and the bundle checksum; `README.md` contains restoration
commands. Verification included `git bundle verify`, fetching into an independent
bare repository, comparing every restored tip, and `git fsck --full`.
The detached learning-probe commit `2c1a752ee` is an ancestor of the archived
private-campaign branch and was separately checked in the restored repository.

| Retired local branch | Archived tip |
|---|---|
| `local/miles-private-campaign-20260918` | `6fd125b28b5e00f299b0f2f697e2f65215f3a7a3` |
| `robertb/olmo3moe-sft-core` | `a1fda307add9816ecd8393d437eb365f03c6cf3e` |
| `robertb/qmoe-int` | `15ead42b4df174537740e9cf8e2a05e6a25cc6b6` |
| `robertb/router-replay-core` | `d0c9e17d000e0a526c729a855c9dac7e2e456427` |
| `archive/robertb-qmoe-int-pre-sft-consolidation-20260801` | `9e602bdcaf1e66da8d795fa852317a742b969247` |
| `archive/robertb-rr-pre-core-20260722` | `a374a3eff040cf369006dac492261324d717837f` |
| `backup/olmo3-pre-rl-before-rebase-20260912` | `68e56fa04ecf91e5e7d0ea917ffb6c925f457e4f` |
| `main` (stale local reference only) | `78d1e5aa3cf80a73ce56fd0775e7ad959faf2660` |

The archive is local and Git-ignored, not an off-machine backup. Private helpers
were neither copied into maintained source nor uploaded. The two removed
worktrees were clean and had no ignored files. Existing experiment result
directories remain. Remote branches, including `origin/main` and the remote
SFT/legacy branches, were left untouched; two local working branches does not
mean this shared repository has only two remote branches.
