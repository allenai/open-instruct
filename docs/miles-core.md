# MILES with an OLMo-core trainer

This branch contains an experimental, executable MILES/Core integration. **It has not met the full GRPO replacement acceptance criteria.** The existing GRPO entrypoints remain available while the missing workflows are implemented and qualified. SFT and DPO retain their existing dependency and entrypoint paths.

MILES owns rollout scheduling, SGLang serving, behavior log probabilities, advantage construction, policy loss, and evaluation dispatch. `open_instruct.miles` connects that runtime to Core's model, backward synchronization, optimizer, and distributed checkpoint APIs. No Megatron or MILES FSDP trainer is selected by this entrypoint. Shared MILES argument and transport helpers are reused.

```mermaid
flowchart LR
    C[Native MILES config] --> D[Core RL driver]
    D --> R[MILES rollout manager]
    R --> S[SGLang]
    S --> V[Open-instruct verifiers]
    V --> L[MILES advantages and loss]
    L --> O[Core train module and optimizer]
    O -->|Versioned HF weights| S
    O --> J[Native checkpoint plus durable prompt cursor]
```

## Reproduce the sources

The lock in `runtime/miles/runtime.lock.json` records exact bases and patch hashes. Core starts at Jacob's `jacobm/moe-v2-core`, `a977e71336da54a16da54c74d04f344d3f100291`. MILES starts at `dbbab1566ae438f7202fff653eae938e07b1d4b6`. The Core patch adds an arbitrary-objective gradient lifecycle, explicit replay across backward recomputation, HF conversion support, and conventional Olmo3Moe model construction. The MILES patch includes the existing olmo-miles compatibility patches plus the explicit Core backend and lifecycle hooks.

```bash
python scripts/miles/prepare_runtime.py runtime/miles/sources
```

For offline development, append `--cache olmo-core=/path/to/OLMo-core --cache miles=/path/to/miles`. The script verifies patches, checks out the exact base, and applies them to new directories. It refuses to replace an existing directory. The reconstructed Core source is also used by the repository type checker; regular SFT/DPO package pins are unchanged.

The runtime is a separate source overlay on the image recorded in the lock. Load that Beaker image into Docker, then:

```bash
python scripts/miles/build_image.py --base-image YOUR_LOCAL_BASE_TAG --tag open-instruct:miles-core
```

The builder verifies the base image's immutable Docker ID. It fetches Core/MILES at the pinned revisions and applies checksum-verified patches inside the image. The image's own sources are used at runtime; sibling development worktrees are unnecessary. The base still contains historical Megatron packages; Core operation has also been checked with Megatron imports deliberately unavailable. Removing unused packages is a separate image-size task.

## Configure and run

`configs/miles/dense.toml` is an example with explicit mounted input/output paths. `configs/miles/prompts.jsonl` and `verifiers.json` show the prepared-data and mixed-verifier contracts. The reward registry is trusted run configuration. Samples select registered verifier names, targets, and weights, and cannot name arbitrary import paths.

```bash
python -m open_instruct.miles plan configs/miles/dense.toml
# In the Core RL image, with actual mounted data/model paths:
python -m open_instruct.miles validate /data/run.toml
python -m open_instruct.miles train /data/run.toml
```

`plan` is CPU-safe and only compiles configuration. `validate` invokes the pinned MILES parser; it does not prove model compatibility or serving readiness. `train` selects Core exclusively. MILES optimizer and loss flags keep MILES semantics. There is no legacy GRPO argument translator.

The current implementation uses one unpadded sequence per microbatch with gradient accumulation. `global_batch_size` must be a multiple of the trainer world size, and each collection must contain complete optimizer batches. `core.max_policy_lag` counts **optimizer steps**, including later steps in a multi-step collection. A collection containing N optimizer batches needs at least N−1 lag budget. The default zero is suitable for a synchronous one-step collection.

Async mode additionally requires `fully_async=true`, a positive lag budget, resident disaggregated rollout engines, and publication after each collection. Groups must carry complete, homogeneous behavior-policy versions. The buffer reserves enough lag for all optimizer steps in the collection, and the actor checks again at consumption.

## Checkpoints and publication

Core checkpoints contain native model/optimizer state, scheduler and per-rank RNG state. MILES persists its data cursor and, for async runs, pristine outstanding prompt groups. `complete.json` and `core-latest.json` are written only after both sides are complete; the manifest hashes the cursor. Resume uses `miles.load` pointing to the checkpoint root and resolves the next rollout before constructing the manager. It currently requires the same trainer world size and model configuration. Interrupted checkpoint directories are preserved under `.incomplete-*` names before retrying that rollout. Generated async responses are regenerated after restart; serving RNG and bitwise-identical future rollouts are not promised.

Publication pauses generation, transfers weights, commits the optimizer-step version, and resumes generation. Dense conversion gathers one parameter at a time. MoE conversion stages the full unsharded model on CPU and still needs GPU space for one gathered expert parameter. This is not yet a measured large-model publication implementation. HF evaluation exports include the tokenizer and MILES completion marker.

## Validation and remaining acceptance work

Verified locally on an RTX 4090:

- 62 focused adapter and existing Core SFT/DPO tests, including four dense HF/Core logits comparisons and exact weight roundtrips, streamed conversion, masked tool-token handling, mixed rewards, and policy clocks.
- Three Core-owned tests for custom-objective accumulation and router replay through backward recomputation.
- Six pinned-runtime tests for a real Core GPU optimizer update with MILES loss, native checkpoint restore, async prompt-cursor crash safety, and policy-group admission.
- A Ray/SGLang/Core smoke run completed two optimizer steps with published versions 0→1→2 and committed checkpoints. A fresh process resumed the cursor/model/optimizer/scheduler/RNG boundary and completed step 3 under version 2. Synthetic rewards test plumbing, not learning quality.
- `make style quality` passes after reconstructing the locked sources.

Repeat the runtime tests inside the built image:

```bash
python -m pytest tests/miles -q
PYTHONPATH=/opt/core-rl/tests/miles:$PYTHONPATH python tests/miles/smoke.py /tmp/core-smoke
PYTHONPATH=/opt/core-rl/tests/miles:$PYTHONPATH python tests/miles/smoke.py /tmp/core-smoke --resume
```

The Beaker smoke launcher uses the repository's required wrapper after committing:

```bash
MILES_BASE_IMAGE=YOUR_LOCAL_BASE_TAG ./scripts/train/build_image_and_launch.sh --miles scripts/train/debug/miles_core.sh
```

This is a training smoke run, not the repository GPU pytest job; its experiment ID must not be used as `GPU_TESTS=` in a PR.

The following remain required before deleting old GRPO code:

| Area | Current state / missing acceptance |
| --- | --- |
| Dense models | Tiny Llama, Qwen2, Qwen3, Olmo2 numerical parity; real training-scale model qualification remains |
| Conventional Olmo3Moe | Factory, native optimizer/checkpoint and export code present; full HF/Core/SGLang and multi-rank EP qualification remains |
| KDA / latent Olmo Hybrid | The selected Core base does not contain the corresponding primitives; not implemented by this adapter |
| Routing replay | Router/recompute gradients tested; serving route alignment, final unscored token's auxiliary loss, and full-model replay qualification remain |
| Tools / environments | Masks survive the adapter; the complete open-instruct multi-turn environment rollout bridge remains to be migrated |
| Mixed rewards | Existing verifier adapter tested with GSM8K; code/judge infrastructure, cleanup and complete registered-task coverage remain |
| Async | Producer lifecycle and durable ledger implemented; multi-GPU end-to-end endurance/failure qualification remains |
| Evaluation | MILES dispatch and HF export wired; full evaluation matrix remains |
| Parallelism | Dense FSDP and MoE EP paths constructed; multi-node qualification remains; TP/PP/CP rejected |
| Offload / packing | Trainer offload, dynamic packing and multi-sequence microbatches rejected |
| Recovery | Same-topology durable resume verified; automatic trainer-cell recovery and exact serving RNG replay unavailable |
| Performance | No throughput/memory acceptance claim; compare against the current olmo-miles Megatron baseline |

The full replacement is therefore unfinished. These are implementation or qualification gaps, not capabilities silently delegated to another trainer.
