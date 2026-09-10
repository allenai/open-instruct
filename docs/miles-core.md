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

The lock in `runtime/miles/runtime.lock.json` records exact bases and patch hashes. Core starts at Jacob's **`jacobm/moe-v2-core-gdn2`**, `169b8f9d06bce0276143876c82f630af483b03b7`. MILES starts at `dbbab1566ae438f7202fff653eae938e07b1d4b6`. The Core patch adds an arbitrary-objective gradient lifecycle, explicit replay across backward recomputation, HF model construction using the branch's existing KDA and latent-MoE components, and dense-model HF conversion support. KDA/latent tensor conversion is already provided by this Core base. The MILES patch includes the existing olmo-miles compatibility patches plus the explicit Core backend and lifecycle hooks.

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

HF is the serving interchange format. SGLang starts from an HF checkpoint directory containing the model config, tokenizer, and weights; olmo-sglang maps those weights into its fused inference layout. The training model is native Core. The adapter can initialize it from the same HF checkpoint, and publishes subsequent policy updates directly as HF-named tensors. A Core-origin model therefore follows `Core checkpoint → HF export → SGLang`; it does not need a Megatron checkpoint or a Megatron conversion step. Updating the policy does not require saving and reloading an HF directory at each step.

HF format alone does not establish model support: the config, tensor mapping, and olmo-sglang implementation must agree on KDA, latent projections, gates, and normalization. Native Core checkpoints remain the resume format because they contain optimizer state that an HF serving export does not carry.

Core checkpoints contain native model/optimizer state, scheduler and per-rank RNG state. MILES persists its data cursor and, for async runs, pristine outstanding prompt groups. `complete.json` and `core-latest.json` are written only after both sides are complete; the manifest hashes the cursor. Resume uses `miles.load` pointing to the checkpoint root and resolves the next rollout before constructing the manager. It currently requires the same trainer world size and model configuration. Interrupted checkpoint directories are preserved under `.incomplete-*` names before retrying that rollout. Generated async responses are regenerated after restart; serving RNG and bitwise-identical future rollouts are not promised.

The pinned image also needs the existing olmo-miles FLA 0.5.2/Triton compatibility shim for KDA; the adapter installs it in each hybrid trainer process. That shim requires one KDA head width per process. Tiny hybrid test models use eight KDA heads to keep native DDP parameter offsets aligned for grouped matmul.

Publication pauses generation, transfers weights, commits the optimizer-step version, and resumes generation. Dense conversion gathers one parameter at a time. MoE conversion stages the full unsharded model on CPU and still needs GPU space for one gathered expert parameter. This is not yet a measured large-model publication implementation. HF evaluation exports include the tokenizer and MILES completion marker.

## Local MoE task and restart check

`tests/miles/local_moe.py` exercises real GSM8K verification, SGLang sampling,
MILES advantages/loss, native Core MoE updates, publication, and restart on one
24 GB RTX 4090. The trained local fixture is
`~/proj/OLMo-core/runs/local-4090-moe/step1090`: 32,447,104 parameters, two layers,
eight experts, top two routing, conventional attention. Its older checkpoint
stores flat FP32 optimizer master parameters. Preparation reads those parameters,
exports BF16 HF weights, checks an exact BF16 roundtrip, and compares native/HF
logits (observed cosine 0.9999961). The source checkpoint is read-only.

Inside the built image, with checkpoint, GPT2 tokenizer, cached public RLVR GSM8K
parquet, and a writable output directory mounted:

```bash
python tests/miles/local_moe.py prepare /validation/local-moe \
  --source /source/step1090 --tokenizer /tokenizer --dataset /data/train.parquet
python tests/miles/local_moe.py run /validation/local-moe
python tests/miles/local_moe.py run /validation/local-moe --resume
python tests/miles/local_moe.py audit /validation/local-moe
```

For a self-contained trial without a trained checkpoint, replace `prepare` with
`bootstrap /validation/public-toy-moe`. Bootstrap initializes a smaller random
conventional MoE and downloads pinned public GPT2 tokenizer and GSM8K revisions.
Use that output path for the subsequent commands. This needs network access only
for bootstrap. Run and audit can execute offline. GPU Docker requires the NVIDIA
container toolkit or explicit device/driver-library mounts on this host.

The task slice strips reference assistant solutions and uses a simple completion
prompt ending in `Answer:`; it is not the production chat template. Four responses
per prompt are scored by the actual `GSM8KVerifier`. Two fresh iterations and one
iteration after restart produced 12 responses under versions 0, 1, and 2. Audit
independently recomputes rewards, checks finite log probabilities, verifies the
prompt cursor and versions, and checks native master parameters changed. All
observed task rewards were zero: policy advantages were zero, and MoE auxiliary
losses drove the updates. This proves lifecycle plumbing, not learning quality.

The image now includes the verifier's missing Python dependencies and validates
positive/negative GSM8K answers during build. Live MoE export must recognize
Core's `MultiGroupDDP` wrapper; runtime regression tests exercise that path.

A bounded public-input Beaker trial uses the required committed-image wrapper:

```bash
MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  ./scripts/train/build_image_and_launch.sh --miles scripts/train/debug/miles_core_moe.sh
```

It requests one Jupiter GPU, a positive 15-minute allocation window, a 20-minute
timeout, and saves restart/audit evidence under `/output`. Saving is enabled here
because native restart is the acceptance objective. The launcher uses Beaker's
CLI schema because the installed mason SDK lacks `minRuntime`. It generates
fresh random weights in the allocation and fetches public task data; no local
trained model is uploaded. It does not run the repository GPU pytest script and
must not be cited as `GPU_TESTS=` evidence. The previous olmo-miles lessons applied
here are a pinned compiled runtime, baked source overlay, local preflight,
bounded workload, explicit allocation window, and checking artifacts after exit.
Disaggregated placement and faster publication need separate qualification.

## Validation and remaining acceptance work

Verified locally on an RTX 4090:

- 62 focused adapter and existing Core SFT/DPO tests, including four dense HF/Core logits comparisons and exact weight roundtrips, streamed conversion, masked tool-token handling, mixed rewards, and policy clocks.
- 13 Core-owned tests for KDA/latent factories and weight conversion, custom-objective accumulation, and router replay through backward recomputation.
- 15 pinned-runtime tests including real Core GPU optimizer updates and exact native checkpoint restores for Qwen3, KDA, and KDA+latent; MILES schedule equivalence; async prompt-cursor crash safety; and policy-group admission.
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
| Conventional Olmo3Moe | Real local 32.45M MoE GSM8K lifecycle, initial serving-weight equality, native/HF parity, updates and separate-process restart verified; larger-model and multi-rank EP qualification remains |
| KDA / latent Olmo Hybrid | Native components and bidirectional weight conversion come from the gdn2 base; factory, exact HF weight roundtrips, GPU updates and native checkpoint restore tested; full HF/Core/SGLang numerical and multi-rank qualification remains |
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
