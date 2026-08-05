# Backend A/B Benchmarks (Part 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce measured, matched-config throughput + loss comparisons between the DeepSpeed and OLMo-core backends for SFT, DPO, and GRPO, published to `docs/algorithms/backend_comparison.md`.

**Architecture:** Three small instrumentation/guard changes to existing entrypoints, then six new launch scripts under `scripts/train/debug/backend_ab/` (one matched pair per stage), a launcher, and an analysis doc. No renames, no moves — Part 2 of the spec is gated on this plan's results.

**Tech Stack:** Python (open_instruct), bash + mason.py Beaker launches, wandb API for analysis.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-05-backend-consolidation-design.md`. Part 1 only — no Part 2 work (renames/moves/shims).
- All six runs use ONE image built from ONE commit via `./scripts/train/build_image_and_launch.sh`.
- All mason-launched scripts MUST include `--no_auto_dataset_cache` before the `--` separator (vllm not installed locally on macOS; per CLAUDE.md).
- `./scripts/train/build_image_and_launch.sh` requires all changes committed first.
- Never use `import logging`; use `logger = logger_utils.setup_logger(__name__)`.
- Imports at top of file; `from package import module` style.
- Run `make style && make quality` before every commit that touches Python.
- CHANGELOG.md entry required (PR convention).
- Do NOT edit the historical sweep scripts (`scripts/train/olmo-hybrid/*.sh`, `scripts/train/qwen/qwen3_4b_dapo_math*.sh`, `scripts/train/olmo2/*.sh`).
- Models per pair (spec decision, amended 2026-08-05): SFT = `allenai/OLMo-2-1124-7B`, DPO = Olmo-3-7B instruct-SFT weka checkpoint (hybrid dropped: `olmo3_hybrid_7B` config missing from the current olmo-core pin), GRPO = `Qwen/Qwen3-4B-Base`.
- Working branch: `backend-parity`.

---

### Task 1: DeepSpeed-only flag guard for `grpo.py`

**Files:**
- Modify: `open_instruct/grpo_utils.py` (add function near the top-level helpers, after `GRPOExperimentConfig`)
- Modify: `open_instruct/grpo.py:117` (call the guard at the top of `main`)
- Test: `open_instruct/test_grpo_utils.py`

**Interfaces:**
- Produces: `grpo_utils.check_olmo_core_compatible_config(args: GRPOExperimentConfig) -> None` — raises `ValueError` listing any DeepSpeed-only flags set to non-default values. Task 6's OLMo-core GRPO script must not trip it.

- [ ] **Step 1: Write the failing tests**

Append to `open_instruct/test_grpo_utils.py` (unittest + parameterized style, matching the file):

```python
class CheckOlmoCoreCompatibleConfigTest(unittest.TestCase):
    def test_default_config_passes(self):
        args = grpo_utils.GRPOExperimentConfig()
        grpo_utils.check_olmo_core_compatible_config(args)  # must not raise

    @parameterized.parameterized.expand(
        [
            ("deepspeed_stage", {"deepspeed_stage": 2}),
            ("deepspeed_zpg", {"deepspeed_zpg": 1}),
            ("deepspeed_offload_param", {"deepspeed_offload_param": True}),
            ("deepspeed_offload_optimizer", {"deepspeed_offload_optimizer": True}),
            ("deepspeed_checkpoint_load_universal", {"deepspeed_checkpoint_load_universal": True}),
            # sequence_parallel_size > 1 requires deepspeed_stage == 3 at construction
            # time (GRPOExperimentConfig.__post_init__), so set both; the guard must
            # still name sequence_parallel_size in its error.
            ("sequence_parallel_size", {"sequence_parallel_size": 4, "deepspeed_stage": 3}),
        ]
    )
    def test_deepspeed_only_flag_raises(self, flag, overrides):
        with self.assertRaisesRegex(ValueError, flag):
            args = grpo_utils.GRPOExperimentConfig(**overrides)
            grpo_utils.check_olmo_core_compatible_config(args)
```

Add `from open_instruct import grpo_utils` to the test file's imports if not present.

Note: if `GRPOExperimentConfig()` turns out to require arguments or do I/O in `__post_init__` that fails locally, construct with the minimal overrides the error demands and record them in the test — do not skip the test.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest open_instruct/test_grpo_utils.py -k OlmoCoreCompatible -v`
Expected: FAIL / ERROR with `AttributeError: ... has no attribute 'check_olmo_core_compatible_config'`

- [ ] **Step 3: Implement the guard in `grpo_utils.py`**

Place after the `GRPOExperimentConfig` class definition:

```python
_DEEPSPEED_ONLY_FLAG_DEFAULTS: dict[str, Any] = {
    "deepspeed_stage": 0,
    "deepspeed_zpg": 8,
    "deepspeed_offload_param": False,
    "deepspeed_offload_optimizer": False,
    "deepspeed_checkpoint_load_universal": False,
    "sequence_parallel_size": 1,
}


def check_olmo_core_compatible_config(args: GRPOExperimentConfig) -> None:
    """Reject DeepSpeed-only flags on the OLMo-core GRPO path.

    grpo.py (OLMo-core) shares GRPOExperimentConfig with grpo_fast.py (DeepSpeed)
    but never reads these flags, so setting them there silently produces a
    differently-configured run. Raise instead.
    """
    violations = [
        f"--{name}={getattr(args, name)!r} (default: {default!r})"
        for name, default in _DEEPSPEED_ONLY_FLAG_DEFAULTS.items()
        if getattr(args, name) != default
    ]
    if violations:
        raise ValueError(
            "These flags are only supported by the DeepSpeed trainer (grpo_fast.py) "
            "and are ignored by the OLMo-core trainer (grpo.py):\n  "
            + "\n  ".join(violations)
            + "\nRemove them, or use open_instruct/grpo_fast.py."
        )
```

(`Any` is already imported in grpo_utils.py's `typing` import.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest open_instruct/test_grpo_utils.py -k OlmoCoreCompatible -v`
Expected: 7 PASS

- [ ] **Step 5: Wire the guard into `grpo.py`**

In `open_instruct/grpo.py`, `main()` body starts at line ~117 with `tokenizer = grpo_fast.make_tokenizer(tc, model_config)`. Insert immediately before that line:

```python
    grpo_utils.check_olmo_core_compatible_config(args)
```

(`grpo_utils` is already imported at grpo.py:34.)

- [ ] **Step 6: Lint and commit**

```bash
make style && make quality
git add open_instruct/grpo_utils.py open_instruct/grpo.py open_instruct/test_grpo_utils.py
git commit -m "Reject DeepSpeed-only flags in OLMo-core GRPO entrypoint"
```

---

### Task 2: Per-log-period TPS in `finetune.py`

**Files:**
- Modify: `open_instruct/finetune.py:776` (add `last_log_time` next to `start_time`)
- Modify: `open_instruct/finetune.py:888-907` (add metric to `metrics_to_log`)

**Interfaces:**
- Produces: wandb metric `per_device_tps_this_log_period` on the DeepSpeed SFT path. Task 9's analysis reads it as the steady-state TPS for runs #1.

No unit test: the change is 4 lines inside a 250-line training loop in a deprecated, lint-excluded file; it is verified on the benchmark run (Task 8 checks the metric exists in wandb).

- [ ] **Step 1: Add the timer**

At `finetune.py:776`, directly after `start_time = time.perf_counter()`:

```python
    last_log_time = time.perf_counter()
```

- [ ] **Step 2: Add the metric**

The logging block (around line 888) builds `metrics_to_log = { ... }`. Immediately BEFORE that dict literal, add:

```python
                    sec_this_log_period = time.perf_counter() - last_log_time
                    last_log_time = time.perf_counter()
```

and add this entry inside the `metrics_to_log` dict, after the `"per_device_tps_including_padding"` entry:

```python
                        "per_device_tps_this_log_period": total_tokens_this_log_period
                        / accelerator.num_processes
                        / sec_this_log_period,
```

Note: `total_tokens_this_log_period` is already gathered across ranks a few lines above (line ~862), and `local_total_tokens_this_log_period.zero_()` resets it each period, so this is genuinely per-period.

- [ ] **Step 3: Sanity-check compile and lint, then commit**

Run: `uv run python -c "import ast; ast.parse(open('open_instruct/finetune.py').read())" && make style && make quality`
Expected: no output from ast, lint passes.

```bash
git add open_instruct/finetune.py
git commit -m "Add per-log-period TPS metric to finetune.py"
```

---

### Task 3: `PerfCallback` for OLMo-core SFT

**Files:**
- Modify: `open_instruct/olmo_core_finetune.py:283-292` (callback wiring in `main`), imports at top

**Interfaces:**
- Consumes: `open_instruct.olmo_core_callbacks.PerfCallback(model_dims, gradient_accumulation_steps, dp_world_size, tensor_parallel_degree)` (existing).
- Produces: wandb metrics `perf/mfu_step`, `perf/tokens_per_second_per_gpu`, `perf/seconds_per_step` on the OLMo-core SFT path. Task 9 reads them for run #2.

No unit test: the wiring mirrors `dpo.py:80-85` (itself untested); a wrong wiring fails loudly at trainer construction, and Task 8 verifies the metrics appear in wandb on the real run. Risk to watch: `PerfCallback.pre_step` calls `padding_free_collator.get_num_sequences(batch)` on numpy-FSL batches — if that raises on this batch format, the fix is to guard `_interval_num_sequences` accumulation, but do NOT preemptively change PerfCallback; let the smoke run tell us.

- [ ] **Step 1: Add imports**

In `olmo_core_finetune.py`, the file already imports `from open_instruct import ...` modules. Ensure these two are present (add to existing import lines, don't create duplicates):

```python
from open_instruct import utils
from open_instruct.olmo_core_callbacks import PerfCallback
```

- [ ] **Step 2: Wire the callback**

In `main()`, after the `trainer_callbacks["garbage_collector"] = callbacks.GarbageCollectorCallback()` line (~line 292), add:

```python
    if use_hf_ckpt:
        trainer_callbacks["perf"] = PerfCallback(
            model_dims=utils.ModelDims.from_hf_config(args.model.model_name_or_path),
            gradient_accumulation_steps=args.training.gradient_accumulation_steps,
            dp_world_size=dp_world_size,
            tensor_parallel_degree=1,
        )
    else:
        logger.warning("Skipping PerfCallback: ModelDims requires an HF checkpoint config.")
```

`use_hf_ckpt` and `dp_world_size` are both already defined earlier in `main()` (lines 123 and 204).

- [ ] **Step 3: Verify existing tests still pass, lint, commit**

Run: `uv run pytest open_instruct/test_olmo_core_finetune.py -v`
Expected: all pass (these test helpers, not main, but they import the module — catches import errors).

```bash
make style && make quality
git add open_instruct/olmo_core_finetune.py
git commit -m "Add PerfCallback to OLMo-core SFT for MFU/TPS metrics"
```

---

### Task 4: SFT A/B pair scripts

**Files:**
- Create: `scripts/train/debug/backend_ab/ab_sft_deepspeed.sh`
- Create: `scripts/train/debug/backend_ab/ab_sft_olmocore.sh`
- Create: `scripts/train/debug/backend_ab/ab_sft_olmocore_cache.sh`

**Interfaces:**
- Produces: three launch scripts, each taking the Beaker image as `$1` (build_image_and_launch.sh convention). The cache script MUST use identical `--model_name_or_path`, `--mixer_list`, `--max_seq_length`, and `--seed` to `ab_sft_olmocore.sh` (the numpy cache dir is keyed on the config hash + seed + seq length).

Matched config for the pair: OLMo-2-1124-7B, `allenai/tulu-3-sft-olmo-2-mixture-0225 60000` (60k examples caps on-node tokenization time; 150 steps × 32 seqs needs only 4,800), seq 4096, bs 1 × grad-accum 2 over 2 nodes × 8 GPU, lr 2e-5, linear, warmup 0.03, wd 0, seed 42, `--max_train_steps 150`, `--logging_steps 1`, tulu chat template, `--add_bos`. Each backend keeps its production memory strategy (DS: gradient checkpointing off per prod script; OC: compile + default AC) — the comparison is "backend as production-configured"; note this in Task 9's doc.

- [ ] **Step 1: Write `ab_sft_deepspeed.sh`**

Pattern from `scripts/train/debug/sft_multinode_test.sh` (mason rewrites accelerate's multinode args; keep `--num_processes 8` as committed scripts do — Task 8 verifies world size 16 in logs):

```bash
#!/bin/bash
# Backend A/B: SFT on DeepSpeed (finetune.py). Pair: ab_sft_olmocore.sh.
# See docs/superpowers/specs/2026-08-05-backend-consolidation-design.md Part 1.
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Backend A/B: SFT DeepSpeed (finetune.py), OLMo-2-7B, 2 nodes." \
    --pure_docker_mode \
    --preemptible \
    --max_retries 0 \
    --num_nodes 2 \
    --gpus 8 \
    --non_resumable \
    --no_auto_dataset_cache \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name ab_sft_deepspeed \
    --model_name_or_path allenai/OLMo-2-1124-7B \
    --tokenizer_name allenai/OLMo-2-1124-7B \
    --add_bos \
    --chat_template_name tulu \
    --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 60000 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --max_train_steps 150 \
    --logging_steps 1 \
    --seed 42 \
    --report_to wandb \
    --with_tracking \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
```

- [ ] **Step 2: Write `ab_sft_olmocore.sh`**

Pattern from `scripts/train/debug/oc_sft_multinode.sh`:

```bash
#!/bin/bash
# Backend A/B: SFT on OLMo-core (olmo_core_finetune.py). Pair: ab_sft_deepspeed.sh.
# Requires ab_sft_olmocore_cache.sh to have completed first (numpy dataset cache).
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Backend A/B: SFT OLMo-core (olmo_core_finetune.py), OLMo-2-7B, 2 nodes." \
    --pure_docker_mode \
    --preemptible \
    --max_retries 0 \
    --num_nodes 2 \
    --gpus 8 \
    --non_resumable \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    -- torchrun \
    --nnodes=2 \
    --node_rank=\$BEAKER_REPLICA_RANK \
    --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
    --master_port=29400 \
    --nproc_per_node=8 \
    open_instruct/olmo_core_finetune.py \
    --exp_name ab_sft_olmocore \
    --model_name_or_path allenai/OLMo-2-1124-7B \
    --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
    --add_bos \
    --chat_template_name tulu \
    --mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 60000 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --max_train_steps 150 \
    --logging_steps 1 \
    --seed 42 \
    --compile_model true \
    --with_tracking \
    --output_dir \$CHECKPOINT_OUTPUT_DIR
```

- [ ] **Step 3: Write `ab_sft_olmocore_cache.sh`**

Pattern from `scripts/train/debug/oc_sft_cache.sh` — CPU-only job, MUST match model/mixer/seq/seed of Step 2:

```bash
#!/bin/bash
# CPU tokenization cache job for ab_sft_olmocore.sh. Run to completion BEFORE it.
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Backend A/B: numpy tokenization cache for ab_sft_olmocore.sh." \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --gpus 0 \
    --non_resumable \
    --no_auto_dataset_cache \
    -- \
    uv run python open_instruct/olmo_core_finetune.py \
    --model_name_or_path allenai/OLMo-2-1124-7B \
    --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
    --add_bos \
    --chat_template_name tulu \
    --mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 60000 \
    --max_seq_length 4096 \
    --seed 42 \
    --cache_dataset_only
```

- [ ] **Step 4: Syntax-check and commit**

Run: `bash -n scripts/train/debug/backend_ab/ab_sft_deepspeed.sh scripts/train/debug/backend_ab/ab_sft_olmocore.sh scripts/train/debug/backend_ab/ab_sft_olmocore_cache.sh`
Expected: no output.

```bash
git add scripts/train/debug/backend_ab/
git commit -m "Add SFT backend A/B benchmark scripts"
```

---

### Task 5: DPO A/B pair scripts

**Files:**
- Create: `scripts/train/debug/backend_ab/ab_dpo_deepspeed.sh`
- Create: `scripts/train/debug/backend_ab/ab_dpo_olmocore.sh`

**Interfaces:**
- Produces: two launch scripts taking image as `$1`. BOTH load the identical checkpoint object — the Olmo-3 7B instruct-SFT weka path that both committed olmo3 DPO production scripts already share.

**Amendment (2026-08-05):** originally Hybrid-7B; substituted with Olmo-3-7B (user decision) because `olmo3_hybrid_7B` is missing from the current olmo-core pin (lost in the #1723 pin bump — tracked separately). Base scripts are now `scripts/train/olmo3/7b_instruct_dpo.sh` (DS) and `scripts/train/olmo3/7b_instruct_dpo_olmocore.sh` (OC).

Matched config: model `/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-7b-instruct-sft-1115`, `--config_name olmo3_7B` on the OC side, mixer `allenai/olmo-3-pref-mix-deltas-complement2-DECON-tpc-kwd-ch-dedup5-lbc100-grafmix-unbal 30000` (single dataset from the production mix, 30k pairs; 150 steps × 128 pairs/step needs 19,200 — identical on both sides is what matters), seq 16384, bs 1 × ga 4 over 4 nodes × 8, lr 1e-6, linear, wd 0, seed 42 (the production pair disagrees: DS default 42, OC 123 — pin both), chat template `olmo123`, `--max_train_steps 150`, `--num_epochs 1`, `--logging_steps 1`, checkpointing_steps 500 (> 150 ⇒ no checkpoint I/O noise), push/eval/beaker-save off. Env vars: use the SAME jupiter-appropriate set on both sides (from the hybrid sweeps) — do NOT copy the DS production script's TCPXO/`/var/lib/tcpxo` env block or its `source ... &&` prefix (that is Augusta/GCP-specific and a confound), and do NOT copy the OC production script's `NCCL_DEBUG=INFO`/`TORCH_LOGS` debug envs. Backend-specific memory strategy per production config: DS keeps `--gradient_checkpointing`; OC keeps `--activation_memory_budget 0.1` + `--compile_model true`. No `--packing` on either side (the olmo3 production pair doesn't use it).

- [ ] **Step 1: Verify the Olmo-3 TransformerConfig and checkpoint exist**

Run: `uv run python -c "from olmo_core.nn.transformer import TransformerConfig; assert hasattr(TransformerConfig, 'olmo3_7B'), 'missing'; print('ok')"`
Expected: `ok`.
Also verify the checkpoint is reachable (weka is not mounted locally, so just record this for Task 8's log check): the path `/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-7b-instruct-sft-1115` must contain `config.json` — the Task 8 verifier confirms it from the job logs.

- [ ] **Step 2: Write `ab_dpo_deepspeed.sh`**

From `scripts/train/olmo3/7b_instruct_dpo.sh` with the LR loop removed, TCPXO env block dropped, eval flags stripped, steps capped:

```bash
#!/bin/bash
# Backend A/B: DPO on DeepSpeed ZeRO-3 (dpo_tune_cache.py). Pair: ab_dpo_olmocore.sh.
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
echo "Using Beaker image: $BEAKER_IMAGE"
MODEL_PATH="/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-7b-instruct-sft-1115"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --description "Backend A/B: DPO DeepSpeed (dpo_tune_cache.py), Olmo-3-7B, 4 nodes, 16k seq." \
    --max_retries 0 \
    --preemptible \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env NCCL_IB_HCA=^=mlx5_bond_0 \
    --env NCCL_SOCKET_IFNAME=ib \
    --env TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
    --env TORCH_DIST_INIT_BARRIER=1 \
    --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
    --num_nodes 4 \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name ab_dpo_deepspeed \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name "$MODEL_PATH" \
    --use_slow_tokenizer False \
    --mixer_list allenai/olmo-3-pref-mix-deltas-complement2-DECON-tpc-kwd-ch-dedup5-lbc100-grafmix-unbal 30000 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --zero_hpz_partition_size 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --checkpointing_steps 500 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --max_train_steps 150 \
    --seed 42 \
    --logging_steps 1 \
    --gradient_checkpointing \
    --chat_template_name olmo123 \
    --push_to_hub False \
    --try_launch_beaker_eval_jobs False \
    --try_auto_save_to_beaker False \
    --with_tracking
```

- [ ] **Step 3: Write `ab_dpo_olmocore.sh`**

From `scripts/train/olmo3/7b_instruct_dpo_olmocore.sh`, same surgery (LR loop removed, debug envs dropped, env set matched to the DS script, seed pinned to 42, steps capped, mixer reduced to the single matched dataset):

```bash
#!/bin/bash
# Backend A/B: DPO on OLMo-core FSDP (dpo.py). Pair: ab_dpo_deepspeed.sh.
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
echo "Using Beaker image: $BEAKER_IMAGE"
MODEL_PATH="/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-7b-instruct-sft-1115"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --description "Backend A/B: DPO OLMo-core (dpo.py), Olmo-3-7B, 4 nodes, 16k seq." \
    --max_retries 0 \
    --preemptible \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env NCCL_IB_HCA=^=mlx5_bond_0 \
    --env NCCL_SOCKET_IFNAME=ib \
    --env TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
    --env TORCH_DIST_INIT_BARRIER=1 \
    --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
    --num_nodes 4 \
    --gpus 8 -- torchrun \
    --nnodes=4 \
    --node_rank=\$BEAKER_REPLICA_RANK \
    --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
    --master_port=29400 \
    --nproc_per_node=8 \
    open_instruct/dpo.py \
    --exp_name ab_dpo_olmocore \
    --model_name_or_path "$MODEL_PATH" \
    --config_name olmo3_7B \
    --chat_template_name olmo123 \
    --mixer_list allenai/olmo-3-pref-mix-deltas-complement2-DECON-tpc-kwd-ch-dedup5-lbc100-grafmix-unbal 30000 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --fsdp_shard_degree 32 \
    --fsdp_num_replicas 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --checkpointing_steps 500 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --max_train_steps 150 \
    --seed 42 \
    --logging_steps 1 \
    --activation_memory_budget 0.1 \
    --compile_model true \
    --push_to_hub False \
    --try_launch_beaker_eval_jobs False \
    --try_auto_save_to_beaker False \
    --with_tracking
```

- [ ] **Step 4: Syntax-check and commit**

Run: `bash -n scripts/train/debug/backend_ab/ab_dpo_deepspeed.sh scripts/train/debug/backend_ab/ab_dpo_olmocore.sh`
Expected: no output.

```bash
git add scripts/train/debug/backend_ab/
git commit -m "Add DPO backend A/B benchmark scripts"
```

---

### Task 6: GRPO A/B pair scripts

**Files:**
- Create: `scripts/train/debug/backend_ab/ab_grpo_deepspeed.sh`
- Create: `scripts/train/debug/backend_ab/ab_grpo_olmocore.sh`

**Interfaces:**
- Consumes: Task 1's guard — the OLMo-core script must pass none of the six guarded flags.
- Produces: two launch scripts taking image as `$1`.

The committed pair `scripts/train/qwen/qwen3_4b_dapo_math.sh` / `_oc.sh` is already matched except backend flags. Copy each verbatim, then apply EXACTLY these changes to BOTH copies:

1. `EXP_NAME` → `ab_grpo_deepspeed` / `ab_grpo_olmocore`.
2. Image handling → `BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"` (delete the OC script's `BEAKER_USER` conditional block and the DS script's `${1:-nathanl/...}` default).
3. `CLUSTER` → pinned `ai2/jupiter` on both (the OC original also lists ceres; same hardware pool is required).
4. `WORKSPACE`/workspace → `ai2/open-instruct-dev` on both.
5. `--total_episodes 128000` → `--total_episodes 19200` (150 steps × 8 prompts × 16 samples).
6. `--local_eval_every 100` → `--local_eval_every -1` (evals add generation noise to timing).
7. `--save_freq 100` → `--save_freq 100000` and `--checkpoint_state_freq 100` → `--checkpoint_state_freq 100000` (no checkpoint I/O inside the measured window).
8. Delete `--send_slack_alerts`.
9. Add `--max_retries 0` to the mason flags (before `--`).

Backend-specific flags stay as committed: DS keeps `--deepspeed_stage 2`; OC keeps `--fsdp_shard_degree 4 --fsdp_num_replicas 1 --activation_memory_budget 0.5`. Everything else (async_steps 4, active_sampling, inflight_updates, bs, pack_length 10240, response_length 8192, temperature 1.0, seed 1, `--load_ref_policy False`) is already identical between the two — do not touch it. Note: the OC original passes `--gradient_checkpointing`, which the OLMo-core path ignores; keep it anyway to stay byte-comparable with the committed script and mnoukhov's production runs.

- [ ] **Step 1: Create both scripts as described**

```bash
cp scripts/train/qwen/qwen3_4b_dapo_math.sh scripts/train/debug/backend_ab/ab_grpo_deepspeed.sh
cp scripts/train/qwen/qwen3_4b_dapo_math_oc.sh scripts/train/debug/backend_ab/ab_grpo_olmocore.sh
```

Then apply edits 1–9 above to each.

- [ ] **Step 2: Verify the pair differs only as intended**

Run: `diff scripts/train/debug/backend_ab/ab_grpo_deepspeed.sh scripts/train/debug/backend_ab/ab_grpo_olmocore.sh`
Expected diff contains ONLY: the exp name, `grpo_fast.py` vs `grpo.py`, and `--deepspeed_stage 2` vs the three fsdp/AC flags. Any other diff line is a bug — fix before committing.

- [ ] **Step 3: Syntax-check and commit**

Run: `bash -n scripts/train/debug/backend_ab/ab_grpo_deepspeed.sh scripts/train/debug/backend_ab/ab_grpo_olmocore.sh`

```bash
git add scripts/train/debug/backend_ab/
git commit -m "Add GRPO backend A/B benchmark scripts"
```

---

### Task 7: Launcher, CHANGELOG, and launch

**Files:**
- Create: `scripts/train/debug/backend_ab/launch_all.sh`
- Modify: `CHANGELOG.md` (under `### Added`)

**Interfaces:**
- Consumes: the six scripts from Tasks 4–6, each taking image as `$1`.
- Produces: six running Beaker experiments; experiment IDs recorded for Tasks 8–9.

- [ ] **Step 1: Write `launch_all.sh`**

The SFT numpy cache job must COMPLETE before `ab_sft_olmocore.sh` starts; everything else launches immediately:

```bash
#!/bin/bash
# Launch all backend A/B benchmark runs. Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/train/debug/backend_ab/launch_all.sh
set -eo pipefail
BEAKER_IMAGE="${1:?usage: launch_all.sh BEAKER_IMAGE}"
DIR="$(dirname "$0")"

echo "=== Launching SFT numpy cache job (blocks ab_sft_olmocore) ==="
CACHE_LOG=$(bash "$DIR/ab_sft_olmocore_cache.sh" "$BEAKER_IMAGE" 2>&1 | tee /dev/stderr)
CACHE_ID=$(echo "$CACHE_LOG" | grep -oE 'https://beaker.org/ex/[a-zA-Z0-9]+' | tail -1 | sed 's|.*/||')
[ -n "$CACHE_ID" ] || { echo "ERROR: no cache experiment id"; exit 1; }

echo "=== Launching the 5 non-blocked runs ==="
bash "$DIR/ab_sft_deepspeed.sh" "$BEAKER_IMAGE"
bash "$DIR/ab_dpo_deepspeed.sh" "$BEAKER_IMAGE"
bash "$DIR/ab_dpo_olmocore.sh" "$BEAKER_IMAGE"
bash "$DIR/ab_grpo_deepspeed.sh" "$BEAKER_IMAGE"
bash "$DIR/ab_grpo_olmocore.sh" "$BEAKER_IMAGE"

echo "=== Waiting for cache job $CACHE_ID ==="
beaker experiment await-all "$CACHE_ID"
echo "=== Launching ab_sft_olmocore ==="
bash "$DIR/ab_sft_olmocore.sh" "$BEAKER_IMAGE"
echo "=== All launched ==="
```

Note: if `beaker experiment await-all` exits nonzero on a *successful* job (check `beaker experiment await-all --help` for status semantics), adjust to `beaker experiment await-all "$CACHE_ID" || true` followed by an explicit `beaker experiment get "$CACHE_ID"` status check.

- [ ] **Step 2: CHANGELOG entry**

Under `### Added` in `CHANGELOG.md`:

```markdown
- Backend A/B benchmark scripts (`scripts/train/debug/backend_ab/`) comparing DeepSpeed vs OLMo-core for SFT/DPO/GRPO, plus `PerfCallback` on OLMo-core SFT, a per-log-period TPS metric in `finetune.py`, and a guard rejecting DeepSpeed-only flags in `grpo.py` (PR link TBD on PR creation).
```

(Replace "PR link TBD on PR creation" with the real PR URL when the PR exists.)

- [ ] **Step 3: Commit, then launch**

```bash
bash -n scripts/train/debug/backend_ab/launch_all.sh
git add scripts/train/debug/backend_ab/launch_all.sh CHANGELOG.md
git commit -m "Add backend A/B launcher and CHANGELOG entry"
./scripts/train/build_image_and_launch.sh scripts/train/debug/backend_ab/launch_all.sh
```

Record all six experiment IDs (plus the cache job's) from the output into `docs/algorithms/backend_comparison.md`'s table (Task 9 creates the skeleton — if running tasks in order, jot them in the task notes and fill the doc in Task 9).

---

### Task 8: In-flight verification and monitoring

**Files:** none (operational task)

**Interfaces:**
- Consumes: the six experiment IDs from Task 7.
- Produces: confirmation that each run is measuring what we think, or an early abort + fix.

- [ ] **Step 1: Within ~15 min of each run starting, verify configuration**

For each experiment (use the monitor-experiment skill or `beaker experiment logs <id>`):

- `ab_sft_deepspeed`: logs contain the accelerate world-size line showing **16** processes (NOT 8 — if 8, the mason multinode rewrite assumption failed; kill, fix `--num_processes`, relaunch).
- `ab_sft_olmocore`: no `FileNotFoundError` about the numpy cache (would mean cache-key mismatch with the cache job — recheck seed/seq/mixer equality).
- `ab_dpo_olmocore`: model builds with `olmo3_hybrid_7B`; reference-logprob cache phase starts.
- `ab_grpo_olmocore`: does not trip Task 1's guard.

- [ ] **Step 2: Verify the new metrics exist in wandb**

After ~25 steps of each SFT run, check the wandb run (URL is in the Beaker experiment description):

- `ab_sft_olmocore` has `perf/mfu_step` and `perf/tokens_per_second_per_gpu` (Task 3 worked; if `PerfCallback` crashed on numpy-FSL batches, the run dies early — see Task 3's risk note, fix `get_num_sequences` handling, relaunch).
- `ab_sft_deepspeed` has `per_device_tps_this_log_period` (Task 2 worked).

- [ ] **Step 3: Step-0 conversion check per pair**

Compare the first logged `train_loss` between the two runs of each pair (data-loader orders differ across backends, so this is a distribution-level sanity check, not a byte check):

- SFT + GRPO: |Δ| within ~5% at step 0–5 average.
- DPO: `dpo_norm` loss starts near `log(2) ≈ 0.693` on BOTH (policy == reference at step 0); also compare `logps/chosen` magnitudes (within ~5%).

A large step-0 gap means the HF→OLMo-core conversion is wrong for that architecture: STOP that pair, report, do not use its throughput numbers.

- [ ] **Step 4: Monitor to completion**

Use the monitor-experiment skill (or `beaker experiment await-all`) for all six. Expected durations: SFT and DPO ≤ ~2h each; GRPO ~4–8h (generation-bound). Preempted runs (max_retries 0) must be relaunched, not resumed.

---

### Task 9: Analysis and `backend_comparison.md`

**Files:**
- Create: `docs/algorithms/backend_comparison.md`
- Modify: `mkdocs.yml` (add to the Training nav section, after `algorithms/olmo_core_sharding.md`)

**Interfaces:**
- Consumes: the six finished wandb runs.
- Produces: the per-stage verdicts that gate Part 2 of the spec.

- [ ] **Step 1: Pull steady-state metrics from wandb**

For each run, average over steps 20–150 (drop compile/warmup). Extraction snippet (fill in the run paths from the Beaker descriptions):

```python
# uv run python scratch_analyze.py  (temp file in scratchpad, not committed)
import wandb

RUNS = {
    "ab_sft_deepspeed": "ai2-llm/open_instruct_internal/<run_id>",
    "ab_sft_olmocore": "ai2-llm/open_instruct_internal/<run_id>",
    "ab_dpo_deepspeed": "ai2-llm/open_instruct_internal/<run_id>",
    "ab_dpo_olmocore": "ai2-llm/open_instruct_internal/<run_id>",
    "ab_grpo_deepspeed": "ai2-llm/open_instruct_internal/<run_id>",
    "ab_grpo_olmocore": "ai2-llm/open_instruct_internal/<run_id>",
}
KEYS = [
    "train_loss", "train/train_loss",
    "perf/mfu_step", "perf/tokens_per_second_per_gpu", "perf/seconds_per_step",
    "per_device_tps_this_log_period",
    "time/training", "learner_tokens_per_second_step",
]
api = wandb.Api()
for name, path in RUNS.items():
    run = api.run(path)
    hist = run.history(samples=2000)
    print(f"== {name}")
    for k in KEYS:
        if k in hist.columns:
            col = hist[k].dropna()
            steady = col.iloc[20:]
            print(f"  {k}: mean={steady.mean():.4g} std={steady.std():.4g} n={len(steady)}")
```

Comparison rules (from the spec):
- SFT/DPO throughput: `perf/tokens_per_second_per_gpu` + `perf/mfu_step` (OC) vs `per_device_tps_this_log_period` (DS SFT) / `perf/*` keys from dpo_tune_cache (DS DPO — it logs the same-named keys per #1719).
- GRPO: `time/training` and `learner_tokens_per_second_*` ONLY. Never `time/total` or wall clock.
- Loss: steady-state `train_loss` curves within ~1% after warmup; step-0 check from Task 8.
- Verdict per stage: OLMo-core wins or ties throughput AND loss tracks ⇒ PASS.

- [ ] **Step 2: Write `docs/algorithms/backend_comparison.md`**

Structure (fill every cell; no TBDs left at commit time):

```markdown
# Backend Comparison: DeepSpeed vs OLMo-core

Matched-config A/B benchmarks (150 steps, steps 20–150 measured, one image,
commit `<sha>`). Scripts: `scripts/train/debug/backend_ab/`. Spec:
`docs/superpowers/specs/2026-08-05-backend-consolidation-design.md`.

Caveat: each backend runs its production memory strategy (DS: ZeRO-3/stage-2 +
gradient checkpointing; OC: FSDP/HSDP + compile + budget AC), and SFT data
semantics differ (padded examples vs packed FSL blocks). This compares
*backends as production-configured*, not isolated kernels. Tokens/sec counts
non-padding tokens on both sides.

| Stage | Model | Backend | tokens/s/GPU | MFU | s/step | steady loss | step-0 loss | Beaker | wandb |
|---|---|---|---|---|---|---|---|---|---|
| SFT | OLMo-2-7B | DeepSpeed | | n/a | | | | [link] | [link] |
| SFT | OLMo-2-7B | OLMo-core | | | | | | [link] | [link] |
| DPO | Olmo-3-7B | DeepSpeed | | | | | | [link] | [link] |
| DPO | Olmo-3-7B | OLMo-core | | | | | | [link] | [link] |
| GRPO | Qwen3-4B | DeepSpeed | | | | | | [link] | [link] |
| GRPO | Qwen3-4B | OLMo-core | | | | | | [link] | [link] |

## Verdicts

| Stage | Verdict | Basis |
|---|---|---|
| SFT | PASS / FAIL | ... |
| DPO | PASS / FAIL | ... |
| GRPO | PASS / FAIL | ... |

A FAIL stage keeps DeepSpeed as primary (see spec failure branch).
```

- [ ] **Step 3: Add to mkdocs nav, build check, commit**

In `mkdocs.yml` nav → Training, after `algorithms/olmo_core_sharding.md`, add `- algorithms/backend_comparison.md`.

Run: `uv run mkdocs build`
Expected: builds without warnings about the new page.

```bash
git add docs/algorithms/backend_comparison.md mkdocs.yml
git commit -m "Add DeepSpeed vs OLMo-core backend comparison results"
```

- [ ] **Step 4: Open the PR**

PR from `backend-parity` → `main` containing everything from Tasks 1–9. Body includes the verdicts table and a "Runs:" numbered list with Beaker links (per CLAUDE.md convention). GPU tests: run `./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh` and include `GPU_TESTS=[EXPERIMENT_ID](https://beaker.org/ex/EXPERIMENT_ID)` in the body (the experiment must be from the GPU test script, not the benchmark runs).
