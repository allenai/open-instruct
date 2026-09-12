# MILES run controls in open-instruct

The researcher interface is `python -m open_instruct.miles {plan,validate,train,run,status} run.toml`.
See the [workflow guide](miles-workflow.md) for structured sections matching
olmo-miles and the one-node Beaker launcher. The original low-level files support
`plan/validate/train`; the native names below describe their resolved controls.
`[miles]` expresses the native MILES/SGLang options using underscores. `[core]`
expresses the settings owned by the OLMo-core adapter. TOML is our integration's
format; it was not the preexisting open-instruct GRPO configuration format.
The Python API is the same `RunConfig(CoreConfig(...), {...})` used by the trial harnesses.

This audit compares `olmo-miles` revision `07887b783ab254577a6656168dc0e0d21aebfe3d`
with the runtime pinned in this repository. The [complete field inventory](miles-knob-inventory.md)
accounts for every `MilesSmokeConfig` field, including controls that belong to
preparation, launch, or diagnostics rather than training arguments. Native parser
acceptance does not establish backend support or GPU qualification.

Start with the [colocated dev/test and disaggregated training profiles](../configs/miles/README.md).
The full-SFT starter now exposes 64 concurrent requests and sizes its graph and
cache limits together. Maintained full-SFT examples now use 8 × 8; async uses
trainer-scored old logprobs plus TIS and buffer factor two. The combined
config-driven exercise passed four updates with independently audited samples; previous measured configs remain frozen.

## Editing and inspecting a run

```bash
python -m open_instruct.miles plan run.toml \
  --set miles.lr=1e-6 \
  --set miles.num_rollout=100 \
  --set 'miles.wandb_group="gsm8k-comparison"'
```

The same repeatable `--set SECTION.KEY=TOML_VALUE` works with `validate` and `train`.
Overrides apply before validation; repeated assignments to the same key use the
last value. String values need TOML quotes protected from the shell. Booleans,
lists, and inline tables use normal TOML syntax. No implicit environment expansion
or config inheritance is performed. Aliases for a single native setting cannot
both appear in a config.

`plan` reports argv, explicit MILES settings, resolved Core settings, and placement,
collection size, optimizer batch size, and the number of updates per collection.
Unspecified serving GPU counts and collection sizes are reported as null; the
installed runtime resolves its own defaults. Planning imports no MILES, SGLang,
CUDA, model code, or dataset. It checks option names, basic types, choices and the
Core restrictions using a snapshot of the pinned parser. `runtime_validated=false`
means installed-runtime checks and model-dependent checks have not run.
For low-level files, `validate` invokes the real MILES parser and its argument/model
checks. Structured files validate the CPU-safe schema and resolved options; the
real parser runs before training. Neither starts engines or certifies GPU memory
fit, numerical equivalence, or training.

Native switches now behave correctly: `use_wandb=false` emits no switch instead
of inventing `--no-use-wandb`. `offload_train=false` uses the real
`--no-offload-train`. `grpo_std_normalization=false` emits
`--disable-grpo-std-normalization`; the existing spelling
`disable_grpo_std_normalization=true` still works. JSON-valued controls accept
structured TOML, for example:

```toml
[miles]
hf_checkpoint = "/data/hf"
global_batch_size = 16
rollout_batch_size = 4
n_samples_per_prompt = 4
train_env_vars = { NCCL_DEBUG = "WARN" }
eval_prompt_data = ["gsm8k", "/data/gsm8k-heldout.jsonl"]
sglang_cuda_graph_config = { decode = { backend = "full", max_bs = 4 }, prefill = { backend = "disabled" } }
```

JSON strings remain accepted. List-valued flags such as `eval_prompt_data` remain
multiple CLI arguments; JSON lists become one JSON argument. Flags using native
append actions accept a list of occurrences (nested lists for an occurrence that
itself takes multiple values). Custom actions without an encoder fail explicitly.

## The shape of a run

All names below are under `[miles]` unless prefixed `core.`.

| Dial from olmo-miles | open-instruct control and semantics |
| --- | --- |
| Placement | `colocate=false` gives separate trainer/serving GPU allocations; they need not be separate physical nodes. `colocate=true` shares GPUs. **Core stays resident**: trainer offload is unsupported. The current olmo-miles colocated compiler also keeps the trainer resident; full-model colocation is qualified there but only tiny-model colocation is qualified here. |
| Trainer GPUs / EP | `actor_num_nodes`, `actor_num_gpus_per_node`; `core.expert_parallel_size` must divide their product. Dense Olmo 3 uses its separate Core backend and EP=1. |
| Inference GPUs / TP / EP | `rollout_num_gpus`, `num_gpus_per_node`, `rollout_num_gpus_per_engine` (TP), `sglang_ep_size`. GPU count per engine determines how many engines fit the allocation. |
| Synchronous / bounded async | `fully_async`; async requires resident disaggregated engines and positive `core.max_policy_lag`. |
| Staleness | `core.max_policy_lag` is measured in **optimizer steps**. If supplied, `max_weight_staleness` must agree; it no longer silently gets overwritten with a different value. |
| Producer capacity | `async_data_buffer_capacity_factor`, `async_max_concurrent_samples`, `async_unused_samples_handler`, `rollout_submission_granularity`. The Core buffer enforces homogeneous policy versions within prompt groups. |
| Collection / optimizer batch | `rollout_batch_size * n_samples_per_prompt` is samples per collection; `global_batch_size` is samples per optimizer step. Collections must contain whole steps and the lag budget must cover their final step. `num_rollout` counts collections. |
| Scoring row specialization | `core.row_specialization` (`static` / `dynamic`), resolved at model construction. Core defaults to static; the starting RL profiles select dynamic to avoid compiling for every activation-buffer capacity. Arithmetic and direct wave/backward callers are unchanged. |
| Standalone scoring pass | Runs only when the recipe needs it: more than one optimizer step per collection (`rollout_batch_size × n_samples_per_prompt / global_batch_size`), a nonzero `kl_coef`, or dropout in the model configuration. Otherwise old log-probabilities are read from the training forward at unchanged weights (or from rollout log-probabilities when `use_rollout_logprobs=true`), the trainer-anchor PPO ratio is 1 at this forward (a rollout anchor can still differ), and the behavior-policy agreement gate runs on the training forward before the optimizer step. `core.scoring_pass_required=true` forces the pass every update. Skipped runs still run the pass on the first update of every process and every `core.scoring_check_interval` updates (default 50; 0 keeps only the first), comparing it to the training forward and failing above `core.scoring_check_tolerance` (default 1e-3 mean absolute). `plan` reports the decision under `scoring_pass`; each optimizer record carries `scoring_pass` as `standalone`, `checked` or `skipped`. |
| Microbatch / recomputation | `micro_batch_size=1`; `core.activation_checkpointing`. Core accumulates unpadded samples; `max_tokens_per_gpu` is rejected because it does not control Core batching. Megatron packing, selective recompute modules and dynamic microbatch selection do not translate directly. |
| Lengths | Set `rollout_max_response_len`, `rollout_max_context_len`, `rollout_max_prompt_len`, `sglang_context_length`, and `core.max_sequence_length` consistently. The baseline's single context knob populated several of these. |
| Admission / cache | `sglang_server_concurrency`, `sglang_max_running_requests`, `sglang_mem_fraction_static`, `sglang_max_total_tokens`, `sglang_max_mamba_cache_size`; `sglang_disable_radix_cache=false` enables radix cache. |
| Serving optimizations | Native `sglang_*` graph, prefill, attention, page-size, speculative decoding and determinism options pass through. Prefer per-phase graph settings; serving compatibility still depends on the pinned SGLang build and architecture. |
| Router policy | `use_miles_router`, `sglang_router_policy`, `router_cache_threshold`, `router_balance_abs_threshold`, `router_balance_rel_threshold`. |

## Data and objective

| Dial | open-instruct control and semantics |
| --- | --- |
| Training input | `prompt_data`, `input_key`, `label_key`, `metadata_key`; `custom_rm_path` selects the reward function. `core.reward_config` configures registered open-instruct verifiers. Prepare prompts with one chat-template application. |
| Task catalog / recipe / manifest | Structured `data.tasks` and supported `data.rl_manifest` invoke explicit preparation/adoption with immutable manifests. Named `data.recipe` selection is rejected. Named tasks currently include GSM8K, math, IF and generated multiplication; see the workflow guide for limits. Raw `miles.prompt_data` still expects prepared JSONL, not a baseline manifest. |
| Held-out in-loop eval | `eval_prompt_data=[name,path,...]`, `eval_interval`, `skip_eval_before_train=false`, `n_samples_per_eval_prompt`; explicit `eval_temperature`, `eval_top_p`, `eval_top_k`, `eval_max_response_len`, `eval_max_prompt_len`. Retain source indices and dataset hashes for comparisons. |
| Learning rate / Adam | `lr`, `lr_decay_style`, `lr_decay_iters`, `lr_warmup_iters` or `lr_warmup_fraction`, `min_lr`, `weight_decay`, `adam_beta1`, `adam_beta2`, `adam_eps`, `clip_grad`. Core uses AdamW; changing `optimizer` to another family is rejected. |
| Policy objective | `advantage_estimator`, `calculate_per_token_loss`, `use_rollout_logprobs`, `grpo_std_normalization`, `eps_clip`, `eps_clip_high`, `entropy_coef`. The baseline uses std normalization off and upper clipping 0.28; these are separate choices. Structured run files and full-SFT starters now apply both defaults; raw MILES parser defaults remain separate. |
| Reference KL | Enable `use_kl_loss`, set `kl_loss_coef`, and provide the reference initialization through `ref_load` as required by MILES. This creates a frozen reference and adds scoring. In raw files a coefficient alone is not the enable switch; the structured compiler enables reference KL for a positive coefficient and initializes the reference from the prepared starting HF model. |
| Off-policy correction | `use_tis`, `tis_clip`, `tis_clip_low`; alternative corrections use `custom_tis_function_path` (the baseline ICEPOP helper is not bundled); explicitly choose the policy-ratio anchor. Structured async defaults to trainer-scored old logprobs with TIS; these are algorithm changes, not just async throughput controls. |
| Router behavior | `core.router_aux_loss_weight`, `core.router_z_loss_weight`; `use_rollout_routing_replay=true` requires `use_miles_router=true`. Trainer-side `use_routing_replay` is a different Megatron feature and is rejected. `core.replay_diagnostics=true` opts into per-layer returned-route and recomputation checks (adds synchronization overhead). |

## Checkpoints, reporting and operational controls

| Dial | open-instruct control and semantics |
| --- | --- |
| Save / restart | `save`, `save_interval`, `load`. Saves are synchronous native Core checkpoints with completion manifests and a rollout/policy cursor. `async_save=true` is rejected. Baseline NVRX saves, retention and token-per-expert cadence are not ported. |
| Native MoE checkpoint writer | Arithmetic metadata planning, compact storage, and balanced replicated ownership are enabled by default. Opt out with `core.checkpoint_constant_memory_planning=false`, `core.checkpoint_compact_storage=false`, and `core.checkpoint_dedup_save_to_lowest_rank=true`. Each switch is independent. Profiling (`core.checkpoint_profile`) and spawned workers (`core.checkpoint_process_count`) remain opt-in; `core.checkpoint_thread_count` controls thread buckets. The separate dense trainer keeps its own checkpoint path and rejects policy overrides. See the [qualification record](measurements/miles-checkpoint-perf-20260911.md). |
| Final HF export | `output.export_hf=true` in structured files requests explicit driver export at completion; `eval_hf_dir` remains snapshot export for evaluation. New workflow export is implemented but outside the completed bounded async qualification. `save_hf` remains rejected because native saves do not produce HF output. |
| Auto resume / launch | Structured `[launch]` controls placement, priority, minimum runtime, mounts and Beaker restart policy. `run` delegates to `build_image_and_launch.sh --miles`; `status` reads a local receipt and queries Beaker. The workflow loads the latest completed Core checkpoint on retry when `auto_resume=true`. The config launcher supports one physical node; raw training files do not submit jobs. |
| Weight publication | `update_weight_buffer_size`, `core.stream_moe_export`, `core.weight_sync_mode` (`flattened` / `per_tensor`); `core.expert_publication` (`per_expert` / `fused`) publishes routed experts as HF per-expert slices or as one stacked tensor per layer and projection in the engine's fused layout, cutting the tensor count by the expert count; HF exports always keep per-expert slices. Colocation uses IPC. Core publishes every collection (`update_weights_interval=1`); skipping publication is rejected. Megatron disk-delta/p2p/rdt transports and pipeline-depth=2 are rejected rather than silently ignored. |
| W&B | `use_wandb`, `wandb_project`, `wandb_team`, `wandb_group`, `wandb_run_name`, `wandb_mode`, `wandb_dir`, `wandb_always_use_train_step`; the Core metrics adapter and rollout hooks determine reported metric definitions. |
| Dashboard / generations | `use_miles_dashboard`, `save_debug_rollout_data` and the custom rollout/eval logging hooks. The existing harnesses retain generations; the baseline sample-count knob remains absent. Structured `status` reports Beaker attempts and config identity, not the complete olmo-miles service/stage dashboard. |
| Contract measurements | `core.diagnostic_interval`, `check_weight_update_equal`, `core.max_train_rollout_logprob_abs_diff`. The last is the active-token **mean absolute** gap in the current implementation despite its legacy name; violations fail. Baseline's configurable warn/fail policy is not ported. |
| Fault tolerance | MILES rollout health/recovery controls pass through (`use_fault_tolerance`, `ft_components`, health intervals/timeouts). The baseline custom retry budget and stage deadline wrapper are not ported. This is not evidence of Core train-actor recovery; fault-injected endurance qualification remains separate. |
| Debugging | `debug_exit_after_rollout`, retained rollout loading, and scoring/diagnostic probes. `debug_disable_optimizer` and `debug_rollout_only` are rejected by the Core training entrypoint because its driver does not implement those shortcuts. |

Inherited FSDP knobs `gradient_checkpointing`, `attn_implementation`, and
`warmup_ratio` are rejected with pointers to the corresponding Core/shared settings,
rather than accepted and ignored.

The configuration audit also found native options with no Core implementation.
Both TOML compilation and direct native CLI parsing now reject optimizer-state
omission/reset, scheduler overrides, disabled advantage computation, the raw
`skip_actor_forward_only` flag, retained old actors, LoRA training, FSDP replication meshes, and
the generic `deterministic_mode` toggle. Supported resume restores optimizer and
scheduler state; use the separate serving determinism and collective diagnostic
controls when appropriate. These rejections prevent silent changes in the claimed
training contract; they do not add the missing capabilities.

`use_rollout_logprobs=true` changes the policy-ratio denominator; scoring-pass
eligibility is decided separately by the recipe checks above. On skipped updates,
the training forward supplies agreement diagnostics before the optimizer step.
TIS still compares detached trainer scores against rollout scores and can clip.
See the [scoring-pass merge review](measurements/miles-scoring-pass-merge-20260912.md)
for the GPU evidence, integration checks, and remaining test dependency gap.
The maintained async starter now selects trainer-scored old logprobs with TIS,
matching olmo-miles' async correction. Full-SFT starters also restore the historical
8 prompts × 8 responses. Earlier measurements used 16 × 4 and rollout logprobs
without TIS; preserve those configurations when interpreting their results. The
new configuration-driven combined exercise passed four updates; see the [workflow evidence](measurements/miles-researcher-workflow-20260911.md).
Core also repeats an initial evaluation after resume when it is enabled, whereas
olmo-miles suppresses that duplicate; account for the extra point and cost.

`save_debug_rollout_data` works. The inherited `save_debug_train_data` field has
no Core writer; `dump_details` must not be interpreted as retaining a Core trainer
payload. Use the adapter's contract JSONL and retained rollout diagnostics.
The [full parity audit](measurements/miles-feature-parity-audit-20260911.md)
separates these interface gaps from measured runtime capabilities.

The adapter owns `train_backend`, `olmo_core_config`, `data_source_path` and
`custom_async_data_buffer_path`; conflicting overrides are rejected. Custom reward,
generation and logging extension paths remain available through MILES. These are
trusted Python objects, not a registration of legacy open-instruct GRPO flags.

## Maintaining the parser contract

`open_instruct/miles/options.json` records the pinned native argparse actions,
including serving flags. Regenerate it inside the runtime with
`scripts/miles/snapshot_options.py`, review the diff, and update it with the runtime
pins. CPU tests check provenance and invalid values; runtime tests compare every
record and round-trip boolean switches plus representative serving, objective,
eval and JSON controls. A parser-only pass is not a new GPU training result.


## Compiler cache reuse

Core RL now enables persistent Triton caches by default. Structured run files use
`[compiler_cache] enabled = false` to opt out; low-level files use
`[core] compiler_cache = false`. The default shared directory is
`/weka/oe-training-default/olmo-miles/compiler-cache/tmp-30d/core-rl`, following
olmo-miles' WEKA TTL naming convention. Custom WEKA roots require a TTL component
and are validated before launch. See the [cache guide](miles-compiler-cache.md)
for measured cold/restored startup, worker compatibility, and remaining scope limits.
