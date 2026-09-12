# Configuration reference

This is the maintained help for the MILES + OLMo-core wrapper. Start from a
[structured example](../../configs/miles/examples/README.md), then inspect `plan`.
The tables below are generated; edit reference-help.json or source definitions
and run `python -m scripts.miles.generate_docs`.

## Interface and precedence

`python -m open_instruct.miles {plan,validate,train,run,status} run.toml` accepts
repeatable `--set SECTION.KEY=TOML_VALUE` and `--debug`. Structured files require
schema_version=1 and model/data/output sections. Low-level files contain [core]
and [miles] and support only plan/validate/train. Structured validate is CPU-safe;
low-level validate invokes the installed native parser. Neither certifies GPU fit.

```bash
python -m open_instruct.miles plan configs/miles/examples/grpo-async-disaggregated.toml \
  --set training.num_rollouts=20 \
  --set 'tracking.wandb_group="my-comparison"'
```

Repeated assignments to the same override key use the last value. Structured
aliases and explicit Core/native options targeting one resolved value must agree;
conflicting values fail. Native aliases cannot be supplied twice under different
spellings. There is no implicit environment expansion or configuration inheritance.
Booleans are unquoted true/false; strings need TOML quotes protected by shell quotes.
Lists and inline tables are encoded according to the pinned native parser.

Python callers may use RunSpec.load(path).compile() or
RunConfig(CoreConfig(...), miles_options). The latter assumes prepared inputs and
has no launch/data workflow. User-input failures raise InputError (a ValueError
subclass); the CLI prints a field-oriented error and exits 2. --debug includes the
traceback. Preparation checks that need data/model files run where those are mounted.

## Which section to edit

| Section | Purpose |
|---|---|
| model / conversion / output | Input identity, preparation and final artifacts |
| data | Tasks, immutable manifests or prepared rows and reward configuration |
| launch | Allocation, mounts, secrets and scheduling |
| training | Collection count, evaluation and saving cadence |
| trainer | Trainer node/GPU geometry, EP, microbatching and recomputation |
| inference | Engine topology, sampling geometry, lengths and SGLang admission |
| optimizer | Learning rate, Adam, clipping, entropy and KL |
| async | Buffering, lag and importance correction |
| tracking | W&B and retained reporting |
| runtime / compiler_cache | Adapter execution and compiler reuse |
| judges / rubrics / judging | Named service identity and verifier bindings |
| core / miles | Explicit backend controls and advanced native escape hatch |

## Defaults and interactions

There are three different sources of values: raw CoreConfig/parser defaults,
structured workflow defaults, and explicit example recipe choices. The generated
Core and native tables identify raw defaults; `plan` shows the effective structured
configuration and leaves unspecified native runtime choices unresolved.

Structured runs default to 8 prompts × 8 responses, global batch equal to the
collection, 100 collections, microbatch one, LR 1e-6 with constant schedule,
Adam betas 0.9/0.95 and epsilon 1e-8, weight decay zero, gradient clip 1, PPO clip
0.2/0.28, no GRPO standard-deviation normalization, KL/entropy coefficients zero.
Do not confuse PPO clipping with TIS clipping or reward normalization.

Context defaults to 6144 tokens during structured compilation. max_context_length
sets Core, SGLang and rollout context together; response must be smaller to leave
prompt space. max_total_tokens defaults to at least 524288 and at least engine
admission × context. This is a requested pool capacity, not measured GPU allocation.

Synchronous colocation is the implicit default; async requires explicit resident
disaggregation. Structured async enables TIS unless rollout log probabilities are
explicitly the anchor. TIS and use_rollout_logprobs cannot both be enabled. Async
uses buffer factor 2, retry, group submissions, and requires an explicit valid Core
lag allowance. The maintained async example selects lag one. KL > 0 enables the
reference pass with the prepared starting model unless ref_load is supplied.

Saving defaults to the final collection; save_checkpoints=false disables cadence
and conflicts with explicit save_interval. Held-out data enables initial and
periodic evaluation (interval 20, greedy, one response), using the training response
cap unless overridden. Tracking stays disabled unless explicitly enabled or a
non-disabled wandb_mode is selected. Example offline tracking is an explicit choice.

Core's raw packing default is false and row_specialization is static. The async
example enables packing and dynamic specialization. Replay is independent:
use_rollout_routing_replay requires use_miles_router; structured compilation supplies
the latter when omitted. The Megatron use_routing_replay option is not the Core switch.

For conditional constraints see [run controls](run-controls.md), [packing](sequence-packing.md),
[topology](topology.md), and [managed judges](managed-judges.md). For all native
flags, choices and source help see the [native appendix](native-options.md).

<!-- Generated by python -m scripts.miles.generate_docs; edit reference-help.json or source definitions. -->

## Workflow fields

| Field | Type, default and behavior |
|---|---|
| compiler_cache.diagnostics | Boolean; maps to core.compiler_cache_diagnostics (default false). |
| compiler_cache.enabled | Boolean; maps to core.compiler_cache (default true). |
| compiler_cache.restore | Boolean; maps to core.compiler_cache_restore (default true). |
| compiler_cache.shared_root | Shared cache path; maps to core.compiler_cache_root. Default shared TTL root documented in cache guide. |
| conversion.hf_output | Prepared HF descriptor/export path; default output.root/prepared/hf. |
| data.eval_prompt_data | List alternating dataset names and held-out JSONL paths; paths resolve relative to TOML. |
| data.prompt_data | Prepared training JSONL path; exclusive data selector. Pair with reward_config. |
| data.recipe | Rejected: named olmo-miles recipes are not ported. Use tasks or an immutable manifest. |
| data.reward_config | Trusted verifier registry path for prepared inputs. |
| data.rl_manifest | Path to immutable prepared manifest with supported verifier contracts; exclusive data selector. |
| data.seed | Nonnegative integer, default 17; controls preparation and default rollout seeds. |
| data.shuffle | Boolean, default true; shuffle prepared training data. |
| data.tasks | Nonempty array of unique task tables; cannot combine with another data selector. |
| data.tasks[].eval_count | Positive integer number of held-out prompts; omit for no held-out rows from this task. |
| data.tasks[].prompt_wrapper | Task rendering wrapper: none, auto or open_instruct_rlzero_answer; default none. See data guide. |
| data.tasks[].task | Required named task: gsm8k, math, ifeval or multiplication. |
| data.tasks[].train_count | Positive integer number of training prompts. At least one task must select training data. |
| judges.NAME.backend | Managed only: sglang (default and only supported value). |
| judges.NAME.chat_template | Managed only: qwen3-no-thinking (default and only supported template). |
| judges.NAME.endpoint | External only: required HTTP(S) endpoint without embedded credentials. |
| judges.NAME.gpus | Managed only: positive GPU count, default 1; must equal tensor_parallel_size. |
| judges.NAME.max_concurrent_calls | Positive client concurrency, default 16. |
| judges.NAME.max_context_length | Positive context tokens, default 40960. |
| judges.NAME.mode | Required managed or external; determines service ownership. |
| judges.NAME.model | Required model identifier used by the judge. |
| judges.NAME.prepared_dir | Managed only: required absolute pre-cached judge directory; prepare before GPU allocation. |
| judges.NAME.revision | Managed only: required immutable 40-character hexadecimal model revision. |
| judges.NAME.tensor_parallel_size | Managed only: positive TP count, default 1. |
| judges.NAME.timeout | Positive seconds per judge request, default 120. |
| judging.bindings.VERIFIER.judge | Required name of a declared judge. |
| judging.bindings.VERIFIER.rubric | Required name of a declared rubric. |
| launch.auto_resume | Boolean, default true; Beaker restart policy. Multi-node/managed jobs require false. |
| launch.budget | Beaker budget string; default ai2/oe-other. |
| launch.cluster | Cluster string; default ai2/holmes for GPU runs. CPU-only WEKA work requires ai2/saturn. |
| launch.coordination | Multi-node coordination timeouts table. |
| launch.coordination.heartbeat_timeout | Positive seconds for replica heartbeat timeout; default 120. |
| launch.coordination.startup_timeout | Positive seconds to await coordinated startup; default 1200. |
| launch.env | Environment-name to nonsecret string mapping, default empty; rank/Ray/CUDA variables are reserved. |
| launch.gpus_per_replica | Positive integer physical GPUs per replica; inferred from topology if omitted. |
| launch.min_runtime | Beaker minimum runtime duration; default 1h. |
| launch.priority | low, normal, high or urgent; default urgent. |
| launch.secrets | Environment-name to Beaker secret-name mapping, default empty; no overlap with env. |
| launch.shared_memory | Shared memory quantity string; default 200 GiB. |
| launch.timeout | Beaker task timeout duration string; default 3h (examples override). |
| launch.weka_mounts | Array of filesystem/mount tables; default oe-training-default at /weka/oe-training-default. |
| launch.weka_mounts[].mount_path | Absolute/resolved destination; distinct nonoverlapping directories below root. |
| launch.weka_mounts[].weka | WEKA filesystem name; unique per mount. |
| launch.workspace | Beaker workspace string; default ai2/open-instruct-dev. |
| model.format | hf (default) or olmo_core. Megatron checkpoint inputs are rejected. |
| model.hf_template | Required only for native olmo_core input; compatible HF architecture/tokenizer template path. |
| model.reference_hf | Rejected baseline conversion-validation field; use miles.ref_load for a frozen KL reference. |
| model.source | Required checkpoint path, resolved relative to the run file; input remains read-only. |
| name | Required run identifier: letters, digits, dots, underscores and hyphens; starts with a letter/digit. |
| output.export_hf | Boolean, default false. Export final HF weights after training. |
| output.hf_dir | Final HF export path; default output.root/export-hf. |
| output.root | Required fresh run directory. Resume requires matching recorded specification; completed runs cannot be overwritten. |
| rubrics.NAME.max_response_tokens | Positive judge output reservation, default 2048; smaller than judge context. |
| rubrics.NAME.profile | Required supported open-instruct/general-* profile; see managed judges guide. |
| rubrics.NAME.temperature | Nonnegative sampling temperature, default 1.0. |
| schema_version | Required integer 1. |

## Structured aliases

Names below work in the organizational sections training, trainer, inference, optimizer, async, tracking and runtime unless stated otherwise. Use the semantically appropriate section shown in examples. These sections are not closed independent schemas: Core field names resolve to Core, other native names resolve to MILES.

| Alias | Resolved target |
|---|---|
| activation_recompute | core.activation_checkpointing |
| collect_dashboard | miles.use_miles_dashboard |
| comparison_id | miles.wandb_group |
| enable_mixed_chunk | miles.sglang_enable_mixed_chunk |
| eval_max_response_length | miles.eval_max_response_len |
| expert_parallel_size | core.expert_parallel_size |
| learning_rate | miles.lr |
| mamba_radix_cache_strategy | miles.sglang_mamba_radix_cache_strategy |
| max_response_length | miles.rollout_max_response_len |
| max_train_rollout_logprob_abs_diff | core.max_train_rollout_logprob_abs_diff |
| max_weight_staleness | core.max_policy_lag |
| num_gpus | miles.actor_num_gpus_per_node |
| num_rollouts | miles.num_rollout |
| replay_rollout_data | miles.load_debug_rollout_data |
| replay_rollout_data_subsample | miles.load_debug_rollout_data_subsample |
| rollout_expert_parallel_size | miles.sglang_ep_size |
| rollout_gpus_per_node | miles.num_gpus_per_node |
| rollout_tensor_parallel_size | miles.rollout_num_gpus_per_engine |
| router_balance_abs_threshold | miles.router_balance_abs_threshold |
| router_balance_rel_threshold | miles.router_balance_rel_threshold |
| router_cache_threshold | miles.router_cache_threshold |
| router_policy | miles.sglang_router_policy |
| samples_per_prompt | miles.n_samples_per_prompt |
| trainer_num_nodes | miles.actor_num_nodes |

| Special control | Behavior |
|---|---|
| disable_radix_cache | Boolean alias for miles.sglang_disable_radix_cache. |
| dynamic_batching | Boolean alias for miles.use_dynamic_batch_size; Core validation still constrains supported modes. |
| gpus | Only trainer/inference: trainer GPUs per node or total rollout GPUs, respectively. |
| max_context_length | Positive token limit applied to Core, SGLang and rollout context. |
| off_policy_correction | tis only; enables use_tis and requires use_rollout_logprobs=false. |
| placement_mode | colocated or disaggregated; resolves miles.colocate. |
| policy_drift_action | fail only; warn is rejected. |
| radix_cache | Boolean; inverse of miles.sglang_disable_radix_cache. |
| recompute_mode | full or off; resolves Core block activation checkpointing. |
| save_checkpoints | Boolean, default true; false removes save cadence and conflicts with explicit save_interval. |
| trainer_diagnostics | Boolean mapped to core.diagnostic_interval as 1 or 0. |
| trainer_flash_attention_version | 2, 3 or 4; selects corresponding Core flash backend, not a hardware qualification. |

## Core fields

These are dataclass defaults for raw CoreConfig. Structured compilation and example files can override them; null means unset.

| Field | Type | CoreConfig default | Meaning |
|---|---|---|---|
| core.max_train_rollout_logprob_abs_diff | float &#124; None | null | Fail when mean absolute active-token trainer/serving log-probability gap exceeds this value; null disables. Despite the name, this is a mean, not a maximum. |
| core.diagnostic_interval | &lt;class &#x27;int&#x27;&gt; | 0 | Interval for trainer contract diagnostics; zero disables periodic diagnostics. |
| core.replay_diagnostics | &lt;class &#x27;bool&#x27;&gt; | false | Retain expert-ID replay diagnostics; does not enable replay itself. |
| core.stream_moe_export | &lt;class &#x27;bool&#x27;&gt; | true | Stream MoE tensors during HF-layout publication to reduce export memory. |
| core.weight_sync_mode | &lt;class &#x27;str&#x27;&gt; | &quot;flattened&quot; | flattened batches tensor transfers; per_tensor is the rollback/reference transport. |
| core.row_specialization | &lt;class &#x27;str&#x27;&gt; | &quot;static&quot; | static specializes no-gradient SwiGLU on capacity; dynamic avoids capacity-specific compilation. Independent of arithmetic flags. |
| core.compiler_cache | &lt;class &#x27;bool&#x27;&gt; | true | Enable persistent compiler-cache lifecycle. |
| core.compiler_cache_root | str &#124; None | null | Shared cache root; null selects maintained default. WEKA custom paths require a TTL component. |
| core.compiler_cache_restore | &lt;class &#x27;bool&#x27;&gt; | true | Restore a compatible cache before worker startup. |
| core.compiler_cache_diagnostics | &lt;class &#x27;bool&#x27;&gt; | false | Retain detailed cache diagnostics. |
| core.checkpoint_profile | &lt;class &#x27;bool&#x27;&gt; | false | Record checkpoint-planning and writer timings. |
| core.checkpoint_thread_count | int &#124; None | null | Optional checkpoint writer thread bucket count; positive integer. |
| core.checkpoint_process_count | int &#124; None | null | Optional spawned checkpoint worker count; positive integer. |
| core.checkpoint_compact_storage | &lt;class &#x27;bool&#x27;&gt; | true | Compact tensor storage before native checkpoint writes. |
| core.checkpoint_dedup_save_to_lowest_rank | &lt;class &#x27;bool&#x27;&gt; | false | Use the lowest rank for duplicated checkpoint entries; independent writer policy switch. |
| core.checkpoint_constant_memory_planning | &lt;class &#x27;bool&#x27;&gt; | true | Use bounded-memory checkpoint planning; separate from async save (unsupported). |
| core.model_config | str &#124; None | null | Optional Core model factory configuration path for supported construction. |
| core.reward_config | str &#124; None | null | Trusted verifier registry path; structured workflow preparation supplies it. |
| core.expert_parallel_size | &lt;class &#x27;int&#x27;&gt; | 1 | Expert-parallel group size; must divide trainer world size. |
| core.attention_backend | &lt;class &#x27;str&#x27;&gt; | &quot;torch&quot; | Core attention implementation: torch, flash_2, flash_3 or flash_4; hardware/model qualification is separate. |
| core.activation_checkpointing | &lt;class &#x27;bool&#x27;&gt; | true | Recompute blocks during backward to reduce activation memory. |
| core.max_sequence_length | &lt;class &#x27;int&#x27;&gt; | 8192 | Trainer context limit in tokens; structured max_context_length sets trainer and serving limits together. |
| core.sequence_packing | &lt;class &#x27;bool&#x27;&gt; | false | Pack complete samples within each optimizer partition using document-isolated attention/KDA and replay alignment. |
| core.packing_max_tokens | int &#124; None | null | Maximum tokens per pack; null uses context limit. Must cover max_sequence_length; samples are never split. |
| core.max_policy_lag | &lt;class &#x27;int&#x27;&gt; | 0 | Maximum optimizer-step age at consumption; multiple updates per collection also consume this allowance. |
| core.router_aux_loss_weight | &lt;class &#x27;float&#x27;&gt; | 0.01 | Coefficient for native Core load-balancing auxiliary loss; with packing, use packed-forward local-batch semantics. |
| core.router_z_loss_weight | &lt;class &#x27;float&#x27;&gt; | 1e-05 | Router z-loss coefficient; replay does not itself disable router losses or gradients. |
| core.scoring_pass_required | &lt;class &#x27;bool&#x27;&gt; | false | Force standalone scoring even when the recipe permits skipping it. |
| core.scoring_check_interval | &lt;class &#x27;int&#x27;&gt; | 50 | Periodic standalone-versus-training score check; startup/resume also checks the first update. |
| core.scoring_check_tolerance | &lt;class &#x27;float&#x27;&gt; | 0.001 | Allowed absolute difference for the standalone/training scoring check. |
| core.expert_publication | &lt;class &#x27;str&#x27;&gt; | &quot;per_expert&quot; | per_expert exports separate HF expert slices; fused publishes stacked serving tensors. Disk HF export remains per-expert. |

## Unsupported olmo-miles controls

| Field | Replacement or limitation |
|---|---|
| accumulate_allreduce_grads_in_fp32 | Core owns reduction precision; this Megatron switch has no Core equivalent |
| capture_generation_samples | use save_debug_rollout_data or collect_dashboard; bounded sampling is not implemented |
| code_service_host | per-run code-service provisioning is not implemented |
| code_service_log | per-run code-service provisioning is not implemented |
| code_service_mode | provision the verifier service externally and pass its environment |
| code_service_port | per-run code-service provisioning is not implemented |
| code_service_python | per-run code-service provisioning is not implemented |
| code_service_source_revision | record externally provisioned service provenance |
| code_service_source_root | per-run code-service provisioning is not implemented |
| code_service_workers | provision the verifier service externally |
| colocated_live_weight_export | Core already owns live IPC export; there is no Megatron patch selector |
| dataset_profile | choose data.tasks, data.recipe or data.rl_manifest |
| determinism_probe_cross_gpu | use retained-input diagnostic scripts |
| determinism_probe_forward_trace | use retained-input diagnostic scripts |
| determinism_probe_l2norm_inputs | use retained-input diagnostic scripts |
| determinism_probe_retune_kda | use retained-input diagnostic scripts |
| determinism_probe_samples | use retained-input diagnostic scripts |
| fla_prewarm | use compiler_cache.enabled; generic FLA prewarming is not implemented |
| fla_prewarm_sequence_length | generic FLA prewarming is not implemented |
| generation_samples_per_rollout | use save_debug_rollout_data or collect_dashboard; bounded sampling is not implemented |
| hardware_profile | choose explicit Core/serving settings and launch.cluster; automatic hardware policy is not implemented |
| hf_checkpoint | use model.source; preparation supplies miles.hf_checkpoint |
| inference_ep_diagnostics | use the separate inference-EP diagnostics |
| megatron_checkpoint | use model.source/model.format or miles.load for a native Core RL resume |
| miles_train_script | this workflow owns the Core driver |
| no_start_ray | the launcher owns Ray startup |
| output_dir | use output.root |
| python_path | install code in the pinned runtime image; the launcher owns PYTHONPATH |
| recompute_modules | Core supports block activation_checkpointing, not Megatron selective modules |
| rl_manifest | use data.rl_manifest |
| rollout_health_diagnostics | use dedicated recovery probes; the baseline diagnostic wrapper is not installed |
| rollout_recovery_max_attempts | Core does not yet implement the baseline driver retry budget |
| rollout_recovery_mem_fraction_static | Core does not implement recovery-time memory overrides |
| rollout_stage_timeout | Core does not yet implement the baseline per-stage deadline |
| rollout_test_fault | use a dedicated fault-injection qualification, not an ordinary run |
| save_retain_interval | native Core checkpoint retention is not implemented |
| save_tokens_per_expert_interval | tokens-per-expert checkpoint capture is not implemented |
| skip_cuda_check | plan is CPU-safe; validate checks the installed runtime |
| start_code_service | per-run code-service provisioning is not implemented |
| trainer_backend | Core selects its native model backend; omit the Megatron optimized/compatibility switch |
| validate_miles_args | use the validate command |
| weight_export_mode | Core exports native HF tensors; use core.stream_moe_export |

## Example recipes

Generated from the actual structured TOMLs. These are recipe choices, not universal defaults or production qualification. GPU columns distinguish per-node trainers from total rollout GPUs.

| Example | Trainer GPUs | Rollout GPUs | Colocated | Prompts × responses | Global batch | Async | TIS | Packing | Collections |
|---|---|---|---|---|---|---|---|---|---|
| grpo-async-disaggregated.toml | 1 × 8 | 8 | False | 64 × 8 | 512 | True | True | True | 100 |
| grpo-basic.toml | 1 × 1 | 1 | True | 8 × 8 | 64 | False | False | False | 2 |
| grpo-disaggregated.toml | 1 × 2 | 1 | False | 8 × 8 | 64 | False | False | False | 100 |
| grpo-multitask.toml | 1 × 2 | 1 | False | 8 × 8 | 64 | False | False | False | 4 |
