# olmo-miles field inventory

Audit of all **136** `MilesSmokeConfig` fields at olmo-miles revision
`07887b783ab254577a6656168dc0e0d21aebfe3d` (`src/olmo_miles/config.py`).

See the [run-control guide](miles-run-controls.md) for semantics and examples, and
the [September 11 capability audit](measurements/miles-feature-parity-audit-20260911.md)
for runtime consumers, qualification, workflow gaps and corrected unsupported controls.
“Native pass-through” means the pinned parser exposes the corresponding option;
it is not a claim that every combination is qualified on Core. Baseline run-spec
sections for task recipes, conversion, HF export and Beaker lifecycle are covered
in that guide separately from this dataclass inventory.

| Baseline field | Core integration |
| --- | --- |
| `hf_checkpoint` | Native pass-through: `miles.hf_checkpoint`. |
| `megatron_checkpoint` | Replaced: initialize from HF; resume from native Core `miles.load`. |
| `prompt_data` | Native pass-through: `miles.prompt_data`. |
| `output_dir` | Split: `miles.save`, debug-rollout paths, W&B directory and launch output paths. |
| `trainer_num_nodes` | `miles.actor_num_nodes`. |
| `placement_mode` | `miles.colocate`; both normal wrappers keep the trainer resident. Core optional trainer offload remains unsupported; full-model Core colocation is unqualified. |
| `rollout_num_gpus` | Native pass-through: `miles.rollout_num_gpus`. |
| `rollout_gpus_per_node` | `miles.num_gpus_per_node`. |
| `num_gpus` | `miles.actor_num_gpus_per_node`. |
| `rollout_tensor_parallel_size` | `miles.rollout_num_gpus_per_engine`. |
| `rollout_expert_parallel_size` | `miles.sglang_ep_size`. |
| `inference_ep_diagnostics` | Separate diagnostic scripts; no generic run toggle. |
| `dataset_profile` | Data preparation workflow; not a MILES argument. |
| `rl_manifest` | Prepare/adopt immutable JSONL through data tooling; no direct baseline-manifest loader. |
| `expert_parallel_size` | `core.expert_parallel_size`. |
| `trainer_backend` | Replaced by architecture-selected Core MoE or standard trainer; no Megatron compatibility/optimized switch. |
| `trainer_diagnostics` | `core.diagnostic_interval`; different measurement contract. |
| `activation_recompute` | `core.activation_checkpointing`. |
| `recompute_mode` | Core block recomputation on/off; no Megatron selective-mode translation. |
| `recompute_modules` | Unsupported: Megatron module names do not describe Core block recomputation. |
| `micro_batch_size` | `miles.micro_batch_size=1`; larger microbatches are rejected. |
| `use_rollout_logprobs` | Selects rollout probabilities as the shared loss anchor. Core still performs trainer scoring for agreement checks; this flag does not skip that pass. |
| `calculate_per_token_loss` | Native pass-through: `miles.calculate_per_token_loss`. |
| `accumulate_allreduce_grads_in_fp32` | Core optimizer/reduction implementation owns precision; no equivalent user switch. |
| `replay_rollout_data` | `miles.load_debug_rollout_data`. |
| `replay_rollout_data_subsample` | `miles.load_debug_rollout_data_subsample`. |
| `trainer_flash_attention_version` | `core.attention_backend` chooses the Core attention implementation (for example torch or flash_4). |
| `dynamic_batching` | `miles.use_dynamic_batch_size`; true is rejected for Core. |
| `data_pad_size_multiplier` | Native spelling exists but does not control Core model padding. Core defaults to 1 and accumulates unpadded samples; Megatron DeepEP padding is not required. |
| `num_rollouts` | `miles.num_rollout`. |
| `debug_exit_after_rollout` | Native pass-through: `miles.debug_exit_after_rollout`. |
| `debug_disable_optimizer` | Rejected: Core always performs its optimizer step. |
| `debug_rollout_only` | Rejected by Core training driver; use a separate serving probe. |
| `save_checkpoints` | Choose `miles.save` and `miles.save_interval`; omit the interval to disable periodic native saves. |
| `save_interval` | Native pass-through: `miles.save_interval`. |
| `save_retain_interval` | Not ported: native Core checkpoint retention policy needs a separate implementation. |
| `save_tokens_per_expert_interval` | Not ported: baseline custom token-per-expert save scheduling. |
| `async_save` | False only; native Core saves are synchronous. |
| `collect_dashboard` | `miles.use_miles_dashboard`. |
| `capture_generation_samples` | Use retained rollout dumps / dashboard and logging hooks; no automatic baseline sampler toggle. |
| `generation_samples_per_rollout` | Not ported as a generic bounded sample-count knob; dumps can retain all responses. |
| `rollout_batch_size` | Native pass-through: `miles.rollout_batch_size`. |
| `samples_per_prompt` | `miles.n_samples_per_prompt`. |
| `rollout_temperature` | Native pass-through: `miles.rollout_temperature`. |
| `rollout_seed` | Native pass-through: `miles.rollout_seed`. |
| `use_fault_tolerance` | Native rollout fault tolerance; custom baseline retry/deadline wrapper is not included. |
| `rollout_health_check_interval` | Native pass-through: `miles.rollout_health_check_interval`. |
| `rollout_health_check_timeout` | Native pass-through: `miles.rollout_health_check_timeout`. |
| `rollout_health_check_first_wait` | Native pass-through: `miles.rollout_health_check_first_wait`. |
| `rollout_recovery_max_attempts` | Not ported: baseline custom recovery budget, not a native MILES flag. |
| `rollout_recovery_mem_fraction_static` | Not ported: baseline recovery-time engine memory override. |
| `rollout_stage_timeout` | Not ported: baseline driver-stage deadline wrapper. |
| `rollout_health_diagnostics` | Not ported as a generic toggle; separate fault probes and native health logging. |
| `rollout_test_fault` | Separate fault-injection experiments; no public training toggle. |
| `determinism_probe_samples` | Separate retained-input serving/routing diagnostic tooling; no generic run toggle. |
| `determinism_probe_forward_trace` | Separate retained-input serving/routing diagnostic tooling; no generic run toggle. |
| `determinism_probe_cross_gpu` | Separate retained-input serving/routing diagnostic tooling; no generic run toggle. |
| `determinism_probe_retune_kda` | Separate retained-input serving/routing diagnostic tooling; no generic run toggle. |
| `determinism_probe_l2norm_inputs` | Separate retained-input serving/routing diagnostic tooling; no generic run toggle. |
| `sglang_enable_deterministic_inference` | Native pass-through: `miles.sglang_enable_deterministic_inference`. |
| `use_routing_replay` | Rejected: Megatron trainer-route replay; use rollout routing replay on Core. |
| `use_rollout_routing_replay` | `miles.use_rollout_routing_replay`; requires `miles.use_miles_router`. |
| `rollout_top_p` | Native pass-through: `miles.rollout_top_p`. |
| `rollout_top_k` | Native pass-through: `miles.rollout_top_k`. |
| `eval_temperature` | Native pass-through: `miles.eval_temperature`. |
| `eval_top_p` | Native pass-through: `miles.eval_top_p`. |
| `eval_top_k` | Native pass-through: `miles.eval_top_k`. |
| `eval_max_response_length` | `miles.eval_max_response_len`. |
| `sglang_chunked_prefill_size` | Native pass-through: `miles.sglang_chunked_prefill_size`. |
| `max_response_length` | `miles.rollout_max_response_len`. |
| `max_context_length` | Set `miles.rollout_max_context_len`, `rollout_max_prompt_len`, `sglang_context_length`, and `core.max_sequence_length` consistently. |
| `global_batch_size` | Native pass-through: `miles.global_batch_size`. |
| `max_tokens_per_gpu` | Rejected: not a Core dynamic-batching control; Core microbatch is fixed at one. |
| `learning_rate` | `miles.lr`. |
| `lr_decay_style` | Native pass-through: `miles.lr_decay_style`. |
| `lr_warmup_iters` | Native pass-through: `miles.lr_warmup_iters`. |
| `weight_decay` | Native pass-through: `miles.weight_decay`. |
| `adam_beta1` | Native pass-through: `miles.adam_beta1`. |
| `adam_beta2` | Native pass-through: `miles.adam_beta2`. |
| `adam_eps` | Native pass-through: `miles.adam_eps`. |
| `clip_grad` | Native pass-through: `miles.clip_grad`. |
| `kl_loss_coef` | `miles.kl_loss_coef` plus `use_kl_loss=true` and reference initialization; coefficient alone does not enable KL. |
| `entropy_coef` | Native pass-through: `miles.entropy_coef`. |
| `eps_clip` | Native pass-through: `miles.eps_clip`. |
| `eps_clip_high` | Native pass-through: `miles.eps_clip_high`. |
| `update_weight_buffer_size` | Native pass-through: `miles.update_weight_buffer_size`. |
| `update_weight_transfer_mode` | Only broadcast accepted. Core transport implementation selected with `core.weight_sync_mode`; colocated uses IPC. |
| `weight_export_mode` | Replaced by Core native HF export and `core.stream_moe_export`; Bridge is not used. |
| `check_weight_update_equal` | Native pass-through: `miles.check_weight_update_equal`. |
| `colocated_live_weight_export` | Core owns live export/IPC; no baseline Megatron monkey-patch switch. |
| `colocated_weight_update_pipeline_depth` | Only 1 accepted; Megatron pipeline overlap is not implemented by Core. |
| `fully_async` | Implemented bounded-async Core path; requires disaggregation and positive lag. Baseline wrapper enables TIS automatically; Core async starter instead selects rollout log probabilities. |
| `max_weight_staleness` | `core.max_policy_lag`, in optimizer steps; explicit native alias must agree. |
| `tis_clip` | Native pass-through: `miles.tis_clip`. |
| `tis_clip_low` | Native pass-through: `miles.tis_clip_low`. |
| `off_policy_correction` | Native TIS: `miles.use_tis`, `tis_clip`, `tis_clip_low`. ICEPOP requires a separately installed/qualified `custom_tis_function_path`; baseline helper is not bundled. |
| `max_train_rollout_logprob_abs_diff` | `core.max_train_rollout_logprob_abs_diff`; active-token mean absolute gap, fail on violation. |
| `policy_drift_action` | Core currently fails on configured threshold violation; warn mode is not ported. |
| `async_max_concurrent_samples` | Native pass-through: `miles.async_max_concurrent_samples`. |
| `async_data_buffer_capacity_factor` | Native pass-through: `miles.async_data_buffer_capacity_factor`. |
| `async_unused_samples_handler` | Native pass-through: `miles.async_unused_samples_handler`. |
| `rollout_submission_granularity` | Native pass-through: `miles.rollout_submission_granularity`. |
| `radix_cache` | Inverse control: `miles.sglang_disable_radix_cache`. |
| `hardware_profile` | Launch/profile selection and measured memory budgets; no automatic olmo-miles hardware-policy resolver. |
| `sglang_mem_fraction_static` | Native pass-through: `miles.sglang_mem_fraction_static`. |
| `sglang_server_concurrency` | Native pass-through: `miles.sglang_server_concurrency`. |
| `sglang_watchdog_timeout` | Native pass-through: `miles.sglang_watchdog_timeout`. |
| `sglang_max_running_requests` | Native pass-through: `miles.sglang_max_running_requests`. |
| `sglang_max_mamba_cache_size` | Native pass-through: `miles.sglang_max_mamba_cache_size`. |
| `sglang_mamba_radix_cache_strategy` | Native pass-through: `miles.sglang_mamba_radix_cache_strategy`. |
| `sglang_attention_backend` | Native pass-through: `miles.sglang_attention_backend`. |
| `sglang_page_size` | Native pass-through: `miles.sglang_page_size`. |
| `sglang_cuda_graph_backend_decode` | Native pass-through: `miles.sglang_cuda_graph_backend_decode`. |
| `sglang_cuda_graph_max_bs_decode` | Native pass-through: `miles.sglang_cuda_graph_max_bs_decode`. |
| `sglang_speculative_algorithm` | Native pass-through: `miles.sglang_speculative_algorithm`. |
| `sglang_speculative_num_draft_tokens` | Native pass-through: `miles.sglang_speculative_num_draft_tokens`. |
| `sglang_speculative_ngram_min_bfs_breadth` | Native pass-through: `miles.sglang_speculative_ngram_min_bfs_breadth`. |
| `sglang_speculative_ngram_max_bfs_breadth` | Native pass-through: `miles.sglang_speculative_ngram_max_bfs_breadth`. |
| `sglang_disable_overlap_schedule` | Native pass-through: `miles.sglang_disable_overlap_schedule`. |
| `use_miles_router` | Native pass-through: `miles.use_miles_router`. |
| `sglang_router_policy` | Native pass-through: `miles.sglang_router_policy`. |
| `router_cache_threshold` | Native pass-through: `miles.router_cache_threshold`. |
| `router_balance_abs_threshold` | Native pass-through: `miles.router_balance_abs_threshold`. |
| `router_balance_rel_threshold` | Native pass-through: `miles.router_balance_rel_threshold`. |
| `code_service_mode` | Launch/environment provisioning for open-instruct code verifiers; no olmo-miles per-job provisioning switch. |
| `code_service_workers` | Code-service deployment setting; not a trainer argument. |
| `code_service_source_revision` | Image/service provenance setting; not a trainer argument. |
| `eval_interval` | Native pass-through: `miles.eval_interval`. |
| `skip_eval_before_train` | Native pass-through. Core repeats restored initial-state eval when false; baseline wrapper automatically skips initial eval on resume. |
| `seed` | Native pass-through: `miles.seed`. |
| `comparison_id` | Experiment metadata / W&B group or run name; no native comparison_id argument. |
| `wandb_project` | `miles.wandb_project` plus explicit `use_wandb=true` to enable tracking. |
| `wandb_team` | Native pass-through: `miles.wandb_team`. |
| `wandb_group` | Native pass-through: `miles.wandb_group`. |
| `wandb_mode` | Native pass-through: `miles.wandb_mode`. |
| `wandb_dir` | Native pass-through: `miles.wandb_dir`. |
