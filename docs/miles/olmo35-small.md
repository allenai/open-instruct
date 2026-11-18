# OLMo 3.5 Small (A4B) on B300

Measured MILES RL layouts for OLMo 3.5 Small ("Hero Small", A4B,
`olmo35-small-e256k8-sh2x`) on `ai2/holmes` B300 nodes. Configs:
[two-node-mech](../../configs/miles/examples/small/two-node-mech.toml),
[four-node](../../configs/miles/examples/small/four-node.toml) and
[eight-node](../../configs/miles/examples/small/eight-node.toml). These are timing runs
(30 updates, no checkpoints), not learning results.

## Model and start checkpoint

- Start checkpoint: H203 SFT export
  `/weka/olmo-3p5-post-checkpoints/abhishekr/a4b-gpu/a4bgpu-s34521/hf_step18410`
  (`Olmo3MoeForCausalLM`, bf16, 72.58B parameters / 145.16 GB).
- 40 layers: 35 KDA plus full attention at layers 7, 15, 23, 31 and 39 (NoPE, head_dim 256);
  256 routed experts top-8 (latent MoE dim 768, `moe_intermediate_size` 3072), shared expert,
  dense layer 0; hidden 1536; untied LM head.
- Export notes: the `emo_*` fields are all null in this SFT export (the base export has
  EMo 8/256/256), so it is used as-is with no config edit. The HF config has
  `vocab_size` 100278 (LM head `[100278, 1536]`); 100352 is Core's padded size.
- Data: robertb's H196 Dolci inputs
  (`/weka/olmo-3p5-post-checkpoints/robertb/miles/h196-dolci-control-20261113/inputs/`,
  84,577 rendered rows). Its rendering was made for Tiny but uses the same tokenizer and chat
  template as this export. There is no held-out split, so do not set `training.eval_interval`.
- Code verifier needs `CODE_EXECUTION_TOKEN`. Each recipe maps it to a placeholder
  Beaker secret name under `[launch.secrets]`; replace that name with a secret in the
  selected workspace. Never put the token value in the TOML. The historical timing
  runs omitted this credential: code samples were ungraded and their groups dropped
  (3-36 groups per update). Runs with code grading enabled have a different effective
  workload; remeasure throughput rather than assuming the historical timings apply.

## Layout and memory

| Config | Nodes | Trainer | Engines | Judge |
|---|---|---|---|---|
| two-node-mech | 2 | 8 GPUs, EP8 | 7 × TP1 | 1 × Qwen3-32B |
| four-node | 4 | 8 GPUs, EP8 | 23 × TP1 | 1 × Qwen3-32B |
| eight-node | 8 | 16 GPUs, 2 nodes × EP8 | 47 × TP1 | 1 × Qwen3-32B |

- Trainer (EP8, 4 nodes): peak reserved 220.5 GiB of 268.6 GiB per B300 (rank 0 peak
  allocated 209.7 GiB). With 2 × EP8 (8 nodes): 169.7 GiB. Packs are 34,816 tokens, below the
  OLMo-core no-grad Triton SwiGLU int32 overflow at 43,690 tokens per forward.
- Engine at R=96: weights 135 GB; KDA state for 128 slots 17.9 GB (SSM 17.64 + conv 0.21);
  KV for 3,342,336 tokens 63.8 GB (K 31.88 + V 31.88); about 50 GB free after pools at
  `mem_fraction_static` 0.88.

## Settings that matter

- `sglang_attention_backend = "trtllm_mha"` for the 5 full-attention layers. On one engine with
  96 forced 4K decodes: 2,881 tok/s vs 2,473 with triton. `fa4` crashes the server: "SM100
  forward with head_dim=256 does not support seqused_q/seqused_k".
- Tuned fused-MoE tiles (`SGLANG_MOE_CONFIG_DIR`, see
  [moe_configs](../../configs/miles/moe_configs/README.md)): kernel -9% at batch 96, decode
  +1.6% at R=96. The tiles only take effect in an image built with this branch's
  `configs/miles`; an older image has no file at that path.
- `OLMO_SGLANG_CORE_COMPAT = "rounding"`. HF vs SGLang greedy parity: mean |Δlogprob|
  0.014-0.019, argmax agreement 98-99%.
- Admission R=96: `sglang_max_running_requests` 96, `sglang_max_mamba_cache_size` 128,
  `sglang_server_concurrency` 192, `sglang_cuda_graph_max_bs_decode` 96, producer budget
  `async_max_concurrent_samples` = 96 × engines.
- Judge: `timeout = 600` and `chat_template = "qwen3-no-thinking"`. The judge bindings use the
  hashed verifier ids in the H196 `verifiers.json`.
- `two-node-mech` enables the initial serving-weight comparison and repeats the
  standalone/training scoring check every update. These qualification checks add
  overhead relative to the original mechanics run and remain off in timing recipes
  (which retain the default startup scoring check).
- The four/eight-node recipes request 4h/3h of protected runtime, respectively,
  based on historical startup and update times plus headroom, with an 8h timeout.
  Revisit these budgets when enabling code grading or other additional work.

## Results

Updates 10-29 unless noted, 128 prompts × 8 samples, max response 32,768 tokens.

| Run | Layout | s/update (wall median) | Gen wait | Train | Publication | Engine tok/s | Notes |
|---|---|---|---|---|---|---|---|
| timing4 | 4-node, R=96 | 298.6 (u10-24) | 109.6 | 171.2 | 8.9 | 2,528 | Failed at update 25: `/weka/oe-training-default` full |
| timing4n | 4-node, R=96 | 309.0 | 125.7 | 174.7 | 9.1 | 2,524 | 30/30, clean exit |
| timing4n-r64 | 4-node, R=64 | 398.9 | 217.5 | 163.8 | 7.2 | 1,946 | 29% slower than R=96 |
| timing8 | 8-node, R=96 | 138.8 overall; ~181 steady | 0.3 | 97.5 | 18.0 | 2,371 | u10-23 trainer-bound on buffered work (~121 s); u24-29 generation-bound (median 181 s) |

- Trainer: ~6.9K tok/s/GPU at EP8 on 4 nodes; ~6.0K tok/s/GPU with 2 × EP8; about 8.6-8.9M
  model tokens per update.
- Response lengths are long: the median of per-update mean accepted lengths is about 8.1-8.4K
  tokens, truncation is 0-6%, and about 25% of groups are zero-std.
- Startup to the end of update 0: 1,953-2,429 s on 4 nodes, 1,543 s on 8 nodes.
- timing8 completed all 30 updates, then exited 1 at teardown because the final generation drain
  hit its 900 s timeout. Terminal refresh cleanup now cancels unused generation through
  the existing backend shutdown API before closing weight transport and disposing
  servers. This change has CPU runtime regression coverage; a clean Small GPU run
  is still required. Evaluation and final HF export retain their generation-drain
  boundaries because they may reuse live serving resources.

## What did not work or was not tested

- `fa4` attention: crashes on head_dim 256 (above).
- R=64: slower (399 vs 309 s/update); engines gain throughput up to R=96. In single-engine
  forced decodes, R=144 matched R=96 and R=192 gave +19% total tok/s with each request ~40%
  slower.
- EP4 trainer: untested.

## Code and runtime

The original timing runs needed no Small-specific adapter or weight-sync changes. timing4 ran on image
01M7A0QM57N9499PGFAT73SMF0 (open-instruct a87a0feac). timing4n and r64 ran on image
01M7GA94Q6GKMP2MV2N36YE83Q, built from open-instruct 5c6645da. That is main 28c18166 plus an
SGLang pin with an Olmo 3 sliding-window KV backport, which Small (no sliding-window layers)
does not use. `plan` on main resolves the timing4n config identically to 5c6645da. These
examples have not been run on an image built from main.

## Qualification before a learning run

Copy `two-node-mech.toml` to ignored `runs/`, set a fresh output path and the actual
workspace/secret names, then inspect `python -m open_instruct.miles plan runs/small-mech.toml`
and `python -m open_instruct.miles validate runs/small-mech.toml` before using the
[committed-image launcher](launching.md). Build the image from the revision being
qualified; `plan` equality cannot establish runtime equality.

Retain the Beaker experiment link, image ID, submitted TOML and completion artifacts.
Verify all four updates, successful code grading, passing initial weight comparison,
standalone/training score agreement, publication acknowledgments and a clean exit.
Inspect `refresh_scores` for mixed-version responses and their historical-prefix and
latest-forward log-ratio diagnostics; see [Core diagnostics](core.md) and
[mixed-policy refresh](async-pipeline.md#mixed-policy-refresh). Those ratios include
policy drift and are not an isolated numerical-parity measurement.

For publication debugging, enable `core.diagnostic_interval = 1` with
`miles.check_weight_update_equal = true` in the run copy to exercise the existing
snapshot/reset/republish checks after optimizer updates. This adds a second transfer
and extra gradient/update diagnostics. The check detects incomplete transfers over
its selected tensors; it does not independently prove the export mapping. Keep a
separate Core-versus-serving score comparison on identical tokens, prefixes and
weights as a qualification requirement. HF-versus-SGLang greedy agreement and an
always-zero training metric do not establish that comparison.

Longer learning runs additionally need native checkpoint/save-resume and evaluation
checks. The timing recipes deliberately disable saving and have no held-out split.

## Open items

- R=160 test (`kf-hero-miles-timing4n-r160`): R=160, server concurrency 320, 192 KDA slots,
  KV pool auto-sized (3,823,488 tokens). Still running at the time of writing; W&B
  <https://wandb.ai/ai2-llm/open_instruct_internal/runs/dm2b8jj8>.
- Judge latency is not logged per call, so the judge's share of generation wait is unmeasured.
- `train_rollout_logprob_abs_diff` reads 0.0 on every update. It is probably not computed in this
  configuration; unverified.
