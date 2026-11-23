# OLMo 3.5 Small (A4B) on B300

Measured MILES RL layouts for OLMo 3.5 Small ("Hero Small", A4B,
`olmo35-small-e256k8-sh2x`) on `ai2/holmes` B300 nodes. Configs:
[two-node-mech](../../configs/miles/examples/small/two-node-mech.toml),
[four-node](../../configs/miles/examples/small/four-node.toml) and
[eight-node](../../configs/miles/examples/small/eight-node.toml). The current recipes
use the Tiny batch/LR/loss baseline from PR #2271; historical measurements below
used the earlier settings. They run four mechanics or 30 timing updates without
checkpoints and do not establish learning quality.

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
- Code verifier needs `CODE_EXECUTION_TOKEN`. The four- and eight-node recipes map it to
  `robertb_MILES_PR2178_CODE_EXECUTION_TOKEN`, the shared code-service token secret in
  `ai2/olmo-instruct`. In another workspace, create or select that workspace's secret and
  put its name there. Never put the token value in the TOML. Check the first updates' driver
  log for `missing code-service bearer token` or 401/403 lines. The historical timing
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
- Engine at R=160 (previous four-node example): KDA state for 192 slots 26.7 GB (SSM 26.39 + conv
  0.31); KV pool auto-sized to 3,823,488 tokens, 72.9 GB (K 36.46 + V 36.46); 32.0 GB free
  after pools and 30.5 GB after CUDA graphs at `mem_fraction_static` 0.88.
- Engine at R=224 (four-node example): 256 KDA slots; KV pool auto-sized to 3,919,232 tokens
  at `mem_fraction_static` 0.92; 19.7 GB free after CUDA graphs; full-attention KV peak 71%.

## Settings that matter

- `sglang_attention_backend = "trtllm_mha"` for the 5 full-attention layers. On one engine with
  96 forced 4K decodes: 2,881 tok/s vs 2,473 with triton. `fa4` crashes the server: "SM100
  forward with head_dim=256 does not support seqused_q/seqused_k".
- Tuned fused-MoE tiles (`SGLANG_MOE_CONFIG_DIR`, see
  [moe_configs](../../configs/miles/moe_configs/README.md)): kernel -9% at batch 96, decode
  +1.6% at R=96. The tiles only take effect in an image built with this branch's
  `configs/miles`; an older image has no file at that path.
- `OLMO_SGLANG_CORE_COMPAT = "rounding"`. The attention trial's HF vs SGLang
  samples have mean |Δlogprob| 0.0155-0.0223 and argmax agreement 98.2-99.4%.
- Admission R=224 on four nodes: `sglang_max_running_requests` 224,
  `sglang_max_mamba_cache_size` 256, `sglang_server_concurrency` 448,
  `sglang_cuda_graph_max_bs_decode` 224, producer budget `async_max_concurrent_samples` =
  224 × engines, `sglang_mem_fraction_static` 0.92, no `sglang_max_total_tokens` key (RunSpec's
  default cap, R × context = 7,798,784 tokens, is above the 3,919,232-token memory-sized pool), and
  `core.refresh_request_timeout = 3600.0` (see [the timeout fix](#request-timeout-at-high-admission)).
  The eight-node example uses the same R=224 serving settings and 3600 s timeout, plus completed-buffer
  factor 1.5, `shared_memory = "400 GiB"` and `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=360000000000`:
  at factor 4 and Ray's 200 GB default, routing-replay payloads in the full completed queue filled the
  producer node's object store and delivery stalled past `refresh_request_timeout` (run failed at update 7).
- Judge: `timeout = 600` and `chat_template = "qwen3-no-thinking"`. The judge bindings use the
  hashed verifier ids in the H196 `verifiers.json`.
- `two-node-mech` enables the initial serving-weight comparison and repeats the
  standalone/training scoring check every update. These qualification checks add
  overhead relative to the original mechanics run and remain off in timing recipes
  (which retain the default startup scoring check).
- The four/eight-node recipes request 4h/3h of protected runtime, respectively,
  based on historical startup and update times plus headroom. They set a 72h/120h job
  timeout, respectively, so a multi-day run is one job; see [Long runs](#long-runs).

## Results

The table below was independently recomputed from retained Beaker artifacts on
November 18, 2026. Updates 10-29 unless noted, 128 prompts × 8 samples, max response
32,768 tokens. A cycle starts at `generation_wait.started_unix` and ends at
`publication.started_unix + publication.seconds` for the same update. Startup,
final drain and checkpoint/evaluation work are excluded. Each column is its own
median; component medians need not sum to the cycle median. Earlier headline cycle
figures (299/309/399/139 seconds) were not reproduced by this definition.

| Run | Layout | s/update (wall median) | Gen wait | Train | Publication | Engine tok/s | Notes |
|---|---|---|---|---|---|---|---|
| [timing4](https://beaker.org/ex/01M7EWK861JCP5Z0G5514S5XJ5) | 4-node, R=96 | 276.2 (u10-24) | 109.6 | 171.2 | 8.9 | 2,528 | Failed at update 25: `/weka/oe-training-default` full |
| [timing4n](https://beaker.org/ex/01M7HF4C8M7XC7ZPVC9QMD8X36) | 4-node, R=96 | 301.7 | 125.7 | 174.7 | 9.1 | 2,524 | 30/30, clean exit |
| [timing4n-r64](https://beaker.org/ex/01M7J0V7M8VHN4X1PJ43BQ9JZQ) | 4-node, R=64 | 396.4 | 217.5 | 163.8 | 7.2 | 1,946 | 31% slower cycle median than R=96 |
| [timing8](https://beaker.org/ex/01M7FV0QBE1CJ2656N5DA0JYY6), u10-23 | 8-node, R=96 | 116.0 | 0.3 | 98.1 | 17.7 | 2,371 | Drains buffered work; not a steady-state topology result |
| timing8, u24-29 | 8-node, R=96 | 185.5 | 73.8 | 96.0 | 18.3 | — | Only six late updates; shutdown subsequently failed |
| [timing4n-r160](https://beaker.org/ex/01M7NHRMV28ZX4S3HXD4BWAQVJ), u10-29 | 4-node, R=160 | 182.7 | 0.3 | 163.8 | 10.0 | — | Early buffered work; 50 updates total |
| timing4n-r160, u30-49 | 4-node, R=160 | 248.9 | 64.8 | 173.3 | 9.8 | — | Later window; shutdown subsequently failed |
| timing4n-r160, u40-49 | 4-node, R=160 | 246.8 | 69.5 | 172.3 | 10.3 | 3,414 | 0 retractions, full KV peak 55% |
| [timing4n-r224](https://beaker.org/ex/01M7R62YRQ4PH9T5YF78MQJB8T), u30-39 | 4-node, R=224 | 350.9 (mean 401) | 171.3 | 165.3 | 11.7 | 4,094 | 883 timed-out group attempts; 40/40, then failed final drain |
| [timing4n-r256](https://beaker.org/ex/01M7R8JNKYN72WEHXY5JGGF9MT), u30-36 | 4-node, R=256 | 313.0 | 160.7 | 152.6 | 9.8 | 4,395 | 622 timed-out group attempts; KV 100%, 51 retractions; aborted at 37/40 |
| [timing4n-r224-t3600](https://beaker.org/ex/01M7V230Y5TM2K6J2X400XGG02), u30-39 | 4-node, R=224, request timeout 3600 s | 215.8 (mean 212.3) | 32.7 | 170.1 | 10.8 | 4,100 | 0 timeouts, 0 stale drops, KV peak 71%, 0 retractions; 40/40, then failed final drain (900 s); basis of `four-node.toml` ([W&B](https://wandb.ai/ai2-llm/open_instruct_internal/runs/6bo8weoa)) |

The engine token rates above retain the original serving summaries; cycle and stage
durations were audited directly. `driver_timing.jsonl`, `rollout_flow.jsonl`,
`pipeline_occupancy.jsonl`, `publication.jsonl`, all-rank `training_contract_rank*.jsonl`
and actor metric events are in each experiment's `run/checkpoints/` results.
The R96 and R64 runs pass `scripts/miles/analyze_throughput.py` with `warmup=10`,
including all eight trainer ranks and completed workflow checks. All 16 timing8
ranks and all eight R160 ranks completed their requested 30/50 optimizer steps,
respectively, without skipped updates; those experiments still failed at shutdown.

### Topology interpretation

- Four-node R96: trainer rollout-wait share is **35.5%** of measured driver-stage
  time over u10-29. R64 increases it to **54.2%**. The R96 trial delivers about
  30.6K useful response tokens/s versus 22.4K for R64, a 37% improvement under the
  strict analyzer. Publication is about 9 seconds/update and is not the largest cost.
- Eight-node R96: early compilation and slow trainer updates build a generation
  backlog. Median completed queue depth falls from 384 groups over u10-23 to zero
  over u24-29. Late trainer wait is **36.5%**; doubling both trainer and inference
  capacity did not eliminate rollout starvation. Late stale-token discard fraction
  is **9.3%** of delivered plus stale-discarded response tokens. Mean sample policy
  age has a per-update median of 8.0 versions in u10-23 and 4.9 in u24-29, versus
  2.2 for four-node R96. These are different operating regimes.
- Four-node R160: the later u30-49 window reduces wait to **24.9%**, with median
  sample age 3.3 versions and no stale-token discards in that window. Its early
  near-zero wait is not a steady-state result. It failed the same final-drain boundary
  and was not repeated; the example moved on to R=224 with a 3600 s request timeout
  (see [the timeout fix](#request-timeout-at-high-admission)), which is also a single,
  unqualified run.
- HTTP permit wait and generation-admission wait are zero at the median in these
  windows; unfinished work is near its configured budget. This does not isolate
  the judge's contribution. Rollout wait includes production, filtering and grading,
  and cannot be attributed entirely to policy GPU inference without more evidence.
- Even the successful four-node R96 run drains for 741 seconds at shutdown and
  leaves 29.4M completed response tokens unused (10.5% of accounted delivered plus
  stale-discarded plus terminal-unused tokens). Warm-cycle throughput is not a
  complete job-cost metric.

The optimization evidence supports the attention backend, tuned tiles and increasing
admission above 64. It does not establish an optimal trainer/engine split. Compare
longer runs with the same workload and objective, useful tokens per allocated GPU-hour,
policy age, stale/unused work and judge time before selecting a production topology.

### Recipe alignment

All three Small recipes, including mechanics, now use the Tiny learning baseline
from [PR #2271](https://github.com/allenai/open-instruct/pull/2271):

| Setting | Current Small baseline | Historical Small measurements |
|---|---|---|
| Prompts × responses per prompt | 256 × 8 | 128 × 8 |
| Responses per optimizer update | 2,048 | 1,024 |
| Learning rate | 6e-6, constant, no warmup | 1e-6, constant, no warmup |
| Loss weighting | Token mean | Response mean |
| Group-standard-deviation normalization | Off | Off |

This is the starting baseline for Small; the historical throughput results do not
measure this combination. The serving geometry is the submitted timing4n-r224-t3600 (four-node) and real-8node-b (eight-node)
geometry. Paths, credentials, runtime budgets and mechanics qualification controls
also have the documented changes above.

The completed-buffer factor is four (1,024 prompt groups / 8,192 responses) except on eight
nodes (1.5: 384 groups / 3,072 responses). Unfinished producer budgets are 672, 5,152 and 10,528
for the two/four/eight-node layouts, respectively (0.328, 2.516 and 5.141 optimizer batches). The
nominal sum of unfinished budget plus completed-buffer capacity is therefore 4.328, 6.516 and
6.641 batches. Finished siblings can live outside that accounting; these figures
are not policy-age guarantees. Inspect measured age and useful throughput when
tuning the new baseline.
Do not copy Tiny's 12.5B inference preset, evaluator profile or cache geometry into
Small: those are model-specific. Checkpoint/export/evaluation differences here are
intentional for disposable timing runs, not a learning-run baseline.

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

The u40-49, r224, r256 and r224-t3600 rows come from the same artifacts via a separate analysis
(engine rates are sampled engine means; r160 sampled 19 of 23 engines), not from the audit above.

### Request timeout at high admission

At batch 1,024 (128 × 8), R=224 and R=256 raise engine throughput by 20% and 29% over
R=160, but each request decodes more slowly: about 21.9 tok/s at R=160, 18.7 at R=224
and 17.5 at R=256. A 32,768-token response then needs about 1,750-1,870 s before queueing
and refresh pauses, which exceeds `core.refresh_request_timeout` = 1800 s. Once the policy
produces long responses (from about update 30), whole groups time out and are retried
(883 timed-out group attempts for R=224 over updates 31-40, 622 for R=256 over 31-37;
none at R=160). Updates became slower (medians 351/313 vs 241-247 s), truncated responses
disappeared from training (0.7%/0.0% vs 3.6-3.8%), R=256 filled the KV pool (100%,
51 retractions) and aborted at update 37 of 40 when the rollout failure rate stayed
above 50%.

The fix is `core.refresh_request_timeout = 3600.0` (the per-request deadline in refresh
rollout; nothing else caps it). A timeout discards the whole group of eight, so a few
slow 32K responses waste the other seven samples too. With 3600 s, R=224 (timing4n-r224-t3600,
updates 31-40, 1-indexed; the table's u30-39) ran at 212 s/update mean vs 243 s at R=160 (-13%)
and 401 s mean at R=224 with 1800 s: 0 timeouts, 0 stale drops, truncation 3.55% (r160 3.79%), and generation
wait fell to 33 s, so the run is now train-bound. 4.7% of completed samples took 1,800-2,000 s
end to end. Risk: the longest (>=30K-token) samples finish at policy lag 8-10 (median 9) against
`max_policy_lag` = 12; if responses grow later in a long run, stale drops are the next limit.
This was measured at batch 1,024; at the shipped 2,048 each update takes longer, so a sample
should span fewer versions, but remeasure. The run completed 40/40 updates and then, on that pre-#2280 image (`5c6645da`), exited 1
when the final generation drain hit `engine_drain_timeout` (900 s). On main, terminal refresh
cleanup cancels unused generation instead of draining (above), while evaluation and final HF
export retain generation-drain boundaries. The four- and eight-node examples set
`engine_drain_timeout = 3600.0` to match `refresh_request_timeout` so those drain boundaries
cover long responses when enabled (untested in a run).
One run per arm; no seed repeat. The eight-node R=224 settings come from one 30-update run
(kf-hero-miles-real-8node-b, H258 start).

## What did not work or was not tested

- `fa4` attention: crashes on head_dim 256 (above).
- R=224 at the default 1800 s request timeout and R=256: slower than R=160 and failing on request
  timeouts (above); R=256 also fills the KV pool. R>96 on 8 nodes: untested.
- R=64: slower (396 vs 302 s/update); engines gain throughput up to R=96. In single-engine
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
weights as a qualification requirement. HF-versus-SGLang greedy agreement and the
always-zero `train/ppo_kl` do not establish that comparison.

`train/train_rollout_logprob_abs_diff` (masked mean |trainer − rollout logprob|) is
nonzero on every update, about 0.008-0.010 in the measured runs;
`collection_score_mean_abs` is about 0.0065-0.0067. These combine numerical and
policy differences and do not replace a controlled same-weight parity check. The
metric that reads exactly 0 is `train/ppo_kl`, which is expected with
`use_rollout_logprobs = false` (the old logprobs are recomputed by the trainer on
the same weights) and refresh mode's one optimizer step per collection.

Longer learning runs additionally need native checkpoint/save-resume and evaluation
checks. The historical timing runs disabled saving; the current four-/eight-node starters
enable bounded recovery saves. Their inputs still have no held-out split.

## Open items

- R160 finished 50 updates, then failed its 900-second final generation drain.
  It used server concurrency 320, 192 KDA slots, CUDA graph batch 160, producer
  budget 3,680 and an auto-sized KV pool. Repeat after the terminal-cleanup fix
  and under the intended code-grading and batch/loss configuration.
- Judge latency is not logged per call, so the judge's share of generation wait is unmeasured.

## Checkpoint settings in the current starters

The four- and eight-node templates now enable 10-update native recovery saves,
automatic resume, retention of the latest two saves and 100-update permanent
milestones, plus a final native save. Their 30-update starting budgets therefore
save at 10/20/30; 100-update retention becomes relevant when the run is extended.
The four-update two-node mechanics check still disables checkpoints/resume.
The reported historical timings used checkpoint-disabled runs and do not include
this new save overhead or qualify restore correctness. Measure current batch 2,048
wall time and checkpoint cost, aim for recovery saves roughly every 1–2 hours, and
verify the first save and resume before relying on these settings for long runs.
See [the shared checkpoint policy](../../configs/miles/examples/README.md#recovery-checkpoints-and-preserved-milestones).

## Long runs

The four- and eight-node starters are set up so that raising `training.num_rollouts`,
and `launch.timeout` when the run is expected to exceed 72h, are the only changes a
multi-day run needs. Check each item below before launch.

### Mounts

`model.source` paths under `/weka/olmo-3p5-post-checkpoints` can be symlinks into
`/weka/olmo-3p5-checkpoints` (the H258 SFT export's weights, config and tokenizer
are). The launcher only checks the paths written in the TOML, and it usually runs
where WEKA is not mounted, so it cannot follow these symlinks. A job without the
target bucket fails at model load, not at submission. The starters therefore mount
both buckets. For another checkpoint, run `ls -l` on its files from a job or
session with WEKA mounted and add a `[[launch.weka_mounts]]` entry for every
bucket the links point into.

### Output path and disk space

Write run outputs to `/weka/olmo-3p5-post-checkpoints/YOUR_USERNAME/miles/...`, not
`oe-training-default` (timing4 failed when that volume filled). One A4B native save
is about 0.81-0.87 TB (model, optimizer and RNG state). With `checkpoint_keep_last
= 2`, `checkpoint_keep_every = 100` and the final save, a 660-update run peaks at
about 2 rolling + 6 milestones + 1 final = 9 saves, ~7.7 TB, plus ~145 GB per
evaluation snapshot if enabled (14 snapshots at interval 50, ~2 TB). Rolling saves
roll off only after the next save completes, so budget one extra save of headroom.
On November 20, 2026 the volume had 81 TB free of 180 TB (56% used); measure it
again before launch with a CPU job that mounts the bucket and runs
`df -h /weka/olmo-3p5-post-checkpoints`. Do not write to `olmo-3p5-checkpoints`
(95% used on the same date).

### Job timeout and preemption

Run a multi-day job as one Beaker job with a `launch.timeout` longer than the expected
run. `four-node.toml` sets `timeout = "72h"` and `eight-node.toml` sets `timeout = "120h"`. On 8 nodes (eight-node.toml) updates took
~264 s vs ~402 s on 4 nodes (updates 11-30 means at batch 2,048), so a 660-update run
is about 2.3 days on 8 nodes and 3.4 days on 4 nodes with startup (about 35-40
minutes to the end of the first update) and saves. The eight-node run uses 120h, about
2x its expected length; for 660 updates on 4 nodes raise it to about 96h. `ai2/holmes` has no maximum job
duration (`beaker cluster get ai2/holmes`), so the spec timeout is the only limit.

Beaker cancels a job when its `timeout` expires. A canceled job is not preempted:
`autoResume` requeues only preempted jobs, and Beaker's task retries apply only to
failed tasks (the launcher does not set them). A timeout therefore ends the run
without a final save, so leave headroom. Preemptions are handled by
`launch.auto_resume`: Beaker restarts the job and the run loads the newest complete
checkpoint under the same `output.root`; the updates since that save are redone.
Every start appends to `attempts.json`; if you set `launch.max_retries`, allow for
preemptions (default -1, no cap).

If a job does hit its timeout, resubmit the unchanged TOML with the same image:
`MILES_EXISTING_IMAGE=<image id from the launch receipt> python -m open_instruct.miles run <toml>`.
The run resumes from the last periodic save. `beaker experiment resume` applies only
to preempted experiments, so it does not restart a timed-out run.

`core.max_run_seconds` is optional, for clusters or budgets that cap job length. It
counts driver wall clock from driver start; at the first completed collection after
it, the run forces a native save and exits with `workflow.json` status `stopped`
(final HF export and final evaluation skipped), and the resubmission above continues
it with no updates redone. Set it below the job timeout with room for the last
collection (~6-8 min at batch 2,048), the forced save (~10-20 min) and teardown, e.g.
`79200` (22 h) under a 24h cap. Each resubmission pays startup again.

All attempts on one `output.root` log to one W&B run. On first start the workflow
writes a random id to `OUTPUT/wandb_run_id` (with `OUTPUT/wandb_run_id.json`
recording when it was created and a hash of the model source, data section and W&B
project, entity and group). Every later start, whether a preemption auto-resume or a
resubmission, reads that file and passes the id to `wandb.init(id=..., resume="allow")`.
An explicit `miles.wandb_run_id` takes precedence, and no file is written. If the
stored hash differs from the current configuration, the log shows a
`W&B RUN CONFIG MISMATCH` warning and the stored run is still continued. A new run
needs a new `output.root`.

This continuity needs `miles.wandb_mode = "online"`. The starters log offline, and
W&B offline mode ignores `resume`: each attempt writes its own local run directory
under `OUTPUT/wandb/` with the same id, so syncing them later does not append one
attempt's history to the previous one automatically.

MILES logs without an explicit W&B step, so updates redone after a resume (those
since the last save) appear again as extra points for those steps; the later points
are the ones the resumed run trained from.

### Evaluation snapshots

Each starter carries a commented `[evaluation]` block in background mode (interval
50, final). It writes trainer-side HF snapshots to
`OUTPUT/eval-snapshots/update-NNNNNNNN/hf` for offline evaluation, without draining the
rollout engines. To enable it, uncomment the block and delete
`training.skip_eval_before_train`, which selects shared-engine evaluation. Snapshot
export is a synchronous trainer collective, and a failed export fails training. It
requires an image built from main at or after #2293, which brought the writer-only
HF-export memory fix (miles 9eff46a4); with older images the snapshot export can run
out of memory. Snapshot export is not yet qualified on Small. The in-run evaluator
must disable the prefill CUDA graph for Small, because KDA prefill-graph padding
crashes it; the example sets `cuda_graph_backend_prefill = "disabled"` in
`provider_kwargs`. Evaluator job failures do not affect training.

### Changing the start checkpoint

The H196 prompt file records the SHA-256 of the chat template it was rendered with.
If the new checkpoint's template differs, data preparation stops with "Prepared
rendering template hash differs from the checkpoint chat template; re-render
inputs". Re-render the prompts with the new checkpoint's tokenizer before launch.

### Engine drain timeout

The four- and eight-node starters set `core.engine_drain_timeout = 3600`, matching
`core.refresh_request_timeout`, so a final drain or a shared-engine evaluation drain
is not cut off at 900 s (the `two-node-mech.toml` mechanics check keeps 900 s). With
`export_hf = false` and background evaluation, no stage that drains generation runs
before shutdown, and terminal refresh cleanup cancels unused generation.
