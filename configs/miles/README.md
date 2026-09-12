# Core run configurations

Start with the [researcher workflow](../../docs/miles/workflow.md) and the
[structured examples](examples). They use the same section layout as olmo-miles
for model, data, trainer, inference, objective, tracking and launch. The profiles
below remain the low-level `[core]` / `[miles]` equivalents.


Use these standalone TOMLs with `python -m open_instruct.miles plan`, `validate`,
and `train`. They describe runtime settings; allocation, mounts and credentials
belong to the Beaker launcher. Copy a file and replace its `/data` paths first.

| Purpose | Config | GPUs / inference capacity |
| --- | --- | --- |
| Default local colocated dev/test | [tiny-resident](profiles/tiny-resident.toml) | One GPU shared by a tiny trainer and TP1 engine; four completions and four requests. Two updates plus checkpoints. |
| Default full-SFT disaggregated training | [train-disaggregated](profiles/train-disaggregated.toml) | Two B300 trainer GPUs plus one dedicated TP1 engine; 64 completions and 64 requests. 100 updates, heldout eval every 20, final checkpoint, offline W&B. |
| Bounded-async full-SFT training | [train-disaggregated-async](profiles/train-disaggregated-async.toml) | Same three-GPU/64-completion shape, one-step lag, buffer factor two/retry, old-policy scoring plus TIS. Same 100-update eval/save schedule. |
| Short full-SFT synchronous check | [sft-b300-ep2-sync](profiles/sft-b300-ep2-sync.toml) | Same three-GPU topology/admission; two updates, initial/final eval, no optimizer save. |
| Short async lifecycle check | [sft-b300-ep2-async-candidate](profiles/sft-b300-ep2-async-candidate.toml) | Same three-GPU topology; 64 requests, shorter 512-token responses, eager decode, four updates, lag at most one. |

The tiny resident profile has passed update and fresh-process resume tests.
The full-SFT model is the existing 18.5B-total KDA/latent checkpoint, not a claim
of hero model qualification. Full-SFT disaggregated runs have completed at
admission four and 64. **The 64-way sizing passed 12 sync and 12 async updates**
in [the B300 trial](https://beaker.org/ex/01M28SRP6G3YZ1MK34D3XEJQ2A), including
independent token/reward, optimizer, publication and policy-lag audits. Warm
response-token throughput was 1856/s sync and 2281/s async across the full cycle,
about 23% higher with async. Both are faster than the prior 16-completion batch;
that comparison includes a batch-size change. This qualifies the measured
capacity/scheduling scope, not the complete 100-update starter's evaluation,
checkpoint cadence, restart or learning quality.
Changing a starter does not rewrite the frozen configurations of previous runs.

Core keeps the trainer resident even in colocated mode. Do not put the full SFT
checkpoint into the tiny profile or copy olmo-miles' full-model colocation memory
fraction and assume it fits. A full-model colocated test must include an optimizer
update, followed by generation with optimizer state resident. Trainer offload,
Megatron microbatch 16 and Megatron async checkpoint writes are not supported
Core substitutes.

## Admission is a set of controls

The previous olmo-miles short-response sweep measured rollout wall time of
11.5 / 9.2 / 6.1 / 4.2 seconds at concurrency 4 / 16 / 32 / 64, with 128
completions split across two engines. Raising engine admission alone was ineffective
because the client semaphore still limited requests. That evidence motivates
raising both controls, not a promised speedup at our 4096-token response limit.
Source: `~/proj/olmo-miles/docs/measurements/rollout-concurrency-screen.md`,
Beaker [01M15KNJ5BGYP2V9FG9QN8NWA5](https://beaker.org/ex/01M15KNJ5BGYP2V9FG9QN8NWA5).

The new 4K-response starter sets these together:

| Setting | Value | Reason |
| --- | ---: | --- |
| `sglang_server_concurrency` | 64 | Client capacity per engine |
| `sglang_max_running_requests` | 64 | Engine admission |
| `sglang_cuda_graph_max_bs_decode` | 64 | Decode graphs through admitted batch size; prefill graphs disabled |
| `sglang_max_total_tokens` | 524288 | More than 64 × 6144 worst-case context tokens; actual pool allocation must be checked |
| `sglang_max_mamba_cache_size` | 128 | Recurrent-state capacity with headroom; radix cache disabled |
| `sglang_mem_fraction_static` | 0.6 | Keep the existing dedicated-GPU policy; do not substitute olmo-miles' 0.85 without measurement |

The collection is `rollout_batch_size=8` × `n_samples_per_prompt=8` = 64,
and `global_batch_size=64` gives one optimizer update per collection. Compared with the previous maintained 16 × 4 starter this preserves the
optimizer batch size while restoring the historical 8 × 8 group geometry. The
earlier 16-completion comparison runs used a smaller optimizer batch; compare
token throughput and do not attribute learning differences solely to admission. Trainer microbatch remains one; structured production async now enables document-isolated packing. These historical low-level profiles are distinct recipes.
The short async candidate reserves 262144 tokens for its 2560-token context.

To add a second dedicated TP1 engine on the same node:

```bash
python -m open_instruct.miles plan /path/to/run.toml \
  --set miles.rollout_num_gpus=2 \
  --set miles.num_gpus_per_node=4
```

Allocate four physical GPUs in the launch too. Trainer EP stays two. Sixty-four
completions then average 32 per engine; this does not guarantee a 2× speedup.
Compare generated tokens/second, tokens/GPU-second, request tails, cache
retractions and weight-publication time, not just update time.

To keep **both** engines busy at 64 requests, supply 128 completions:
set `miles.rollout_batch_size=16` and `miles.global_batch_size=128` (with eight
samples per prompt). More generally, `collection >= engines × concurrency`
is necessary to occupy every slot in a synchronous collection. Keeping one
optimizer step per collection means growing `global_batch_size` with collection
size. Smaller optimizer batches introduce multiple steps and require sufficient
policy-lag allowance. Watch request retractions, actual full-attention and recurrent
pool capacities, memory after graph capture, and end-to-end throughput at each size.
The historical 64-way result used short responses; it does not establish a safe
or optimal 4K-response configuration on every checkpoint/GPU.

## Async, evaluation and recovery

Use [train-disaggregated-async.toml](profiles/train-disaggregated-async.toml)
for the updated bounded-async recipe (8 × 8, TIS, buffer factor two). The
previous measured async arm used 16 × 4 and rollout behavior log probabilities.
The original `train-disaggregated.toml` remains the synchronous starter. Equivalently, apply these overrides to it,
keeping lengths, batch size and objective coefficients fixed:

```bash
python -m open_instruct.miles plan /path/to/run.toml \
  --set miles.fully_async=true \
  --set core.max_policy_lag=1 \
  --set miles.async_data_buffer_capacity_factor=2.0 \
  --set 'miles.async_unused_samples_handler="retry"' \
  --set 'miles.rollout_submission_granularity="group"' \
  --set miles.use_rollout_logprobs=false \
  --set miles.use_tis=true
```

Use the same overrides with `validate` and `train`. Async changes the data schedule
and uses trainer-scored old-policy log probabilities with TIS correction. It
requires disaggregated, resident engines. Replay and reference KL remain separate
choices. This recipe change does not rewrite prior measurements.

Prepare an HF descriptor with the exact tokenizer/chat template, rendered
`train.jsonl`, disjoint `eval.jsonl`, and trusted open-instruct `verifiers.json`.
These low-level profiles require prepared inputs; the structured workflow
examples support task preparation. Neither format substitutes `/data` paths automatically.
Initial and every-20-update heldout evaluation use greedy decoding with the same
4096-token response cap. Full rollout dumps and offline W&B are retained.

The training starter saves once at update 100. Core currently writes synchronous
native checkpoints; the measured full-SFT checkpoint was about 222 GB. The qualified optimized
writer saved it in 118.5 seconds versus 437.1 seconds for its matched baseline;
fresh-process load took 144.3 seconds. These are the fixed-input EP2 results in
the [checkpoint qualification](../../docs/miles/measurements/checkpoint-perf-20260911.md). Set a more frequent recovery cadence
when appropriate and budget that cost. Do not copy olmo-miles' `async_save=true`;
Core rejects it. Automatic Beaker restart and HF export are not supplied by this
TOML. See the [run-control guide](../../docs/miles/run-controls.md) for supported
resume settings and the [compiler-cache guide](../../docs/miles/compiler-cache.md)
for cache reuse with provenance.

On Beaker, launch through the committed `scripts/train/build_image_and_launch.sh --miles` workflow. GPU defaults are `ai2/holmes`, workspace `ai2/open-instruct-dev`,
urgent, with positive minimum runtime sized to the job (at least one hour for
these trials). CPU-only preparation requiring WEKA goes to Saturn. Run the
existing FA4 forward/backward preflight for B300 Core training.

## Evaluation concurrency on the next ordinary run

Shared-engine evaluation uses the same `GenerateState` client semaphore and the
same SGLang engine as rollout generation. The 64-way baseline therefore applies
to evaluation already; there is no separate four-request eval cap to override.
It does not require changing training batch size just to evaluate 128 prompts.
For a future Core/Megatron pair, use the same serving controls from the admission
table in **both** arms, including graph/cache limits, lengths, radix policy and
memory fraction. Historical comparison configurations stay frozen at four.

Evaluate the same immutable 128 heldout question IDs, tokenizer/chat template and
4096-response-token limit, with one greedy answer per question. Compare update
zero at the same checkpoint before comparing learned policies. Retain responses,
per-question rewards, lengths and cap hits; inspect paired answer flips rather
than assuming every score difference is noise. Keep the existing autotune/cache
provenance so unrelated kernel-selection changes are visible.

New Core runs record `evaluation` events in `driver_timing.jsonl`, with initial
versus periodic phase and configured serving limits. This is blocking end-to-end
eval time, including data/loading, scoring rewards and retaining outputs. Native
MILES `eval_rollout` timing remains the narrower generation/reward measurement.
Cold initial evaluation is reported separately from later points. Snapshot-based
evaluation instead records `evaluation_dispatch`, which measures submission and
must not be counted as completed evaluation latency.

The target is **under 60 seconds for warm 128-question evaluation**. It is a
performance target, not a timeout or correctness condition: one 64-slot engine
still needs at least two admission waves, and long tails/batching can keep it
above that target. Report tokens/second, response lengths, engine occupancy,
actual allocated KV/recurrent pools and retraction messages with wall time.
The TOML pool sizes are requests, not proof of the allocation SGLang obtained.
Observe this on the next ordinary run that includes evaluation; no separate
allocation is required. The current 12-update admission trial has no heldout eval
and cannot establish this evaluation speedup.
