# Length guidance exercise — September 13, 2026

Status: both serving sweeps passed (exit 0); the 16K and 32K RL probes are running.

Runtime image: `01M2CD5CH7MBVHZK50AXAPYGFB` (image C from the colleague
readiness campaign). The standalone probe embeds its committed source; ordinary
RL launches retain their exact run specification. The image's baked modules are
not implicitly updated by a branch change.

Model:
`/weka/oe-training-default/robertb/olmo-miles/checkpoints/olmoe3-kda-1.2b-dolci-think-sft-65536-router-bf16-autocast-v2-hf`.

| Exercise | Purpose | Status |
|---|---|---|
| [Single-GPU serving sweep](https://beaker.org/ex/01M2E0SRNC49XZZY31VWPFE2SM) | Exact synthetic inputs of context minus 256 tokens at 16K/32K/64K; greedy output capped at 128; cold/warm and increasing concurrency | Passed |
| [Saturn math preparation](https://beaker.org/ex/01M2E0VGC7HJQMZRHEAMJHQQST) | Retain the existing eight train/two held-out natural math rows; token budget and disjointness checks | Passed |
| [16K RL config](../../../../configs/miles/qualification/colleague-20260912/length-math-16k-20260913.toml) | Two updates, EP2 + one engine, packing/replay, recomputation, 2 × 4 responses; response cap 14,336 | [Running](https://beaker.org/ex/01M2E15KM2J249V5BQ719PNX95) |
| [32K RL config](../../../../configs/miles/qualification/colleague-20260912/length-math-32k-20260913.toml) | Same recipe with response cap 30,720 | [Running](https://beaker.org/ex/01M2E15MNA9HZV0V54JSD5JC32) |

The inference probe uses a fresh server for each context, an explicit 131,072 KV
token ceiling, 16 recurrent state slots, radix off, chunked prefill 2,048 and decode
graph cap eight. Concurrencies are 1 (cold), 1 (warm), 2, then 8/4/2 by context.
A separate server uses the same local compiler cache within this allocation;
context changes are not independent cold-cache trials.

The RL probes use sync scheduling and admission two to isolate long-response
training memory. They preserve natural prompts and stopping, so actual lengths
may fall short of configured limits. Advance to a 64K training probe after the
32K evidence is reviewed. Do not force minimum response lengths and report that
as natural RL performance.

Retain `lengths.json`, per-context `server.log` and device memory samples from the
serving job. For RL retain rollout dumps, training contracts, replay/packing
records, driver stage times and generations. Separate startup, cold shape compile,
warm request latency, engine queueing and optimizer execution in the analysis.

Reproduce the serving probe from a clean committed worktree:

```bash
MILES_EXISTING_IMAGE=01M2CD5CH7MBVHZK50AXAPYGFB \
  bash scripts/train/build_image_and_launch.sh --miles \
  scripts/train/debug/miles_readiness_lengths.sh --receipt /tmp/lengths-receipt.json
```

The receipt pins the submitted source and immutable image. Use a fresh output
root when copying an RL configuration. These dated probes are not starter defaults.

## Initial serving results

All 24 responses produced 128 finite-scored tokens; every response hit the short
probe cap. This is execution evidence, **not a retrieval accuracy pass**.
No OOM or retraction event was found in the server logs. The requested 131,072
KV tokens were actually allocated (1 GiB K + 1 GiB V) at every context.

| Configured context | Actual prompt | Warm single request | Largest tested concurrent group | Group time | Peak sampled device memory |
|---|---|---|---|---|---|
| 16,384 | 16,128 | 1.24 s | 8 | 7.65 s | 38.51 GiB |
| 32,768 | 32,512 | 2.05 s | 4 | 6.48 s | 38.72 GiB |
| 65,536 | 65,280 | 3.94 s | 2 | 7.22 s | 38.72 GiB |

Times include prefill and 128 decode tokens through a local HTTP client. Larger
groups have not each received a repeated warm measurement. Startup took 270 s
for the first server and 62–63 s for the next two, which reused this allocation's
compiler cache. Device memory is sampled every second and includes startup;
it is not a precise allocator peak. This model/hardware leaves substantial
headroom, so these admission limits are conservative test points, not maxima.

The [expanded sweep](https://beaker.org/ex/01M2E1EC64MT0BKP4JYJD4JHEB) requests
524,288 KV tokens, admission/graph cap 32 and 64 recurrent slots. Its largest
groups will be 32 at 16K, 16 at 32K and eight at 64K. It uses the same pinned
runtime and prefill chunk size. Results are below.

Raw short generations and per-token scores are retained in
[serving-small-generations.json](serving-small-generations.json); compact timings,
commands, memory and log hashes are in
[serving-small-summary.json](serving-small-summary.json). Full logs and memory
samples remain in the Beaker result dataset. The preparation report confirms
8 train and 2 held-out natural math prompts with disjoint token hashes.

## Expanded serving results

The larger-pool sweep passed with 68 finite-scored responses, each capped at 128
new tokens. The actual allocation was 524,288 KV tokens (4 GiB K + 4 GiB V).
There were no logged OOMs or retractions.

| Context | Submitted concurrent requests | Largest logged running count | Largest logged KV occupancy | Group time | Peak sampled device memory |
|---|---|---|---|---|---|
| 16K | 32 | 31 | 502,913 tokens | 21.52 s | 46.10 GiB |
| 32K | 16 | 16 | 521,728 tokens | 21.91 s | 46.33 GiB |
| 64K | 8 | 8 | 523,008 tokens | 22.65 s | 46.33 GiB |

The running count is periodically logged, not an exact peak. Submitted concurrency
is measured by the client barrier. Near-equal group times correspond to roughly
equal aggregate input tokens; they do not make individual 64K requests as cheap
as 16K requests. Warm single-request times were 1.22, 1.91 and 3.72 seconds.
The first server took 316 s to start; subsequent servers took 62–64 s.

See [serving-large-summary.json](serving-large-summary.json) for exact commands,
artifact hashes and measurements, and [representative generations](serving-large-examples.json).
Full per-token scores and memory samples remain in the Beaker result dataset.
The larger test points have ample memory headroom on this B300; they are not a
maximum-capacity search or a throughput guarantee for natural long reasoning.

## RL progress and acceptance boundary

The 16K run's first training batch had eight responses averaging 13,612 tokens;
seven reached 14,336. All eight rewards were zero, so this batch cannot establish
a nonzero policy-gradient update. It still exercises long forward/backward
computation, including the router auxiliary objective. Rank one's first packing
record had four samples in four packs: packing did not reduce the forward count.
The run is not marked passed until optimizer, publication and final evaluation
complete and retained-sample/replay checks pass.

The 32K run is still generating long responses. A local follow-up monitor will
submit an exact-source `lengths` audit on Saturn after the GPU jobs terminate,
for workflows whose Beaker exit code is zero. Its launch receipt is retained at
`/tmp/readiness-length-final-audit-receipt.json` on the submitting host. Audit
results and a final training recommendation are pending; 64K training remains
unqualified by this exercise.
