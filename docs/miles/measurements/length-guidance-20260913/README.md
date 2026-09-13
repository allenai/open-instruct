# Length guidance exercise — September 13, 2026

Status: in progress. No new capacity result is claimed yet.

Runtime image: `01M2CD5CH7MBVHZK50AXAPYGFB` (image C from the colleague
readiness campaign). The standalone probe embeds its committed source; ordinary
RL launches retain their exact run specification. The image's baked modules are
not implicitly updated by a branch change.

Model:
`/weka/oe-training-default/robertb/olmo-miles/checkpoints/olmoe3-kda-1.2b-dolci-think-sft-65536-router-bf16-autocast-v2-hf`.

| Exercise | Purpose | Status |
|---|---|---|
| [Single-GPU serving sweep](https://beaker.org/ex/01M2E0SRNC49XZZY31VWPFE2SM) | Exact synthetic inputs of context minus 256 tokens at 16K/32K/64K; greedy output capped at 128; cold/warm and increasing concurrency | Running |
| [Saturn math preparation](https://beaker.org/ex/01M2E0VGC7HJQMZRHEAMJHQQST) | Retain the existing eight train/two held-out natural math rows; token budget and disjointness checks | Preparing |
| [16K RL config](../../../../configs/miles/qualification/colleague-20260912/length-math-16k-20260913.toml) | Two updates, EP2 + one engine, packing/replay, recomputation, 2 × 4 responses; response cap 14,336 | Awaiting preparation |
| [32K RL config](../../../../configs/miles/qualification/colleague-20260912/length-math-32k-20260913.toml) | Same recipe with response cap 30,720 | Awaiting preparation |

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
