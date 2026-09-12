# Publication transport profile (phase 0)

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

Point-in-time evidence, 2026-09-12. Beaker experiment
[01M29R6B0W5TJQ0YAQPXFS4B2V](https://beaker.org/ex/01M29R6B0W5TJQ0YAQPXFS4B2V),
three B300 GPUs on `holmes-cs-aus-503`, source `8f84ff3ed` (branch
`robertb/miles-publication-profile`), frozen GSM8K parity checkpoint (512 routed
experts, 29,669 HF tensors, 37.0 GB BF16). Zero optimizer updates. Driver:
`scripts/miles/publication_profile.py`; launcher
`scripts/train/debug/miles_publication_profile.sh`. Raw records in
[this directory](publication-profile-20260912): `profile.json`,
`publication.jsonl`, `final-weight-comparison.json`, `nccl-transports.txt`,
`nccl-gpu-topology.txt`.

The job exited 1 in its cleanup only (disposal without first retiring the
weight-update NCCL group, fixed in the following commit); all 16 publications
and the final serving-weight equality check completed and were written first.

## Method

Each publication record now splits every bucket into the NCCL broadcast wait
(request dispatch, engine receive posting, transfer) and the engine wait (bucket
reconstruction and `load_weights` until the engine responds), with the bucket's
tensor count and bytes. After the ordinary initial publication the unchanged
weights were republished three times at each of five bucket sizes. NCCL ran
with `NCCL_DEBUG=INFO` writing per-process files.

## Results

| Bucket size | Buckets | Total s | Broadcast s | Engine s |
|---|---:|---:|---:|---:|
| 256 MiB | 139 | 3.50 | 0.072 | 2.64 |
| 512 MiB | 70 | 2.91 | 0.040 | 2.13 |
| 1 GiB (production) | 35 | 2.60 | 0.023 | 1.87 |
| 2 GiB | 18 | 2.94 | 0.032 | 1.81 |
| 4 GiB | 9 | 3.69 | 0.020 | 1.99 |

Totals at 2 and 4 GiB include one slow outlier each (a 3.3 s transport and a
2.4 s export pack); their medians are 2.54 and 3.02 s. Export pack is 0.4 to
0.6 s throughout; finalize under 0.1 s. The initial cold publication took
5.05 s, of which 1.15 s was first-use NCCL connection setup inside the
broadcast wait.

**Regression of engine seconds per bucket** on tensor count and bytes across
all 813 bucket observations (`profile.json` → `summary.engine_seconds_regression`):

| Term | Value | Per publication |
|---|---:|---:|
| per tensor | 40.6 µs | 1.20 s (29,669 tensors) |
| per gigabyte | 17.6 ms | 0.65 s (37 GB) |
| per bucket | 4.3 ms | 0.15 s at 35 buckets |
| R² | 0.98 | |

Effective broadcast rate across all buckets: 989 GB/s. The final full
serving-weight comparison after the sweep succeeded.

**NCCL transports** (`nccl-transports.txt`): the trainer-rank-0 to engine
update group (bus 0x10000 to 0x5e000) used `P2P/CUMEM` on 32 channels, that is
NVLink through NVSwitch. The trainer's own two-rank communicators (0x10000 to
0x38000) used `NET/IB/GDRDMA` on 8 channels. NCCL's topology for that GPU
(`nccl-gpu-topology.txt`) lists no NVSwitch link at all, while the other two
GPUs on the node do, and NVLS multicast is reported available on device 0
only. No NCCL environment override was set.

## Interpretation

- **Transport is not the cost.** The 37 GB broadcast takes about 25 ms on
  NVLink. The 11 GB/s figure inferred from the training runs was the engine's
  loading rate, not a transfer rate.
- **Engine-side per-tensor work dominates.** 40.6 µs per tensor matches the
  1.0 s per publication measured for the adapter loader's substring scan alone
  at 512 experts, plus dispatch overhead. The per-gigabyte term is consistent
  with 24,576 small device copies at low efficiency. Together they are about
  1.85 s of the 1.87 s engine wait at 1 GiB.
- **Bucket count matters only at the extremes.** 1 to 2 GiB is the flat
  region; 256 MiB costs 0.8 s more; 4 GiB adds allocation cost.
- **Unexplained gap to training.** This idle republish costs 2.6 s per
  publication versus 3.6 to 3.9 s in the 500-update runs. The split timers are
  now permanent in the actor, so the next training run will show whether the
  extra second is in the engine wait or elsewhere. Do not treat 2.6 s as the
  training-time number.
- **Node finding, separate from publication.** On `holmes-cs-aus-503` the GPU
  at bus 0x38000 has no NVLink path, so the trainer's EP collectives, gradient
  reductions and export all-gathers ran over InfiniBand. The 500-update runs
  were on `holmes-cs-aus-488` (Core) and `holmes-cs-aus-521` (Megatron) and are
  not affected by this node. Any job that lands on that GPU will see slower
  trainer-side communication; this should be reported for the node.

## What this decides

Phases 1 and 2 of the [publication transport plan](../plans/publication-transport-plan-20260912.md)
target the right cost. Phase 3 (bucket sizing) is closed: keep 1 GiB, or 2 GiB
if buckets are ever the limiting factor. NCCL path work is unnecessary for
publication.

## Phases 1 and 2 results

Same driver, same checkpoint, same bucket sweep, three B300s on Holmes. Each run
ends with the full serving-weight equality check, which passed in all three.
Evidence: `phase1/` and `phase2/` subdirectories beside the baseline records.

- Phase 1, memoized name resolution in the olmo-sglang loader:
  [01M29T86JM7PKTJ2JV48YHRXZN](https://beaker.org/ex/01M29T86JM7PKTJ2JV48YHRXZN),
  olmo-sglang `a5447d1f`.
- Phase 2, routed experts published as one stacked tensor per layer and
  projection in the engine's fused layout, on top of phase 1:
  [01M29TSXRMNEN7DSHS6JBMHEVF](https://beaker.org/ex/01M29TSXRMNEN7DSHS6JBMHEVF),
  Core `cfc42934`, olmo-sglang `02ccb5dc`, open-instruct `2154a1af`.

| Run | Tensors | 1 GiB total s | 1 GiB engine s | 2 GiB total s | 2 GiB engine s | µs per tensor |
|---|---:|---:|---:|---:|---:|---:|
| baseline (per-expert, scanning loader) | 29,669 | 2.60 | 1.87 | 2.94 | 1.81 | 40.6 |
| phase 1 (memoized names) | 29,669 | 2.21 | 1.64 | 2.38 | 1.57 | 21.4 |
| phase 2 (fused experts + phase 1) | 523 | 0.49 | 0.26 | 0.35 | 0.14 | 34.3 |

Phase 1 halved the per-tensor coefficient; the name scan was about half of the
per-tensor cost on the B300 host, the loader call and its small copy the rest.
Phase 2 removes the per-tensor work almost entirely: 523 tensors instead of
29,669, export-and-pack drops from 0.49 s to 0.16 s because the trainer no
longer slices slabs per expert, and the engine wait drops to 0.26 s at 1 GiB.
Each fused expert tensor is 1.25 GB, larger than a 1 GiB bucket, so buckets
hold one such tensor each; 2 GiB buckets halve the bucket count and give
0.35 s total, 4 GiB gives 0.32 s. The full-SFT starter profiles now select
`core.expert_publication = "fused"` with 2 GiB buckets. The phase 2 regression
has R² 0.11 because per-tensor and per-byte costs are now both negligible
against the 6 ms per-bucket floor.

Against the training-run figure of 3.6 to 3.9 s, this is a saving of roughly
3.3 s per update, or about 0.45 h per 500-update run, with the unexplained
training-versus-idle gap of about 1 s still to be read from the next training
run's per-bucket records.
