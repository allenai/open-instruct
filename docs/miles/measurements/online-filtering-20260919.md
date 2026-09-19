# Online group filtering: small qualification — September 19, 2026

Image **`01M2XGZM2N1V4DQVMYHM52KBHZ`** contains application commit
`a8f8aca84a565a944a30bb6cd877a18e23dc77dd` on `robertb/miles-olmo-core`.
The filtering implementation is commit `6e7e4e6d3`; the later commit relocates the
synthetic qualification reward to an importable script. Runtime dependency pins
are unchanged. Both runs used the committed-image wrapper, without source overlays.

## Results

| Small variant | Optimizer updates | Reported all-zero drops | Reported all-one drops | Retained training responses |
|---|---:|---:|---:|---:|
| [Barrier](https://beaker.org/ex/01M2XGZX0NTG9KTYH92HAJ0H2J) | 4 | 16 | 15 | 32 |
| [Refresh](https://beaker.org/ex/01M2XH174ZYYQJBVP4S1QD7B4C) | 4 | 19 | 19 | 32 |

Both jobs exited zero, completed initial and periodic evaluation, published weights,
and saved their final native checkpoints. Drop totals sum the four per-collection
metric reports; asynchronous work after the final report is outside these totals.

An independent [read-only CPU audit on Saturn](https://beaker.org/ex/01M2XH8ATYYY917NVK04HSM5VE)
loaded all eight saved training batches. Every batch contained four distinct prompt
groups and eight responses; every group had rewards `[0, 1]`. No constant-reward
group reached training. Two responses in the refresh run's final batch spanned
multiple policy versions, so that check exercised actual mixed-policy responses.

Each run retained exactly one committed checkpoint with four completed steps and
a matching dataset-cursor checksum. Pending ledgers contained 16 and 17 groups,
respectively, with no already-trained groups. Inflight prompts remain eligible for
retry and can later receive constant rewards; they are not prematurely filtered.
The focused runtime tests separately verify that dynamic rejections retire their
ledger entries and do not enter the stale/aborted retry handler.

## Configuration and scope

Both configurations were copied from `configs/miles/examples/small.toml`: one
trainer GPU and one TP1 serving GPU, EP1, four prompts with two responses each,
256 response tokens, 512 context tokens, four updates and offline W&B. They enabled
`training.filter_zero_std_groups=true` and used
`scripts.miles.filtering_smoke_reward.score`, which cycles all-zero, all-one and
mixed groups. The refresh variant additionally enabled refresh publication and
the MILES router. Checkpoints were retained specifically for the cursor audit;
HF export was disabled.

The checkpoint is the previously qualified two-layer, four-expert tiny random MoE
under `/weka/oe-training-default/robertb/open-instruct/runs/standard-examples-20260918/tiny/hf`.
These synthetic rewards test rejection, replenishment and training mechanics;
they establish no GSM8K accuracy or learning improvement. Full-policy filtering,
engine-drain GPU execution, sustained throughput and interrupted-run recovery were
not exercised here. The maintained tiny dev/small examples keep filtering disabled
because their ordinary real-task rewards can all be zero; learning defaults remain
enabled. Offline preprocessing remains compatible.

The first barrier attempt failed before training because `tests.miles` resolved to
the backend's test package. The concurrent refresh attempt was cancelled after
that diagnosis. Moving the fixture into `scripts.miles` resolved the import;
corrected runs used fresh output paths and the rebuilt image above.

The final image passed 84 focused CPU runtime tests. The optional Megatron dumper
plugin was disabled only for those CPU tests because the local Docker host lacks
`libcuda`; GPU runs used the image unchanged. Source checks, lint, generated docs
and the documentation build also passed.

Submitted TOMLs, receipts, logs, metrics and downloaded audit results are retained
locally in ignored `runs/online-filter-small-20260919/`. Durable run artifacts are
under `/weka/oe-training-default/robertb/open-instruct/runs/online-filter-small-20260919/`
in `barrier-v2` and `refresh-v2`; the audit report is in its Beaker results.
