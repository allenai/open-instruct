# Throughput qualification — September 13, 2026

The EP2/batch-128 graph-enabled run kept the trainer supplied: completed-buffer
waiting was effectively zero, with no dequeued-token drops. Its 3,750 useful
response tokens/s compares with 744/s for the same batch and six engines without
decode graphs. All 16 graph updates and 24 control updates completed, with clean
shutdown. Full decode graphs remain paired with disabled prefill graphs in this
qualification.

**Final allocation recommendations are being completed.** The eight-trainer graph
run, the reduced-producer batch-32 run, and a four-engine batch-128 comparison
finish the selection. Use the [throughput guide](../throughput-profiles.md) for
current profile scope, and the [campaign log](throughput-campaign-20260913.md) for
the earlier measurements and repairs.

## Measured scope

All full-model arms use the same 18.5B-total full-SFT KDA/latent MoE HF checkpoint,
frozen prepared GSM8K fixture, response cap 4096, context limit 6144, four responses
per prompt, learning rate 1e-6, PPO clipping 0.2/0.28, auxiliary/z coefficients
0.01/1e-5, no GRPO standard-deviation normalization, FIFO whole groups, and maximum
policy lag two. Batch size is explicit in every comparison. This is throughput
and runtime-contract qualification, not a learning or held-out-accuracy study.

The fixture is retained at
`/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1`.
The [machine-readable report](../results/throughput-profiles-20260913.json) includes
per-run commit, immutable image, archive checksum, full resolved run specification,
allocation, per-update timing, token/drop/age records, and sampled occupancy.

All GPU runs use urgent Holmes placement, workspace `ai2/open-instruct-dev`, with
one-hour minimum runtime for the full-model trials. W&B project is
`ai2-llm/olmo-rl-comparison`, group `throughput-profiles-20260913-v1`.

## Interpreting the percentages

Training/scoring plus publication is the occupied fraction of the **driver cycle**,
not hardware GPU utilization. The enclosing `generation_wait` stage includes
completed-buffer gets and other collection/handoff work. The report separately
retains these components; completed-buffer get time includes expiry filtering.

For the original 2T/4I graph run at batch 32, 221 measured seconds comprised about
11 seconds awaiting/filtering eligible groups, 7 seconds other collection/handoff,
and 203 seconds training/scoring/publication. It delivered 320 response attempts
and discarded 112: 25.9% of attempts, representing 35.7% of tokens. Excess work and
waiting coexist because the supply is bursty and stale completed groups cannot
fill the next eligible batch.

At batch 128 with six graph-enabled engines, completed-buffer gets took only
0.014 seconds across ten measured updates. Nearly all 4.7% collection time was
outside those gets. Adding engines cannot remove that handoff or trainer work.
Every delivered group in that window was age two; no consumed tokens were sampled
under the current trainer policy. Zero drops and high throughput do not imply
on-policy sampling.

Discard fractions cover dropped plus delivered response attempts/tokens at
completed-queue dequeue. `retry` requeues the prompt after discarding its old
responses. Unfinished work and shutdown leftovers are outside this denominator.

## Correctness and lifecycle boundaries

The original 2T/4I graph trial consumed 176 mixed responses. Both trainer-rank
audits passed across all 19 routed layers, with 512 checked microbatches per rank
covering scoring, training and backward recomputation. The six-engine batch-128
trial's audit also passed, but it consumed no mixed responses because generation
finished before publication. Audit results are retained in the JSON report.

The analyzer requires every expected optimizer update on every trainer rank,
no skipped updates, delivered-token accounting agreement, complete driver stages,
and a completed workflow. The earlier five 12-update measurements failed final
shutdown and remain labeled as such. A repaired outer deadline now honors the
configured generation-drain budget; later trials passed drains longer than five
minutes and exited cleanly.

Normal-cycle performance excludes model/engine startup, compilation warmup,
checkpoint saving, evaluation, export and final drain. Tiny dev/tiny trials passed
four updates and saves; this basket does not qualify resume or HF export. Graph
qualification enables extra route diagnostics; the graph-disabled controls do not.
This is an additional trainer-overhead difference to retain when comparing cycles.

## Reproduce the measurements

Frozen benchmark templates live in `configs/miles/qualification/throughput-*-base.toml`.
`scripts/miles/throughput_basket.py` derives each named case without changing those
historical inputs when researcher examples evolve. Launch from a committed tree
through the repository image/launch wrapper; see the campaign log for image and
overlay scope. Use a new run identity for every attempt.

Download a completed Beaker result and analyze it with:

```bash
python -m scripts.miles.throughput_basket /path/to/downloaded/run --warmup 6
```

The normal analyzer rejects failed workflows. The first six updates are an initial
warmup exclusion; inspect per-update scoring/training traces before deciding the
remaining window is stable. This is timing evidence, not an exhaustive compiler
variant count.
