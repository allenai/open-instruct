# Olmo 3.5 inference configuration study

This study is in progress. The measurements below do not yet qualify a new
default. It isolates fixed-policy MILES inference, with no trainer, publication,
evaluation or reward computation. See the [benchmark contract](inference-benchmark.md)
and [tracking discussion](https://github.com/allenai/oe-planning/issues/31#issuecomment-6401660900).

## Checkpoint and workload

The checkpoint is `olmo35-fixedtok-sft-20260921-4t-non-emo-dolci-think/emo/step5402/hf`
under `/weka/olmo-3p5-checkpoints/scratch/olmo35-fixedtok-sft-20260921/`.
Its retained configuration has hidden size 1,024, 16 layers (14 linear-attention
and two full-attention), four KV heads of dimension 128, 512 routed experts and
16 selected experts per token. Serving uses BF16 weights, PyTorch sampling,
Triton attention, full decode graphs, extra-buffer Mamba radix caching and routing
replay payloads. Optional olmo-sglang arithmetic switches remain at their pinned
defaults throughout the scheduling study.

The immutable workload contains 2,048 prompts: 1,024 math, 512 GSM8K and 512
IFEval. Preparation used the checkpoint tokenizer/template, excluded overlength
prompts without truncating them, and retained source identities and exclusion
records. Prompt lengths are at most 2,048; each prompt generates four responses
at temperature one with a 32,768-token response cap and 34,816-token context.
This illustrative mixture does not represent every production workload.

The admission panel uses open-instruct image source
`adab333343460a5dd18b13e024310b0e2387d950`, MILES
`4772a6767f1079eb3548d680c59fe98e481d6ff5`, and olmo-sglang
`3d701d8bdad348d7ae4b5ef7166290ec595cc5a5`. Runtime and image identities,
resolved server settings and the workload manifest accompany the artifacts.

## Initial B300 admission panel

[Run](https://beaker.org/ex/01M5T4R1E0HF47Z0VVY9WBDKXE): one B300 on Holmes,
serial fresh-engine cases at admission 64, 128, 256 and 64 again. Decode graph
coverage and full-context KV capacity scale with admission. HTTP permits are
twice admission; producer sample capacity is four times admission. Prefill chunks
are 8,192 tokens with native maximum prefill 16,384. Each case warms for at least
180 seconds and measures for at least 600 seconds, ending at complete collection
boundaries. Startup and drain are outside the reported rates.

The panel completed successfully. These rows were recomputed from downloaded
collection and response records; all requested captures are present. A separate
source-side WEKA recovery found no missing result content or hash differences;
only five empty workflow-lock files were absent from the original collection.

| Case | Delivered tokens | Actual seconds | Tokens/s | Mean delivered response length |
|---|---:|---:|---:|---:|
| Admission 64, opening control | 6,160,116 | 672.958 | 9,153.8 | 8,021.0 |
| Admission 128 | 10,166,610 | 730.100 | 13,925.0 | 7,942.7 |
| Admission 256 | 11,816,780 | 616.823 | 19,157.5 | 7,693.2 |
| Admission 64, closing control | 6,128,678 | 706.526 | 8,674.4 | 7,980.0 |

Admission 128 delivers about 52% more tokens per second than the opening control;
256 adds about 38% over 128. This compares coupled admission/graph/pool/request-supply settings,
not an isolated admission change. Completion ordering, unfinished responses and
work crossing boundaries require the retained inventory and timing audit before
selecting a setting. Higher admission can improve aggregate throughput while
increasing individual response latency. The closing control is 5.2% below the
opening control; these four serial cases do not establish a run-to-run confidence
interval.

| Case | Mean running requests | GPU memory high-water (GiB) | Mean engine execution per delivered response (s) | Mean initial engine queue (s) |
|---|---:|---:|---:|---:|
| 64, opening | 63.84 | 48.78 | 58.17 | 57.15 |
| 128 | 127.71 | 57.54 | 77.02 | 73.24 |
| 256 | 255.60 | 82.30 | 104.17 | 102.44 |
| 64, closing | 63.90 | 48.90 | 57.32 | 56.99 |

Occupancy and GPU observations cover the full measurement windows. All measured
responses have matched request timing records and report zero retractions.
These per-request intervals overlap and exclude unfinished responses; they are
not an engine wall-time partition. The native forward-time metric was unavailable,
not zero. The twice-admission HTTP limit kept an approximately admission-sized
engine queue, making reduced HTTP headroom a worthwhile independent comparison.

## Memory evidence and the higher-admission range

The engine reports approximately 23.31 GiB of weights. BF16 full-attention KV
storage is 4 KiB per token for this architecture: two full-attention layers ×
K/V × four KV heads × 128 dimensions × two bytes. This matches the engine's
17 GiB KV allocation for 4,456,448 tokens at admission 128. A 1,024-slot recurrent
pool occupies approximately 14.34 GiB (convolution and SSM states together).

Those values are specific to this checkpoint. They differ substantially from
larger-model examples elsewhere in the documentation. Full-context KV capacity
at admission 256, 512, 768 and 1,024 would be 34, 68, 102 and 136 GiB respectively,
before recurrent pools, weights, graphs, workspaces and runtime overhead. These
are capacity calculations, not proof that the corresponding serving settings
are safe or fast. Explicit pool caps also mean raising `mem_fraction_static`
alone need not allocate or use more memory.

The [running higher-admission panel](https://beaker.org/ex/01M5T88CTXEP3GQG2AVRYGRCYK)
brackets admission 512 with admission-256 controls, using at least 600 seconds of
warmup and 900 seconds of measurement. Admission 768/1,024 remain candidates
if measured gains and memory headroom justify them. HTTP and producer headroom
will then be varied separately, followed by prefill controls and fleet scaling.

## Correctness and retained evidence

The [diagnostic smoke](https://beaker.org/ex/01M5T6RNDNE1YRK4ZW2KEC0QKA)
successfully forced eight saved 512-token trajectories through two replay waves,
checking exact token reproduction, finite logprobs and routing-payload lengths.
It covers only two prompt clusters and qualifies the diagnostic machinery, not
32K correctness. The two waves differed in probabilities and routing hashes;
the later wave had a warmer cache. This is not established to be harmless or an
irreducible numerical floor. Finalist comparisons must match corresponding waves
across repeated controls and examine cache/batching effects.

Later diagnostics retain compressed routing matrices and compare response-
conditioning expert slots and tokens with prompt-clustered intervals. Repeated
trajectory copies or continuous background traffic can fill higher admission
without pretending to supply more independent prompts. The
[loaded replay smoke](https://beaker.org/ex/01M5T9ZMY62AQSWDSE8G69XY1D) passed two
waves, retaining all 16 routing matrices and joining all 32 outstanding background
responses after each wave's abort. Its short waves and five-second observations
do not establish sustained admission; inspect coverage during the long comparisons.
The [long qualification panel](https://beaker.org/ex/01M5TB865YZRZG4BR87B70TWNW)
now compares 64/256/512/64 with two loaded waves each on the same 256 captures,
explicit Core-compatible rounding and fresh engines between cases. These
diagnostic timings are excluded from throughput measurements.

The [forward-only Core reference](https://beaker.org/ex/01M5T83FJNW3Q87VYF468YMH1H)
completed all 256 opening-control captures, spanning 64 prompt clusters and
1,976,865 response tokens, including a 32,768-token response. It uses the pinned
Core revision `7e97bc43ff25d080ae4fd0eab1050a79caf80e55`, MILES model preparation,
BF16/FlashAttention 4, natural routing and one causal document per request at EP1.
For the original admission-64 sampled behavior scores, absolute logprob error
against this reference averages 0.03396 (prompt-clustered 95% interval
0.03317–0.03479); 1.807% of selected-token Core/serving probability ratios lie
outside 0.8–1.2 (interval 1.726–1.876%). The 99th-percentile absolute error is
0.2394 and the maximum is 5.1974. The mean does not make those tails harmless.
These establish the starting discrepancy, not the correctness of higher admission
or equivalence to packed expert-parallel training. Candidate comparisons will
report paired changes against this same reference and repeated serving controls.

All finalized attempts, including failed startup attempts, are downloaded under
`runs/inference-throughput-20261025/archives/` with full job logs, Beaker metadata,
result files and SHA-256 inventories. The ledger and launch receipts are retained
alongside them. The initial panel predates the unlimited benchmark report
collector; its [WEKA inventory/recovery](https://beaker.org/ex/01M5TA4FSVAK5VR84GPQABPMD4)
verified all 1,194 source files (excluding the checkpoint symlink trees), totaling
249,077,758 bytes. `analysis/fit-observations.jsonl` preserves the four initial
measurements with full model geometry, settings, hardware observations and hashed
analysis/source references. Its records explicitly retain pending numerical
qualification rather than presenting every completed case as a recommendation.
Future fitting should retain model architecture, GPU/runtime identity, pool sizes,
admission/graph/prefill settings, queue controls, fleet size, observed lengths,
memory/utilization and measurement coverage for every row.

## Remaining qualification

Finish the admission sweep and closing controls; audit full results; test higher
admission while gains continue; independently tune request supply and prefill;
compare frozen-prefix numerical behavior; and validate the selected setting on
multiple engines. An H100 check needs its own memory-feasible pools. Only then
should examples carry a measured serving recommendation. This study supplies no
new evidence for trainer batch size, policy-lag limits, publication intervals or
evaluation placement.
