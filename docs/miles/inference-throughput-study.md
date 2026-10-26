# Olmo 3.5 inference configuration study

This study is in progress. The measurements below do not yet qualify a new
default. It isolates fixed-policy MILES inference, with no trainer, publication,
evaluation or reward computation. See the [benchmark contract](inference-benchmark.md)
and [tracking discussion](https://github.com/allenai/oe-planning/issues/31#issuecomment-6401660900).

## Working candidate for this pass

Use **512 running requests per engine** as the ballpark candidate for the
checkpoint and B300 workload below. Its completed fixed-policy case delivered
23,956 tokens/s at 139.03 GiB peak memory. Increasing to 768 delivered 7.84% more
tokens/s at 188.35 GiB; 1,024 did not improve delivered throughput further.
Those coupled configurations and their measurement limits are detailed below.
The existing upper-end observations are sufficient for this pass; additional
high-admission sweeps are not needed to select a practical starting point.

| Control | Candidate or measured control | Qualification still needed |
|---|---|---|
| Running admission / decode graph maximum | 512 / 512 per engine, full decode graphs | Closing numerical control, fleet check and production refresh |
| HTTP concurrency | 512 per engine is the interim candidate; both 512 and 1,024 measured | Finish the closing control and check fleet refill behavior |
| Producer sample budget | 2,048 for the measured single engine | Compare 1,024 against 2,048; check fleet and policy-age effects separately |
| Full-attention KV pool | 17,825,792 tokens = 512 × 34,816 | Inspect resolved capacity and retractions for each shape |
| Recurrent pool | 3,072 slots, extra-buffer radix caching | Recheck memory and cache behavior after changing model or context |
| Static memory fraction | 0.7 | This measured B300 setting is not an H100 recommendation |
| Prefill | Chunk 8,192; native maximum 16,384; prefill graphs disabled | Long-prefix cache/chunk comparison |
| Response / context caps | 32,768 / 34,816 tokens | Short-response and long-prefix panels |
| Engine fleet | One TP1 engine measured | Two-engine scaling, then actual production refresh |

These are a reproducible fixed-policy candidate and pending decisions, not a
production preset. HTTP capacity 1,024 left roughly 511 requests waiting inside
the engine in addition to 511 running requests. Capacity 512 retained almost the
same running occupancy with essentially no engine queue. The producer budget
still needs its separate comparison. The trainer, global batch 256 and
lag limit 6 stay fixed during production qualification; do not increase the lag
limit merely to accommodate more queued work. Fully asynchronous mid-sequence
weight refresh must pass before promoting the candidate to a learning example.

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

The [first higher-admission attempt](https://beaker.org/ex/01M5T88CTXEP3GQG2AVRYGRCYK)
completed its opening 256 control, then was preempted during 512 warmup. Its
completed control delivered 18,157,581 tokens in 953.303 seconds (19,047.0 tokens/s).
Native decode execution counters give 18,767.1 tokens/s over a 953.687-second
scrape window, including unfinished responses and excluding prefill-selected
first tokens. The difference illustrates boundary inventory rather than a second
independent throughput observation. The interrupted 512 warmup is not a result.

The [replacement admission panel](https://beaker.org/ex/01M5TC1Q6KB701FQJH5TCRYSVT)
scheduled 512/768/1,024/512/256 on one B300, with at least 600 seconds of warmup and
900 seconds of measurement per case. This combines the remaining admission work
without repeating the completed opening 256 case. The 1,024 case reserves 5,376
recurrent slots, exceeding the five-slot reservation with prefix headroom. HTTP
and producer headroom are being varied separately, followed by prefill controls
and fleet scaling.

Its completed opening 512 case was [snapshotted independently](https://beaker.org/ex/01M5TEXAQKZJAA8HCH21T5H7Q8)
while later cases continued. All 298 copied files match the source-job hash
inventory. It delivered **22,045,514 tokens in 920.238 seconds (23,956.3 tokens/s)**;
native decode counters give 23,582.8 tokens/s. This is about 26% above the longer
256 control, before the new bracket's closing controls and numerical qualification.
Mean running occupancy was 511.30/512, with 99.24% observation coverage, and peak
memory was 139.03 GiB. All 2,816 measured responses have timing records and report
zero retractions. Mean response length was 7,828.7 tokens; startup took 386.53 s
and final drain 409.45 s, outside measurement.

The completed 768 case was also [snapshotted and audited](https://beaker.org/ex/01M5THB5QNVAB1H78N19RM82P3):
all 298 files / 70,341,529 bytes match the source-job inventory. It delivered
**24,967,280 tokens in 966.461 seconds (25,833.7 tokens/s)**, 7.84% above 512.
Native decode counters give 24,850.4 tokens/s, a smaller 5.38% gain; the two measures
have different boundary inventory. Mean occupancy was 767.22/768 with complete
observation coverage, peak memory 188.35 GiB, and all 3,072 measured responses
reported zero retractions. Mean response length was 8,127.4 tokens. Mean initial
engine queue/execution times rose to 227.36/247.82 seconds; startup was 237.44 s
and final drain 542.50 s.

The [1,024 snapshot](https://beaker.org/ex/01M5TM05XJZG4KQRGV0SWZKVWJ)
also passed full source-hash reconciliation: 298 files / 72,389,601 bytes. It
delivered **22,585,224 tokens in 901.851 seconds (25,043.2 tokens/s)**, 3.06%
below 768. Native decode counters instead rose 1.39% to 25,195.4 tokens/s;
work on unfinished responses explains why these measures need not rank adjacent
cases alike. Mean occupancy was 1,023.12/1,024 with 96.89% observation coverage,
and peak memory was 241.42 GiB. All 2,816 measured responses have valid timing
records and report zero retractions. Mean engine queue/execution times were
271.11/306.76 s; startup and drain took 236.44/705.52 s outside measurement.
The marginal gains are small relative to the added memory and response latency;
closing controls and numerical qualification remain necessary.

The parent was preempted by allocated work during the closing 512 warmup.
The [replacement controls](https://beaker.org/ex/01M5TMC3NPE6WVRYV1WM164CBE)
were also preempted before their 512 measurement completed. All 302 archive files
are retained, but this incomplete case is excluded from the qualified results.
For this ballpark pass, use the supply panel's opening and closing 512 cases for
replication rather than retrying another dedicated 512/256 panel. Its 600-second
measurement and 60-second keep-alive differ from the earlier admission panel;
retain those distinctions in comparisons. A different node/attempt cannot restore
the original same-node bracket. More memory or a longer queue is not itself
evidence of more useful RL throughput.

Retained process CPU counters give an average of 0.277–0.279 CPU cores at
admission 512 and 0.335–0.337 at 768 for the producer/consumer process. These
are bounds from observations bracketing the measurement, not confidence
intervals. At 1,024, the bound is 0.372–0.376 cores. Routing payloads alone
account for 23.83, 25.70 and 24.91 MiB/s respectively.
The counters include this process's validation and serialization, but exclude
the separate router and engine processes; they do not isolate individual checks
or rule out brief event-loop stalls. `analysis/consumer-cpu.json` and the fitting
records retain the evidence. Fleet scaling must measure the shared producer and
router as well as per-engine throughput.

The 512 case carried 511.10 queued engine requests on average. Mean initial
engine queue time was 168.48 s and mean engine execution 170.94 s; these overlapping
request durations are not a wall-time partition. The [supply panel](https://beaker.org/ex/01M5TQTNPNT6GVW8YR28HZK7VT)
holds admission/graphs/pools at 512 and compares HTTP/producer capacities
1,024/2,048 → 512/2,048 → 512/1,024 → 1,024/2,048. Each case uses 600 s warmup and
600 s measurement, with the production 60-second engine keep-alive. The first two
attempts were preempted during image pull before application startup; the third
was preempted during warmup and its full 149-file archive is retained. The fourth
retains the same settings and 150-minute timeout but requests 150-minute protection
on its one B300, at high priority with no automatic restart. A bounded scheduler
simulation did not resolve completion times and had incomplete external-demand
history; the change follows the actual preemption losses, not a predicted success
probability. This panel isolates whether the large queue and producer budget
improve throughput.

The fourth attempt's opening case completed and was [fully snapshotted](https://beaker.org/ex/01M5TT85287XVYE221YJTSJKQY)
while the panel continued. It delivered 14,177,712 response tokens in 610.754 s
(23,213.5 tokens/s); native decode counters give 24,107.9 tokens/s. Peak memory was
138.93 GiB, with mean running/queued requests 511.42/511.20 and full observation
coverage. All 1,792 measured responses have timing records and report zero
retractions; all 256 requested captures are present. Startup and drain were
373.99/392.89 s outside measurement. This repeats the approximate 23–24K delivered
tokens/s range at 512. Different collection windows and
unfinished work account for different delivered/native rates.

The second case, HTTP 512 / producer 2,048, also completed and was
[fully snapshotted](https://beaker.org/ex/01M5TVWWJH2H0WGZPTCKX9FPBK), with all
298 source-file hashes verified. It delivered 16,742,615 tokens in 656.308 s
(25,510.3 tokens/s); native decode was 24,637.7 tokens/s. Mean running/queued
requests were 510.98/0.015 with full observation coverage. All 2,048 measured
responses have valid timing records and zero reported retractions; all 256
captures are present. Peak memory remained 138.93 GiB.

Compared with the opening case, native decode rose 2.2% while delivered throughput
rose 9.9%; these are different counters, not independent estimates of one gain.
The smaller HTTP capacity kept admission almost full and removed the engine queue,
but waiting moved to the producer: mean producer wait rose from 295.4 to 450.4 s,
while producer-to-response duration remained about 625.3 versus 620.8 s. This
supports testing less queued work, not a claim that total policy age improved.
The smaller producer case and closing control remain pending. Full records are
in `analysis/supply-headroom-interim.json` and the ten-row fit dataset.

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
The [second long qualification attempt](https://beaker.org/ex/01M5TC1T09R3M7NFESE49VJKA1)
completed and drained its first 256-trajectory wave at admission 64, then failed
on a router-to-engine `ReadError`/HTTP 503 while starting the next wave. All first-
wave scores and 256 routing matrices survived and passed the local integrity
audit. Mean running occupancy was 63.07/64 over 887 seconds with 99.93% observation
coverage. Its Core mean absolute logprob error was 0.03389 (95% prompt-clustered
interval 0.03307–0.03474), with 1.831% of ratios outside 0.8–1.2 and maximum error
5.1752. This is supplemental control evidence from a failed workflow, not a
qualified candidate. The transport failure's cause remains unresolved.

The failed benchmark omitted `SGLANG_TIMEOUT_KEEP_ALIVE=60`, already present in
the production examples. The pinned engine's five-second default equals HTTPX's
reuse window; the [documented connection-close race](operations.md#http-keep-alive-margin)
is consistent with the observed timing. This attribution awaits the rerun.
A [single-wave workaround](https://beaker.org/ex/01M5TE1ZDE09W1SRYR0VCQWSBP) was
stopped before its first wave completed. The [corrected replacement](https://beaker.org/ex/01M5TEFKHVQE06R7QGZKW2N912)
restores the original two-wave 64/256/512/64 protocol with the production engine
keep-alive value. It uses source `6201ba12a400` and image
`01M5TEFD3AF7VY06EE0TJ1ZZPQ`, including incremental score retention and a separate
fix that joins all foreground tasks after a diagnostic failure. Diagnostic timings
are excluded from throughput measurements.
The replacement completed both opening-control waves and advanced to admission
256 without a transport exception or HTTP 503. Its
[control snapshot](https://beaker.org/ex/01M5TGQ7XC1R8A9T7EBD8FGJPF) retains all
535 source files (1,687,245,055 bytes), verified against the source-job hashes.
Each wave retained 256 score vectors and routing matrices; all expert-ID range,
top-k and uniqueness checks pass, excluding dense-layer padding. Both waves joined
their 128 outstanding background requests in under 0.48 seconds. This supports
the transport correction; the remaining numerical cases and closing control
are still pending.
Its [first attempt](https://beaker.org/ex/01M5TB865YZRZG4BR87B70TWNW) was canceled
when the node was cordoned for an unrecoverable Xid 31 error, before completing a
wave. The application log contains no preceding CUDA exception; that does not
identify the cause of the node event. Both failed attempts are fully archived.

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

A same-token tail audit of the original draws and the completed replay wave found
59 and 57 tokens, respectively, with absolute Core logprob error above one; 18
token positions occur in both sets. The largest discrepancy recurs at sample
171, response offset 974 (a space token): original/replayed serving logprobs are
-0.49356/-0.51578 versus Core -5.69095. This is a persistent admission-64 baseline
discrepancy, not evidence that higher admission caused it. Its cause and practical
effect remain unresolved. The full token identities, surrounding text, scores and
threshold counts are retained in `analysis/baseline-tail-repeatability.json`; keep
this issue visible when assessing candidates rather than relying on mean error.

The corrected control's two waves have Core mean absolute errors 0.033925 and
0.033954. Their paired change is +0.000029, with a 95% prompt-clustered interval
of [-0.000038, +0.000093]. Core ratio-outlier rates are 1.8326% and 1.8446%; the
paired change is +0.0120 percentage points [-0.0069, +0.0320]. The serving waves
themselves differ by 0.02690 mean absolute logprob, and 2.092% of expert memberships
change (18.575% of ordered slots). Thus similar aggregate Core errors do not mean
identical probabilities or routing. The second wave has a warmer cache; these
observations do not establish an irreducible numerical floor. Compare candidates
against matching waves and the closing control.

The completed admission-256 case retains two full loaded waves, with all 535
[source snapshot files](https://beaker.org/ex/01M5TK0MRHFJQH40PM3H2JR9D1)
verified. Relative to matching admission-64 waves, the paired Core MAE change is
+0.0000041 [-0.0000534, +0.0000621], while the ratio-outlier fraction increases
by 0.0220 percentage points [0.0037, 0.0402]. These prompt-clustered intervals
include both waves without treating copies as independent prompts. Expert
membership differs at 3.126% of compared token/layer positions; ordered slots
differ at 27.210%. No invalid expert IDs were observed.

Stable average error conceals a large change at the persistent tail token:
sample 171 / response offset 974 has serving scores -0.75204 in both 64 waves,
but -6.81363/-6.75120 at 256, versus Core -5.69095. Expert memberships differ
at its conditioning position in layers 6–15. This association does not establish
causality. Admission 256 is closer to Core for this token, but neither average
agreement nor this one improvement qualifies it. The full matched report is
`analysis/replay-v4-64-vs256.json`. The parent was preempted during 512;
the [first replacement](https://beaker.org/ex/01M5TMC66743TX59RVBJGKZ8N9)
was also preempted during its first 512 wave. Its full 278-file archive retains
251 aligned score vectors and matching route hashes; these partial observations
do not qualify the setting. The next attempt was also preempted during its first
512 wave; all 126 archive files and 97 completed score vectors are retained.
The [protected replacement](https://beaker.org/ex/01M5TRRQMV4AHS79SNBDHRDSGQ)
retains 512 and closing 64 with two waves each, explicitly across attempts. It
requests 150-minute protection on one B300 at high priority, matching its existing
150-minute execution cap, with no automatic restart. It uses application image
`01M5TNPB5BXFE2675SD2XNJAWB` (source `5241f527a`):
the added optional Core route scorer and publication diagnostic changes are
inactive in this serving-only replay, and the serving replay path is unchanged.

Its admission-512 case has now completed both loaded waves and clean background
shutdown. The [full snapshot](https://beaker.org/ex/01M5TWA7HCKRAXPM6E96R91F14)
retains 535 source files (1,689,661,500 bytes), all hash-verified locally. Each wave
has 256 trajectories, 64 original prompt clusters and 1,976,865 response tokens.
Mean running occupancy was 505.93 and 507.95, with 99.86% and effectively 100%
observation coverage. No invalid expert IDs or duplicate selected experts were
found in either complete prompt/response routing payload.

| Matched comparison | Change in Core mean absolute logprob error, 95% interval | Change in Core ratio-outlier rate, percentage points, 95% interval |
|---|---:|---:|
| 512 versus earlier 64 | +0.00000416 [-0.0000621, +0.0000788] | +0.0161 [-0.0037, +0.0356] |
| 512 versus earlier 256 | +0.000000069 [-0.0000473, +0.0000538] | -0.0059 [-0.0224, +0.0103] |

Intervals resample paired prompt clusters, preserving siblings and waves. They
support no detected increase in these aggregate Core discrepancies; they do not
prove equivalence or cover machine/day variation. The closing 64 control remains
pending, and the natural Core reference uses EP1 rather than packed EP8 training.

Routing and individual scores still differ: 512 versus 64 changes 3.125% of expert
memberships, with mean absolute serving-score difference 0.03305. The maximum
serving difference is 6.87347 at the previously localized sample 171 / offset 974.
At that position, 512 scores -6.06491/-7.62550 in its two waves versus Core
-5.69095 and the earlier control's -0.75204. The maximum Core error across the
entire 512 case is 3.29321. Keep these tails visible despite the stable averages;
no additional tail-probe jobs are planned for this pass. Full comparisons and
threshold counts are retained in `analysis/replay-v7-numerical-summary.json`,
`analysis/replay-v7-64-vs512.json`, `analysis/replay-v7-256-vs512.json` and
`analysis/replay-v7-tail-audit.json`. Fleet behavior and actual mid-sequence
publication remain separate gates.

Across the original draws, the retained earlier replay wave and both corrected
control waves, eight token positions exceed absolute Core logprob error one in
all four observations. The same space token remains the largest discrepancy;
both corrected waves score it at approximately -0.75204 versus Core -5.69095.
`analysis/baseline-tail-controls.json` retains these matched observations. A
bounded localization probe uses four short-prefix examples: compare full-document
and truncated-prefix Core scoring, then recurrent decode and prefill scoring of
the identical target token. The [Core probe](https://beaker.org/ex/01M5TMBHEDG5AZS0QAGT7HSXV9)
completed all twelve records and is fully archived. The worst recurring token
(sample 171 / offset 974) has exactly the same Core score in all three arms and
the original reference. Two other selected targets also remain identical. At
sample 42 / offset 588, however, Core changes from -5.21493 for the full sequence
to -3.24734 for the truncated prefix. Both identical-prefix arms agree exactly.
Thus Core sequence length does not explain the worst recurring space-token
case, but sequence-dependent numerical behavior matters for another selected
case. This does not by itself demonstrate noncausal attention; kernels and MoE
routing can change with shape. Full selected-token evidence is retained in
`analysis/tail-core-localization.json`. The [serving counterpart](https://beaker.org/ex/01M5TN03YDESZRFSNXQ6PJ5A6X)
completed both twelve-record waves at admission 16 with radix caching disabled;
all 52 archive files are saved. The space token scores -0.58334 in both full
generation waves and -0.81241 in both prefilled-prefix waves. Its truncated
generation scores instead change from -0.58334 to -7.43811 across waves, while
Core remains -5.69095. At the target's conditioning position, expert membership
changes in layers 4–15. All other selected target scores repeat exactly between
waves. This shows that high admission and prefix caching are not required to
observe the discrepancy; it does not establish a root cause.

Both waves passed offline route hash, shape, expert-range and unique-top-k checks.
No invalid expert IDs were observed. `analysis/tail-core-serving-localization.json`,
`analysis/tail-serving-route-differences.json` and
`analysis/tail-serving-route-audit.json` retain the scores and route evidence.
The [native Core routing-replay diagnostic](https://beaker.org/ex/01M5TQJ0WEX7KSS941KEK7RTM4)
completed the same twelve records using the first serving wave's expert IDs and
Core's recomputed expert weights. Its 19-file archive and source hashes are
verified. At the full-generation space target, Core moves from -5.69095 to
-0.63627, versus serving -0.58334. Across the 30,387 scored tokens in these
related selected trajectories, mean absolute error falls from 0.031038 to
0.013084 and maximum absolute error from 5.10761 to 0.24637. Routing therefore
accounts for much of this selected discrepancy, while a residual arithmetic
gap remains. `analysis/tail-r3-wave0.json` retains every target and full-vector
comparison. The [second-wave diagnostic](https://beaker.org/ex/01M5TQY1HE1RJNKHN8WP9EESP1)
also completed, with all 19 archive files saved. For the truncated space target,
serving scores -7.438107967 and Core with that wave's routes scores -7.438108444,
an absolute difference of 4.77e-7. Selected-trajectory mean error is 0.013093 and
maximum error remains 0.24637. Thus the recorded route change reproduces the
large score change in Core; this does not establish why serving changed routes
between identical requests. `analysis/tail-r3-paired-summary.json` compares both
waves. Stop the targeted tail probes here and retain broader admission and
production qualification as separate gates.

Observed native route-context setup took 5.467 and 9.748 seconds on the first
calls of the two runs and 2.82–5.04 milliseconds on subsequent calls. This includes native padding,
conversion and existing GPU checks, with no added synchronization; it is not
isolated per-check cost or a packed EP8 trainer estimate. The first call includes
cold setup. These selected EP1 diagnostics do not estimate a population error
rate, qualify production or replace the async refresh test.

An offline CPU microbenchmark of the existing checks used 256 retained captures
(1,960,609 response tokens) on the development host. Native `record_response`
took 235–236 microseconds per response; benchmark `sample_record` took 37.6–38.0.
These are five hot-cache trial ranges, not confidence intervals or production
speedup estimates. They exclude JSON/base64 parsing, route deserialization, GPU
work, contention and mixed-version span counts. The actual native sample class
and captured token/logprob arrays were used; routing placeholders supplied only
shape metadata. `analysis/check-cost-microbenchmark.json` retains provenance and
all timings. These measurements do not justify removing transport integrity
checks; no runtime checks were removed or profiling hooks added.

The runtime-check audit also found an opt-in publication diagnostic race. With
refresh publication and a positive `core.diagnostic_interval`, the round-trip
check could reset weights after the publisher had resumed live decoding. The
driver now pauses every engine before the diagnostic snapshot/reset; the normal
publisher restores weights and resumes the preserved requests. Failure paths
leave decoding paused. CPU regression tests exercise the driver and the mutation
ordering; GPU validation remains pending. The current production draft uses
interval zero, so this path adds no recurring work to that configuration. The
issue is tracked as `ITP-009` in the retained issue log.

All finalized attempts, including failed startup attempts, are downloaded under
`runs/inference-throughput-20261025/archives/` with full job logs, Beaker metadata,
result files and SHA-256 inventories. The ledger and launch receipts are retained
alongside them. The initial panel predates the unlimited benchmark report
collector; its [WEKA inventory/recovery](https://beaker.org/ex/01M5TA4FSVAK5VR84GPQABPMD4)
verified all 1,194 source files (excluding the checkpoint symlink trees), totaling
249,077,758 bytes. `analysis/fit-observations.jsonl` preserves ten audited
measurements with full model geometry, settings, hardware observations and hashed
analysis/source references. Its records explicitly retain pending numerical
qualification rather than presenting every completed case as a recommendation.
Future fitting should retain model architecture, GPU/runtime identity, pool sizes,
admission/graph/prefill settings, queue controls, fleet size, observed lengths,
memory/utilization and measurement coverage for every row.

## Remaining qualification

Correctness and robustness are production gates. Track encountered failures in
`runs/inference-throughput-20261025/issues.json`, with evidence, fixes and open
regression checks. A fixed-policy sweep cannot qualify weight publication:
exercise the ordinary fully async production path with live generations crossing
native weight refreshes, checking behavior-logprob preservation, policy-version
spans, final replay routes, publication pauses and recovery. Keep trainer settings
unchanged; a synchronous or barrier-drained substitute does not meet this gate.

For this pass, use admission 512 as the ballpark tuning point. The completed
768 and 1,024 measurements are sufficient upper-end exploration; no 1,536/2,048
runs or fine search for the knee are planned. Finish the remaining controls and
numerical qualification, independently tune request supply and prefill, and
validate the selected setting on multiple engines. Exercise shorter response
budgets on the same prompts and a separately identified longer-prefill workload,
retaining actual length and truncation distributions. Check whether admission
recommendations transfer across those workloads rather than assuming the 32K-cap
result is universal.

The [2K-response panel](https://beaker.org/ex/01M5TR6117TJFDWMNC7BBM9Q6D)
runs independently of the 32K supply comparison. It uses the original prompts,
a 4,096-token context and admission 512 → 256 → 512. HTTP capacity stays at twice
admission and producer budget at four times admission; graph batch and full-context
KV capacity scale with admission, while the 3,072-slot KDA pool stays fixed.
Each case has 300 seconds of warmup and 600 seconds of measurement. One B300 is
protected for the 90-minute timeout, at high priority with no automatic restart;
Ray log deduplication is disabled to retain every engine observation. This is a
controlled workload-shape comparison, not a claim that the final queue setting
has been selected. The intermediate 8K-response panel remains prepared but
unlaunched.

The short-response attempt was stopped after shutdown stalled. Its complete
305-file archive retains `generation_paused` with zero active generation calls,
1,021 parked waiters and no `shutdown_complete`; the workflow never completed and
does not qualify a throughput row. A pinned-runtime CPU probe reproduced producer
cancellation being swallowed while retiring completion tasks. Two deterministic
regressions fail on the old runtime and pass with MILES fix
[`ff7be7a`](https://github.com/allenai/miles-olmo-core/commit/ff7be7a31d4c2b05eb75d443314a7e2991df9b7a).
The fix also passes 33 existing boundary/refresh tests and 36 full producer-loop
trials with synthetic generations. It changes completion cleanup, not mid-sequence
publication. The rebuilt image `01M5TVRMG3856ZFJ5R67KB39ZZ` passes all 35 tests
without source overlays. The [replacement panel](https://beaker.org/ex/01M5TVTEZQ6MJMEGF31T2JCG45)
uses the same workload and 90-minute one-B300 budget with this pin; GPU transition
qualification remains pending.

The [continuation preparation](https://beaker.org/ex/01M5TDG8K8CHYRXQVR4QDHXE9D)
retains 194 exact 8,192-token prefixes from 32 original prompt clusters. All
prefixes were verified locally using the downloaded tokenizer. Ninety-five other
length-eligible captures lacked an exact text/token boundary and were excluded.
This small corpus can make repeated cache-on runs prefix-hit dominated; include a
separately qualified cache-off comparison for uncached prefill mechanics.
The [long-prefill panel](https://beaker.org/ex/01M5TSJ42J1TZRK9KBYWB3ZE6Z)
uses these prefixes with a 2K response cap and 10,240-token context at admission
512. Its cases are cache-on/chunk-8192 → cache-off/chunk-8192 →
cache-off/chunk-4096 → cache-on/chunk-8192, keeping pools, graphs and request
headroom fixed. It has the same 300-second warmup, 600-second measurements and
protected one-B300, 90-minute execution cap as the short-response panel. This
separates uncached prefill chunking from reuse on the small repeated corpus.
Its first attempt failed before generation: continuation preparation omitted
`label`, which the pinned native dataset loader requires even with rewards
disabled. The preparation fix emits an explicit null label when targets exist
only in verifier metadata. A native CPU regression reproduces the old failure
and loads all 194 corrected rows, preserving their exact text and token IDs.
The failed attempt is fully archived and supplies no throughput result. The
[immutable corpus repair](https://beaker.org/ex/01M5TTN7VXYST3J4DKZHF4XBRF)
passed the native loader check and preserves the original corpus. The
[replacement panel](https://beaker.org/ex/01M5TTYXF8JSB8RTZ246MA3F73) retains
the same runtime, serving settings and resource cap.
An H100 check needs its own memory-feasible pools. Only then
should examples carry a measured serving recommendation. This study supplies no
new evidence for trainer batch size, policy-lag limits, publication intervals or
evaluation placement.
