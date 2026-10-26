# Olmo 3.5 inference configuration study

This study is in progress. The measurements below do not yet qualify a new
default. Its capacity panels isolate fixed-policy MILES inference, with no trainer,
publication, evaluation or reward computation. A separate production-path run
qualifies actual updates and fully async mid-sequence refresh. See the [benchmark contract](inference-benchmark.md)
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
| Running admission / decode graph maximum | 512 / 512 per engine, full decode graphs | Production refresh; one/two-engine fixed-policy numerical controls complete |
| HTTP concurrency | 512 per engine is the candidate; both 512 and 1,024 measured | Check fleet refill behavior |
| Producer sample budget | 1,024 for one engine is the candidate | Qualify fleet-wide budget and policy age separately |
| Full-attention KV pool | 17,825,792 tokens = 512 × 34,816 | Inspect resolved capacity and retractions for each shape |
| Recurrent pool | 3,072 slots, extra-buffer radix caching | Recheck memory and cache behavior after changing model or context |
| Static memory fraction | 0.7 | This measured B300 setting is not an H100 recommendation |
| Prefill | Chunk 8,192; native maximum 16,384; prefill graphs disabled | Realistic prompt-stream cache comparison; long-prefix bracket complete |
| Response / context caps | 32,768 / 34,816 tokens | Realistic-mixture transfer; short-response and long-prefix panels complete |
| Engine fleet | One and two TP1 engines measured; two retain 98% of ideal scaling | Larger-fleet capacity and actual production refresh |

These are a reproducible fixed-policy candidate and pending decisions, not a
production preset. HTTP capacity 1,024 left roughly 511 requests waiting inside
the engine in addition to 511 running requests. Capacity 512 retained almost the
same running occupancy with essentially no engine queue. Reducing the producer
budget to 1,024 retained that occupancy and roughly halved request residence time.
The trainer, global batch 256 and
lag limit 6 stay fixed during production qualification; do not increase the lag
limit merely to accommodate more queued work. Fully asynchronous mid-sequence
weight refresh must pass before promoting the candidate to a learning example.
The [submitted production check](https://beaker.org/ex/01M5V5YP9DPRB9J2VDKQKV2PMC)
uses the default node topology: two EP2 trainer GPUs and six inference engines on
one eight-B300 Holmes node. It preserves the default global producer budget of
768 unfinished samples, batch 256 and lag 6, for 16 optimizer updates.
Partly completed sibling groups and completed buffers can retain additional work;
this budget does not bound total retained samples or oldest-token age. It also
cannot fill six engines' combined 3,072 admission slots. That check measures
actual publication, freshness and useful consumption; the separate fixed-policy
fleet check measures saturated serving capacity. The run has high priority,
90-minute allocation protection and execution timeout, and no automatic restart.
It retains the studied optimized runtime/cache settings, full rollout captures
and refresh diagnostics. The old unlaunched 16-GPU proposal is superseded.

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
Full records are in `analysis/supply-headroom-interim.json` and the fit dataset.

The HTTP 512 / producer 1,024 case also completed, with its full
[snapshot](https://beaker.org/ex/01M5TX6W2P7KDPE3T21PG51TRK) retained. It delivered
16,091,649 tokens in 659.292 s (24,407.5 tokens/s), with native decode 24,216.3
tokens/s. Mean running/queued requests were 510.60/0.008, with full coverage and
zero reported retractions across all 2,048 measured responses. Native decode was
1.7% below the producer-2,048 case; mean producer-to-response duration fell from
620.8 to 326.6 s, and producer waiting from 450.4 to 161.7 s. This favors the
smaller budget for this ballpark pass. Do not blindly
multiply it by fleet size in training: the global budget interacts with batch 256
and lag 6, and this fixed-policy result has no version-age filtering.

The closing HTTP 1,024 / producer 2,048 control completed successfully. It delivered
23,674.2 tokens/s with native decode 24,197.7 tokens/s, just 0.37% above the opening
native rate. Mean running/queued requests were 511.24/511.27; all 2,048 measured
responses have valid timings and zero reported retractions. All four cases used
the same GPU. The complete panel's 1,203 archive files have verified local hashes,
and the collector skipped no files. This supports carrying admission 512,
HTTP 512 and producer 1,024 per engine into the fleet check, rather than retaining
the much larger queues for a small unreplicated throughput difference.

The [two-engine check](https://beaker.org/ex/01M5TZ4D4NA60T95YR28M2BGRV) uses two
B300s on one Holmes node, with global HTTP capacity 1,024 and producer budget
2,048. It preserves the checkpoint, 32K response cap, 34,816-token context,
sampling, graph and pool settings. Warmup is 600 seconds and measurement 900
seconds, with per-engine counters and occupancy retained. Its high-priority
protected window and execution cap are both 75 minutes, with no automatic restart.
The image includes the cancellation and resource-observer fixes; the node with
observed disk exhaustion is excluded. This run is still a fixed-policy capacity
check and does not qualify production policy-age filtering or weight refresh.

The run completed successfully. All 304 archived files passed local hash checks;
all 298 collector-declared reports are present, with no skipped files. It delivered
44,755,862 response tokens in 935.283 seconds: **47,852.7 tokens/s**, or 23,926.4
per GPU. Native decode counters report 23,976.7 and 23,545.9 tokens/s for the two
engines, totaling 47,522.6. Relative to the selected single-engine case, these are
1.961× delivered and 1.962× native throughput, approximately **98% of ideal doubling**.
These are separate runs on different GPUs, not paired hardware replicates or
confidence intervals. The fleet image also includes cancellation and telemetry
fixes; serving arithmetic is unchanged.

Both engines have complete measured gauge coverage, with mean running counts
510.33 and 510.70 out of 512 and mean queued counts 0.006 and zero. GPU observations
also cover the full measurement, with memory maxima 138.93 GiB on each device.
All 5,632 delivered samples have matched timing records and zero reported
retractions. Mean response length is 7,946.7 tokens; mean producer-to-response
residence is 339.5 seconds. Startup took 336.9 seconds and final drain 262.8 seconds,
outside the measurement. Shutdown leaves no active generation, producer or HTTP
tasks or waiters; retained completed siblings are recorded separately.

The consumer used 0.594–0.598 CPU-core equivalents while processing 47.62 MiB/s of
uncompressed route payloads. That excludes separate router and engine processes
and does not establish eight-engine CPU headroom. There were no failed resource
queries; sampled temporary storage retained at least 546.5 GiB available. No logged
slow compilation overlapped measurement; the last prior warning was about 596
seconds earlier. See `analysis/two-engines-summary.json` and the full archived data.

The [two-engine numerical check](https://beaker.org/ex/01M5V1MYRJ9BE2C0JTC7TET8X9)
completed separately. An opt-in router response header records the actual engine
serving each replay trajectory. Offline reports require attribution, show coverage
per engine and wave, and compare matched serving/Core scores for each engine's
subset. The change adds no per-token check or routing change. Forced-token replay
timings are diagnostic overhead, not throughput measurements. All 543 archive
files (1,723,832,559 bytes) passed hash verification, with no collector omissions.
Both engines served foreground trajectories in both 256-trajectory waves, from
64 prompt clusters. Background streams joined before each subsequent wave.
No invalid or duplicate expert IDs were found. Core-relative mean absolute
log-probability error changed by -0.00001554 versus single-engine 512, with paired
prompt-cluster 95% interval [-0.00006977, +0.00004022]. Both per-engine and both
overall per-wave error-change intervals include zero. One exploratory stratum
(second wave, response positions 1,024–8,191) has a +0.0271 percentage-point
ratio-outlier change with an unadjusted interval excluding zero; retain it for
follow-up alongside the aggregate, whose interval includes zero. Stratum intervals
are not adjusted for multiple comparisons.

The reference is natural-routing Core EP1; this does not establish bitwise
identity, packed EP2 training equivalence or learning quality. Single-engine replay
used HTTP 1,024, while the fleet uses HTTP 512 per engine and includes the later
cancellation/attribution fixes. It qualifies that coupled transfer for the bounded
production check, not an isolated engine-count effect. See
`analysis/fleet-replay-gate-decision.json` and the full comparison.

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
| 512 versus closing 64 | -0.00000542 [-0.0000654, +0.0000559] | +0.0020 [-0.0132, +0.0189] |
| Closing 64 versus earlier 64 | +0.00000958 [-0.0000422, +0.0000615] | +0.0140 [-0.0029, +0.0304] |

Intervals resample paired prompt clusters, preserving siblings and waves. They
support no detected increase in these aggregate Core discrepancies; they do not
prove equivalence or cover machine/day variation. Both cases in the protected
panel completed, with all 1,081 archive files hash-verified and no collector
omissions. Closing 64 and 512 share a GPU; the earlier 64 control was a separate
attempt. Closing waves also passed route-ID/uniqueness and background-join checks.
The natural Core reference uses EP1 rather than packed EP8 training. See
`analysis/replay-v7-closing-audit.json` and the matched comparison reports.

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
249,077,758 bytes. `analysis/fit-observations.jsonl` preserves completed audited
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
through both 512 → 256 and 256 → 512 cases succeeded, and the whole panel exited
successfully. All 905 archive files have verified hashes. Each case reached
`shutdown_complete` with zero active producer tasks, unfinished samples, generation
calls, admission waiters and HTTP requests. The final cancellation join after
generation paused took less than one second in each case. Completed and partially
completed sibling inventory remains recorded separately from active tasks.

| 2K-response case | Delivered tokens/s | Native decode tokens/s | Mean response tokens | Truncated |
|---|---:|---:|---:|---:|
| 512, opening | 19,182.7 | 19,641.4 | 1,869.4 | 79.39% |
| 256 | 14,742.2 | 14,758.2 | 1,872.4 | 79.65% |
| 512, closing | 21,513.7 | 19,422.6 | 1,875.3 | 79.86% |

The native 512 rates exceed 256 by 31.6–33.1%, with closing-control drift of
-1.11%. Delivered rates differ more because of completed-group ordering and
inventory across collection boundaries; they are not independent gain estimates.
All measured responses have timing records and zero reported retractions. Opening
512 engine gauges cover only 85.06% of its window, and consumer CPU is higher
(0.767–0.774 core equivalents versus 0.426–0.431 in closing 512). Retain these
limitations rather than attributing the change to admission alone. No logged slow
compilation overlaps the measured windows. This panel still uses the earlier
HTTP/producer headroom; it establishes a shorter-response comparison and actual
shutdown regression, not the final queue recommendation for every length.
See `analysis/short2k-summary.json` and all three complete per-case reports.

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

Its opening cache-on case completed and was
[fully snapshotted](https://beaker.org/ex/01M5TWXRCJ3NM5P95A7203KA0G). It delivered
7,679,371 tokens in 686.376 s (11,188.3 tokens/s), with native decode 12,900.9
tokens/s. All 3,840 measured prompts had 8,192 tokens; mean response length was
1,999.84, maximum 2,046, and 93.26% were truncated. All requested captures and
response timings are present, with zero reported retractions. Native prefill
counters attribute 98.12% of tokens to cache hits, so this small repeated corpus
does not measure uncached 8K prefill. The following cache-off case failed during
shutdown with local disk exhaustion, so the cache comparison remains incomplete.

The 2,046-token maximum is explained by the pinned SGLang request limits: its
worker advertises `context_length - 1`, and the scheduler allows at most that
limit minus input length minus one. With an 8,192-token prefix and 10,240 context,
the effective maximum is 2,046 despite the requested 2,048 cap. This is the
runtime's context-budget arithmetic, not evidence of lost generated tokens.
Retain actual lengths when comparing shapes; do not alter a running bracket or
remove native guards to obtain the nominal cap. Source evidence is saved in
`analysis/context-boundary-cap.json`.

Resource sampling stopped early: GPU observations cover only 2.85% of the
measurement, and engine gauges cover 45.93%. Do not interpret those gaps as idle
time, infer peak memory, or extrapolate consumer CPU. The resource observer could
exit permanently on one `nvidia-smi` timeout or OS error and report it only during
cleanup. The fix records failed samples and retries at the existing cadence,
while preserving cancellation and child-process cleanup. Two failure regressions
reproduce the old behavior; all 935 MILES host tests pass with the fix. Existing
missing observations cannot be recovered. Engine HTTP scrape failures also
contributed gaps; that observer already retries, and their underlying cause is
not yet established. See `ITP-012`, `analysis/long8k-cache-on-summary.json` and the
full retained logs.

The full failed attempt is retained under `archives/length-long8k-response2k-v2`.
Ray reported zero available GB on its 878.155 GB temporary filesystem at
00:51:50 UTC, before the first inference-engine launch at 00:52:17 UTC. Later
worker logs and shutdown failed with `ENOSPC`; the final exception occurred while
Ray tried to create its spill directory during cleanup. This is evidence of
storage pressure predating generation, not evidence that this workload spilled
enough objects to fill the disk. File ownership and exact cause remain unknown.
`ITP-013` retains the timeline, node and scheduler evidence. Subsequent benchmark
images sample temporary-filesystem byte and inode availability alongside existing
resource observations; they do not change inference concurrency or spill policy.

The [replacement long-prefix panel](https://beaker.org/ex/01M5TZ8JVY0Y8S2ZQ6MWA8Q7Z3)
uses the selected HTTP 512 / producer 1,024 supply settings, retaining admission
512 and the same four cache/chunk cases, corpus, context, sampling and pools.
The image includes cancellation and resource-observation fixes. It excludes the
affected node and retains the one-B300, high-priority, protected 90-minute window
and execution cap, without automatic restart. It completed all four cases and
exited successfully. All 1,203 archived files (277,746,082 bytes) have verified
hashes; the collector omitted no files. Every case reached `shutdown_complete`
with zero active producer/generation/HTTP tasks, admission waiters and unfinished
samples. The failed attempt remains separately retained.

| 8K-prefix / 2K-response case | Delivered tokens/s | Native decode tokens/s | Prefix cache hit fraction |
|---|---:|---:|---:|
| Cache on, chunk 8192, opening | 13,593.7 | 14,085.1 | 98.11% |
| Cache off, chunk 8192 | 8,295.7 | 8,657.9 | 0% |
| Cache off, chunk 4096 | 6,994.5 | 7,288.4 | 0% |
| Cache on, chunk 8192, closing | 15,137.4 | 14,632.1 | 99.17% |

The closing native control drift is +3.88%. Chunk 4096 is 15.82% below chunk
8192 in the uncached comparison, so retain 8192 for this measured shape. Cache
hit fraction is cached tokens divided by cached plus uncached input tokens;
the input-token counter alone is not the denominator. These exceptionally high
hit rates reflect the tiny repeated continuation corpus and do not justify a
general cache recommendation. All measured responses have timing records and
zero reported retractions. Engine-gauge coverage is 83.71–89.41%, while GPU
sampling covers 97.08–99.80%; missing observations are not idle time. Native
counter and delivered-group windows differ, particularly by about 50 seconds in
the cache-off/8192 case. Their rates are complementary measurements, not an
identical time partition. See `analysis/long8k-v3-summary.json` and the complete
per-case reports.

### Realistic prompt-stream cache comparison

The next cache comparison uses the full, pinned
[`allenai/Dolci-Think-RL-7B` mixture](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B/tree/0fb6466d31ef3a9dd16985ef635e6429e05a6491).
This published Olmo 3 workload is a representative proxy, not an established
future Olmo 3.5 production distribution. Preparation retains all 102,014 source
rows, source identities and answers, natural duplicates and prompt lengths.
It verifies message text against the published prompt before rerendering with
the target checkpoint's chat template. Any reconstruction mismatch fails the
preparation instead of silently filtering data. These artifacts are explicitly
reward-free benchmarks, not training-ready reward conversions.

Freeze a deterministic prompt order and retain four sibling responses per prompt.
Use `data.shuffle=false`, join response `group_index` to the retained order, and
verify that the measured run never rolls over the dataset. Choose context limits
from the audited length distribution; do not silently truncate or drop long
prompts. Compare cache-on/off/on at the selected 512 admission and HTTP limits,
1,024 producer budget and 8,192 prefill chunks, keeping pool geometry and sampling
fixed within a bracket. Retain source mix and prompt/response length distributions
for each measured window, prefix hits, retractions, occupancy coverage, memory,
startup/drain times and raw outputs. Different response caps are separate panels.
The fixed-policy bracket measures serving capacity; the production qualification
must also exercise cache behavior across fully async mid-sequence weight refresh.

Preparation code is `scripts/miles/prepare_realistic_cache_workload.py`.
The [Saturn CPU attempt](https://beaker.org/ex/01M5V5P2A2P9FEVZYRW87VVA30)
was canceled before execution after scheduler events reported no free slots.
The [Jupiter fallback](https://beaker.org/ex/01M5V5R3DZVPKTKWE8TZJ4S4ZV)
also had no free slots and was canceled before execution. Local preparation then
completed under a 30-minute cap. All 102,014 rows passed exact text reconstruction;
all artifact hashes and source-order joins were verified. The pinned native loader
accepted every row without filtering or reordering. Prompt token lengths are
median 158, p99 1,130 and maximum 2,013, so the existing 2,048 prompt limit fits
the full mixture. There are 94,881 unique rendered prompts; all 7,133 duplicate
rows remain. Full source records, order, prepared inputs, tokenizer and hashes
total 2,761,893,977 bytes before the hash-list file. The committed Beaker artifact
`01M5V69A61TA9ASGC75HX0FCX0` contains all 15 files, including that hash list;
every uploaded size and SHA-256 digest matches the local copy.

The [realistic-cache bracket](https://beaker.org/ex/01M5V6YDN8FKT670Q4Y6Z7WH76)
has been submitted and assigned one Holmes B300. It uses high priority, 120-minute
allocation protection and execution timeout, and no automatic restart. Cache
on/off/on arms each get 600 seconds warmup and 900 seconds measurement at the
32K response cap, with the studied runtime, admission/HTTP 512, producer 1,024,
3,072 recurrent states and 17,825,792 KV tokens. Host RAM is explicitly 128 GiB
in the Beaker spec; the immutable image predates the new `launch.memory` field.
The whole immutable prompt artifact is mounted directly, with no WEKA copy.
No realistic-cache GPU result is available yet.

### Completed H100 admission measurements

The [H100 transfer panel](https://beaker.org/ex/01M5V23DSCQE90HWD86944WHE9)
completed 128/192/128/256 admission, followed by the separately submitted
[224 probe](https://beaker.org/ex/01M5V8ASGNV8H28GNSBBFQKDPK). Both exited
successfully. All 1,506 files (353,895,162 bytes) were downloaded and hash-verified,
with no collector omissions. Five complete case records extend the fitting table
to 26 observations. Full reports are in `analysis/h100-throughput-summary.json`,
`analysis/h100-transfer-v1/` and `analysis/h100-admit224-v1/` in the study archive.

| Admission | Delivered tokens/s | Native decode tokens/s | Peak GPU MiB | KV tokens | Recurrent slots |
|---|---:|---:|---:|---:|---:|
| 128, before | 9,341.1 | 8,788.3 | 55,136 | 4,456,448 | 768 |
| 192 | 11,648.1 | 11,416.9 | 69,676 | 6,684,672 | 1,152 |
| 128, after | 9,016.2 | 8,528.9 | 55,256 | 4,456,448 | 768 |
| 224, separate job | 13,100.7 | 12,468.9 | 66,312 | 4,456,448 | 1,536 |
| 256 | 14,322.6 | 13,321.0 | 66,356 | 4,456,448 | 1,536 |

All cases use one active H100 80GB engine, HTTP permits equal to admission,
producer capacity twice admission, context 34,816, response cap 32,768,
static fraction 0.9, cache on with `extra_buffer`, and prefill chunk 8,192.
Decode graphs cover admission; 224 retains the 256 graph ceiling. The runtime
image is `01M5TYRW60SC2JVCHJX1DHZFK9`, source `bf831286198807db97cea379fb049139348aa6dd`.
Each case warms for at least 300 seconds and measures at least 600 seconds,
ending at a complete collection boundary. Native counters include work on
unfinished responses and have their own scrape boundaries. No paired or
independent-replicate throughput confidence interval is claimed.

Mean running requests are 99.7% or more of admission. Timing records cover all
measured delivered responses, with zero reported retractions. The repeated 128
native control drifts -2.95%. At 256, native throughput exceeds 224 by 6.83%, and
delivered throughput by 9.33%. The comparison crosses hosts and CPU/RAM allocations:
the original panel reserved two GPU slots because of host RAM but used only one;
224 reserved one GPU with a 128-GiB host-memory request. Both actual allocated and
active-engine denominators are retained. Pool sizes also vary across the panel;
this is a configuration search, not an isolated causal effect of admission.

The 256 case's 17-GiB KV pool holds only 128 full-length contexts. It is a bounded
pool candidate for the measured length distribution, not 256 guaranteed 34K
contexts. Its maximum observed full-token usage was 47.5%, leaving room for a
bounded higher-admission probe. The [320 follow-up](https://beaker.org/ex/01M5VA8KDF63EJKHV8MNMHWH4J)
retains that KV pool, raises the state pool to 1,920 and decode graphs to 320,
and uses HTTP 320 / producer 640. It requests one Jupiter H100, urgent allocated
45 minutes with a 45-minute execution cap and no automatic restart. Estimated
memory from the 256 observation plus extra states is about 70.2 GiB before the
additional graph/workspace delta; actual fit and throughput remain unverified.

256 is currently the best measured H100 candidate, not an established peak.
Numerical and fully async production-refresh qualification remain pending.
This study supplies no new evidence for changing trainer batch size, policy-lag
limits, publication intervals or evaluation placement.
