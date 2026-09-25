# Provisional lag-six default and small comparison

September 25, 2026. **Both 24-update arms completed successfully.** Six is a deliberate
operating starting point, not an empirically established optimum. The structured
async fallback changes from one to six, and medium/large change from two to six.
Explicit limits and synchronous dev/small behavior are preserved. See
[lag semantics and TIS](../async-pipeline.md#policy-lag-and-tis).

## Why this archived workload

The [September 14 six-GPU run](fast-2t4i-20260914.md) completed 48 updates on the
original full-SFT policy with real GSM8K rewards. At lag two it discarded
23–31% of response tokens across the reported windows; all age-binned drops in
the final window were age three. It provides measured age pressure on a small
allocation. The more recent September 23 two-GPU refresh checks use synthetic
rewards and are less informative about useful training.

Historical measurements alone cannot isolate a lag effect after runtime changes.
The new experiment therefore runs a fresh lag-two control followed by lag six
in the same six-GPU allocation, with separate fresh models/optimizers/output
roots. Both arms use the same immutable image and differ only in identity and
lag. Stochastic async scheduling means their sampled batches need not match.

## Matched comparison

[Beaker](https://beaker.org/ex/01M3BHSY240VJHQ5JR4EG37E4S)

| Setting | Both arms |
|---|---|
| Allocation | One Holmes task, six B300 GPUs: EP2 trainer plus four TP1 engines |
| Duration | 24 optimizer updates per arm, sequential; one-hour minimum, two-hour hard timeout |
| Model/data | Original full-SFT HF policy and prepared GSM8K inputs under `gsm8k-parity/20260910-core-megatron-v1` |
| Batch | 32 prompt groups × four responses = 128 responses/update |
| Length | 4,096 response tokens; 6,144 context/pack tokens |
| Optimizer | Constant LR 1e-6, router auxiliary/z coefficients zero, no KL/entropy objective |
| Training | Packing, dynamic rows, no recomputation, router replay, guarded scoring skip; explicit scoring check every five updates |
| Serving | 32 requests/decode-graph batch per engine, full decode graphs, eager prefill, radix caching |
| Async | Refresh; 512 unfinished samples; completed queue one batch; whole-group submission/retry; TIS |
| Artifacts | Online W&B, full accepted-group rollout capture, phase timings and contracts; no model checkpoints, HF export or evaluation |
| Image | `01M3APTWSR4VH2PFYMMSPQ8TD2` |
| Application/MILES | `96e215b334a43d40a2e6d0a942438dbbd2d681b6` / `9ecde5bc80cdadf2348600c49056449c74fb4ec3` |

The explicit 32-request admission reproduces the archived small workload; it
does not change the maintained production examples' 64-request starting point.
Constant-reward filtering is disabled, as in the archived workload. This lets
the age comparison retain its original batch accounting, but rewards and
nonzero-advantage fractions must be considered when interpreting throughput.
Both new arms disable auxiliary losses, unlike the September 14 source run;
only the new matched arms can isolate lag with the current objective/runtime.

The ordinary committed MILES task renderer supplies both single-node commands.
A disposable launch script executes them sequentially, stops Ray between arms,
and retains results separately under `/output/lag2` and `/output/lag6`.
The committed-image wrapper ran from a clean checkout at the image revision;
unrelated uncommitted local instrumentation was not included. No image rebuild
is needed to exercise the explicitly submitted lag values.

`plan`/`validate` passed for both arms. Before submission, assertions checked
the matched settings, one six-GPU task, immutable image, Holmes placement,
WEKA mount, minimum runtime and execution limit. Submitted metadata confirms
job `01M3BHSY5SJXN8ZC3Y1NV38SQ1` requests six GPUs in one task. At 05:52 UTC,
scheduler events report the shared `ai2/oe-scaling` allocation group at 160/160
slots. The job subsequently ran from 06:31:38 to 07:42:06 UTC and exited zero.

Local validation: 140 configuration, throughput-plan and documentation tests
passed; changed Python files pass Ruff lint/format checks, generated references
are current, and MkDocs builds successfully. The documentation check also found
and repaired two missing descriptions of existing compiler-cache fields; no
compiler-cache behavior changed. A read-only collector retains scheduler status
and job logs while this comparison waits/runs.

Disposable inputs, rendered and submitted specs, receipts and monitoring evidence
are under ignored `runs/lag-six-20260925/`. Remote output roots are
`/weka/oe-training-default/robertb/open-instruct/runs/lag-six-20260925/lag2`
and the corresponding `lag6` directory. W&B group: `lag-six-20260925`.

## Analysis contract

Compare updates 7–24 after reporting startup and early compilation separately:
useful response tokens/second, update cycles, trainer wait, training/publication
time, stale groups/tokens, accepted ages, reward/nonzero-advantage fractions,
response lengths, Core/behavior probability gaps and TIS clipping. Verify that
the lag-six arm actually trains on age >2 data; otherwise the test does not
exercise the changed allowance. Keep terminal unused work separate from
ordinary stale filtering. Cache warming and fixed arm order remain timing
confounds even on the same node.

This short single-seed screen has no held-out evaluation and cannot establish
learning-quality equivalence or the right lag for the larger hero workload.
The next learning comparison must measure quality per update and per hour.
The existing PPO `train/ess_ratio` is not a TIS effective-sample-size metric.

The completed hero qualification retained its submitted lag two. Its observed
updates 19–23 discarded 328 stale groups while accepting 320 groups, motivating
the investigation; that observation does not prove those discarded groups
would all survive reward filtering or contribute useful gradients at lag six.


## Results and operating decision

Both arms completed 24 updates. W&B: [lag two](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/wcopz6da),
[lag six](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/9g5y5nmn).
The following window is **updates 7–24**, inclusive. Cycles use elapsed driver
wall time from the first generation wait to the last publication, divided by 18.

| Measurement | Lag 2 | Lag 6 |
|---|---:|---:|
| Mean update cycle | 32.26 s | 26.25 s |
| Mean generation wait | 10.86 s | 6.00 s |
| Mean training | 18.07 s | 17.20 s |
| Mean publication | 3.33 s | 3.05 s |
| Accepted response tokens/s | 7,409 | 9,483 |
| Accepted response tokens | 4,302,487 | 4,479,991 |
| Stale response tokens discarded at completed queue | 1,476,463 | 0 |
| Stale fraction of completed-queue decision tokens | 25.55% | 0% |
| Stale groups discarded | 123 | 0 |
| Maximum accepted policy age | 2 | 5 |
| Accepted responses at age >2 | 0/2,304 | 1,012/2,304 |
| Mean Core/behavior absolute token log-probability difference | 0.01133 | 0.01253 |
| Mean TIS clipping fraction | 0.000676% | 0.000709% |
| Maximum update TIS clipping fraction | 0.001781% | 0.001802% |
| Mean gradient norm | 0.06434 | 0.06656 |
| Maximum gradient norm | 0.10464 | 0.07450 |
| Mean rollout reward | 0.8190 | 0.8021 |
| Mean response length | 1,867 | 1,944 |
| Response cap fraction | 13.67% | 15.67% |
| Mean constant-reward groups per 32-group batch | 21.22 | 19.56 |

Lag six improved accepted response-token throughput by **28.0%** and reduced
cycle duration by 18.6%. Accepted tokens include constant-reward groups: these
are not all advantage-bearing tokens. The reward/length shifts and different
nonzero-advantage mix are reasons not to treat this as a pure kernel speed test
or evidence of equal learning quality. Of lag-six responses, 972 had age three,
36 age four and four age five; the screen did not exercise age six.

Startup placement/serving/trainer/initial publication took approximately 509 s
for lag two and 454 s for lag six, separately from the first six update stages
(1,004 s and 956 s). Fixed arm order and warm caches remain confounds. Despite this,
the measured stale-waste reduction and comparable gradients/TIS clipping support
**lag six for the next hero duration exercise**. They do not establish an optimum
or justify changing learning rate, group size or length simultaneously.

Raw phase/queue artifacts are in Beaker result dataset
`01M3BHSY290Y5R0NDA5BH8HT03`, under `lag2/run/checkpoints/` and
`lag6/run/checkpoints/`; local retained histories and summary are under ignored
`runs/lag-six-20260925/`. Use `driver_timing.jsonl`, `rollout_flow.jsonl` and the
W&B histories to reproduce the table. Terminal unfinished work is not included
in the ordinary stale-drop counts.

## Eight-hour hero follow-up

[Submitted long run](https://beaker.org/ex/01M3BRJR449STQ9D1SVT45JH2Q) uses lag six
and a fresh copy of the same non-EMO SFT baseline. This preserves an independent
startup evaluation and learning curve; the completed R6 checkpoint remains
available for continuation. Runtime/image and the remaining training/admission
settings are unchanged from the qualified R6 recipe. Driver budget 8h,
protected minimum 8h, hard job timeout 10h; startup is charged to the driver budget.
Three nodes were scheduled together at 07:48:47 UTC. Job metadata verifies ranks
0/1/2, one replica group `01M3BRJR4CPF7YK2P5BY8XRN7P`, leader selection and
three distinct hosts.

The [completed hero run](hero-long-startup-20260924.md#completed-three-hour-qualification)
spent most steady-state time waiting for usable rollouts. Keep EP4 plus 19 policy
engines, 64 requests/decode graphs per engine,1,216 unfinished samples, judge
concurrency 16, recomputation and the other qualified optimizations. Changing lag
first targets measured discarded work; higher concurrency or a larger unfinished
budget is not yet qualified by this workload. Keep native saves every 25,
retention 2, final native/HF outputs, and background 128-example GSM8K/IFEval at
startup/every 50/final. The run stays in `ai2/open-instruct-dev`; background eval
allocations may be extra. Its rendered config and launch receipt are under
ignored `runs/hero-non-emo-long-lag6-20260925/`.


### Follow-up startup and evaluation repeatability

The three-node long run started at 07:56:18 UTC. Initial publication passed in
50.6 s, including exact tensor checks. The first full driver training stage took
916.1 s versus 936.1 s in R6; the actor's 584.9 s training timer excludes the initial
scoring work and must not be compared with the full driver stage. At 08:45 UTC,
eight updates had completed; accepted sample ages had reached five, with no
stale-group drops. Four transport failures were recovered by regenerating
pristine groups, and the recent failure window was zero. These are startup
observations, not a qualification of the full eight-hour duration.
W&B: [rlzb9wwq](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/rlzb9wwq).

The [startup evaluation](https://beaker.org/ex/01M3BT3VD87R5QM1E9BGRC8HFN)
completed successfully: GSM8K 53/128, IFEval strict prompt 24/128 and loose 29/128.
GSM8K’s 53/128 contrasts with R6’s 72/128 on the same original SFT checkpoint.
An artifact audit found **identical 128requests, native IDs, task settings and
pinned evaluator image**, but different automatically selected server RNG seeds
(299960870 versus 522344102) with temperature 0.8. Both evaluations hit the response
cap 13 times; mean response lengths were 2,432 and 2,541 tokens. Paired correctness:
41 both right,44 both wrong,31 R6-only and 12 new-only. This is observed baseline
variation; seed and asynchronous execution differ, and a seed-only causal
explanation has not been isolated. The small changes in R6's final evaluation
must not be read as established learning gains.

A separate [full GSM8K greedy baseline](https://beaker.org/ex/01M3BVS0SF91RY4NH7V6EAKZ1E)
was launched to support a less noisy final-checkpoint comparison: all 1,319 test
examples, zero shot, temperature 0, `do_sample=false`, server seed 17 and the same
10,240-token cap. It uses one extra Holmes GPU, one-hour minimum and three-hour
hard timeout. The default task's completion prompt is retained; this is not a
chat-template evaluation. Greedy decoding does not establish bitwise determinism
across batch schedules or kernel choices. A matching final-checkpoint evaluation
is planned. These scores are kept separate from the sampled periodic W&B series
to avoid overwriting identical metric keys. Local receipts/results live under
`runs/hero-non-emo-long-lag6-20260925/greedy-full-gsm8k/`.

A startup-efficiency issue was also identified in source: `startup_cache.prepare`
includes `max_run_seconds` in its compiler-cache identity. Changing three hours
to eight hours therefore changes the namespace despite identical compilation
semantics; current worker receipts report cache misses. This is a candidate for
a narrow future cache-key fix. No patch or restart was imposed on this running
experiment. The shared cache currently covers Triton; this observation does not
establish reuse of all TileLang, CUDA-graph or other startup work.

### First 50 updates and prompt-format audit

At 09:40 UTC, the long run had completed 50 updates and retained a native
checkpoint with clock 50/50/50, all four rank states, model metadata and a matching
dataset-cursor checksum. Saving it took 77.1 s. The
[update-50 background evaluation](https://beaker.org/ex/01M3BYVBTB84QPEYNTG51RMTF2)
started successfully. Updates 43–47 averaged 64.5 s per cycle: 31.6 s waiting,
24.6 s training and 8.4 s publishing. That window delivered 17,197 response
tokens/s with no stale-token drops. Treat this as an early window, not a final
sustained-throughput result; checkpoint/export milestones add overhead.

Training means through update 50:

| Updates | Raw reward | Response tokens | At length cap | Absolute log-probability gap | TIS clipped fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–10 | 0.447 | 4,414 | 13.09% | 0.02022 | 0.000055% |
| 11–25 | 0.461 | 4,527 | 15.23% | 0.02288 | 0.000399% |
| 26–50 | 0.443 | 4,437 | 13.63% | 0.02375 | 0.000619% |

These mixed-task, filtered training batches do not establish a learning trend.
Router load CV averaged about 0.54 after the first ten updates; no dead experts
were reported. Ten recoverable transport-error attempts had occurred by 09:27,
including repeated attempts for the same group; all three replicas continued,
with no fatal error. Errors at 09:22:29 and 09:23:55 occurred during training,
not the checked publication windows. Their underlying network cause remains
unisolated; successful recovery is not evidence that higher admission is safe.

The full greedy **raw-completion** baseline finished with 602/1,319 correct
(45.64%), but 490 responses (37.15%) reached 10,240 tokens. Mean length was 3,996
and median 398. Some capped responses invent subsequent `Question:`/`Answer:`
exchanges or repeat reasoning. Only 44 of those 490 capped responses scored
correctly. The task's normal completion stop sequences were removed for this
long-reasoning recipe, and the task never applies the checkpoint's chat template.
This is therefore a poor measure of chat response-length behavior; the final
comparison must retain that limitation rather than treating the cap rate as an
RL regression.

A separate [chat-template audit](https://beaker.org/ex/01M3BZ9QF7PTCDF9F0CEW38A81)
now compares the original SFT and immutable update-50 HF snapshot on all 1,319
GSM8K test questions. It uses the existing committed `gsm8k_test_eval.py`, the
checkpoint's own chat template and `GSM8KVerifier`, greedy decoding, seed 17,
10,240 response tokens and one additional B300. The historical script's context,
KV-token capacity and Mamba-state capacity are explicitly configured to 12,288,
786,432 and 1,024; its remaining serving settings include concurrency/decode
graphs 64 and disabled radix caching. The script is carried byte-for-byte from
commit `96e215b334a4` because the minimal runtime image omits it. A first audit
attempt failed immediately on that missing import, before inference; the linked
retry includes the script. The runtime image is unchanged. This audit also
changes serving batch settings and the scorer relative to the raw audit, so it
is not a controlled prompt-only attribution. Results and the eventual matching
final-checkpoint chat evaluation belong to a separate series.

### Full chat audit and hundred-update milestone

The chat audit completed successfully at 10:21 UTC. It confirms that greedy
length degeneration is also present with the proper chat template; changing
prompt format alone does not explain the raw audit's length failures.

| Full GSM8K chat audit, 1,319 questions | Original SFT | Update 50 |
| --- | ---: | ---: |
| Correct by last-number verifier | 688 (52.16%) | 701 (53.15%) |
| Reached 10,240-token cap | 723 (54.81%) | 707 (53.60%) |
| Correct **and finished before the cap** | 538 (40.79%) | 562 (42.61%) |
| Capped responses scored correct | 150 | 139 |
| Capped responses that closed `</think>` | 3 | 3 |
| Mean response tokens | 6,277 | 6,180 |
| Median response tokens | 10,240 | 10,240 |
| Generation time, one B300 | 881.7 s | 846.2 s |

Inspected capped chat responses repeatedly reconsider arithmetic instead of
finishing the answer. The last-number verifier can credit an unfinished thought,
so ordinary accuracy and correctly finished answers should both be reported.
There were 195 baseline-only and 208 update-50-only correct answers; the paired
exact McNemar p-value is 0.55. For correctly finished answers the corresponding
counts are 168 and 192, p=0.23. The direction is mildly favorable, but neither
comparison establishes a learning gain. This is greedy decoding on GSM8K;
the temperature-1 mixed-task RL batches have different length behavior.
Engine startup was 277.3 s for the baseline and 61.3 s for update 50 in the same
allocation, so startup timing is confounded by order and warmed caches.
Per-question responses, timing and paired summaries are retained in the linked
Beaker result and locally under `chat-gsm8k/results/` in this run directory.

The sampled 128-question update-50 panel also completed: GSM8K 65/128 versus
53/128 at startup, IFEval strict 22/128 versus 24/128, and loose 27/128 versus
29/128. Those small panels are mixed and subject to the baseline variability
already described.

At 10:52 UTC, the main run had completed 101 updates. Its update-100 native
checkpoint passed clock 100/100/100, four-rank and cursor-checksum checks; the
[update-100 evaluation](https://beaker.org/ex/01M3C2ZR69F9HB4JMZAZW4V57A) was
running. The milestone training stage, including its HF snapshot, took 127.2 s,
and the separate native save took 81.8 s. No main-fleet restart had been needed.

| Training-cycle window | Cycle | Wait | Training | Publication | Stale decision-token fraction | Accepted response tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lag 6, updates 26–50 | 72.4 s | 33.1 s | 31.2 s | 8.1 s | 0.27% | 15,694 |
| Lag 6, updates 51–75 | 80.4 s | 45.1 s | 26.1 s | 9.2 s | 0.53% | 14,123 |
| Lag 6, updates 29–48 | 75.7 s | 38.7 s | 28.9 s | 8.1 s | 0.35% | 14,870 |
| Prior R6 lag 2, updates 29–48 | about 233 s | 198.5 s | 28.1 s | 6.7 s | 53.19% | 4,920 |

Windows start at the first generation wait and end after the last publication;
checkpoint writes outside those endpoints are excluded. Training includes any
HF evaluation snapshot within its stage. These are separate hero runs with
stochastic sampling and filtering, not a controlled large-scale A/B. The small
matched comparison above remains the controlled throughput screen. The longer
windows nevertheless support retaining lag six for the ongoing duration test.
