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
