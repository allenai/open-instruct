# Provisional lag-six default and small comparison

September 25, 2026. **Submitted; no comparison results yet.** Six is a deliberate
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

## Submitted comparison

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
slots. The experiment remains queued in `ai2/open-instruct-dev`.

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

The ongoing hero qualification retains its submitted lag two. Its observed
updates 19–23 discarded 328 stale groups while accepting 320 groups, motivating
the investigation; that observation does not prove those discarded groups
would all survive reward filtering or contribute useful gradients at lag six.
