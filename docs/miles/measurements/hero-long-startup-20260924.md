# Hero non-EMO long-run startup investigation

The cache-fix attempt completed **six optimizer updates** on September 24, 2026,
then failed at router readiness confirmation after a successful weight transfer.
It produced baseline background evaluations, but no trained checkpoint (saving
was scheduled every 25 updates). A replacement with longer infrastructure
deadlines, warnings and a router response-forwarding fix is being prepared.
The earlier [four-update hero smoke](hero-rl-smoke-20260924.md) is separate evidence.

## Run and preparation

The intended exercise uses the corrected-tokenizer non-EMO Dolci Think SFT
step5402 checkpoint:

```text
/weka/olmo-3p5-checkpoints/scratch/olmo35-fixedtok-sft-20260921/olmo35-fixedtok-sft-20260921-4t-non-emo-dolci-think/emo/step5402/hf
```

The fleet is one native Beaker replica group of three eight-GPU Holmes nodes:
four trainer GPUs with EP4, nineteen TP1 policy engines, and one Qwen3-32B judge.
Each policy engine admits 64 requests and captures full decode graphs through
64; prefill graphs are disabled. The producer budget is 1,216 unfinished samples,
with a one-batch completed queue and judge concurrency 16. Training uses
64 prompts × four responses, a 10,240-token response cap, 2,048-token prompt cap,
12,288-token packs, recomputation, dynamic rows, guarded scoring skip, refresh,
TIS and zero router auxiliary/z loss. The intended duration is eight hours with
a ten-hour hard job limit, native saves every 25 updates, retention of two saves,
and a final HF export. Background GSM8K and IFBench evaluation is requested at
startup, every 50 updates and completion, with 128 examples per task.

[CPU preparation](https://beaker.org/ex/01M3A0GJ8YGWAJWBEYTTBBHTGH) succeeded.
It rerendered the frozen Dolci mixture for the hero checkpoint and retained
101,434 training rows and four 128-example holdouts, with no additional
overlength exclusions. Earlier baseline preparation had already applied its
filters and holdout selection. Outputs are under:

```text
/weka/oe-training-default/robertb/open-instruct/data/hero-non-emo-basket-20260924
```

## Attempts

| Attempt | Image / application source | Outcome |
|---|---|---|
| [Initial long run](https://beaker.org/ex/01M3A1BG4RQP012BY5PAD822Z6) | `01M3A0F9W9K6C4RNJGP6814SHE` / `999a9a08c455` | All engines and trainers initialized; initial publication timed out after 180 seconds |
| [Instrumented retry](https://beaker.org/ex/01M3A4EWJHJ9ZZEBMSFV17WCAW) | `01M3A4D5CNA72MH1TVEN7D8NFR` / `aca051d369ea` | One engine crashed while importing a partial cached configuration; stopped before publication |
| [Cache-fix retry](https://beaker.org/ex/01M3A6D4H0ZVVW07GGHCCWBX2Q) | `01M3A6AV5HDMP64ZMY8HXH46H0` / `cacc219018ee` | Six updates; initial sync and all later transfers succeeded; router readiness HTTP read timed out after update six |

The cache-fix retry preserves the training recipe, adds a 600-second publication
allowance and diagnostics, and incorporates the evaluator settings below.
One-off TOMLs, submitted specifications, receipts and logs are retained in ignored
`runs/hero-non-emo-long-20260924{,-r2,-r3}/`, with the corresponding run output
roots under `/weka/oe-training-default/robertb/open-instruct/runs/`.

## Original initial-publication stall

The inference engines first load the HF checkpoint from storage and complete
serving startup. This run enables `check_weight_update_equal`: MILES saves a CPU
snapshot of the loaded engine tensors, then deliberately replaces the checked
tensors with random values. Nonpersistent buffers and explicitly excluded tensors
are skipped consistently by reset and comparison. The failed run's logs confirm
both `snapshot` and `reset_tensors` actions before trainer publication.

The trainer independently loads the checkpoint and builds the Core training
representation. Before admitting training rollouts, the driver publishes the
trainer's current weights through the same transport used after updates, then
compares them against the initial serving snapshot. Resetting prevents an omitted
tensor from passing just because its original HF value remained intact. Thus the
engines do initially load HF weights, but in this checked configuration they
require the trainer publication to restore valid serving weights. Publication
also makes restored trainer state authoritative when resuming. The failed attempt
did not complete publication or reach comparison. See
`open_instruct/miles/driver.py:train`,
`open_instruct/miles/models.py:build_train_module`, and the pinned MILES
`miles/ray/rollout/server_cell.py:_tick_when_initializing`.

All timestamps in this table are UTC on September 24:

| Initial attempt event | Time |
|---|---|
| All three job processes started | 15:50:17 |
| Nineteen policy engines ready | About 16:08:26 |
| Trainer initialization completed; initial-publication timer began | 16:14:26 |
| All nineteen engines acknowledged update-group setup and begin-update | 16:14:34 |
| Publication deadline expired | 16:17:26 |
| Driver exited after cleanup | 16:19:49 |

The model/service startup time did **not** consume the publication deadline.
Serving readiness has its own 30-minute allowance. The refresh driver uses
`core.engine_update_timeout` for both initial and later publications; there is
currently no separate first-publication setting. The earlier barrier smoke did
not apply this deadline.

For comparison, the earlier H100 EP4/one-engine **non-EMO** smoke measured:

| Operation | Measured time |
|---|---:|
| First actor export/transfer/finalization | 4.85 s |
| Subsequent actor export/transfer/finalization | 1.30–2.01 s |
| Complete initial-publication stage, including validation | 18.59 s |

Each transfer carried 23,441 tensors, 24,992,380,160 bytes, in 24 buckets.
These are observations from the one-engine smoke, not performance guarantees for
nineteen engines across three nodes. The initial long attempt recorded no completed
publication. Trainer GPU 0 was idle while its other three GPUs were busy during
the stall; no specific straggler or root cause has been established.

Opt-in diagnostics now identify host/PID, trainer rank/GPU UUID, engine address,
individual engine request start/completion/error, and export/synchronization/
broadcast/loading progress. Actor Python stacks are dumped every minute while
publication is active. The retries enable NCCL `INFO` logging for `INIT,NET`.
See [operations](../operations.md#failure-triage). The longer deadline is an
investigation allowance, not a claimed fix.

### Scale versus regression audit

Comparing the successful hero smoke application revision `d5b60f2ebbd2` with
the first failed long-run revision `999a9a08c455` shows no changes to `actor.py`,
`publication.py`, `models.py` or `moe_models.py`. The driver's changes add graceful
run deadlines and final save/export/evaluation handling; its initial-publication
path is unchanged. Core, olmo-sglang and the runtime base image retain the same
pins. MILES moved from `cd0cbe5cc08d` to `19393c0672c3`: runtime changes concern
rollout capture sampling, with no transfer implementation changes. The later
publication logging and Python-module cache fixes postdate the first failure.

The experiments nevertheless change several relevant conditions together:
one H100 node / one engine / barrier publication becomes three B300 nodes /
nineteen engines / refresh publication, with larger serving allocations.
Barrier mode did not enforce the refresh deadline, but the hero smoke's measured
initial transfers were about five seconds, so its success was not simply due to
waiting longer than 180 seconds. Tiny-model refresh passed on both GPU types in
the [September 23 checks](refresh-sanity-20260923.md). An older checkpoint and
runtime completed 100 updates with cross-node publication to eight engines in
the [September 12 run](two-node-async-gsm8k-20260912.md); that run's earlier
publication-boundary timeout was producer cancellation, a different failure.

This is the first recorded occurrence of the specific hero initial-transfer
stall in the reviewed evidence. Topology/transport at scale is a plausible
suspect, not an isolated cause. The cache-fix retry subsequently completed initial
publication and six updates at this topology; that does not isolate the original stall.
A subsequent reproduction should preserve the phase and conditions identified by
its traces, rather than assume that an additional one-engine smoke rules out a
nineteen-engine or inter-node problem.

## A distinct Python-module cache race

In the instrumented retry, one engine on node `10.93.1.244`, physical GPU 5,
crashed before loading its model. Eighteen other engines loaded successfully.
Its traceback ended in the cached `configuration_olmo3moe.py` with:

```text
NameError: name 'dense_mlp' is not defined
```

The retained checkpoint source is 8,800 bytes, with SHA-256
`49ed9aca7406dd62cc43f317ce81e13eaa26d94adb310f1af88c6e10dd2fd7ba`.
Its first **4,096 bytes** end exactly at:

```python
self.dense_mlp_intermediate_size = dense_mlp
```

In the pinned runtime, executing the full source and constructing the config
succeeds; executing that 4 KiB prefix reproduces the exact observed `NameError`.
The pinned Transformers implementation uses `shutil.copyfile` to populate local
custom-code cache entries, without atomic replacement or an interprocess lock.
This strongly supports a partial-copy race between concurrent importers. The
original node's transient file contents were not captured during the write.
There is no evidence that the checkpoint weights were damaged.

The Open Instruct fix assigns a private `HF_MODULES_CACHE` under `/tmp` to each
trainer worker and serving engine, independently of Triton caching. The serving
parent loads its config and tokenizer before launching SGLang children, completing
their shared module copies before concurrent imports. Model/download caches remain
shared. This fixes our worker startup path; it does not change Transformers'
copying behavior for unrelated applications or standalone serving launches.

Validation: 27 focused tests passed, including four fresh concurrent child
interpreters loading a primed custom config/tokenizer with any module-copy attempt
made fatal. The final tokenizer-mode preservation change passed that integration
test again. Ruff and formatting checks passed. The cache-fix retry subsequently
initialized all nineteen engines without reproducing the import failure.

## Evaluator execution fixes

The independent evaluator smoke uncovered three issues before a real mechanics
pass:

- Implicit prefill graphs hit a KDA shape error; explicitly disable prefill graphs.
- JSON `stop_sequences=[]` made frozen sampling parameters unhashable; olmo-eval
  commit `e2f3ef0ef185040d68ab3f37f0611abce520918a` normalizes these lists to tuples.
- Chat requests inherited invalid `top_k=0` from generation defaults. The evaluator
  now uses `--sampling-defaults openai` and explicit `do_sample=true`, `top_k=-1`,
  `top_p=1.0`. Open Instruct also rejects missing model outputs even when the
  harness exits zero and reports no failed instances.

The [successful smoke](https://beaker.org/ex/01M3A35E12VARJSK8JC8FY109A) used
evaluator image `01M3A2FDS2036RBJMXP2YB4XDT` and produced actual responses for both
GSM8K examples and both IFBench examples. Its two examples per task and 256-token
cap establish execution only. The scheduled full evaluations retain the requested
128 examples per task and 10,240-token cap; no full baseline score is reported here.


## Six-update result and distinct router timeout

[Cache-fix run](https://beaker.org/ex/01M3A6D4H0ZVVW07GGHCCWBX2Q)
([W&B](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/ashx9ttq)) ran from
18:06:14 to 18:57:26 UTC. All three replicas shared one native replica group on
three distinct Holmes B300 nodes. All nineteen engines initialized.

| Measurement | Observed |
|---|---:|
| Initial actor publication | 15.158 s |
| Initial publication stage including HF snapshot comparison | 43.017 s, passed |
| Actor publications after updates 1–6 | 7.453, 6.928, 5.900, 5.869, 7.695, 5.705 s |
| Completed optimizer updates on each of four trainer ranks | 6; none skipped |
| Trainer stage duration, updates 1–6 | 871.18, 311.65, 253.71, 79.10, 62.08, 61.34 s |
| Peak allocated trainer memory, rank 0–3 | 102.67, 120.98, 138.18, 119.06 GiB |
| Last mean behavior/Core log-probability error | 0.020372 over 676,251 active tokens |
| Startup GSM8K exact match | 0.484375 (128 examples) |
| Startup IFBench strict / loose prompt accuracy | 0.2109375 / 0.234375 (128 examples) |

These evaluation scores describe the starting SFT checkpoint, not trained-model
quality. Initial training times include cold setup; the final three stages are
not enough to establish sustained throughput.

Every publication delivered 24 buckets, 23,441 tensors and 24,992,380,160 bytes.
The last transfer finished at 18:55:27.661 UTC. The subsequent call chain was
`train/group.update_weights → inference_controller.end_update_weights →
server_cell.mark_weights_ready → router_api_client.add_worker`. The router
`/add_worker?...&weights_ready=true` request hit its separate **30-second HTTP
read timeout** at 18:55:57.68. The publication stage lasted 35.784 seconds,
not the configured 600-second Core deadline. Cleanup then correctly refused to
reuse engines after incomplete readiness confirmation. No save/final export ran.

### Response forwarding hypothesis and patch

The router parsed and serialized each generation response as JSON on the same
async event loop that handles readiness requests. Routing replay makes those
responses large: at 12,288 tokens, 16 layers and top-16 expert indices, the base64
routing field alone is about 16.8 MB. The patch forwards the HTTPX-decoded body
bytes directly, retaining status/content type and dropping obsolete length,
transfer and content-encoding headers. Tests cover exact JSON bytes, binary
responses, errors and compressed upstream bodies.

A local CPU synthetic burst of already-ready responses measured event-loop
scheduling delay of 2.641 s for 64 responses, 10.342 s for 256 and 51.264 s for
1,216 with the old JSON conversion; raw forwarding measured 0.000308, 0.000663
and 0.002991 s respectively. This demonstrates a plausible mechanism exceeding
30 seconds. **It is not a measured production burst or a proven root cause.**
Raw receipts are retained under ignored `runs/hero-non-emo-long-20260924-r3/`.

Readiness confirmation now logs router/worker/attempt/elapsed time and retries
transport timeouts or network failures up to three attempts. Only the idempotent
`weights_ready` confirmation is retried; HTTP rejection remains fatal. Router
logs record readiness, active requests and worker epoch. Confirmation still occurs
after every publication because it can readmit a quarantined engine.

### Replacement timeout policy

The replacement sets `MILES_INFRA_TIMEOUT_MULTIPLIER=5` in the replica environment.
The code default is 1. Infrastructure waits warn at the earlier of 30 seconds or
the old deadline, repeat every 30 seconds, and report completion after a slow wait.
Warnings identify the operation, endpoint/request/cell where available, elapsed
time and effective deadline. Driver/startup stage warnings also run from a thread
so blocking GPU/Python work cannot silence those stage warnings. Async request
watchdogs use tasks, not a thread per generation.

| Wait | Previous | Replacement |
|---|---:|---:|
| Router control HTTP read/write/pool | 30 s | 150 s per attempt |
| Router control connect | 10 s | 50 s |
| Core publication | 600 s | 3,000 s |
| Refresh request | 1,800 s | 9,000 s |
| Engine drain | 900 s | 4,500 s |
| Cell initialization tick | 120 s | 600 s |
| Engine startup state deadline | 1,800 s | 9,000 s |
| Fleet readiness | 3,600 s | 18,000 s |
| Judge HTTP request | 600 s | 3,000 s |
| Replica heartbeat expiry | 120 s | 600 s |
| Replica startup | 1,200 s | 6,000 s |
| Background evaluation submission | 30 s | 150 s |
| Trainer distributed timeout (explicit run option) | 10 min | 50 min |
| SGLang watchdog (explicit run option) | 300 s | 1,500 s |

HTTP health probes, abort discovery/requests, producer joins and cleanup waits
are also multiplied. Existing unbounded waits stay unbounded. Polling cadence,
verifier execution budgets, generation token caps and the eight-hour training /
ten-hour Beaker execution budgets retain their meanings; this does not globally
rewrite third-party internal timers. The run retains concurrency 64, budget
1,216, judge concurrency 16 and recomputation. Native saves move to every five
updates (keep two) to retain progress; measure their overhead separately.
