# Hero non-EMO long-run startup investigation

As of September 24, 2026, 17:13 UTC, these long-run attempts have completed **no
optimizer updates** and produced no trained checkpoint or long-run quality result.
The latest cache-fix attempt is queued. The earlier
[four-update hero smoke](hero-rl-smoke-20260924.md) remains separate evidence.

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
| [Cache-fix retry](https://beaker.org/ex/01M3A6D4H0ZVVW07GGHCCWBX2Q) | `01M3A6AV5HDMP64ZMY8HXH46H0` / `cacc219018ee` | Queued at the timestamp above; native three-replica grouping verified |

The cache-fix retry preserves the training recipe, adds a 600-second publication
allowance and diagnostics, and incorporates the evaluator settings below.
One-off TOMLs, submitted specifications, receipts and logs are retained in ignored
`runs/hero-non-emo-long-20260924{,-r2,-r3}/`, with the corresponding run output
roots under `/weka/oe-training-default/robertb/open-instruct/runs/`.

## Publication timing remains unresolved

The inference engines first load the HF checkpoint from storage and complete
serving startup. The trainer then independently loads that checkpoint and builds
the Core training representation. Before admitting training rollouts, the driver
publishes the trainer's current weights through the same transport used after
updates. Thus the initial publication updates already-populated engines; it is
not their initial checkpoint load. This also makes restored trainer state
authoritative when resuming. This run enables `check_weight_update_equal`, so a
fresh run compares the published weights against the initial serving snapshot
before proceeding. The failed attempt did not complete publication or reach
that comparison. See `open_instruct/miles/driver.py:train` and
`open_instruct/miles/models.py:build_train_module`.

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
test again. Ruff and formatting checks passed. The nineteen-engine runtime
qualification is still pending capacity.

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
