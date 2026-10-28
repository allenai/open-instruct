# Research

## Temporary Olmo Hybrid MILES port

Status: ACTIVE.

Can the five released MainlineLadder sizes serve correctly under SGLang and then
support native Core RL without maintaining another inference fork?

Decision: reuse the release FLA Gated DeltaNet and keep its negative-eigenvalue
semantics; use SGLang RadixAttention for full attention. Restrict the temporary
bridge to eager TP1 with no prefix sharing or chunked prefill. GDN states are
per request slot and outside SGLang's allocator. This trades serving throughput
for a smaller implementation and a direct correctness check.

CPU finding: the actual pinned Core runtime builds the 275m native descriptor on
meta and matches all 192 released HF tensor names and shapes. Strict interchange
regressions pass. The corrected 275m GPU serving test passed (including reload, long prompts, and slot reuse); the expanded queued-request/FP32-state 275m check also passed. All five scales now pass with the BF16 reference-backbone fallback; end-to-end RL qualification remains pending.

Harness finding: the pinned SGLang returns a prompt-logprob suffix with an initial
`None` even when logprob_start_len is nonzero. Align by original token position
instead of dropping that entry and shifting the reference. This caused the first
batch attempt to fail independently of model correctness; CPU regressions cover it.

See [the GPU launch batch](experiment.md#five-scale-sglang-midtrain-qualification-2026-10-27)
and [the implementation guide](docs/miles/hybrid-mainline.md).


Serving finding (2026-10-27): all five final-midtrain sizes passed the BF16
reference-backbone fallback on Rhea: 160 functional requests and 40 dense probes,
including long prompts, queued requests, slot reuse, and disk reload. Maximum
selected-token logprob error stayed below 5e-4 and greedy agreement was 100%.
This selects `MILES_HYBRID_REFERENCE_BACKBONE=1` with BF16 logits for the temporary
recipe. It localizes the earlier discrepancy to differences introduced by the
faster packed/RadixAttention path, without isolating one specific operation.
That path remains unqualified across the whole family. The fallback is slower
(roughly 62–67 output tokens/s at 275m and 20–22 at 2.7b for these warm batches).
HF cached-versus-full discrepancies remain, so native RL publication, importance
ratios, restart recovery, and learning evidence are still pending; the overall
MILES integration question remains ACTIVE.


## GSM8K 500-update comparison of Core/vLLM and MILES

Status: ACTIVE — 450m smoke passed; both 500-update jobs scheduled; 275m deferred.

Do the two stacks show useful learning and acceptable throughput on the275m and
450m final-midtrain releases? Four fresh runs use500 updates, eval/checkpoint
every100, BF16, and a hard one-update policy-age limit. Preserve existing reward
and optimizer differences per user instruction, except match PPO upper clipping
to0.272. Compare held-out scores, valid reward groups, wall time/update, serving
throughput, consumed policy age/stale drops, and checkpoint/restart behavior.
The initial serving matrix qualifies the BF16 HF fallback, not the native RL path.

See [the preparation batch](experiment.md#gsm8k-infrastructure-comparison-275m-and-450m-500-updates-2026-10-27).


450m-first update (2026-10-27): user approved only the two 450m stacks; 275m GPU runs deferred. Run one two-update MILES smoke on two Rhea A6000s, then the two 500-update jobs (four GPUs peak) only if it passes. Initial smoke [01M5ZFP2Z9D41ZFW6YW4AJR85C](https://beaker.org/ex/01M5ZFP2Z9D41ZFW6YW4AJR85C) was stopped before training: the hard-coded 256 GiB RAM request caused three GPU slots to be allocated. Launcher now supports explicit host memory; comparison requests use 128 GiB RAM and 32 GiB shared memory. Native RL validation is still pending.

450m native smoke [01M5ZFZGNFPMQ8F3PPVXG3EACQ](https://beaker.org/ex/01M5ZFZGNFPMQ8F3PPVXG3EACQ) acquired exactly two A6000s and passed the BF16 attention forward/backward preflight, then failed before training: Ray manager rendered SGLang ServerArgs before registering mainline_ladder. Register the hybrid Transformers config in that manager before command rendering; a runtime regression checks registration order. Long runs remain gated on a successful retry.

Registration retry [01M5ZGAG9ZB0NSFGN1172QJCNB](https://beaker.org/ex/01M5ZGAG9ZB0NSFGN1172QJCNB) passed manager registration, but SGLang dtype=auto chose a non-BF16 computation type and the bridge rejected startup. No learner updates. The controller then waited until the one-hour job timeout. Set sglang_dtype=bfloat16 explicitly and opt these recipes into a 600-second serving-startup deadline.


### 450m-first launch update (2026-10-27, queue refresh 2026-10-28 UTC)

User narrowed the approved comparison to 450m: one two-A6000 MILES smoke,
then two 500-update jobs, one learner plus one inference A6000 each (four peak),
only after the smoke succeeds. All GPU jobs use ai2/rhea, normal/preemptible,
minRuntime 0, budget ai2/oe-other and workspace ai2/open-instruct-dev.
Main jobs retain auto-resume, three retries, 24h per attempt, BF16,
max policy lag 1, eval/checkpoint every 100, and PPO clip lower0.2/upper0.272.
275m configs are prepared but no new275m GPU runs are authorized by this batch.

Smoke attempts:
- [01M5ZFP2Z9D41ZFW6YW4AJR85C](https://beaker.org/ex/01M5ZFP2Z9D41ZFW6YW4AJR85C): stopped before training; hard-coded256GiB host RAM requested a third slot. Fixed configurable128GiB RAM/32GiB shm; later attempts acquired exactly two GPUs.
- [01M5ZFZGNFPMQ8F3PPVXG3EACQ](https://beaker.org/ex/01M5ZFZGNFPMQ8F3PPVXG3EACQ): BF16 attention forward/backward passed; failed before training because Ray manager rendered ServerArgs before registering mainline_ladder. Fixed in pushed miles-olmo-core hybrid branch, commit a5ec0c61.
- [01M5ZGAG9ZB0NSFGN1172QJCNB](https://beaker.org/ex/01M5ZGAG9ZB0NSFGN1172QJCNB): registration passed, but SGLang dtype=auto selected non-BF16; server rejected startup and controller waited until the one-hour timeout. Fixed explicit sglang_dtype=bfloat16 plus a600-second serving-startup deadline in companion commit3f0933565.
- [01M608040GER8BMAC4E8DRQ7Q7](https://beaker.org/ex/01M608040GER8BMAC4E8DRQ7Q7): corrected BF16 smoke, currently queued because Rhea lacks two free slots. Image01M607Z48PYPNN0KHVZ5PGYJEG.

Native MILES learning remains unvalidated. A one-time Hammond controller,
`scripts/miles/hybrid_compare_450m.py` in the companion checkout, waits for this
smoke and checks two completed updates, successful publication/evaluation,
committed native checkpoints and final HF export before submitting the two main
runs through build_image_and_launch.sh. It fails closed on missing evidence,
worker failure or an interrupted submission. Durable controller status and launch
receipts live in `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/runs/compare-450m-controller-20261028`.

2026-10-28 update: corrected BF16 smoke 01M608040GER8BMAC4E8DRQ7Q7 succeeded on its replacement job 01M609PQ4J14ACC3QY9GWY60M5 (04:33–04:45 UTC); the first attempt failed its node healthcheck before starting. Durable gate verified two updates, publication/eval and committed checkpoints/HF export. Main submission 01M60D8RYBPESB5NCH5PPVQA3C was immediately stopped before scheduling: Beaker retry policy was missing despite launch.max_retries=3. Emit finite retry caps in the Beaker spec alongside the existing native attempt ledger; retry the authorized controller under compare-450m-controller-20261028-retry.


2026-10-28 05:04 UTC launch update: corrected BF16 smoke completed both learner
updates with successful publication/evaluation stages, two committed native
checkpoints and final HF export. The retry-policy guard caught a missing Beaker
retry field; its first main submission was stopped before allocation. Fix pushed
in companion commit f22385319 (100 focused tests and lint/type checks passed).

The approved main pair is now submitted and scheduled, each with exactly two
A6000s on Rhea. Verified autoResume=true, minRuntime=0s, allowedTaskRetries=3,
500 updates, eval/checkpoint100, BF16, max policy lag1. No275m comparison submitted.
- MILES: [01M60E822XVYFNHWJ97N6WDBY2](https://beaker.org/ex/01M60E822XVYFNHWJ97N6WDBY2), image01M607Z48PYPNN0KHVZ5PGYJEG.
- Core/vLLM: [01M60E9H8YVK2VJNWPE4BHBFM0](https://beaker.org/ex/01M60E9H8YVK2VJNWPE4BHBFM0), image01M6087KP6KYPHQ9MMZTEX9SBJ.
Controller status: both_submitted, under runs/compare-450m-controller-20261028-retry
on Weka. Main-run optimizer progress and learning quality are not yet verified.

Separate Saturn275m/1k run01M5ZAPEQFTT3YEMK107ZSVFK4 has suffered six scheduler
preemptions by urgent allocated workloads. Current job01M604JS88XG6RFSGNF2Z0FWB1
has waited almost three hours; scheduler reports no slots on any of27 eligible
nodes. Cluster snapshot:216slots, zero available,161queued jobs. Run unchanged.
