# Experiments

## Five-scale SGLang midtrain qualification (2026-10-27)

- User requested GPU serving tests at all five hybrid scales, midtrain first,
  and selected Rhea A6000 instead of Saturn A100 for this qualification.
- [Beaker](https://beaker.org/ex/01M5ZA7ZXSFH1T18AQFMX8P9RQ), image
  `01M5ZA7AP531QVWGSAGSC8M5B8`, source MILES `ce1d9564`, release plugin
  scaling-ladders `0fb485ce38733103f7da73973cdf0e74b0fc75db`.
- Cluster ai2/rhea; 1 RTX A6000 48GB, normal priority, minRuntime 0,
  zero retries, timeout 2h; budget ai2/oe-other; workspace ai2/open-instruct-dev.
  Verified Rhea driver 580.173.02 supports the CUDA 13 base runtime.
- BF16, TP1/PP1, Triton full attention, release FLA GDN, no graphs or radix cache,
  no chunked prefill. max_running_requests=4, mem_fraction_static=0.35.
- Final midtrain steps: 275m/450m 0038147, 810m 0023842,
  1.4b/2.7b 0011921, all on the general hybridish-local HF tree.
- Prompt lengths 1/3/31/127/511/2047/3071; mixed batches; 32 greedy cached
  decode steps per request; slot reuse; full disk weight reload followed by repeat.
  HF probabilities on the served continuation must differ by <=0.08 logprob;
  >=95% of generated tokens must be within 0.01 logprob of the HF argmax.
- Every scale runs separately (20-minute cap each), with per-size JSON/logs and
  aggregate failures. Durable output:
  `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/qualification/rhea-ce1d9564`;
  reports also copied into Beaker `/results/hybrid` after each model.
- Status at submission: scheduled, pulling image. No GPU pass claimed.


### Interactive continuation

- The initial batch attempt loaded/decoded the first three sizes but failed in
  the harness's prompt-logprob alignment. SGLang prepends a `None` entry to the
  returned suffix even for nonzero logprob_start_len. Dropping it shifted the
  comparison. Fixed by retaining token positions; three CPU regressions pass.
  The initial attempt was stopped; it provides no numerical parity pass.
- At the user's request, moved to interactive session
  `01M5ZB6KJDNFG7KYKWHN8X948W`, Rhea node `rhea-cs-aus-473.reviz.ai2.in`,
  one A6000, normal priority, min-runtime 0, timeout 90m, same workspace/budget.
  BF16 CUDA matmul passed. The installed image is unchanged; experiments import
  the live Weka Git checkout through PYTHONPATH, with no further image rebuilds.
- Runtime checkout: `/weka/oe-adapt-default/michaeln/miles-olmo-core-hybrid`;
  companion checkout: `/weka/oe-adapt-default/michaeln/open-instruct-miles-hybrid`.
  Both use branch `codex/hybrid-mainline`. Experiments run in tmux `hybrid-tests`.
- Corrected 275m baseline, source `3682f0f0`: PASSED 10 requests before and 10
  after disk weight reload; max selected-token logprob error 0.069979, minimum
  greedy agreement 1.0. Warm short batches about 88 output tokens/s including
  prefill and request overhead. This is serving evidence, not RL training.
- Expanded matrix source `d023770c` adds six queued requests with different
  completion lengths (8/32/16/48/8/32), exceeding max_running_requests=4, and
  asserts FP32 recurrent state during every GDN forward. This exercises new
  prefills while older requests retain state. The expanded 275m test passed;
  larger scales are running. Seven CPU tests pass in the real GPU image.
- Reports: `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/qualification/rhea-interactive-d023770c`.
  Per-model JSON includes source/config hashes and exact GPU/toolchain versions.

### Reference and precision diagnostics

- Cached-reference matrix `c3d46f12`: 810m PASSED all 32 functional requests
  (16 before/16 after reload) plus eight dense probes; max selected logprob
  error0.0779502, greedy agreement1.0, max KL(served||HF-cached)0.000466866.
  Other scales failed strict numerical checks, despite successful load/decode.
  Reports: `.../qualification/rhea-cached-c3d46f12`.
- Independent HF self-comparison at 810m reproduced cached-versus-full KL
  0.00176434 at prefix1, TV0.0244253, exactly matching the SGLang-versus-full
  discrepancy. FP32 output logits reduced this to 0.00151127, not below1e-3.
- FP32-head matrix `e4bedecf` retained every numerical case and reload phase.
  275m failed selected-token checks (max about0.0864); 450m failed a dense probe
  (max KL 0.00117072). 810m and1.4b passed. Interrupted2.7b before completion
  after the smaller-scale failures established this was not a universal fix.
  `STOPPED.txt` marks the partial matrix; 2.7b has no FP32-head pass.
- User confirmed BF16 training. Recipe explicitly keeps core.fp32_lm_head=false.
  Added opt-in `MILES_HYBRID_REFERENCE_BACKBONE=1`: the release HF/SDPA backbone
  and request-local hybrid cache under SGLang scheduling/sampling/reload APIs.
  It retains TP1/eager/no-prefix/no-chunking constraints; both GDN and full-attention
  KV live outside SGLang's memory planner. This is slower than the RadixAttention
  path, which remains available with the variable unset.
- BF16 fallback matrix `8bb6da4a` running with `--reference-backbone`, no FP32-head
  override, same checkpoints/resources/criteria, reports
  `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/qualification/rhea-hf-bf16-8bb6da4a`.

### Completed BF16 reference-backbone qualification

All five final-midtrain sizes PASSED on the same A6000 and harness/adapter
revision `8bb6da4a`, with BF16 model computation and BF16 logits (no FP32-head
override). Each passed 16 functional requests before plus 16 after disk reload,
and four dense-distribution probes before plus four after reload: 160 functional
requests and 40 dense probes total. Greedy agreement was 1.0 in every case.

| Size | Result | Max selected logprob error | Warm output tokens/s |
| --- | --- | ---: | ---: |
| 275m | PASSED | 1.23978e-05 | 62.1–67.1 |
| 450m | PASSED | 3.00407e-05 | 42.2–45.0 |
| 810m | PASSED | 0.000198603 | 39.2–45.0 |
| 1.4b | PASSED | 0.000464261 | 28.9–32.0 |
| 2.7b | PASSED | 9.9659e-05 | 19.8–22.4 |

Throughput ranges are complete warm batches after reload, including prefill and
request overhead; these are short synthetic serving measurements, not RL throughput.
Maximum measured cached-reference KL was0 after nonnegative clamping at float32
resolution; the small nonzero logprob errors above mean this is not exact equality.
HF cached-versus-full discrepancies remain (450m up to 0.00240536; 810m0.00176434),
so training/rollout importance-ratio calibration remains a separate GPU requirement.

Reports: `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/qualification/rhea-hf-bf16-8bb6da4a`.
Harness SHA256: `e0f9d51f60e463149cffd72ed6f7bec92e0d013ce12eec82830a49387c37307c`.
Adapter SHA256: `73e345ec258aded60a37bea5e93d70a5bc5f753199bc41b004c809a6ebf4df62`.
No checkpoint weights were modified. This qualifies the HF-backbone fallback;
the packed/RadixAttention path does not pass the same matrix at every scale.
Native MILES learner updates, live publication, restart recovery, and learning
quality remain pending. CPU validation:19 runtime tests and148 companion tests
passed; scoped Ruff and companion formatting/lint/compile/type checks passed.

Saturn rerun status refresh (2026-10-27): initial job
`01M5ZAPEVK9K2S43GCB1JPT57B` was preempted by an urgent allocated workload
while starting at 19:34 UTC. autoResume created replacement
`01M5ZDP3XP4YA4V7WASNNGC0RT`, scheduled at 19:41 UTC. No learner logs or
completed update yet. This verifies scheduler requeue, not policy/optimizer
checkpoint restoration. The experiment and durable output/W&B identity are unchanged.


## GSM8K infrastructure comparison: 275m and 450m, 500 updates (2026-10-27)

Status: PREPARED; GPU allocation awaiting resource-plan approval.

Four fresh runs: Core/vLLM and MILES at each of275m and450m, final general-tree
midtrain step0038147. Each uses BF16, four prompts x four completions (batch16),
500 optimizer updates (8,000 completions), full GSM8K training and platinum
evaluation, prompt512/response2048, packed length4096, lr1e-6, seed1, no KL or
entropy penalty, and retains zero-variance groups. Eval at0/100/200/300/400/500;
native checkpoints and HF saves every100. Fresh output roots per run, auto-resume,
normal preemptible priority, minRuntime0, three retries, timeout24h per attempt.

User requested a hard maximum of one optimizer update off-policy. MILES uses
fully_async=true, max_weight_staleness=1 (Core max_policy_lag=1), barrier publication,
one group/four samples in flight and one batch of completed buffering. The old
stack uses async_steps=1, inflight_updates=false, and max_result_age_steps=1.
The hard guards reject older groups; queue sizing is not the age guarantee.

User chose to retain the stacks' existing reward and optimizer differences, then
explicitly requested matching the upper PPO clip: lower0.2/upper0.272 in both.
MILES keeps numeric GSM8K answer comparison, zero warmup, grad clip1 and eval
top_p0.95; Core/vLLM keeps its exact-string GSM8K answer comparison, existing
warmup/gradient behavior and eval top_p1. This is an end-to-end stack comparison,
not an isolated causal test of the inference backend. Sampling is not bit-identical.

Recommended resources: four jobs x two Rhea RTX A6000 48GB, one learner plus
one inference GPU per job, TP1, all on the same hardware. Eight GPUs peak.
Rhea had26 free slots at preparation, including two empty eight-GPU nodes;
Saturn A100 80GB is the alternative if Rhea availability changes. No extra
learner GPUs recommended for these small models. Budget ai2/oe-other, workspace
ai2/open-instruct-dev; mounts oe-adapt-default and oe-training-default.

MILES retains the qualified HF-backbone SGLang fallback, no graphs/prefix cache/
chunked prefill, max_running_requests4 and memory fraction0.35. Core/vLLM keeps
its existing eager/FLASH_ATTN setup with FP32 SSM cache and memory fraction0.6.
The new MILES native learner/publication path needs a short two-update lifecycle
smoke for each size before launching the four longer jobs. Proposed smoke cap:
two GPUs sequentially, at most1h per size, zero retries, small held-out eval slice,
checkpoint every update; then the full runs only if both smokes pass.

Launch files: MILES `configs/miles/benchmarks/hybrid_{275m,450m}_gsm8k_compare_500.toml`;
Core/vLLM `scripts/train/olmo-hybrid-small/gsm8k_infra_compare_500.sh` with
MODEL_SIZE=275m or450m. Run-file schema planning passed; images/data are prepared
on the existing Hammond CPU session, with no local Docker build.
