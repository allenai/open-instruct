# Learning confidence across models and frameworks

This is the working report for the broad comparison program, not a GSM8K-only
replacement. It reuses existing olmo-miles runs, records incomplete experiments,
and identifies the missing matched original Open Instruct control. The current
MILES/Core GSM8K run is one independent learning control while code-service
reliability is investigated.

## Active identities — September 14, 21:07 UTC

| Run | Experiment | Status / target |
| --- | --- | --- |
| Fully SFT MoE, four domains | [01M2GVVKXSS2E1X1FHYNQ8P555](https://beaker.org/ex/01M2GVVKXSS2E1X1FHYNQ8P555) | Scheduled; qualified 128K judge; 200 updates |
| Dense Think-SFT, same four domains | [01M2GVVT4CMSPS4TQSYTYSE9TG](https://beaker.org/ex/01M2GVVT4CMSPS4TQSYTYSE9TG) | Scheduled; same 128K judge; 200 updates |
| Dense GSM8K protected control | [01M2GQK5F7T1YTVPPVD9E3S43Q](https://beaker.org/ex/01M2GQK5F7T1YTVPPVD9E3S43Q) | Running; 200 updates, eval every 50, save every 25, 48-hour ceiling |
| Original framework hardware qualifier | [01M2GVP4SZTA36EV579ST4NDCB](https://beaker.org/ex/01M2GVP4SZTA36EV579ST4NDCB) | Scheduled; 4 H100 trainers + 4 inference; 3 driver iterations of 512 responses |

[Repair launch receipts](repair-launches.json) supersede `active-launches.json`
for the broad benchmarks. The former MoE broad run failed the 40,960-token judge
context guard. The former dense broad run was stopped at zero optimizer updates
to adopt the same qualified judge. Both new profiles keep 32K policy responses;
only the judge capacity/concurrency/timeout changes. Initial evaluation repeats.

The GSM8K r2 control completed its initial 441/512 evaluation before a fresh r3
restart with saves every 25 updates and a 48-hour limit. R3 skips repeating initial
evaluation; verify prepared hashes before combining its curve with the r2 score.

Historical full-checkpoint preparation checked all 64 norm shapes and 6512 frozen
prompt token hashes: [proof](original-retrofit-preparation.json). The old automatic
original-framework continuation stopped on its failed B300 trial. **No automatic
200-update original run is armed.** H100 qualification uncovered zero-advantage
filtering that can skip driver iterations; the new overlay counts completed
training calls separately and fixes export-only generation metadata. Its larger
smoke batch is a hardware/mechanics exercise, not a matched learning benchmark.
No new 200-update endpoint exists yet.

Read-only local monitor: `/tmp/learning-confidence-repairs-watch/status.json`.
See the dated repair section below for all failures, fixes and qualification.

## Current runs — September 14, 19:24 UTC

The first 32K MoE attempt stopped before training when one symbolic math grading
request exceeded its 45-second subprocess budget. Code fallback worked at its
own boundary, but did not cover this separate local verifier timeout. The dense
basket was deliberately stopped during initial evaluation so both restarts use
the same timeout policy; neither 32K basket attempt had reached an optimizer step.

The corrected policy kills/replaces the stuck math worker and records a tagged
zero reward. Unexpected worker/configuration failures and cancellation still
propagate. Six new tests, including a real hanging subprocess followed by a
successful request, and four existing metrics tests passed in the MILES image.
`rollout/math_verifier/timeout_fraction` separates these ungraded answers from
ordinary incorrect answers; `OI_MILES_MATH_TIMEOUT_POLICY=raise` restores strict
behavior. Data, model weights, optimization, lengths and topology are unchanged.

- [MoE basket restart](https://beaker.org/ex/01M2GP10EKTRDDC4MD924FGDRM).
- [Dense basket restart](https://beaker.org/ex/01M2GP1D9Q4ZNGVJF3ATN1V9VN).
- [Dense GSM8K control](https://beaker.org/ex/01M2GK81TBNVB6DEEAXC2Q3XTY) remains running unchanged.

The two basket restarts use source `724a3baa1` over base image
`01M2CJG5RQQ93GEYNYAS7ASCQJ`. [Immutable receipts](robust-launches.json) retain
full resolved configurations. They are running experiments, not completed
learning comparisons.

## Restarted broad baselines — September 14, 18:35 UTC

Code-verifier failures now default to logged/tagged zero rewards rather than
terminating the run, matching original Open Instruct's continuation behavior.
Strict service-failure handling remains opt-in. The implementation passed 28
focused tests including real HTTP retry exhaustion and recovery; four additional
rollout-metric tests passed in the actual MILES runtime image. See the
[historical service audit and policy](historical-code-service.md).

| Run | Beaker | Allocation | Workload |
| --- | --- | --- | --- |
| Fully SFT MoE | [01M2GK6NNB1BV8C50AXBVW319D](https://beaker.org/ex/01M2GK6NNB1BV8C50AXBVW319D) | EP8 + 7 inference + 1 judge | Frozen math/IF/code/general basket, 200 updates |
| Dense Olmo 3 Think-SFT | [01M2GK7CYF7BKJZR97QSA5ZJF4](https://beaker.org/ex/01M2GK7CYF7BKJZR97QSA5ZJF4) | FSDP8 + 7 inference + 1 judge | Same frozen basket identities, 200 updates |
| Dense GSM8K control, retry | [01M2GK81TBNVB6DEEAXC2Q3XTY](https://beaker.org/ex/01M2GK81TBNVB6DEEAXC2Q3XTY) | FSDP2 + 4 inference | 6000 training / 512 held-out RLVR rows, 200 updates |

All were allocated on urgent Holmes with four-hour minimum runtime. By 19:06 UTC,
both basket runs had reached their initial math evaluation and the GSM8K control
was halfway through its initial held-out evaluation. No new learning endpoint is
available yet.

Both basket restarts preserve all four domains and their original frozen inputs.
The response budget increases from 4096 to 32768, with context/packing 34816,
activation recomputation, and eight running sequences per inference engine.
Dense KV-cache capacity rises from 131072 to 294912 tokens. These are explicit
memory adjustments for the larger budget, not the previously qualified 4K
throughput configuration. Batch size remains 256, learning rate 1e-6, same seed
17. Evaluate and checkpoint every 50 updates. The MoE keeps qualified decode
graphs/radix/replay; the dense path keeps graphs/radix/replay disabled. See the
[complete immutable launch receipts](restart-launches.json).

The first GSM8K attempt exited before training because `NCCL_SOCKET_IFNAME=ib`
selected an interface absent in its single-node bridge network. The fresh retry
removes this restriction and lets NCCL choose an available interface. It has no
code-service or judge dependency. The multi-node basket runs use host networking.
All three submissions use source `dc8c2661d6a3c97566662c157f2039a05e5093d0` and
base image `01M2CJG5RQQ93GEYNYAS7ASCQJ` with the recorded source overlay.

## What the completed runs tell us

| Starting model and evaluation | MILES/Core | olmo-miles/Megatron | Evidence |
|---|---:|---:|---|
| Light SFT, native held-out GSM8K, 200 updates | 32 → 79 /128 (+36.72 pp) | 26 → 77 /128 (+39.84 pp) | Complete curves; same historical questions and sampling recipe, different runtimes/topology |
| Light SFT, separate raw-prompt official test, 200 updates | 229 → 268 /1319 (+2.96 pp) | 230 → 274 /1319 (+3.34 pp) | All 2638 retained Core answers independently rescored; zero disagreements |
| Heavy SFT, first 100 updates of the 500-update pair | 99 → 107 /128 (+6.25 pp) | 98 → 106 /128 (+6.25 pp) | Recovered logged evaluations |
| Heavy SFT, final 500-update endpoint | 99 → 101 /128 (+1.56 pp) | 98 → 104 /128 (+4.69 pp) | Both jobs exited zero; full curves retained |

![Recovered learning and truncation curves](learning-curves.png)

The light-SFT runs show a substantial native-evaluation learning signal and
similar endpoint performance in both implementations. Their separate raw-prompt
full-test gains also agree closely. These are encouraging bounded results, not
statistical equivalence: there is one training seed per backend, inference
implementations and topology differ, and the native and raw-prompt evaluations
must remain separate. The raw-prompt format is not an estimate of the model's
best instruction-following capability.

The heavy-SFT trajectories are nonmonotonic. More updates were not automatically
better. The final accuracy gap is three questions, but Core's final capped-answer
fraction was 40.625%, versus 9.375% for Megatron. Some capped responses can still
receive credit, so this behavior difference must not be erased by summarizing
accuracy alone. The plot retains it without claiming a cause.

For the Core light-SFT full-test endpoints, matched IDs and labels were checked:
80 questions changed from wrong to correct, 41 from correct to wrong, 188 stayed
correct and 1010 stayed wrong. Net gain: 39. This independent rescore does not
revalidate all historical optimizer, publication or checkpoint contracts.
The heavy-SFT recovery checks policy-version/update alignment and all 26 scheduled
evaluation records per arm; its retained generations have not been independently
re-audited in this report yet.

The older 100-update campaign remains separate: Core 97→93, Megatron 96→107.
It used an earlier Core scoring path and other different settings. It must not be
silently combined with the later 500-update pair as a single learning trajectory.
See [its report](../gsm8k-results-20260911.md) and
[configuration inventory](../gsm8k-configuration-differences-20260911.md).

## Broad task coverage: available starts, missing learning endpoints

Both recent basket attempts used the same 101434 training rows, 512 held-out
questions and four-domain reward setup, but different policy models/tokenizers.
These are **update-zero mean rewards**, not improvements or framework comparisons.
IF and judged rewards are fractional; they are not binary accuracy percentages.

| Domain (128 questions each) | Heavy-SFT MoE initial reward | Dense Olmo 3 Think-SFT initial reward | MoE / dense fraction capped at 4K |
|---|---:|---:|---:|
| Math | 0.0078125 | 0.0078125 | 92.97% / 91.41% |
| Instruction following | 0.20208 | 0.28034 | 72.66% / 42.97% |
| Code, function and stdio | 0.109375 | 0.109375 | 71.88% / 76.56% |
| General judged quality | 0.67422 | 0.77578 | 34.38% / 25.00% |

The MoE attempt completed 18 optimizer updates and the dense attempt 11, then
both exhausted external code-execution HTTP retries. Neither reached its first
scheduled post-training held-out evaluation. They establish partial integration
execution, not broad-task learning. We do not have final checkpoint learning
numbers from either run. The 4K cap also left math near a reward floor for both
models. That budget was inherited from efficiency tests; it should not be treated
as a universal learning benchmark setting.

The follow-up keeps a matched response budget across arms and measures completion
and mixed-reward groups before committing to long trajectories. The current dense
GSM8K control uses a 32K ceiling and local rewards. Broader math/IF/code/judge
learning remains in scope; external-service failures will not be converted to
incorrect-answer rewards to make a run finish.

## Coverage and remaining work

| Comparison / purpose | Status | Next required evidence |
|---|---|---|
| Light MoE SFT: Core versus Megatron | Both 200-update runs complete; report recovered | Preserve configuration caveats; no new Megatron run needed just to duplicate this signal |
| Heavy MoE SFT: Core versus Megatron | Both 500-update runs complete; full curves recovered | Generation/length analysis and retained-sample audit of the extension |
| Current dense Think-SFT: MILES/Core GSM8K | Submitted for 200 updates on 2 trainers + 4 engines | Initial reward distribution, learning curve, final common evaluation and resumable save |
| Dense Think-SFT: original Open Instruct versus MILES/Core | Original image downloaded; pair not yet launched | GPU qualification of the reference, matching frozen data/optimizer/exposure, common final evaluator |
| Dense and MoE: broad Dolci task basket | Incomplete after 11 and 18 updates | Diagnose code timeouts, useful response budget, 200-update learning endpoints per domain |
| Current optimized versus historical runtime | Historical results give context, not exact parity | Separate changes in packing, replay, async scheduling, scoring and sampler settings |

For the new comparison, 200 steps is an initial observation point. Extend toward
400 from resumable checkpoints if the signal is inconclusive and updates are
healthy. A persistent directional gap calls for diagnosis and possibly another
seed, not automatic indefinite extension. Match response/sample exposure as well
as nominal optimizer steps; historical and current batches are different sizes.

## Cost and performance boundaries

The heavy-SFT 500-update runs used three B300 GPUs each. Scheduled-to-exit cost
was **31.46 GPU-hours Core** and **32.15 GPU-hours Megatron**, including startup,
evaluation and checkpointing, excluding queue time. This is total campaign cost,
not a claim of equal training throughput. Those Core runs predate later compiler
and checkpoint optimizations, so they are not the current performance baseline.

The light-SFT comparison used different allocations: four GPUs for Core versus
two colocated GPUs historically for Megatron. Do not infer backend efficiency
from their wall times alone. Current phase-level optimization measurements remain
in [the 2-trainer/4-engine report](../fast-2t4i-20260914.md); they are a different
workload from these historical learning comparisons. Broad-run time spent in
inference, training, verification and orchestration still needs a common report
once sustained runs complete.

## Provenance and reproducibility

- Light Core200: [Beaker](https://beaker.org/ex/01M27EWCG7P03CNY40E5WSXX1D).
- Historical light Megatron200: [Beaker](https://beaker.org/ex/01M12Y23YZS5ZJWBK45CKQJ7DP).
- Heavy Core500: [Beaker](https://beaker.org/ex/01M279ZFM6RBC223RJJ6QHN9MP).
- Heavy Megatron500: [Beaker](https://beaker.org/ex/01M278B5E9HME181B04HT6391P).
- MoE basket, incomplete: [Beaker](https://beaker.org/ex/01M2F7N19DQ3YJMRJAXJ1K4H59).
- Dense basket, incomplete: [Beaker](https://beaker.org/ex/01M2FBGGE8K8XJ7WJKCTE4KHMB).
- Failed first dense GSM8K launch: [Beaker](https://beaker.org/ex/01M2GDHWYJX9VH4J94RS2QQWK3),
  source `44a893788`; exited before training because of the NCCL interface restriction.
- Active dense GSM8K retry: [Beaker](https://beaker.org/ex/01M2GK81TBNVB6DEEAXC2Q3XTY),
  source `dc8c2661d`, immutable base `01M2CJG5RQQ93GEYNYAS7ASCQJ` with committed overlay.

[Machine-readable evidence](recovered-evidence.json) contains curves, original
log/answer SHA256 digests, job status, evaluation boundaries and paired counts.
[The light-SFT protocol](../light-sft1000-gsm8k.md) and
[heavy-SFT protocol](../learning-comparisons-20260911.md) retain model/data/runtime
identities and known differences. Generation samples and original-framework
results remain pending additions; this report marks those gaps explicitly.


## Original Open Instruct comparator qualification

`scripts/miles/launch_original_baseline.sh` uses the committed-image wrapper to
run the actual original image `01K7B0Z1KKP8AFKV2YKENMQ53B` (October 2025), not
MILES. Its CPU preparation runs on Saturn and adapts the current Core GSM8K
control's 6000 training and 512 held-out rows. It checks every prompt token hash
through the original `rlvr_tokenize_v2` transform, preserves source identities and
labels, and rejects changed filtering or train/eval overlap. A tokenizer-only
passthrough template prevents adding a second template to already-rendered
prompts; it does not change model weights.

A three-update, six-GPU qualifier uses 64 training prompts and eight held-out
questions before the full 200-update, full-heldout comparator is authorized to
start. Both use two DeepSpeed learners and four vLLM engines, response 32768,
pack/context 34816, 16 prompts x four samples, LR 1e-6, centered advantages,
clipping 0.2/0.28, KL zero, training temperature 1, greedy evaluation and seed17.
The trainer adjustments align Adam beta2 from its hardcoded 0.999 to 0.95 and
make `eval_on_step_0` actually schedule an initial evaluation with interval 50.
The exact original source hash and each unique replacement are checked and
recorded; seven tests pass in the original image. The periodic original evaluations
sample the policy before updates 50/100/150/200, hence completed-update labels
49/99/149/199. They must not be plotted as post-update 50/100/150/200. The exported
final model is post-update 200 and can be evaluated separately.
All proposed CLI options were checked against the image's dataclasses before
submission; this is not a claim of runtime qualification.

Remaining differences include vLLM/SGLang numerics, DeepSpeed/HF/Core execution,
packed token-mean versus response reduction, and the historical GSM8K verifier.
They remain visible in the invocation receipt; this is a rough implementation
comparison rather than a claim of exact algorithmic parity. Original-framework
GPU execution is still unqualified until that separate smoke succeeds.

The [CPU preparation retry](https://beaker.org/ex/01M2GMYARVE95SRYZDTH8P6Z12)
passed on September 14 at 19:06 UTC: all 6000 training and 512 evaluation prompt
token sequences match Core exactly. [Input-parity proof](original-input-parity.json)
retains the artifact digests. The first attempt failed because `str.splitlines()`
split Unicode separators embedded in valid JSON strings; file-line iteration fixes
that reader bug. All six comparator tests passed in the original image.

The [three-update GPU trial](https://beaker.org/ex/01M2GN0YT6XVX0MAX7PC635YZQ)
has acquired six Holmes GPUs and is pulling the original image. Its launch revision
is `2806460af`; [receipts](original-launches.json) distinguish preparation from
GPU execution. The full original 200-update run remains gated on the trial.

The first GPU trial stopped before model loading: the historical CLI defaults to
`push_to_hub=True` and required an HF credential. Publishing is now explicitly
disabled. The [retry](https://beaker.org/ex/01M2GNC6R97113K90EM8ND6GXF) has started
Ray and model actors (19:15 UTC); source `5b9c43245`. It uses evaluation interval
one, so it already exercises initial-policy evaluation without the interval-50
scheduling adjustment needed by the full run. Neither failed startup performed
an optimizer update.

The original GPU retry exposed a checkpoint/model-class naming mismatch before
training: the historical `Olmo3ForCausalLM` prototype has per-head Q/K norms, while
the published checkpoint has global Q/K norms. The published Think recipe actually
used `Olmo2RetrofitForCausalLM`, with global norms and mixed sliding/full attention.
Compatibility qualification now targets that implementation in its published-run
image `01KA3FGCMVYGVEX2NG7Q2JWZ8E`; no checkpoint tensors will be reshaped to fit
the prototype. The original 200-update comparison is still not launched.

The metadata alias passed a CPU equivalence check against public Transformers
4.57.0: all 3968 logits are identical and maximum Q-normalization gradient error
is 1.46e-11. The two-layer probe includes full/sliding attention, grouped-query
attention, nonuniform global norm weights and YaRN. [Proof](original-retrofit-parity.json).
Full-checkpoint preparation checks every Q/K norm shape, changes only config
`model_type`/`architectures`, and links unchanged source weights. Export restores
public Olmo 3 config and the original chat template. Nine tests passed in the
historical Think image. A fresh CPU preparation and GPU trial are still required.

At 19:45 UTC the dense GSM8K control completed its initial evaluation: **441/512
(86.13%)**, median response 1525 tokens, mean 4805.83, and 49 capped responses
(9.57%). All samples report policy version zero. The long response tail made this
evaluation take 3718 seconds. [Recorded metrics](dense-gsm8k-initial.json). This is
a starting score, not an RL gain; training-cycle measurements and an endpoint are
still pending.

The historical retrofit's full-checkpoint CPU preparation passed on Saturn:
[Beaker](https://beaker.org/ex/01M2GPZSQFFGFNM545229AYB19). All 64 normalization
shapes and all 6512 prompt token hashes pass. Its three-update GPU trial is
[Beaker](https://beaker.org/ex/01M2GQ7M7QZHDZS4AKM488BJ9S). A bounded local
continuation in `/tmp/original-baseline-sequence.py`, pinned to detached checkout
`.worktrees/miles-original-executor` at `bbfdb130f`, launches the original
200-update run only after zero exit, three completed updates and restored public
HF exports. State and receipts are in `/tmp/original-baseline-sequence/`; a failed
gate stops the sequence without automatic retries. The 200-update original run
is not yet launched.

The corrected MoE run exercised the timeout fallback live at 19:53:15 and
19:53:33 UTC: both 45-second symbolic-math timeouts were logged as zero rewards
and evaluation continued. This confirms the fix reached the running image; it is
not a final timeout-rate estimate. The retained per-sample diagnostics, rather
than duplicated driver warning lines, must supply the final grading coverage.

## September 14: judge-context and historical hardware repairs

The MoE 32K robust run failed before training because a judge request needed
48,181 input tokens plus 2,048 output, exceeding the configured 40,960 total.
These are judge-tokenizer tokens, not the policy's generation tokens. The judge
also includes the question, rubric and (for reference grading) reference answer.
The strict context guard rejected the complete request; it did not truncate it.
A candidate explicit `qwen3-yarn-128k` extension follows Qwen's published YaRN
configuration. It remains opt-in and requires a standalone long-context GPU
qualification before a broad-run restart. It changes judge numerics, so paired
broad runs must use the same judge configuration. Original checkpoint files stay
unchanged. The candidate reduces judge concurrency from 16 to 4 and increases
request timeout from 120 to 600 seconds for long prefills.

The original-framework B300 smoke failed in its hardware-name lookup. The first
Jupiter H100 retry, 01M2GT3ER9V1J4VRH4D3RDZB6B, instead exhausted device memory
while initializing Adam on two trainers: a 3.77 GiB allocation with 2.74 GiB free.
The next candidate uses four H100 trainers and four inference GPUs, keeping the
64-response global batch, checkpoint, frozen data, objective and LR unchanged.
Do not compare its raw trainer timings with the two-B300 Core control as if the
hardware/topology were equal.

The first CPU budget audit (01M2GT8HMKAC4462240XBX73KT) exited successfully but
returned invalid two-token counts because Transformers returned a dictionary.
Those aggregate counts are rejected; the corrected audit explicitly requests
lists of token IDs and verifies that rendering preserves its input.

The corrected CPU audit, [01M2GTQY3AEX7JHS0HBJBR6SHT](https://beaker.org/ex/01M2GTQY3AEX7JHS0HBJBR6SHT),
passed and covered all 20,489 judged training rows and 128 judged held-out rows.
Largest static grading prompt: 7,477 training tokens / 1,455 held-out tokens,
including rubric and framing, excluding candidate answer. No static prompt alone
exceeds the old budget. Thus the failed held-out request's size is dominated by
candidate re-tokenization, not an unusually large reference. See `judge-budget.json`.
The pinned Qwen config advertises 40,960 positions and has no rope scaling.

The first isolated GPU probe, 01M2GTQTR1RZ4WFFHKSX5ZJKBX, failed before model load:
its reused CPU launcher set LD_LIBRARY_PATH to CUDA compatibility libraries,
causing CUDA error 803 on a GPU node. The GPU variant now leaves driver library
selection to the image/runtime, with a regression test. Its retry is
[01M2GTY8B5ZKMVVXE2Z6DNXRNG](https://beaker.org/ex/01M2GTY8B5ZKMVVXE2Z6DNXRNG).
The four-trainer/four-engine H100 retry is
[01M2GTQPTPSEQCY3W3W2PZXDM9](https://beaker.org/ex/01M2GTQPTPSEQCY3W3W2PZXDM9).
Neither was qualified at submission; inspect final markers and exit status.

The second judge probe reached CUDA but failed the serving context guard. In the
pinned SGLang build, YaRN with `original_max_position_embeddings` does not multiply
the configured maximum; Transformers 5 also reads `rope_parameters`. The candidate
now explicitly sets the serving maximum to 131072 and both legacy/new RoPE fields.
A CPU check using the actual Qwen config and pinned SGLang helpers confirmed
`get_context_length == 131072` and effective factor-4 YaRN at theta 1,000,000.
No unsafe longer-context environment override is used.

The four-H100 original smoke cleared initialization and first weight sync (2.951s).
Its first collection then exposed another comparison difference: the historical
trainer filters zero-variance groups (15/16 here) and skips the driver iteration
if the remaining packed samples cannot fill all ranks. Its old completion marker
incorrectly called driver iterations optimizer updates. Future launches now record
successful training calls separately; the already-running smoke's marker must not
be trusted as an optimizer count. No automatic full run is armed from that marker.

The small H100 smoke later failed at HF export: the historical
`get_olmo3_generation_config` supplied sampling temperature/top-p with
`do_sample=False`, which its Transformers version rejects when saving. The
benchmark overlay now sets `do_sample=True` on that **export-only** generation
config; training and vLLM decoding keep their explicit sampling settings.
The next hardware smoke draws 128 prompts x four samples from the full frozen
training pool to exercise filtering/packing on four ranks. This larger smoke is
not the matched 64-response learning benchmark. Full-run defaults remain separate;
its batch/filtering difference must be resolved before arming a 200-update run.

The final judge GPU gate [01M2GVCA5QT9G32743Z5CSX4MH](https://beaker.org/ex/01M2GVCA5QT9G32743Z5CSX4MH)
passed with exit 0: 12 complete, parseable grades across both rubrics and short,
50K and 100K backgrounds; all six correct/incorrect pairs ranked correctly.
Actual largest request: 100,270 input tokens plus 2,048 reserved output. Per-request
elapsed times at 100K were 4.08–11.68s; these are sequential canaries with prefix
reuse, not a concurrent throughput benchmark. Over-budget requests still failed
before inference. Details are in `judge-yarn-qualification.json`. This checks the
long-context mechanism and simple grading discrimination, not broad judge quality.

Both broad-run replacement profiles now use this same qualified judge, four
concurrent calls and 600-second request timeout. Policy response budgets, frozen
data, trainer settings and objectives remain unchanged. The prior dense broad
run is being stopped before its first optimizer step to avoid mixing judge
configurations across the two arms. Initial evaluation is repeated in the fresh
run; no optimizer checkpoint is being resumed.
