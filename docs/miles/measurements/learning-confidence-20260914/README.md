# Learning confidence across models and frameworks

This is the working report for the broad comparison program, not a GSM8K-only
replacement. It reuses existing olmo-miles runs, records incomplete experiments,
and identifies the missing matched original Open Instruct control. The current
MILES/Core GSM8K run is one independent learning control while code-service
reliability is investigated.

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
