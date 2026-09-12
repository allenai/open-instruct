# Next MILES/Core qualification gates

> Historical proposal. For current operating instructions, start at the [MILES guide](../index.md).

Current experiment status and the restored matched Megatron comparison are tracked
in [the active-work ledger](active-work.md). The architecture-specific hero
port is documented in [hero support](../hero-support.md).

This is a proposed experiment sequence, not a record of additional launches.
It builds on the [feature inventory and profiles](../feature-parity.md) and
[training contract](../core.md#training-contract-checks). Run the smallest
unmet gate first; do not repeat completed gates without a relevant code or
configuration change. Status below is as of September 10, 2026.

## Established and pending evidence

| Gate | Current evidence |
| --- | --- |
| Independent loss, reduction, auxiliary gradients and lifecycle tests | [90 pinned-runtime tests](../measurements/core-contract-final-20260910.json) plus 29 targeted host verifier tests passed. This is not full-model or cross-backend parity. |
| Native Core EP consistency | [Twelve arms](../measurements/core-native-ep-20260910.json): EP1/EP2 × policy/auxiliary/combined × recomputation off/on passed. Maximum category relative L2 error was 5.79e-5; fixed routes came from Core. This was one response-averaged tiny batch without a clipping stress case, not exact update parity. |
| Actual serving replay | [Tiny SGLang/Core replay](../measurements/core-replay-local-20260910.json) passed two updates, a fresh-process third update and a 12-response audit. Full-model replay and the final unscored token's auxiliary-routing semantics remain open. |
| Public tiny default and resume | [The 512-token resident profile](../measurements/core-tiny-default-20260910.json) passed two updates and a separate-process third update with schema-2 checkpoints and a 12-response audit. |
| Real SFT task learning path | [GSM8K](../measurements/core-sft-20260910.json) passed two EP2 updates and 64 audited responses, including mixed-reward groups. Sixteen held-out questions cannot establish learning improvement. |
| Additional individual sources | [Local math and legacy IF](../measurements/core-datasources-local-20260910.json) each passed two updates and 64-response audits. All training advantages were zero; updates came from auxiliary losses. |
| Full-SFT math and IF | [Trial 01M26GC6F3TRRQEXR9HJQR0XGG](https://beaker.org/ex/01M26GC6F3TRRQEXR9HJQR0XGG) is in progress. Require each task's audit and final exit status before promoting its scope. The tasks start independently from the same checkpoint. |

## Bounded experiment sequence

Budgets are proposal ceilings, not runtime predictions. GPU limits exclude queue
wait. Use the committed-image launch wrapper, preserve exact inputs/arguments,
and stop on a numerical/provenance violation. A timeout or partial artifact is
not a pass. New infrastructure mentioned here still needs implementation.

| Priority / experiment | Concrete shape | Required evidence and stop boundary |
| --- | --- | --- |
| **1. Finish current math/IF gate** | Existing EP2 + one engine allocation; two updates and 64 responses per task | Check independent rewards, immutable source/template hashes, policy versions, contract denominators and diagnostic republications. Report mixed-reward groups and cap hits by source. Do not silently relaunch with shorter responses to manufacture a pass. |
| **2. Mixed-source collection** | GSM8K + math + legacy IF, 2 prompts/source × 4 responses = 24 samples/update; two updates on EP2 + one engine; eight held-out prompts/source before and after. At most 60 minutes. | 48 training + 48 evaluation responses, each independently audited. Preserve single-source four-response groups; never normalize advantages across different prompts. Verify all source/target identities and no train/eval overlap. At least one mixed-reward training group per source is required to claim policy-gradient coverage for that source; otherwise mark only its plumbing passed. |
| **3. Matched Megatron numerical contract** | Identical tiny KDA/latent weights and fixed rollout artifact; EP1 first, then EP2; policy-only, auxiliary-only and combined; one update, then a fixed five-update sequence. Start with recomputation off; add it after parity. No live generation. Up to 45 minutes per two-GPU allocation. | Compare independent losses, pre-clipping gradients, clipping factors, Adam moments, FP32 masters and BF16 model weights in canonical HF layout. Agree the semantic contract below before launching. Stop at the first unexplained mismatch and isolate it; a similar final reward is not numerical acceptance. |
| **4. Full-model durable continuation** | Qualified SFT/task on EP2 + one engine. Four planned updates with a fixed four-step LR horizon; save after update 2, exit, then fresh-process resume through update 4. Separate uninterrupted fixed-batch control. At most 60 minutes per attempt. | Verify schema, topology/model metadata, model and optimizer state, LR/clock and cursor before the next update. Compare the next fixed batch to uninterrupted execution. Check complete-marker and cursor hashes plus clean shutdown. Size available storage from actual state inventory and retain both boundaries until the audit passes. Live subsequent samples need not be bitwise identical. |
| **5. Bounded async against sync** | Same three-GPU shape, same source slice, 2560 context/512 response, four updates, four samples/prompt; lag ≤1, one buffered collection, one optimizer step/collection, replay off, eager decode. Compare matched synchronous control. At most 60 minutes per arm. | Every consumed group has one behavior version; measured learner lag ≤1; publications and diagnostic repeats have correct versions. Count completed, accepted, aborted and retried groups with no lost prompt identities. Require mixed rewards for learning coverage. Report useful tokens and accepted samples/GPU-second, not raw decode rate alone. |
| **6. Resident colocation fit and parity** | Two Core EP ranks + **two** TP1 engines: two physical GPUs colocated versus four physical GPUs disaggregated. Match engine count and workload. Two updates initially; eager decode, no replay, trainer resident. At most 60 minutes per arm. | Measure peak memory after Adam state exists, then generation after the update. Fix total KV/recurrent capacity in both arms; do not copy the dedicated memory fraction. Require serving equality, probability checks, reward audit and complete teardown. A successful generation-only run is insufficient. Test rollout offload and decode graphs as separate subsequent changes. |
| **7. Compiler-cache lifecycle** | Implement fingerprinted restore to node-local storage and immutable publish on success, following olmo-miles. Compare two fresh allocations of the same two-update workload, cold then restored; repeat with an intentionally incompatible key. | Same outputs/contracts within existing numerical tolerances; cache hit/miss and compilation phase timings recorded; incompatible cache rejected; concurrent writers cannot corrupt a generation. Separate startup savings from steady training throughput. No shared writable compiler hot directory. |

The targeted EP1/EP2 unequal-length **token averaging** and **active gradient
clipping** follow-up now [passed](../measurements/core-ep-stress-20260910.json):
canonical pre/post gradients, independently counted global norms, Adam moments,
and exact reconstructed state agreement between both EP2 ranks were checked.
This tiny fixed-route gate does not establish exact update agreement near zero
gradients or cross-backend agreement. Use the full-model
restart gate for native EP continuation. EP-plus-DP at world size four remains
a later topology gate, not a prerequisite for the present EP2 default.
The mixed-source manifest, cross-backend fixed-batch runner and cache lifecycle
are proposed additions; the existing individual-source launcher does not
implement them automatically.

The full-SFT math slice reached the 4096-token cap in 63 of 64 responses.
Before using that shape for math-quality comparisons, measure a longer-response
configuration with matching context, KV capacity and admission limits. Keep the
original bounded result as evidence; do not reinterpret it as an uncapped score.

For the mixture, start with equal numbers of prompt groups to expose each
verifier. This is a diagnostic mixture, not the released Olmo 3 recipe. Record
proposed and accepted prompts, response tokens, reward scale, variance and
policy-gradient contribution per source. A later production mixture should
preserve its own pinned source proportions and selection rules; equal prompt
counts do not imply equal token or gradient contribution.

## Contract for the Megatron comparison

The comparison target is our customized `~/proj/olmo-miles` trainer, including
its Olmo model adapter and efficient weight sync. Weight transport is outside the
first numerical gate. Freeze these items in a machine-readable comparison
manifest before collecting measurements:

- **Model:** identical source checkpoint and tensor inventory, tokenizer, KDA/full
  attention pattern, latent dimensions, norms/gates and dense/shared experts.
  Match BF16 stored parameters, router compute precision, gradient accumulation,
  FP32 optimizer masters/moments and post-update rounding. Verify actual operator
  dtypes rather than assuming autocast has the intended effect.
- **Rollout artifact:** exact tokens, next-token shift, prompt/response lengths,
  masks, rewards, behavior log probabilities and fixed actor/reference anchors.
  Fix GRPO standard-deviation normalization, clipping bounds, entropy/KL/TIS
  choices and response-versus-token reduction. Initially disable optional terms;
  exercise them one at a time after the basic objective matches.
- **Auxiliary objective:** match per-sequence versus batch aggregation, selected
  expert counts, all-token versus response-token scope, denominator and replica
  reduction. Core is unpadded; the Megatron path must exclude artificial padding
  equivalently or compare an explicitly matched equal-length fixture first.
  The coefficients 0.01/1e-5 alone do not establish equivalence.
- **Replay:** use one immutable route artifact, verify each routed layer/token
  mapping, and agree which final token is processed. Core currently supplies
  deterministic IDs for the last unscored token; it still contributes auxiliary
  loss. Exclude or reproduce that contribution consistently before asserting
  replay equivalence. Then separately compare native, non-replayed routing.
- **Optimizer and distribution:** same global batch, microbatch accumulation,
  EP/data-parallel sample ownership, world averaging, gradient clipping, Adam
  betas/epsilon/weight decay, LR used on each step, initialization of moments and
  scheduler horizon. Compare FP32 moments: first-step Adam updates can hide a
  uniform gradient-scaling error.

Establish repeatability within each backend first. Use exact checks for IDs,
masks, counts and clock state. Preserve the independent FP32 reference tolerances
already used by the contract suite. For BF16 cross-backend comparisons,
predeclare bounds from within-backend noise and operator differences; the native
EP screen's 0.05 relative-L2 limit is not automatically a Megatron acceptance
threshold. Report error by router/expert/dense category and layer, including
absolute norms when a reference gradient is close to zero. An auxiliary semantic
difference is an algorithm difference to resolve, not numerical noise to hide
by increasing a tolerance.

The baseline's
`docs/measurements/backend-parity-post-training-study.md` documents meaningful
operator-induced differences and the router BF16-storage/FP32-compute contract.
Its results are evidence for how to design this comparison, not measurements of
our present Core/MILES adapter. The baseline
`docs/measurements/async-multistep-20260907.md` provides the separate multi-update
policy-clock and restart protocol to reproduce after single-update async passes.

## Matched held-out learning comparison

The current Core GSM8K result (14/16 before, 15/16 after two updates) is a small
in-loop evaluation check. No matched learning-curve comparison with olmo-miles
has run. After resolving the fixed-batch backend contract, freeze one shared SFT
checkpoint, training manifest and at least 128 disjoint held-out GSM8K prompts.
Run 100 optimizer updates and evaluate the identical prompts at updates
0, 20, 40, 60, 80 and 100 in both backends, following the
[frozen comparison design](gsm8k-parity-design.md). Match response/context budgets, temperature,
samples per prompt, seed, LR schedule, effective global batch and auxiliary
semantics. Retain prompt-level correctness and truncation at every boundary.

Plot accuracy against optimizer updates, consumed training tokens and elapsed
GPU time separately. A single-seed screen can reveal a regression but cannot
establish equal learning efficiency; repeat seeds before drawing that conclusion.
The held-out set is disjoint from this RL training manifest, not automatically
certified absent from SFT. Do not compare the new curve directly with historical
olmo-miles curves that used different prompts, response caps or update budgets.
This comparison is proposed work, not an additional submitted job.

## Promotion criteria

**Synchronous default:** complete the individual and mixed-source gates with
honest per-source policy-signal reporting; pass the numerical contract and
full-model restart; then complete 20 consecutive updates at the intended length
with no non-finite values, skipped updates or unexplained drift. Keep a fixed
held-out set and inspect the full checkpoint/audit, not just exit zero. A bounded
20-update run establishes operation, not model quality or multi-node support.

**Async default:** first promote its matched synchronous shape, then pass the
four-update async screen and a 20-update run with bounded lag, finite rewards,
measured retry waste, and clean shutdown. Add a fresh-process restart and one
controlled serving failure with preserved prompt accounting. Promote multiple
optimizer steps per collection only after reproducing the baseline 0/2/4 policy
clock and restart protocol; increase the lag budget explicitly. Do not combine
initial async qualification with replay, prefix caching or offload changes.

**Colocated default:** pass resident full-model two-update fit, then ten updates
and restart on the chosen memory budget. Add rollout offload, then decode-only
graphs, one at a time; each must survive publication, resumed generation and
shutdown. Compare total GPU-hours per accepted update with equal engine counts.
Trainer offload, multi-sequence microbatches, dynamic packing and trainer TP/PP/CP
remain unsupported until separately implemented and tested.

For all performance promotions, distinguish cold compilation, warm training,
generation, checkpoint writes, publication and diagnostic overhead. Current
`diagnostic_interval=1` adds a same-version snapshot/reset/republish transfer;
count it, and use a separately identified diagnostic-disabled arm for throughput
measurement only after correctness passes. Preserve cheap runtime invariants.
Reuse the baseline 1 GiB bucket and bounded admission as starting points, not
universal optima. Record queue delay separately from allocated GPU runtime.
