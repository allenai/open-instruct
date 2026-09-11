# GSM8K Core/Megatron discrepancy: investigation handoff

As of 2026-09-11 05:25 UTC. The immediate problem is that **the two serving paths already generate different responses before training**. The completed 100-update learning comparison is useful operational evidence, but cannot yet isolate trainer learning behavior from inference differences. Investigate update-zero equality first. No cause has been established.

## Systems and immutable inputs

- Core arm: open-instruct task/reward facade → MILES orchestration → specialized OLMo-core MoE trainer → live HF-layout weight publication → SGLang.
- Megatron arm: olmo-miles → MILES orchestration → repaired olmo-megatron trainer → live HF-layout weight publication → SGLang.
- Both used the older approximately 18.5B Dolci-think SFT model, Abhishek native `step23607` from the `sft-65536` run. 65536 denotes context, not SFT updates. This is not the newer 12.5B hero model and not the lightly SFT1000 checkpoint.
- Shared HF artifact: `/weka/oe-training-default/robertb/olmo-miles/checkpoints/olmoe3-kda-1.2b-dolci-think-sft-65536-router-bf16-autocast-v2-hf`.
- Original campaign root: `/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1`.
- 400 fixed ordered training questions, four prompts × four responses/update, 100 updates; the same 128 official-test questions evaluated at 0/20/40/60/80/100. Train temperature1, eval temperature0, one evaluation completion/question. Response cap4096, context6144; LR1e-6 constant, GRPO std normalization off, clipping0.2/0.28, router balancing0.01 and z1e-5. Router replay OFF.
- Both original allocations: three B300 GPUs, two EP2 trainer GPUs and one dedicated TP1 SGLang GPU; synchronous disaggregated loop.

| Frozen arm | Experiment | Image | Relevant sources |
|---|---|---|---|
| Core100 | `01M26P6XX6SN886DCVZ68WMQK2` | `01M26N80T0V9PREQTS87J849P8` | OI `c3a41598e434d50058f929e6c55b82e34aa4ff09`; Core `b7f1e5296704779e7deb06de4c4242be9729d1a9` |
| Megatron100 r3 | `01M26YNP5E64YGNXR85TA2RP4Q` | `01M26TT1EQTB84RV0MM650W0YE` | olmo-miles `0e648108c70c5d5256a9b93a88b4f8d610e44ea0`; olmo-megatron `ba5615df741a24ba4aee678d0e853306b3502282` |

Original SGLang package/adapter versions matched. Serving settings did not fully match: Core explicitly used the PyTorch sampler and32768 token pool; Megatron used FlashInfer and automatic pool sizing. Core's original resolved prefill chunk is not yet proved. Do not attribute greedy differences to random sampling simply because sampler implementations differ.

## Established results

Both100 runs exited0. Independent audit `01M279KT4PQNC2HGZ7JVB1TDBJ` passed exact question membership, prompt/token proofs, reward regrading, response lengths, policy versions and optimizer counts. Full initial serving-weight comparisons passed. Those are checks of defined contracts, not proof that two backends compute equivalent activations/gradients or learn identically.

| Updates | Core correct/128 | Megatron correct/128 |
|---:|---:|---:|
|0|97|96|
|20|101|93|
|40|94|98|
|60|102|99|
|80|97|105|
|100|93|107|

Final pair:84 both correct,9 Core-only,23 Megatron-only,12 neither. Core lost16 initially correct questions and gained12; Megatron lost8 and gained19. Eleven of Core's16 losses hit the final token cap. Training mean reward is nearly identical (~0.806), despite different held-out behavior. This is one seed and a128-question subset; no between-run variance estimate exists.

## Update-zero findings

- **Chat template difference ruled out for these inputs:** all128 rendered prompt strings, labels and prompt token IDs are identical in the retained generations. All six evaluations in both arms use this same set. Cross-evaluating different sets is unnecessary.
- Only **7/128 complete generated token sequences match exactly** at update zero.
- Initial correctness disagrees on17 questions:9 Core-only,8 Megatron-only. Initial aggregate97 vs96 hides that disagreement.
- Median common generated prefix is232.5 tokens. On33,137 emitted tokens before the first different token, stored log-probabilities differ by mean absolute0.004326823 and maximum0.303771973. These compare identical tokens under identical preceding tokens.
- Thus different effective inference computations are visible before any RL update. Their source is unknown: engine settings, kernel selection, scheduling, cached state/buffers, and publication-induced state are candidates. Initial stored-weight equality does not establish activation equality.
- Actual examples include repeated self-questioning to the cap, omitted quantities, wrong arithmetic and changes between reasoning and final answer. There are reverse examples where Core succeeds and Megatron fails. This is not uniformly a format-only phenomenon.

## Router precision and replay

The intended precision detail WAS carried over: BF16 router parameter storage; explicit FP32 activation and weight operands; FP32 projection/logits, softmax/top-k weights and auxiliary scalars. The specialized Core OLMoDDP trainer rejects outer autocast. Repaired Megatron uses both `moe_router_dtype=fp32` and `moe_router_use_torch_mm=true`.

Actual frozen Core100, current Core500 and repaired Megatron operator probes on fixed BF16 inputs/weights exactly matched an explicit FP32 reference, with no differing expert sets. These are bounded operator tests, not whole-checkpoint activation equivalence. An intentionally imposed outer BF16 autocast broke both helpers; this is not the active Core trainer configuration. See the router precision report for historical fixes and raw evidence.

Replay was disabled in all four heavy comparison arms (100/500). Retained first Core100 and Core500 training batches have no route IDs. `_score(use_replay=True)` does not enable replay when its runtime flag is false. Therefore a failure inside replay cannot directly explain these runs. This **does not rule out natural serving/training routing disagreement**, which has not been measured here.

Existing replay gates test forced IDs, differentiable router weights, auxiliary/policy/combined gradients, cleanup, recomputation and native EP1/EP2. Tiny live SGLang replay ran, but had zero policy advantages. No full-SFT replay qualification exists. A replay-only final-unscored-token dummy-ID auxiliary limitation remains; it is inactive in these replay-off comparisons.

## Other concrete differences and limitations

1. Old Core100 no-grad SwiGLU scoring uses a different BF16 rounding path from its gradient-enabled forward. This was fixed in Core500's Core `290d2ca4521373bef0bf7fe4244673cc79dcc004`. It is a real training contract discrepancy, but does not explain SGLang's update-zero divergence. The new500 runs also change serving settings, so they are not a single-variable ablation of this fix.
2. Router auxiliary objectives differ: Core weights by real token counts; Megatron uses sequence means and includes padding. In95 matched warm batches Megatron executes5.71M padded positions for3.10M real tokens. Equal auxiliary coefficients are not equal objectives. Existing fixed-input auxiliary gates used equal lengths/no padding and EP1; they do not settle real variable-length EP2 behavior.
3. Core21/100 and Megatron29/100 updates have no current policy-advantage signal because every sampled group has uniform rewards. Auxiliaries and Adam momentum can still move weights. This does not support a simple Core-only policy-gradient-starvation account.
4. The shared GSM8K verifier compares its last extracted number as a string: `42.00` fails target `42`. Independent audits faithfully reproduce that rule. A diagnostic numeric-equivalence regrade gives Core101→96 and Megatron101→107, rather than97→93 and96→107. Three of the14 final questions of gap are formatting-sensitive;11 remain. Active rewards have not been changed. This is not a counterfactual training experiment.
5. Neither original100 run saved a resumable final checkpoint or final HF export. We retain the actual generated responses at all evaluations. New500 runs save checkpoints every100.

## Diagnostics just submitted

- Core: `01M27ENY7T955XE1W116D7M4BM`, job `01M27ENYCQJMJNXTJ8MPMQMTWN`.
- Megatron: `01M27EPG7T5PKJ9B7P6J1GPASD`, job `01M27EPGBAVD31VD3GHJ5XB7DX`.
- Launcher source `f0bcf2f51`; immutable original images and original Megatron source bootstrap preserved. Core first attempt failed before model initialization because the diagnostic serialized unsupported `--no-use-wandb`; a tested retry is being prepared. Megatron passed its exact source bootstrap and parser and is starting the runtime. No model trace results yet at this snapshot.
- Each runs zero optimizer steps. Four frozen prefixes end immediately before original divergent next tokens (test IDs341,975,1039,605).
- Stages: direct-HF serving controls/captures/repeat → full weight snapshot/reset → actual native trainer initialization/publication → full weight equality check → same controls/captures/repeat.
- Captures: all-prefix router logits, selected expert IDs/weights/margins; bounded embedding, layer inputs, attention/MLP/layer outputs, final norm/logits; dtypes, execution controls and loaded source hashes.
- Hooks trace **prefill only**; original decode graphs remain enabled. Requests are serial and sent directly to the engine using token IDs. This isolates fixed-prefix computation but does not recreate original continuous-batching decode. If fixed-prefix traces agree, investigate incremental decode next.
- Raw output: campaign root + `/update-zero-20260911-v1/{core,megatron}`. JSON evidence copied to Beaker results; raw tensor captures on WEKA.
- Both use urgent Holmes/open-instruct-dev,3B300,1h minimum/90m timeout.

## Longer runs and parallel program

Core500 `01M279ZFM6RBC223RJJ6QHN9MP` and Megatron500 `01M278B5E9HME181B04HT6391P` are active. Initial scores99/128 and98/128. They align sampler/token-pool/prefill settings more closely, and Core has the scoring-rounding fix. SG adapter revisions still differ with new hero features configured off. Do not label the recipes exactly identical. Leave these runs unchanged while diagnosing.

Light-SFT200 is a separate historical-comparison campaign. Its first attempt failed on a stale offline-evaluator router port; the retry reached both eval paths but failed a strict tokenizer-proof gate before training. The runtime tokenizer restores the checkpoint's Sequence pretokenizer while bare Transformers preparation reconstructed ByteLevel. A Saturn CPU probe verified runtime MILES/SGLang/checkpoint tokenization agreement for all8,920 prompts; proofs are being regenerated correctly. No successful light-Core learning result yet.

Performance is a separate investigation: original warm cycles91.1s Core vs68.2s Megatron; weight publication3.73s vs5.88s (no per-update HF disk conversion). Recurring Core SwiGLU compilation has a qualified isolated fix: five real successive batches,154,531 exact logprobs across arms; subsequent-batch scoring63.28s→15.22s. That candidate is NOT in active learning runs. Remaining FLA compilation and full-loop speedup remain to measure.

## Evidence and workspaces

Primary working tree: `/home/robert/proj/open-instruct/.worktrees/miles-hero-integration` (`robertb/miles-hero-support`). Core: sibling `miles-core-hero`; MILES fork: sibling `miles-runtime` (`allenai/miles.git`); baseline: sibling `olmo-miles-gsm8k-parity`; repaired Megatron: sibling `olmo-megatron-gsm8k-parity`. Inspect frozen SHAs above rather than current defaults. Do not edit dirty original olmo-miles/olmo-megatron checkouts.

Start with:

- `docs/measurements/miles-gsm8k-results-20260911.md`: completed learning/performance analysis.
- `docs/measurements/miles-gsm8k-configuration-differences-20260911.md`: exact recipes and causal followups.
- `docs/measurements/miles-gsm8k-generation-behavior-20260911.md`: paired examples, update-zero and verifier sensitivity, companion JSON evidence.
- `docs/measurements/miles-router-precision-20260911.md`: exact dtype source paths and operator checks.
- `docs/miles-update-zero-diagnostics.md`: launch protocol and limitations.
- `/home/robert/proj/open-instruct/.artifacts/miles-gsm8k-generations-20260911/comparison-reader.html`: offline searchable paired reader with all1,536 complete responses. CSV, original prepared data and verification report alongside it.

Useful independent agent questions: where do update-zero captures first diverge; are stored parameters equal but execution controls/buffers different; do routes diverge before/after activation differences; does equal-input variable-length EP2 auxiliary math agree; how much does ordinary within-engine decode nondeterminism explain? Do not assume one-shot aggregate accuracy equality implies inference equivalence, or that probability drift itself proves a route mismatch.
