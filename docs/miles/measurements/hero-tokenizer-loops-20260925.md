# Hero tokenizer continuity and reasoning loops

September 25, 2026. The corrected non-EMO hero SFT tokenizer survives RL preparation,
HF checkpoint saves and both serving/rollout loaders. We reproduced an unsafe
legacy class-loading path, but it is **not the path used by this run**. Greedy
reasoning loops already occur in the original corrected SFT checkpoint. The
[completed 330-update run](policy-lag-20260925.md#completed-eight-hour-duration-exercise)
reduces their practical impact on chat-formatted GSM8K; temperature-1 accuracy
changes little, while responses become shorter and cap hits less common.

## Exact checkpoint and lineage

The starting policy is **4T non-EMO, corrected Dolci Think SFT, step 5402**:

```text
/weka/olmo-3p5-checkpoints/scratch/olmo35-fixedtok-sft-20260921/olmo35-fixedtok-sft-20260921-4t-non-emo-dolci-think/emo/step5402/hf
```

It has **12,496,190,080 total parameters and 794,081,920 active parameters**.
This is distinct from the earlier approximately 18.5B-total/1.2B-active full-SFT
model used in the small lag comparison. The extra `/emo/` directory component
is present in both hero exports; the selected model's pretraining ancestry is
non-EMO. EMO routing is disabled during its MT, LC, SFT and this RL run.

The [original qualification](hero-sft-20260923.md#checkpoints-and-source-recipes)
records the SFT experiments and native configs. Corrected SFT uses
`allenai/Dolci-Think-SFT-32B` revision
`7668c638cc84100951973b456069a1a462d6d915`, two epochs/5,402 updates,
LR 5e-5, and fresh tokenized arrays and packing. It initializes from native
4T→MT→LC weights, rather than continuing the older incorrectly tokenized SFT.
The source explicitly requires retokenization and retraining; changing an export
alone cannot fix weights trained on incorrect token IDs.

Authoritative upstream sources, pinned at `444025b6de8013025c351d9c0015b760ef866217`:

- [Corrected campaign design](https://github.com/allenai/OLMo-core/blob/444025b6de8013025c351d9c0015b760ef866217/src/examples/olmo_ddp/CORRECTED_SFT_20260921.md).
- [Tokenizer policy and failure gates](https://github.com/allenai/OLMo-core/blob/444025b6de8013025c351d9c0015b760ef866217/src/examples/olmo_ddp/olmoe3_tokenizer_policy.py).
- [Data preparation and actual loader checks](https://github.com/allenai/OLMo-core/blob/444025b6de8013025c351d9c0015b760ef866217/src/examples/olmo_ddp/olmoe3_corrected_sft_data.py).

## What was checked

The audit ran inside the qualified application image
`01M3APTWSR4VH2PFYMMSPQ8TD2`: Transformers 5.12.1 and tokenizers 0.22.2.
It compared serialized backends, loaded backends, vocabulary mappings, special
IDs, eight encoding probes, chat rendering and asset SHA256 values. Probes cover
long integers, code, whitespace, contractions, Chinese and reasoning markers.

| Stage | Evidence | Result |
| --- | --- | --- |
| Pretraining family / LC parent | Saved native LC config identifies `allenai/dolma2-tokenizer`; pinned upstream Dolma2 backend compared with the production SFT tokenizer | Identical BPE vocabulary/merges, pre-tokenizer and added-token mappings |
| Corrected SFT data | Retokenization manifest, tokenizer proof, data plan and native SFT config | Fresh arrays; raw/backend/HF probes and loader checks passed; native config points to the corrected tokenizer folder |
| SFT training tokenizer → HF export | Actual files loaded in the RL image | Same token IDs and special IDs; training-mode conversation rendering identical |
| Original SFT HF → RL `prepared/hf` | Asset hashes and runtime probes | Tokenizer, template and generation files byte-identical |
| RL HF snapshots 50, 100, 150, 200, 250, 300 → final 330 | Reload every tokenizer and compare assets/probes | Same backend, IDs, chat template and generation metadata |
| SGLang and MILES loaders, prepared and final | Invoke the actual installed loader functions | `TokenizersBackend`; correct digit IDs; no BOS; serving chat matches |
| Rollout → Core trainer | Pinned MILES generation code and Core batch path | Prompt IDs sent directly; returned output IDs appended directly and consumed by training, without decoding/re-tokenizing the response |

The upstream comparison pins Dolma2 at
[`5292e5d6c0f40b67cc765fe41bec991cf4345b5c`](https://huggingface.co/allenai/dolma2-tokenizer/tree/5292e5d6c0f40b67cc765fe41bec991cf4345b5c)
and the production reference `allenai/Olmo-3-7B-Think-SFT` at
[`6ff857587e040d6d523a3d5f3a56e918f5401d66`](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT/tree/6ff857587e040d6d523a3d5f3a56e918f5401d66).
The saved LC config does not pin a tokenizer revision. Thus the audit establishes
agreement with that declared tokenizer family and retained native metadata;
it does not reconstruct the historical pretraining corpus or prove every
pretraining shard was produced correctly. Earlier intermediate native source
copies were not available at the inspected paths. Keep this distinction from
the direct, artifact-by-artifact SFT/RL verification.

All actual corrected paths have 100,278 vocabulary entries, EOS 100257, PAD
100277 and no BOS. The number `9078563412` encodes as
`[23505, 25505, 16546, 17]`, corresponding to `907 | 856 | 341 | 2`.
HF saving changes only `local_files_only: true → false` in `tokenizer_config.json`;
its changed hash therefore does not indicate changed tokenization.

SFT training and inference templates are intentionally not byte-identical:
the exported template adds the assistant's `<think>` prefix when generating.
The trained system prompt and rendering of a complete training conversation
are preserved. `<think>`/`</think>` are ordinary token sequences, not EOS IDs.
Generation metadata retains EOS IDs `[100265, 100257]` (`im_end`, `endoftext`).
The installed SGLang model configuration unions model and generation-config EOS
IDs. A response that remains inside repetitive reasoning has not simply failed
to stop on an already-emitted `</think>`; that marker transitions to the answer.

### Reproduced unsafe loading behavior

Loading the original reference payload through `AutoTokenizer` in Transformers
5.12.1 selects `GPT2Tokenizer`, replaces the serialized pre-tokenizer, and makes
the digit probe produce `[1954, 2495, 3487, 1958, 717]`. It also exposes BOS 100257.
The vocabulary mapping remains identical, so checking vocabulary size or even
its mapping hash would miss this problem.

The corrected SFT preparation reads the canonical raw tokenizer backend and
writes a safe generic tokenizer class. The production export gate accepts
`TokenizersBackend` or `PreTrainedTokenizerFast` and checks actual segmentation.
All actual corrected SFT/RL artifacts passed those behavioral checks. Do not
replace their tokenizer config with the unsafe original reference config.
This reproduction identifies a real hazard, not evidence that it caused the
loops observed in these corrected checkpoints.

### What survives native checkpoint saves

HF exports are self-contained for tokenization: the actor calls
`AutoTokenizer.save_pretrained` and copies the chat template and generation
config into each snapshot. The final exported policy was independently loaded
for both full GSM8K evaluations.

Native RL checkpoints retain model/optimizer, rank RNG/scheduler states, model
configs and the committed rollout cursor. They do **not** embed the tokenizer
assets. Restoring this run also requires its retained `prepared/hf` tokenizer
and prepared data/configuration. Keep the complete run directory for native
resume; distribute `export-hf` for inference. This audit verified the retained
assets and final native manifest, not a new trainer resume experiment.

## Full GSM8K behavior before and after RL

Each row covers all 1,319 official test questions, with a 10,240-token cap.
The chat comparisons use the checkpoint's own template and the training
`GSM8KVerifier`; raw completion uses the separate evaluator's formatter/scorer.
Their absolute scores are not interchangeable.

| Protocol / checkpoint | Correct | Correct and finished | Hit cap | Mean response tokens |
| --- | ---: | ---: | ---: | ---: |
| Chat, greedy, original SFT | 705 (53.45%) | 556 (42.15%) | 709 (53.75%) | 6,174 |
| Chat, greedy, final RL 330 | 838 (63.53%) | 751 (56.94%) | 464 (35.18%) | 4,481 |
| Chat, T=1, original SFT | 1,046 (79.30%) | 1,040 (78.85%) | 101 (7.66%) | 4,404 |
| Chat, T=1, final RL 330 | 1,053 (79.83%) | 1,050 (79.61%) | 54 (4.09%) | 3,306 |
| Raw completion, greedy, original SFT | 602 (45.64%) | 558 (42.30%) | 490 (37.15%) | 3,996 |
| Raw completion, greedy, final RL 330 | 596 (45.19%) | 564 (42.76%) | 456 (34.57%) | 3,742 |

“Correct and finished” requires a non-capped response as well as verifier
success. Ordinary correctness can count a final number inside an unfinished
thought. In the original greedy chat baseline, **149 capped responses were
scored correct and none of the 709 capped responses closed `</think>`**.
The final model still has 87 capped-but-correct responses and 462 capped
responses that never close thinking. This remains a substantial behavior issue.

Paired chat greedy outcomes favor final RL: 255 questions become correct versus
122 becoming incorrect (exact McNemar p=6.44e-12); correct-and-finished counts
are 283 versus 88 (p=5.41e-25). Sampled correctness changes little: 119 gains
versus 112 losses (p=.69). Raw completion has 160 gains versus 166 losses
(p=.78). These are single-run checkpoint comparisons, not replicated training
ablations. Repeated greedy baseline accuracy was 688 in the earlier audit and
705 here; batched serving is not guaranteed bitwise deterministic. The final
chat movement is much larger than that observed baseline difference.

Full chat generation time fell from 856 to 644 seconds for greedy and 667 to
524 seconds for T=1, on one B300 with the same settings. Shorter responses
change the amount of work; these timings do not establish a faster kernel.
Baseline/final server startup was 285/81 seconds, confounded by order/cache warming.
T=1 requests explicitly use top-p 1 and unrestricted top-k. Chat serving uses
64 admissions/decode graphs, eager prefill, Triton attention and PyTorch sampling.

Runs:

1. [Full chat greedy and T=1, original/final](https://beaker.org/ex/01M3CN9D8MGXC0RKEXSHJZN2AE).
2. [Raw greedy original](https://beaker.org/ex/01M3BVS0SF91RY4NH7V6EAKZ1E).
3. [Raw greedy final](https://beaker.org/ex/01M3CN9SZFB3VEZW9JSDNGZZJB).

## Qualitative examples

These are the first four dataset-order cases satisfying original greedy-capped,
T=1-finished-correct, with thinking closed. Selection is deliberately illustrative;
it is not an estimate of how often sampling rescues an arbitrary failure.

| GSM8K test ID | Greedy behavior through the 10,240-token cap | T=1 behavior |
| --- | --- | --- |
| 0, Janet's eggs | Repeatedly computes `(16−3−4)×2=18`, then questions whether “four” means eggs or muffins | Eventually commits to 18 and closes thinking; 4,970 tokens |
| 1, robe fabric | Computes `2+1=3`, then repeatedly reopens whether “half that much” refers to blue or total fabric | Resolves the reference and returns 3 |
| 2, house flip | Repeatedly computes `200,000−130,000=70,000`, interleaved with rejected alternatives | Returns 70,000 and closes thinking; 6,910 tokens |
| 3, weekly sprints | Oscillates between 180 and 540 while inventing ambiguity about “3 sprints 3 times a week” | Chooses `3×3×60=540`, closes thinking; 1,213 tokens |

This looks like repeated deliberation and failure to commit, sometimes settling
into nearly verbatim cycles. It is not merely missing arithmetic knowledge.
However, different complete generations cannot establish whether T=1 escaped
an existing loop or avoided it earlier. The controlled continuation experiment
below tests that narrower question.

## Controlled continuation from an existing loop

[One-B300 diagnostic](https://beaker.org/ex/01M3CRMAXFKK3ZKA0WE6MZJ1Q2)
completed successfully in 6m39s, including serving startup, on the same immutable
image. This avoids changing precision to squeeze the full BF16 model into the
local 24GB GPU. No training or checkpoint mutation was involved.

The test regenerated the four selected questions greedily, retaining exact
returned token IDs. Janet's eggs finished in 1,701 tokens on this fresh run and
was excluded. For the other three, it forked after 4,096 generated tokens and
allowed **4,096 additional tokens**, using identical input IDs at each temperature.
There was one greedy continuation and four at each positive temperature per
prefix: 39 responses total. The server RNG seed was 17; replicates draw from its
stream, rather than using independent per-request seed overrides. All use
unrestricted top-p/top-k. There is no repetition penalty or inserted instruction.

The three prefixes were visibly repetitive: 64–73% of their overlapping 32-token
windows were repeats. All three new greedy continuations exactly matched the
next 2,048 tokens of the original fresh greedy trajectory, the entire available
overlap. Thus re-prefilling the prefix did not itself break these tested loops.

| Temperature | Finished correctly within continuation budget | Closed thinking | Result by problem |
| --- | ---: | ---: | --- |
| 0 | 0/3 | 0/3 | All three remain unfinished |
| 0.3 | 0/12 | 0/12 | All three remain unfinished |
| 0.6 | 0/12 | 0/12 | All three remain unfinished |
| 1.0 | 4/12 | 4/12 | Robe 4/4; house flip 0/4; sprints 0/4 |

The robe completions finish after 779, 1,334, 1,623 and 2,503 additional tokens.
Each closes `</think>`, answers 3, and stops on token **100257**, demonstrating
that EOS handling works on these continuations. One escape explicitly resolves
the repeated ambiguity: “there’s only blue fiber mentioned,” then commits to
the total and produces the final answer.

Higher temperature changes the text even when it does not finish: all T=1
replicates have distinct output token sequences. In the sprint example, one
continuation leaves the verbatim cycle and argues for 540, then resumes checking
alternatives and hits the cap. The house-flip example often continues the same
argument with wording variations. Breaking exact textual repetition and
successfully terminating are therefore different outcomes. The table uses the
stricter termination criterion, and its budget is shorter than the full 10,240-token
evaluation allowance.

This establishes a causal temperature-dependent escape for **one selected
prefix**, not a general rescue rate. The full-from-start T=1 evaluation is much
healthier than greedy, but T=1 is not a guaranteed repair once thousands of tokens
of repetitive reasoning are already in context.

## Working diagnosis and next choice

The direct evidence supports a decoding-sensitive reasoning/termination problem
already present after corrected SFT. It does not implicate tokenizer drift across
RL saves. Low-temperature decoding repeatedly selects the same continuation;
sampling can open another continuation, but the repeated context can continue
to pull the model back into checking alternatives. Determining why SFT learned
this behavior would require a separate investigation of training examples,
objective and optimization; these experiments do not isolate that upstream cause.

Keep the corrected tokenizer bundle, checkpoint chat template, T=1/top-p=1,
10,240-token response cap and recommended optimized serving path for RL.
Do not treat greedy as a proxy for the sampling workload or replace the tokenizer
to fix these loops. Report both ordinary correctness and correct-and-finished
rates, response lengths, thinking closure and cap hits. The long run's clearest
quality movement is improved chat termination; sampled answer accuracy and the
small IFEval panels do not establish a broader capability gain.

Before another long training sweep, a useful narrow follow-up is to compare
termination behavior of the source SFT's earlier checkpoint or alternative SFT
recipe on the same prompts/decoding settings, and inspect its reasoning-length
and repetition distribution. That is a proposed follow-up, not an experiment
completed here. Avoid changing tokenizer, numerical compatibility, temperature,
and training hyperparameters together: it would obscure which change mattered.

## Evidence and reproduction

The [machine-readable audit](hero-tokenizer-loops-20260925.json) preserves
per-stage asset hashes, probes, loader checks and both evaluation summaries.

Local ignored evidence lives under
`runs/hero-non-emo-long-lag6-20260925/tokenizer-loop/`; full evaluation outputs
are alongside it under `chat-gsm8k/` and `greedy-full-gsm8k/`.
`tokenizer-audit.json`, `tokenizer-extra.json`, `pt-production-comparison.json`,
`qualitative-cases.json`, `controlled-summary.json` and `final-eval-summary.json` (in the parent directory)
retain the checks and counts. The original retokenization proof records
Transformers 5.4.0/tokenizers 0.22.2, successful 20-text/three-chat reload probes,
and fresh data/loader checks; this audit repeats the relevant checks in the
newer actual RL image.

Key asset hashes:

| Artifact | SHA256 |
| --- | --- |
| Corrected SFT / RL tokenizer.json | `18e309ad7f9c60037eaf26aca401a96128187cad2bdfa01a508dca7526bf6300` |
| Vocabulary mapping | `a5a0d5e00a14d7ce506e9977acdd6cc8d6e76815b29f7f143c2f8779250a448c` |
| Serialized pre-tokenizer | `ba2cf544ccca9022f2033fe108be1c340633f0f30f50c0dd6835bb49371ac6a1` |
| HF chat template | `43b0c225dd327d4af450809bfb3abfbd828eb19d6546d3e9ae33782464874d6a` |
| Retokenization manifest | `43f122058d6cb177cc7e18094104ad309089979f24ea27c608f45d06fa06d5ab` |
| Saved native SFT config | `74587b59dde2cb434cf8b19849b492187f8f752e89b1c7b98ba022cb7550776b` |

The controlled job's result dataset is `01M3CRMAXZ84K75RZ1HTZZ6DDS`, including
`fresh-greedy.json`, `prefixes.json`, `continuations.jsonl`, `design.json`, engine
logs and tokenizer-loader evidence. WEKA copy:
`/weka/oe-training-default/robertb/open-instruct/runs/hero-non-emo-long-lag6-20260925/tokenizer-loop/controlled-v1`.
The metadata helper initially used an obsolete SGLang import path; its corrected
read-only rerun passed without changing the serving process or model. The
original submitted helper and its failure log are retained alongside the
successful `tokenizer-extra.json` / `runtime-extra.json` results.
