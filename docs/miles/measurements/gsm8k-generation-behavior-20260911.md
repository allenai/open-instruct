# Paired GSM8K generations and routing investigation

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

The completed Core100 and Megatron100 comparison used the **same 128 held-out questions**, with identical rendered prompts, targets, and prompt-token hashes. All six evaluations use this same set. There is no different question set to cross-evaluate: the final results already compare the two trained policies on exactly the same questions.

We extracted all **1,536 actual generations** (128 questions × six evaluations × two backends). The reader validation matched every response hash, score, token length, status, and policy version to the passing independent audits, and every original dump hash to its audit. The original `eval.jsonl` hash is `6008c065c2555667cc38449057b822bc011001530fd7585f5ab03b3e67af3804`. The extracted JSON hash is `0ffb6a8efd9f563a1984b7d8ac511b257e6173a2f05e63aa3180bc9eb46def19`.

## Review artifacts

Local artifacts are in `/home/robert/proj/open-instruct/.artifacts/miles-gsm8k-generations-20260911/`:

- `comparison-reader.html`: standalone offline reader with paired initial/current responses, all intermediate evaluations, question search, and final-outcome filters.
- `question-outcomes.csv`: exact question IDs, targets, scores and response lengths across evaluations.
- `generations.json`: all unmodified response strings plus prompt and source metadata.
- `eval.jsonl`: original prepared evaluation data bytes.
- `generation-verification.json`: full extraction verification and source dump hashes.
- `selected-generations.md`: six selected cases with both backends' complete initial and final responses.
- `behavior-evidence.json`: full selected responses and literal-format counts.

The committed [behavior evidence](gsm8k-generation-behavior-20260911.json) retains counts, selected question IDs, response hashes and lengths. The reusable reader generator is `scripts/miles/compare_gsm8k_generations.py`. Raw source dumps remain under the campaign WEKA root recorded in the [configuration audit](gsm8k-configuration-differences-20260911.md).

## What the actual responses show

These are deliberately selected qualitative examples, not a random sample or a classifier applied to every error. All IDs below refer to the official test source; the reader retains full prepared IDs.

| Source ID | Target | Core after 100 | Megatron after 100 | Observation |
|---|---:|---|---|---|
| gsm8k-test-000346 | 9 | Incorrect, 4096 tokens | Correct, 423 tokens | Core computes 36−20−7=9, then repeatedly doubts whether the wording supplies enough information and exhausts the budget. |
| gsm8k-test-000177 | 350 | Answers 600, 1815 tokens | Answers 350, 1360 tokens | Core reaches 350 in its reasoning but changes fuel from one-third of salary to one-third of the post-rent remainder in its final explanation. This is an actual changed calculation, below the cap. |
| gsm8k-test-001239 | 50 | Answers 250, 3872 tokens | Answers 50, 1587 tokens | Core divides all five days' cleaning work by one day's available time. Megatron includes all five days. |
| gsm8k-test-001029 | 34 | Answers 34, 1534 tokens | Answers 27, 2003 tokens | A reverse case: Megatron omits the seven green balloons; Core includes them. |
| gsm8k-test-000809 | 50 | Incorrect, 4096 tokens | Correct by verifier, 4096 tokens | Both become repetitive and hit the cap. Megatron's saved response still contains an extractable correct answer. Correctness and completion are distinct. |
| gsm8k-test-000547 | 81 | Incorrect, 4096 tokens | Answers 81, 1482 tokens | Core turns a four-round multiplication problem into a long numbered call list, still enumerating calls at truncation. |

The cookie, salary and phone-tree questions were initially correct in both systems. Core improved on the balloon problem while Megatron remained wrong. The entire final paired result is 84 both correct, 23 Megatron-only correct, nine Core-only correct, and 12 neither correct.

A simple structural count reinforces the observed completion behavior:

| Literal output property, out of 128 | Core initial | Core final | Megatron initial | Megatron final |
|---|---:|---:|---:|---:|
| No closing `</think>` marker | 17 | 23 | 19 | 17 |
| Incorrect, capped, and no closing `</think>` marker | 14 | 21 | 15 | 11 |

Every response missing that marker in these dumps also reaches the cap. The marker count is a transparent string check, not a reasoning-quality metric. It does not establish that allowing more tokens would recover an answer. The full intermediate counts are retained in the JSON. Also, an `Answer:` marker is not required by every accepted verifier extraction pattern, and can occur inside unfinished reasoning; marker presence alone is not correctness.

## Router replay and the next causal checks

**Rollout router replay was disabled in both completed 100-update runs and both current 500-update runs.** Core's `_score(..., use_replay=True)` is permission to enter its replay wrapper; the wrapper checks the actual runtime flag and returns a no-op context when disabled. The first retained Core100 and Core500 training batches contain no captured expert IDs, corroborating the effective configuration. Therefore, these learning runs do not qualify replay and a failure inside replay cannot directly explain their difference.

That does **not** exclude natural routing disagreement between serving and training, or differences between the two trainers' routing. Since route IDs were not captured in these comparisons, log-probability drift alone cannot measure route mismatch. A controlled same-weight, same-prefix expert-ID comparison is the appropriate way to investigate that hypothesis.

Current CPU replay tests were rerun: six passed, covering the disabled no-op, forced IDs with nonzero router gradients, cleanup after exceptions, and policy/auxiliary/combined gradients with activation recomputation on and off. Prior native EP1/EP2 and tiny live SGLang tests cover more of the integration, but do not establish full-SFT replay correctness. In particular, the tiny live case had no nonzero policy advantages. Router weights remain trainable when replay is used; it fixes expert choices, not their differentiable mixture weights.

The most useful controlled followups are:

1. **Scoring versus gradient-enabled forward:** the old Core100 SwiGLU rounding discrepancy is a concrete implementation difference, fixed in Core500. The longer comparison also changes serving settings, so it will not isolate this fix causally on its own.
2. **Auxiliary objective on actual variable-length batches:** current Core and Megatron use different length weighting, and Megatron includes padded positions in router auxiliaries. Equal coefficients are not equal objectives. Existing equal-length, unpadded tests intentionally omit this difference. Measure real-batch policy, load-balancing and z-loss gradients separately, using matching weights and token inputs.
3. **Initial serving and natural-route differences:** 17 question outcomes already disagree at update zero despite similar aggregate scores. Hold weights, prefixes, engine settings and batch scheduling fixed when tracing logits and expert IDs; do not attribute greedy differences to random sampling alone.

The observed loops and changed arithmetic are real output differences. They narrow the behavior to explain; they do not yet identify a single implementation cause. Preserve the ongoing 500-update runs while qualifying isolated changes separately.

## Update-zero divergence and grader sensitivity

Only **7/128 initial generated token sequences are identical**. All 128 input token sequences are identical. The median common generated prefix is 232.5 tokens; comparison of 33,137 generated tokens before the first token divergence finds mean absolute stored log-probability difference 0.004326823 and maximum 0.303771973. These compare the same emitted token under the same preceding tokens; they establish different effective probability calculations, without identifying which kernel, scheduling, or state difference causes them. There are 88 both-correct, nine Core-only, eight Megatron-only, and 23 both-wrong initial outcomes. See [token evidence](gsm8k-initial-tokens-20260911.json) and [question outcomes](gsm8k-initial-generations-20260911.json).

The initial checkpoint and chat-template/prompt tokens agree, but the original inference configurations were not completely identical: the frozen Core arm explicitly selected the PyTorch sampler and a 32,768-token pool, while Megatron resolved FlashInfer and an automatically sized pool. Greedy evaluation does not make those configuration differences a demonstrated cause. Exact-prefix activation and expert-ID diagnostics before and after initial trainer publication are being prepared with the original images.

Manual reading also exposed a **shared verifier formatting limitation**: its extracted final number is compared as a string, so `42.00` does not match target `42`. Both pipelines and the independent audits use that same rule. For example, the initial lemonade response from Megatron is mathematically correct but rejected for decimal formatting. A counterfactual Decimal comparison of that same last extracted number changes Core initial/final from 97/93 to 101/96 and Megatron from 96/107 to 101/107. Thus numerical formatting explains three of the 14-question final gap, but leaves an 11-question gap. This regrade is a sensitivity analysis, not a change to the official run rewards, and does not establish that the models would train identically under a different verifier. Every extracted string was first checked to reproduce its stored original score. All cases are retained in the [format sensitivity artifact](gsm8k-numeric-format-20260911.json).
