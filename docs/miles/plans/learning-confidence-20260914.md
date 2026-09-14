# Learning confidence campaign

User authorization: available workspace GPUs may be used for matched learning
runs, initially 200 updates, extending to 400 or longer if inconclusive. Isolated
branch: robertb/miles-learning-confidence-20260914.

## Response budget correction

The dense Dolci200 run inherited a 4096-response-token / 6144-context-token limit
from the short MoE efficiency basket. This is an experimental cap, not a model
or backend limit. The original recipe in scripts/train/olmo3/7b_think_rl_no_pipeline.sh
uses 32768 response tokens and starts from Think-DPO. Our requested comparison
starts from Think-SFT; do not claim reproduction of the paper's recipe.

At update zero, the dense run scored 1/128 on math while 117/128 answers were
truncated at 4096. That score is heavily confounded by the response budget.
Use 32768 response tokens, 2048 prompt tokens and 34816 total context for the
Olmo 3 comparison. Inspect initial generations and completion/cap fractions before
spending on a long trajectory. Reduce serving concurrency and use a dense-appropriate
KV token pool to fit the larger context; a MoE pool size is not portable.

## Workstreams

1. Recover and independently audit existing Core/Megatron 500-update and light-SFT
   200-update evidence. Keep native chat/sampled and raw/greedy evaluations separate.
2. Run a small matched Think-SFT GSM8K learning pair through original Open Instruct
   and MILES/Core, using identical frozen train and official-test held-out questions,
   rendered prompts, reward definition, response budget and optimizer exposure.
   Use deterministic local rewards to separate learning from code/judge reliability.
3. Resume the broad Dolci integration campaign after diagnosing code execution
   failures. Both recent basket runs exhausted existing HTTP retries: MoE after
   18 updates, dense after 11. Do not replace unavailable rewards with zeros.
4. At 200 updates inspect the whole curve, mixed-reward group fraction, response
   lengths/caps and paired question outcomes. Extend from saved checkpoints if
   healthy learning remains inconclusive; investigate a persistent opposite-direction
   result before blindly extending. Use another seed when trajectory variance is
   the main uncertainty. Compare equal training responses/tokens, not just updates.

Retain initial and final exports for one common serving/evaluation harness.
Report planned endpoints and whole curves, effective config/source/image/data
provenance, allocated GPU-hours and generated/trained tokens. No equivalence or
paper-level quality claim follows from a single short seed.

## Current state

Dense experiment 01M2FBGGE8K8XJ7WJKCTE4KHMB failed on the external code endpoint
at 11 optimizer updates, before its first scheduled post-training evaluation.
Original Open Instruct reference image 01K7B0Z1KKP8AFKV2YKENMQ53B was downloaded
and inspected: torch 2.7.0, transformers 4.54.0.dev0, custom vLLM. Compatibility
with the published Olmo 3 checkpoint still needs qualification; the original-vs-Core
learning pair has not launched. Tool approval capacity failures interrupted work.
