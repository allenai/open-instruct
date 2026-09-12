# Two-node async GSM8K: first EP8 run at 64 × 8

Run file: [`configs/miles/qualification/two-node-async-gsm8k-20260912.toml`](../../configs/miles/qualification/two-node-async-gsm8k-20260912.toml).
Model: the full-SFT KDA/latent-MoE checkpoint (`sft-65536` step 23607, HF artifact).
Layout: one Holmes B300 node of eight Core trainer ranks at expert parallelism 8, one
node of eight TP1 SGLang engines. Collections of 64 prompts × 8 responses = 512, one
optimizer step per collection, bounded async with lag ≤ 2 and buffer factor 2, truncated
importance sampling, fused expert publication with 2 GiB buckets, skipped standalone
scoring pass with a check every 20 updates, shared compiler cache root.

| Attempt | Beaker | Source | Outcome |
| --- | --- | --- | --- |
| 1 | [01M2A9FYT22G9WC95EMMRKD1DJ](https://beaker.org/ex/01M2A9FYT22G9WC95EMMRKD1DJ) | `016318180` | 13 updates, then `TimeoutError` at the 13th publication boundary; both replicas exit 1/143. |
| r2 | [01M2AEF27DB9TZXXSXHDA41AM3](https://beaker.org/ex/01M2AEF27DB9TZXXSXHDA41AM3) | `c91e63716` | **100 updates, four checkpoints, final HF export, exit 0 on both replicas.** [W&B](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/77seq45c) |

Evidence: [r2 driver timings](miles-two-node-async-gsm8k-20260912/r2-driver_timing.jsonl),
[r2 publication records](miles-two-node-async-gsm8k-20260912/r2-publication.jsonl),
[r2 workflow state](miles-two-node-async-gsm8k-20260912/r2-workflow.json), and the
per-step summaries produced by `scripts/miles/analyze_async_run.py` from the Beaker
logs ([attempt 1](miles-two-node-async-gsm8k-20260912/attempt1-summary.json),
[r2](miles-two-node-async-gsm8k-20260912/r2-summary.json)). Run root:
`/weka/oe-training-default/robertb/open-instruct/runs/two-node-async-gsm8k-20260912-r2`
(checkpoints at updates 25/50/75/100, `export-hf/`, retained rollouts and eval outputs).

## The failure in attempt 1 and its fix

After twelve clean publications, the driver raised `TimeoutError` from
`RolloutManager.core_publication_boundary`: the async producer did not join within the
30 s `rollout_health_check_timeout` budget after being cancelled, and the boundary only
aborted the engines' in-flight requests *after* that join. Commit `d22f5ae27` makes the
boundary abort every engine's requests when the first join times out, then retry the join
with a 180 s budget, and logs the join duration.

In r2 the retry fired exactly once, at update 27: `Async producer join exceeded 30s with
0 active group(s); aborting engine requests and retrying`, then `joined ... in 30.02s`.
The producer had no active generation tasks at the time, so the cancellation itself was
not delivered for 30 s. The likely mechanism is a synchronous call blocking the producer's
event loop (a candidate is a durable cursor or debug-rollout write to WEKA); this is not
yet confirmed. The run continued normally. Attempt 1 also logged three router
`Failed to send typed request` errors to different engines in the four minutes before
its failure; r2 logged none.

## Timing (r2, warm steps 3–100)

| Quantity | Value |
| --- | ---: |
| Optimizer step, mean / median (64 microbatches per rank) | 45.4 s / 44.9 s |
| Seconds per sequence forward+backward | 0.70 s |
| Same quantity from the earlier 16-sequence runs (5.4 s / 8 per rank) | 0.68 s |
| Cadence between optimizer steps, mean / median | 60.3 s / 49.0 s |
| Trainer waiting for data (`generation_wait`), mean / median | 8.1 s / 2.2 s; over 30 s in 10 of 100 steps |
| Trainer busy fraction, (step + publication) / cadence | 0.76 (0.87 in attempt 1) |
| Publication, 37.0 GB fused, one trainer node → eight engines on the other node | 0.96 s mean, 1.01 s max; 25.1 s initial |
| Publication broadcast / engine load / transport load | 0.09 s / 0.25 s / 0.59 s |
| Checkpoint save (native, EP8, about 222 GB), each of four | 104–111 s |
| Held-out evaluation, 128 questions greedy, shared engines | 53 s initial; 64–80 s periodic |
| Final HF export | 129 s |
| Serving startup / trainer startup / first (cold) step | 492 s / 137 s / 590 s |
| Driver wall, 100 updates | 133 min (90.5 min training, 6.6 min eval, 13.4 min waiting) |
| Warm response-token throughput, eight engines | 13,900 tokens/s (8.5 samples/s) |

**The training step cost is per forward, not per step.** The step scales linearly with
sequences per rank: 0.70 s each at 64 per rank against 0.68 s at 8 per rank. That settles
the question parked in the optimization pass: sequence packing is the lever for the
trainer, and at 512 sequences the trainer, not generation, bounds the cadence.

## Learning signal (r2)

| Update | Held-out correct (of 128) | Held-out truncated |
| ---: | ---: | ---: |
| 0 | 106 (0.828) | 14.8% |
| 20 | 108 (0.844) | 10.2% |
| 40 | 109 (0.852) | 7.8% |
| 60 | 104 (0.813) | 6.3% |
| 80 | 110 (0.859) | 3.9% |
| 100 | 101 (0.789) | 7.0% |

| Collections | Training reward | Response tokens | Truncated | Groups all-correct | Groups all-wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–9 | 0.725 | 2188 | 21.7% | 40% | 7.5% |
| 40–49 | 0.824 | 1715 | 11.2% | 56% | 3.4% |
| 90–99 | 0.819 | 1149 | 4.7% | 55% | 3.6% |

Training reward rose from 0.73 to 0.82 while responses halved in length and truncation
fell from 22% to 5%. Held-out accuracy moved within ±5 questions of its start and ended
five below it; with one seed and 128 greedy questions this is not a learning verdict in
either direction. About 55% of groups carried no policy-gradient signal (all eight
responses correct) by the second half, versus 40% at the start; zero-signal groups are not
filtered in this stack. Trainer-versus-behavior mean log-probability gap was 0.020
throughout at lag ≤ 2 (0.009 at lag 1 in earlier runs); TIS clipping was negligible
(3e-5). All five standalone-versus-training scoring checks were exact.

## Full GSM8K test evaluation of the start and update-100 checkpoints

The 128-question in-run eval draws from the RLVR training pool and cannot resolve a
few-point change. [`scripts/miles/gsm8k_test_eval.py`](../../scripts/miles/gsm8k_test_eval.py)
served each checkpoint on eight TP1 engines with the training run's serving settings
(prefill CUDA graphs disabled; the default `breakable` prefill backend pads token counts
and crashed every engine on its first batch in
[01M2B1K8D969NJDVTAXZ0WHTD7](https://beaker.org/ex/01M2B1K8D969NJDVTAXZ0WHTD7)) and
scored all 1,319 official test questions with the run's `GSM8KVerifier`, greedy and with
eight temperature-1 samples each. Prompt rendering was checked against all 128 prepared
eval rows before generation. Job: [01M2B27XAEKC9V7F08GXR8KVRD](https://beaker.org/ex/01M2B27XAEKC9V7F08GXR8KVRD);
[summary](miles-two-node-async-gsm8k-20260912/gsm8k-test-eval-summary.json); responses
retained under the run root in `gsm8k-test-eval-20260912-r2/`.

| Checkpoint | Greedy correct / 1319 | Greedy truncated at 4096 | Greedy mean tokens | pass@1 (T=1, n=8) | pass@8 | All 8 correct | None of 8 correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| start (SFT-65536) | 967 (0.733) | 20.6% | 2065 | 0.668 | 0.913 | 29.6% | 8.7% |
| update 100 | 1021 (0.774) | 8.3% | 1115 | 0.744 | 0.942 | 42.7% | 5.8% |

Paired greedy comparison: 839 correct in both, 182 correct only after training, 128
correct only before, 170 in neither; net +54 questions (+4.1 points), exact McNemar
p = 0.0026. The change is real, and the 128-question curve (106 → 101) had its sign wrong.

Where the gain comes from: 130 of the 182 newly correct answers were truncated at the
4096-token cap before training. On the 1,004 questions that neither checkpoint
truncated, greedy accuracy moved from 0.897 to 0.871 (26 questions worse); that subset is
selected on the outcome, so it is suggestive rather than conclusive, but it says the
policy got terser rather than more accurate on problems it could already finish. Sampled
pass@8 rose 0.913 → 0.942, and of the 115 questions the start never solved in eight
samples, the trained model solves 62 at least once and 30 greedily, so the change is not
only sharpening; the start's 26% sampled truncation rate is the confound there too.

Note that the official test split is harder for this model than the in-run pool: 0.733
greedy against 0.828 on the 128 prepared questions.

## Qualification scope

Established by r2: EP8 startup, 100 finite optimizer steps with gradients in dense,
expert and router groups, cross-node publication to eight engines, four native EP8
checkpoint saves, blocking shared-engine evaluation, final HF export, clean two-node
teardown, and the shared compiler-cache root receiving its first publication (7 trainer
workers published; the cold miss was expected on a fresh root).

Not established: an eval of the update-25/50/75 native checkpoints (they need HF export first); resume from an EP8 checkpoint (multi-node runs require
`auto_resume=false`; a manual relaunch against the saved root is the test), learning
quality, the root cause of the one 30 s producer join stall, and a second-run compiler
cache hit at this shape.
