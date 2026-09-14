# Published RL timing anchors and what they imply for our priorities

Reviewed September 14, 2026. These are scope-labeled reference points, not a
hardware-normalized leaderboard. No matched KDA/latent-MoE trainer benchmark or
measured FLOP utilization is available for our current run.

## Concrete published numbers

| Source / workload | Hardware | Published measurement | Interpretation |
|---|---|---|---|
| AReaL, dense 1.5B math | 128 H800 | 1,000 reported PPO steps in 14.8 h | Derived 53.3 s/reported PPO step, full training workflow |
| AReaL, dense 7B math | 192 H800 | 1,000 reported PPO steps in 25.4 h | Derived 91.4 s/reported PPO step |
| AReaL, dense 14B code | 256 H800 | 320 reported PPO steps in 21.9 h | Derived 246.4 s/reported PPO step |

These are Table 1 in the pinned [AReaL v1 paper](https://arxiv.org/html/2505.24298v1#S7.T1).
Section 7.1 assigns 75% of GPUs to inference. Its effective-throughput definition
counts consumed generated tokens after warmup. Model size, sequence budgets,
minibatch structure, allocation and objective differ from ours; the derived
seconds are not standalone forward/backward timings. Its interruptible generation
also rebuilds cache state after weight updates (section 4.1), a relevant systems
comparison to our refresh approach.

[DeepSeek-R1's supplement, section 2.4.4](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-09422-z/MediaObjects/41586_2025_9422_MOESM1_ESM.pdf)
reports **512 H800 and approximately 198 hours for R1-Zero**, and approximately
80 hours for R1. The [main paper's training details](https://www.nature.com/articles/s41586-025-09422-z)
give 10,400 R1-Zero policy updates, 512 responses/update, and 8,192 responses per
rollout split into 16 minibatches. Dividing the reported elapsed time yields
**68.5 seconds per R1-Zero update averaged over the run**, including amortized
rollout work. It does not establish a 68.5-second trainer forward/backward, or
provide a matched token denominator for our workload. Do not mix its RL cost
with DeepSeek-V3's much larger pretraining cost.

A closer architecture category, though still a different model and hardware, is
NVIDIA's [NeMo RL v0.6 H100 BF16 benchmark](https://docs.nvidia.com/nemo/rl/latest/about/performance-summary.html#nemo-rl-v0-6).
For **Qwen3-30B3A on 32 H100s**, the table reports **1,102 tokens/s/GPU and 192 s**
on-policy, or **1,414 tokens/s/GPU and 152 s** with one-step off-policy operation.
Generation batch is 2,048; training batch is 512; average generation length is
about 3,200 tokens. These are total-step figures, not a standalone trainer
comparison. Hardware, active geometry, token accounting and batch/update ratio
must be aligned before ranking implementations. This is an official reproducible
framework benchmark, not a research-paper result.

## Our corresponding measurements

The [matched trainer screen](trainer-capacity-20260914.md) uses EP2 on two B300s,
128 responses/update and approximately 271k input-plus-response model tokens.
The best arm averaged **22.0 s/update and 6,162 model tokens/s/trainer GPU**;
selected common batches without new cubins averaged 14.6 s and 9,341. The latter
is a conditional view, not a sustained warm-cache run. These exclude generation,
publication and other RL stages, so comparing them directly with NVIDIA's
1,102/1,414 or AReaL's full-cycle times would be invalid.

The subsequent [24-update live qualification](full-sft-basket-20260914.md) puts
that faster trainer into EP2 + two inference engines. Updates 6–23 delivered
**5,089 consumed response tokens/s across the four-GPU allocation**: approximately
**1,272 response tokens/s/allocated GPU**, excluding startup/eval/save/shutdown.
This is closer in metric scope to full RL throughput, but still uses a different
token denominator and architecture. It is not evidence of performance parity
with the larger H100 benchmark.

## Priority decision

Our own controlled measurements are more useful than cross-paper rankings.
After accelerating the trainer, its median forward/backward/optimizer phase was
16.4 s, but batch waiting occupied 58.1% of the warm cycle. Useful throughput
rose only modestly from the preceding packed c32 run (4,930 to 5,089 response
tokens/s; different sampled trajectories, not a paired speedup estimate).

For the internal trial, prioritize: (1) correct mixed-domain rewards, held-out
measurement and resume/export; (2) reliable serving and useful batch supply;
(3) publication, judge/code service and queue overhead on the critical path;
(4) cache reuse/startup; (5) further trainer kernel optimization unless the
16-GPU run shows that the trainer is now the bottleneck. These are an engineering
judgment for this workload, not a universal ordering. Improving training could
still reduce the trainer allocation and cost, even when throughput is inference
limited; that requires rebalancing and measuring the resulting topology.
