# Reward by response length on the four-domain basket

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

Every retained training sample of the dense and MoE basket arms, binned by
response length: does reward arrive above 10K or 16K tokens, and what does the
32K cap cost? Saturn CPU job
[01M2NHK01KZBKP9B3152C7K74T](https://beaker.org/ex/01M2NHK01KZBKP9B3152C7K74T)
([script](length-reward-20260916/bin_reward_by_length.py),
[full bins](length-reward-20260916/length-reward.json),
[plot](length-reward-20260916/length-reward.png)). Dense: updates 0–67 across the
single-node, g16, kv, c24, c24b and c24c roots, 17,408 samples. MoE: updates
0–101 across the robust and c16–c16f roots, 26,112 samples. Later roots override
earlier attempts of the same update.

| | Dense math | Dense code | Dense IF | Dense general | MoE math | MoE code | MoE IF | MoE general |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean tokens | 18,666 | 14,395 | 4,642 | 2,425 | 22,102 | 16,170 | 7,801 | 3,926 |
| Mean reward | 0.29 | 0.29 | 0.30 | 0.56 | 0.13 | 0.13 | 0.25 | 0.43 |
| Reward share above 10K tokens | 87% | 45% | 9% | 5% | 81% | 20% | 21% | 10% |
| Reward share above 16K tokens | 51% | 29% | 3% | 1% | 58% | 8% | 11% | 5% |
| Capped at 32K | 7.5% | 6.0% | 0.6% | 0.3% | 18.9% | 9.0% | 5.3% | 2.2% |
| Reward rate of capped responses | 0.02 | 0.005 | 0.03 | 0.00 | 0.01 | 0.00 | 0.06 | 0.08 |

Math needs the 32K budget on both models: half of all math reward is earned
above 16K tokens, and dense math accuracy peaks at 0.42 in the 12–16K band. Code
diverges by model: dense code accuracy declines gently with length (0.47 under
2K, 0.19 at 24–32K), while the MoE's collapses (0.40 under 2K, 0.03 above 16K),
so 40% of MoE code samples earn 8% of its code reward. Instruction following and
general are short on both models and lose under 10% of reward above 16K.

Capped responses earn almost nothing and are the batch stragglers: on dense
they are 3.5% of samples and about 11% of generated tokens; on the MoE 9.5% of
samples and roughly a quarter of tokens. A per-domain response budget (32K math,
16K IF and general, 16K code on the MoE or 24K on dense) would remove most of
that cost with the loss confined to code. Length is also predictable from the
domain (70% of dense general responses finish under 2K; 35% of math lands in
16–24K), which is the basis for length-aware admission in the producer.
