# Holmes Acceptance Test Conclusions

## Scope

These notes summarize Holmes/B300 benchmark runs for Olmo inference acceptance testing and compare them against the H100 baseline table in `HOLMES_HANDOFF.md` where the configuration matches.

## H100 Baselines From Handoff

| Model | TP | Generation length | H100 TPS |
| --- | ---: | ---: | ---: |
| Olmo 3 32B | 1 | 8k | 300.50 |
| Olmo 3 32B | 4 | 8k | 1302.62 |
| Olmo 3 32B | 8 | 32k | 928.45 |

The handoff target is roughly 2.4x H100 inference throughput on GB200/B300-class hardware.

## Completed Holmes Runs

| Config | Beaker experiment | Status | TPS | Comparison |
| --- | --- | --- | ---: | --- |
| Olmo 3 32B, TP=1, 8k generation | https://beaker.org/ex/01KV6R7BW9KHM3HCWW7KZ8WXGH | Passed | 216.36 | 0.72x the documented 32B TP=1 H100 baseline; below the 2.4x target. |
| Olmo 3 32B, TP=4, 8k generation | https://beaker.org/ex/01KV6SNJBCGEQ8P5BBX67X5G1B | Passed | 3588.51 | 2.75x the documented 32B TP=4 H100 baseline; above the 2.4x target. |
| Olmo 3 7B, TP=1, 8k generation | https://beaker.org/ex/01KV74GD5HSA24ZQT59JQ9VHWM | Passed | 432.68 | No documented 7B H100 baseline in the handoff. This is 2.00x the B300 32B TP=1 run, but that is not a hardware comparison. |

## In Progress Or Next Runs

| Config | Goal |
| --- | --- |
| Olmo 3 32B, TP=8, 32k generation | https://beaker.org/ex/01KV79XA9B3A89SCTE18K9BR5Q launched with the explicit 4h Beaker task timeout added in `scripts/benchmarking/launch_benchmark_single_node_holmes_tp8.sh`. |

## Current Conclusions

- The strongest passing result so far is Olmo 3 32B TP=4 at 8k generation: 3588.51 TPS, which is above the documented 2.4x target.
- The Olmo 3 32B TP=1 8k run is below the documented target.
- The Olmo 3 7B TP=1 8k run passed at 432.68 TPS, but the handoff does not include a matching H100 7B baseline.
- TP=8 at 32k still needs a clean passing run before drawing conclusions.
