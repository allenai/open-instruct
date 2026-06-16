# Holmes Acceptance Test Conclusions

## Scope

These notes summarize Holmes/B300 benchmark runs for Olmo inference acceptance testing and compare them against the H100 baseline table in `HOLMES_HANDOFF.md` where the configuration matches.

## H100 Baselines From Handoff

| Model | TP | Generation length | H100 TPS |
| --- | ---: | ---: | ---: |
| Olmo 3 32B | 1 | 8k | 300.50 |
| Olmo 3 32B | 4 | 8k | 1302.62 |
| Olmo 3 32B | 8 | 32k | 928.45 |

The handoff target is roughly 2.4x H100 inference throughput on GB200/B300-class hardware. The handoff identifies `scripts/benchmarking/launch_benchmark_single_gpu_holmes.sh` as the PDF's single-GPU inference benchmark path for the TP=1, 8k baseline. The TP=4 and TP=8 rows use the single-node launchers.

## Completed Holmes Runs

| Config | Beaker experiment | Status | TPS | Comparison |
| --- | --- | --- | ---: | --- |
| Olmo 3 32B, TP=1, 8k generation, 1 GPU | https://beaker.org/ex/01KV6R7BW9KHM3HCWW7KZ8WXGH | Passed | 216.36 | 0.72x the documented 32B TP=1 H100 single-GPU baseline and below the 2.4x target. This run used `--vllm_enforce_eager`. |
| Olmo 3 32B, TP=1, 8k generation, 1 GPU, non-eager mode | https://beaker.org/ex/01KV7G8MMFZH1YAC2Z862M8CRR | Passed | 419.68 | 1.40x the documented 32B TP=1 H100 single-GPU baseline. Removing `--vllm_enforce_eager` improved this cell by 1.94x. |
| Olmo 3 32B, TP=1, 8k generation, 8 GPUs | https://beaker.org/ex/01KV7B3WYBB2F05R5NAD0BTEZ0 | Passed | 3383.68 | Useful 8-GPU node diagnostic, but not the documented TP=1 single-GPU baseline path. Per-GPU aggregate is 422.96 TPS, and this run did not use `--vllm_enforce_eager`. |
| Olmo 3 32B, TP=1, 8k generation, 8 GPUs, eager mode | https://beaker.org/ex/01KV7CS3JT0M8SP6SWP1DPDVSK | Passed | 1765.72 | Useful 8-GPU node diagnostic. Per-GPU aggregate is 220.72 TPS; eager mode is 0.52x the matching TP=1 8-GPU non-eager diagnostic. |
| Olmo 3 32B, TP=4, 8k generation | https://beaker.org/ex/01KV6SNJBCGEQ8P5BBX67X5G1B | Passed | 3588.51 | 2.75x the documented 32B TP=4 H100 baseline; above the 2.4x target. |
| Olmo 3 32B, TP=4, 8k generation, eager mode | https://beaker.org/ex/01KV7G8HPAGD3X236NXTM6EPE0 | Passed | 612.39 | 0.47x the documented 32B TP=4 H100 baseline. Eager mode is 0.17x the matching TP=4 8k non-eager run. |
| Olmo 3 32B, TP=4, 32k generation | https://beaker.org/ex/01KV7F54ZQMRBDDB7ZE93S1GP1 | Passed | 2641.33 | 8-GPU long-generation diagnostic using two TP=4 engines. This is 1.16x the matching TP=8 32k non-eager result, but it is not the documented TP=8 baseline shape. |
| Olmo 3 32B, TP=8, 32k generation | https://beaker.org/ex/01KV7GZP2DHZQ7P6TMHVM7KPFQ | Passed | 2271.36 | 2.45x the documented 32B TP=8 32k H100 baseline; above the 2.4x target. This corrected run removed the Ray executor override so TP=8 used the local `mp` executor path. |
| Olmo 3 7B, TP=1, 8k generation | https://beaker.org/ex/01KV74GD5HSA24ZQT59JQ9VHWM | Passed | 432.68 | No documented 7B H100 baseline in the handoff. |

## Throughput Grid

All TPS values are aggregate benchmark throughput across the configured vLLM engines, not per-engine throughput.

| TP | H100 doc baseline | B300 1 GPU eager 8k | B300 1 GPU non-eager 8k | B300 8 GPU non-eager 8k | B300 8 GPU eager 8k | B300 8 GPU non-eager 32k | B300 8 GPU eager 32k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 300.50 TPS, 8k | 216.36 | 419.68 | 3383.68 | 1765.72 | n/a | n/a |
| 4 | 1302.62 TPS, 8k | n/a | n/a | 3588.51 | 612.39 | 2641.33 | running: https://beaker.org/ex/01KV7G8K55GVS71BAJ8W5XBT2X |
| 8 | 928.45 TPS, 32k | n/a | n/a | n/a | n/a | 2271.36 | running: https://beaker.org/ex/01KV7GZTHQFRFHSRKHX47KFRVG |

## Failed Or Superseded Runs

| Config | Beaker experiment | Outcome |
| --- | --- | --- |
| Olmo 3 32B, TP=8, 32k generation | https://beaker.org/ex/01KV79XA9B3A89SCTE18K9BR5Q | Failed during vLLM engine initialization after the hardcoded 1200s internal wait, despite the 4h Beaker task timeout. Superseded by later timeout retries. |
| Olmo 3 32B, TP=8, 32k generation | https://beaker.org/ex/01KV7BJM4Q6GJK83ZE0R8202KH | Failed at vLLM engine startup after 20m22s. The launcher used unsupported `VLLM_ENGINE_INIT_TIMEOUT_S`; vLLM warned it was unknown, so this did not raise the internal vLLM engine readiness timeout. |
| Olmo 3 32B, TP=8, 32k generation | https://beaker.org/ex/01KV7DC5W515CN1YCMDHEMVKGN | Failed during vLLM EngineCore startup after vLLM could not create its nested Ray placement group requiring 8 GPUs within 1800s. Root cause was the TP=8 launcher overriding the Open Instruct TP>1 default and passing `--vllm_distributed_executor_backend ray`. Superseded by corrected `mp`-executor retries. |
| Olmo 3 32B, TP=8, 32k generation, eager mode | https://beaker.org/ex/01KV7G8NP4VR4GE5RTYGDNPSJS | Canceled because it used the same nested Ray executor override as the failed TP=8 non-eager run. Superseded by the corrected eager retry. |

## In Progress Or Next Runs

| Config | Goal |
| --- | --- |
| Olmo 3 32B, TP=8, 32k generation, eager mode | https://beaker.org/ex/01KV7GZTHQFRFHSRKHX47KFRVG launched from commit `402852b2` with image `01KV7GYNMS32XJZF7F39Q6V8C4`, official `VLLM_ENGINE_READY_TIMEOUT_S=7200`, and Open Instruct wrapper `OPEN_INSTRUCT_VLLM_ENGINE_INIT_TIMEOUT_S=7500`. This removes the nested Ray executor override so TP>1 uses the local `mp` executor path; logs reached repeated vLLM health checks. |
| Olmo 3 32B, TP=4, 32k generation, eager mode | https://beaker.org/ex/01KV7G8K55GVS71BAJ8W5XBT2X launched from commit `4d8e0694` with image `01KV7G7CN191WZMNATXKDYX4ZZ` as the eager comparison for the completed TP=4 32k run. |

## Current Conclusions

- The comparable Olmo 3 32B TP=1 8k single-GPU non-eager run passed at 419.68 TPS, which is 1.40x the documented H100 single-GPU baseline. The eager version passed at 216.36 TPS, only 0.72x H100.
- The Olmo 3 32B TP=1 8k 8-GPU diagnostic passed at 3383.68 TPS non-eager and 1765.72 TPS eager, but this should not be used as the primary comparison for the documented TP=1 single-GPU baseline row.
- Olmo 3 32B TP=4 at 8k generation passed at 3588.51 TPS, which is 2.75x the documented H100 baseline and above the 2.4x target. The matching eager run was much slower at 612.39 TPS.
- Olmo 3 32B TP=8 at 32k generation passed at 2271.36 TPS with the corrected `mp` executor path, which is 2.45x the documented H100 baseline and above the 2.4x target.
- For maximizing throughput on 8 GPUs, TP=4 with two TP=4 engines is the best completed setting so far. At 32k it reached 2641.33 TPS, compared with 2271.36 TPS for TP=8 with one TP=8 engine. The reported TPS is aggregate across both TP=4 engines, not per engine.
- The Olmo 3 7B TP=1 8k run passed at 432.68 TPS, but the handoff does not include a matching H100 7B baseline.
- The previous TP=8 failures were caused by vLLM startup configuration, especially forcing the Ray executor and triggering a nested Ray placement-group request for 8 GPUs. Removing the Ray executor override lets TP>1 use the local `mp` executor path, which produced the clean TP=8 32k pass.
