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
| Olmo 3 32B, TP=1, 8k generation, 8 GPUs | https://beaker.org/ex/01KV7B3WYBB2F05R5NAD0BTEZ0 | Passed | 3383.68 | Useful 8-GPU node diagnostic, but not the documented TP=1 single-GPU baseline path. Per-GPU aggregate is 422.96 TPS, and this run did not use `--vllm_enforce_eager`. |
| Olmo 3 32B, TP=4, 8k generation | https://beaker.org/ex/01KV6SNJBCGEQ8P5BBX67X5G1B | Passed | 3588.51 | 2.75x the documented 32B TP=4 H100 baseline; above the 2.4x target. |
| Olmo 3 7B, TP=1, 8k generation | https://beaker.org/ex/01KV74GD5HSA24ZQT59JQ9VHWM | Passed | 432.68 | No documented 7B H100 baseline in the handoff. |

## Failed Or Superseded Runs

| Config | Beaker experiment | Outcome |
| --- | --- | --- |
| Olmo 3 32B, TP=8, 32k generation | https://beaker.org/ex/01KV79XA9B3A89SCTE18K9BR5Q | Failed during vLLM engine initialization after the hardcoded 1200s internal wait, despite the 4h Beaker task timeout. Superseded by later timeout retries. |
| Olmo 3 32B, TP=8, 32k generation | https://beaker.org/ex/01KV7BJM4Q6GJK83ZE0R8202KH | Failed at vLLM engine startup after 20m22s. The launcher used unsupported `VLLM_ENGINE_INIT_TIMEOUT_S`; vLLM warned it was unknown, so this did not raise the internal vLLM engine readiness timeout. |

## In Progress Or Next Runs

| Config | Goal |
| --- | --- |
| Olmo 3 32B, TP=8, 32k generation | https://beaker.org/ex/01KV7DC5W515CN1YCMDHEMVKGN launched from commit `8f23df74` with image `01KV7DAS2FYP5V1C0668NW46K6`, official `VLLM_ENGINE_READY_TIMEOUT_S=7200`, and Open Instruct wrapper `OPEN_INSTRUCT_VLLM_ENGINE_INIT_TIMEOUT_S=7500`. |
| Olmo 3 32B, TP=4, 32k generation | https://beaker.org/ex/01KV7F54ZQMRBDDB7ZE93S1GP1 launched from commit `980d84bf` with image `01KV7F2H4N917JGWVV5F8XNKAJ`. This uses the previous TP=4 single-node shape with 8 GPUs, `vllm_num_engines=2`, `vllm_tensor_parallel_size=4`, and `response_length=32768`. |
| Olmo 3 32B, TP=1, 8k generation, 8 GPUs, eager mode | https://beaker.org/ex/01KV7CS3JT0M8SP6SWP1DPDVSK launched from commit `c91adf64` with `--vllm_enforce_eager` to compare against the previous non-eager 8-GPU TP=1 diagnostic. |

## Current Conclusions

- The comparable Olmo 3 32B TP=1 8k single-GPU run passed at 216.36 TPS, which is 0.72x the documented H100 single-GPU baseline and below the 2.4x target.
- The Olmo 3 32B TP=1 8k 8-GPU diagnostic passed at 3383.68 TPS, but it should not be used as the primary comparison for the documented TP=1 baseline row.
- Olmo 3 32B TP=4 at 8k generation passed at 3588.51 TPS, which is 2.75x the documented H100 baseline and above the 2.4x target.
- The Olmo 3 7B TP=1 8k run passed at 432.68 TPS, but the handoff does not include a matching H100 7B baseline.
- TP=8 at 32k still needs a clean passing run before drawing conclusions for that H100 baseline row. TP=4 at 32k is also running as an additional fallback/diagnostic for long-generation behavior.
