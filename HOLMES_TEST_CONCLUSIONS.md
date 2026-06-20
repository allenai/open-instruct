# Holmes Acceptance Test Conclusions

## Scope

These notes summarize Holmes/B300 benchmark runs for Olmo inference acceptance testing and compare them against the H100 baseline table in `HOLMES_HANDOFF.md` where the configuration matches.

Note: the original inference tables below were collected before the latest Holmes image/environment work (CUDA 13, FlashAttention 4, vLLM CUDA-13 wheel path, and Mason compile-cache defaults). Treat the older inference results as potentially obsolete until the refreshed grid finishes. The older results remain useful as historical diagnostics, but the refreshed runs should be preferred for final acceptance conclusions.

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
| 4 | 1302.62 TPS, 8k | n/a | n/a | 3588.51 | 612.39 | 2641.33 | superseded; see refreshed grid |
| 8 | 928.45 TPS, 32k | n/a | n/a | n/a | n/a | 2271.36 | superseded; see refreshed grid |

## Failed Or Superseded Runs

| Config | Beaker experiment | Outcome |
| --- | --- | --- |
| Olmo 3 32B, TP=8, 32k generation | https://beaker.org/ex/01KV79XA9B3A89SCTE18K9BR5Q | Failed during vLLM engine initialization after the hardcoded 1200s internal wait, despite the 4h Beaker task timeout. Superseded by later timeout retries. |
| Olmo 3 32B, TP=8, 32k generation | https://beaker.org/ex/01KV7BJM4Q6GJK83ZE0R8202KH | Failed at vLLM engine startup after 20m22s. The launcher used unsupported `VLLM_ENGINE_INIT_TIMEOUT_S`; vLLM warned it was unknown, so this did not raise the internal vLLM engine readiness timeout. |
| Olmo 3 32B, TP=8, 32k generation | https://beaker.org/ex/01KV7DC5W515CN1YCMDHEMVKGN | Failed during vLLM EngineCore startup after vLLM could not create its nested Ray placement group requiring 8 GPUs within 1800s. Root cause was the TP=8 launcher overriding the Open Instruct TP>1 default and passing `--vllm_distributed_executor_backend ray`. Superseded by corrected `mp`-executor retries. |
| Olmo 3 32B, TP=8, 32k generation, eager mode | https://beaker.org/ex/01KV7G8NP4VR4GE5RTYGDNPSJS | Canceled because it used the same nested Ray executor override as the failed TP=8 non-eager run. Superseded by the corrected eager retry. |

## Earlier Eager Follow-up Runs

The older eager 32k follow-up placeholders were superseded by the refreshed-grid retries below, which used the new image and longer vLLM/benchmark batch timeouts.

## Refreshed Inference Grid

These runs use the newer Holmes image/settings after the CUDA 13 / FlashAttention 4 / vLLM compatibility work and after restoring Mason `VLLM_DISABLE_COMPILE_CACHE=1`. Prefer this table over the older inference grid. Initial failures in several cells were superseded by retries with longer vLLM / benchmark batch timeouts; all refreshed cells now have resolved TPS summaries.

| Config | Beaker experiment | Status | TPS | Notes |
| --- | --- | --- | ---: | --- |
| Olmo 3 32B, TP=1, 8k generation, 1 GPU, non-eager | https://beaker.org/ex/01KVE6XSRDTMWRF6HDSFTNRB8H | Passed | 421.62 | Retry of the earlier init-timeout cell; essentially matches the older Holmes single-GPU non-eager result. |
| Olmo 3 32B, TP=1, 8k generation, 1 GPU, eager | https://beaker.org/ex/01KVCNME9HPP7V1R82JBHKCQR2 | Passed | 281.86 | New-image refreshed result; lower than the older non-eager single-GPU result and above the older eager result. |
| Olmo 3 32B, TP=4, 32k generation, 8 GPUs, non-eager | https://beaker.org/ex/01KVE6Y8AQNZYF2AXYP3BZV4A4 | Passed | 2669.13 | Retry of the earlier init-timeout cell; two TP=4 engines on one 8-GPU node. |
| Olmo 3 32B, TP=4, 32k generation, 8 GPUs, eager | https://beaker.org/ex/01KVEFVZT84QQ2JNZ9QZS49Z3Y | Passed | 961.60 | Retry with longer batch/completion timeouts; much slower than TP=4 non-eager but faster than TP=8 eager. |
| Olmo 3 32B, TP=8, 32k generation, 8 GPUs, non-eager | https://beaker.org/ex/01KVCMXNYZT453NSDXZZSCQGHJ | Passed | 2028.12 | New-image refreshed result; still well above the 928.45 H100 baseline, but lower than the older TP=8 32k pass at 2271.36 TPS. |
| Olmo 3 32B, TP=8, 32k generation, 8 GPUs, eager | https://beaker.org/ex/01KVEFWGY7PCQZ84TP58436AV4 | Passed | 517.16 | Retry with longer batch/completion timeouts; this is the slowest refreshed 32k 8-GPU cell. |

## RLVR Training Throughput Notes

The current 28-node Holmes Think RLVR run is `https://beaker.org/ex/01KVCM995YVBCGT1E2DG4P8K81` with W&B run `l534x7sh`. The closest older Jupiter comparison we checked is W&B run `29h723j6`.

Key known differences between the current Holmes run and the older Jupiter comparison:

- Trainer backend changed from the older `grpo_fast.py` / DeepSpeed path to `grpo.py` / OLMo-core FSDP.
- Hardware changed from H100/Jupiter to B300/Holmes.
- The current Holmes run does not use `--vllm_enforce_eager`; the old Jupiter run logged `vllm_enforce_eager=true`.
- The logged generator/learner allocation may also differ: the old W&B config reports 20 TP=8 vLLM engines and 8 learner nodes, while the current Holmes launch command uses 6 TP=8 vLLM engines and 12 learner nodes. Treat cross-run attribution carefully.

| Run | Step | Actor TPS | Learner TPS step | Learner TPS overall | Step tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Holmes current | 1 | 6849.73 | 2868.88 | 2868.88 | 1604666 |
| Holmes current | 2 | 9217.38 | 12258.52 | 5467.63 | 2623906 |
| Holmes current | 3 | 7249.14 | 4580.76 | 4935.42 | 5316105 |
| Holmes current | 4 | 5556.19 | 7243.51 | 5529.69 | 4857399 |
| Holmes current | 5 | 5489.14 | 93239.59 | 7393.43 | 5272167 |
| Holmes current | 6 | 2557.60 | 59847.29 | 8238.98 | 2609228 |
| Holmes current | 7 | 2214.55 | 6735.54 | 7907.21 | 5158289 |
| Holmes current | 8 | 1598.39 | 93508.63 | 9230.18 | 5094164 |
| Holmes current | 9 | 1709.39 | 10781.50 | 9419.48 | 5281969 |
| Holmes current | 10 | 1549.26 | 103436.27 | 10597.54 | 5269677 |
| Holmes current | 11 | 1459.42 | 8420.95 | 10306.97 | 5274879 |
| Old Jupiter | 1 | 5039.49 | 3912.87 | 3912.87 | 3516883 |
| Old Jupiter | 2 | 4792.07 | 11389.63 | 6787.86 | 6523532 |
| Old Jupiter final summary | 1685 | 3829.55 | 9206.84 | 10813.68 | 9786193 |

Current estimates across Holmes Think RLVR steps 1-11: token-weighted actor/generator throughput is about 2451.31 TPS, learner overall throughput is 10306.97 TPS, median actor TPS is about 2557.60, and median learner step TPS is about 10781.50. Final pre-stop W&B summary at step 14: actor TPS 1026.55, learner step TPS 12392.57, learner overall TPS 11098.88, step tokens 5201326, mean sequence length 10536.95, max sequence length 32768. The actor/generator throughput was strongest in early steps and dropped sharply once step token counts and long sequence lengths stabilized; the learner estimate is now roughly comparable to the old Jupiter final summary.

## RL0 Debug Throughput Notes

The 9-node Holmes RL0 debug run is `https://beaker.org/ex/01KVCQ2HWSDPNMQT2SPZWN28RV` with W&B run `46vnaj5i`.

| Run | Step | Actor TPS | Learner TPS step | Learner TPS overall | Step tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| RL0 debug | 1 | 5648.10 | 1345.38 | 1345.38 | 364982 |
| RL0 debug | 2 | 5170.93 | 49360.90 | 3075.72 | 500608 |
| RL0 debug | 3 | 4848.01 | 39197.05 | 4570.11 | 476068 |
| RL0 debug | 4 | 5037.03 | 21126.23 | 5704.76 | 456327 |
| RL0 debug | 5 | 4580.91 | 24223.99 | 7668.76 | 905730 |
| RL0 debug | 6 | 4756.91 | 23635.82 | 9195.68 | 881158 |

Current estimates across RL0 debug steps 1-6: token-weighted actor/generator throughput is about 4889.48 TPS, learner overall throughput is 9195.68 TPS, median actor TPS is about 4942.52, and median learner step TPS is about 23929.91. Final pre-stop W&B summary at step 30: actor TPS 5044.43, learner step TPS 40363.08, learner overall TPS 30022.55, step tokens 1782897, mean sequence length 7200.03, max sequence length 16384. This small run is healthy enough to use for the large RL0 launch planning; its learner/generator balance is still based on the original RL0 4-learner-node / 5-generator-node split.


## DeepSpeed 28-Node Think RLVR Retry Notes

These runs retry the 28-node Think RLVR shape with the DeepSpeed backend. The non-eager run is `https://beaker.org/ex/01KVDYZZ5ENSBA57S2DYGBTC69` with W&B run `1jl40yhi`; the eager run is `https://beaker.org/ex/01KVDZ0HQ1MDRTKJKMC2TYW392` with W&B run `3vfsddan`. Both jobs were still running at the time these metrics were recorded. W&B was lagging behind Beaker logs, so the table below is taken from Beaker leader log summaries.

| Run | Logged steps | Average actor TPS | Average learner step TPS | Latest learner overall TPS | Short summary |
| --- | ---: | ---: | ---: | ---: | --- |
| DeepSpeed non-eager | 1-3 | 9373.32 | 276805.92 | 122277.09 | Actor/generator throughput is much higher than the eager retry so far. Step 4 had started, but no step-4 metric summary had flushed yet. |
| DeepSpeed eager | 1-4 | 2245.94 | 414033.86 | 183859.17 | Learner step TPS is higher than non-eager on the logged rows, but actor/generator throughput is much lower. |

Per-step TPS:

| Run | Step | Actor TPS | Learner step TPS | Learner overall TPS |
| --- | ---: | ---: | ---: | ---: |
| DeepSpeed non-eager | 1 | 7647.75 | 172185.84 | 0.00 |
| DeepSpeed non-eager | 2 | 10506.35 | 292953.12 | 60349.70 |
| DeepSpeed non-eager | 3 | 9965.87 | 365278.81 | 122277.09 |
| DeepSpeed eager | 1 | 1988.78 | 111394.52 | 0.00 |
| DeepSpeed eager | 2 | 2414.77 | 530648.90 | 67831.87 |
| DeepSpeed eager | 3 | 3096.56 | 444569.39 | 120041.39 |
| DeepSpeed eager | 4 | 1483.65 | 569522.63 | 183859.17 |


## Current Conclusions

- Refreshed inference grid status: all updated Holmes/B300 inference cells have now passed after retrying the failed cells with longer vLLM and benchmark batch timeouts.
- On the refreshed image, TP=1 8k single-GPU non-eager reached 421.62 TPS, or 1.40x the 300.50 H100 baseline; TP=1 eager reached 281.86 TPS, or 0.94x H100.
- On the refreshed image for 32k generation, TP=4 non-eager remains the best 8-GPU setting at 2669.13 TPS. TP=8 non-eager reached 2028.12 TPS, or 2.18x the 928.45 H100 TP=8 32k baseline.
- Refreshed eager 32k remains much slower: TP=4 eager reached 961.60 TPS and TP=8 eager reached 517.16 TPS.
- Historical pre-refresh note: the comparable Olmo 3 32B TP=1 8k single-GPU non-eager run passed at 419.68 TPS, which is 1.40x the documented H100 single-GPU baseline. The eager version passed at 216.36 TPS, only 0.72x H100.
- The Olmo 3 32B TP=1 8k 8-GPU diagnostic passed at 3383.68 TPS non-eager and 1765.72 TPS eager, but this should not be used as the primary comparison for the documented TP=1 single-GPU baseline row.
- Olmo 3 32B TP=4 at 8k generation passed at 3588.51 TPS, which is 2.75x the documented H100 baseline and above the 2.4x target. The matching eager run was much slower at 612.39 TPS.
- Olmo 3 32B TP=8 at 32k generation passed at 2271.36 TPS with the corrected `mp` executor path, which is 2.45x the documented H100 baseline and above the 2.4x target.
- For maximizing throughput on 8 GPUs, TP=4 with two TP=4 engines is the best completed setting so far. At 32k it reached 2641.33 TPS, compared with 2271.36 TPS for TP=8 with one TP=8 engine. The reported TPS is aggregate across both TP=4 engines, not per engine.
- The Olmo 3 7B TP=1 8k run passed at 432.68 TPS, but the handoff does not include a matching H100 7B baseline.
- The previous TP=8 failures were caused by vLLM startup configuration, especially forcing the Ray executor and triggering a nested Ray placement-group request for 8 GPUs. Removing the Ray executor override lets TP>1 use the local `mp` executor path, which produced the clean TP=8 32k pass.
