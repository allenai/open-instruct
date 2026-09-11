# Light SFT1000 GSM8K comparison

The historical anchor is [the successful 200-update run](https://beaker.org/ex/01M12Y23YZS5ZJWBK45CKQJ7DP), W&B [35ehrf3h](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/35ehrf3h). Its actual manifest records olmo-miles `99c10d2891b3ee6cc0c40b050ff946c177d3dd38`; the experiment name contains another revision and should not be used as source provenance. The earlier 100-update run's native evaluations returned empty model outputs and its attempted full-test evaluation failed, so it is not the quality anchor.

The checkpoint is **20-block latent-KDA, hidden 1280, latent 640, 512 experts/top16**, with 16 KDA and four full-attention blocks. It received 1000 SFT optimizer updates from latent-KDA midtrain step63802. The attached HF descriptor is `/weka/oe-training-default/robertb/olmo-miles/rl-sft1000/sft-v1/checkpoints/.olmo-miles/hf` (Megatron iteration999). This is neither the misleadingly named older 31-layer SFT1000 export nor the recent Dolci-think/hero checkpoint.

[The frozen historical descriptor](../configs/miles/reference/light-sft1000-gsm8k-historical.json) contains the full original comparison manifest, source/config/template/data digests, result IDs, serving settings recovered from the resolved server log, evaluation outcomes, and timing provenance. Preparation verifies historical config, template, native manifest/train/eval bytes, and frozen full-test requests. Safetensors file sizes, offset bounds and header digests are checked and retained; full weight-payload digests are **not** claimed. Source router tensor storage dtypes/shapes are retained. Core loads BF16, and current SGLang routers inherit the engine model dtype; both compute the router projection in FP32. Any original FP32 router values therefore need their BF16 import rounding recorded as an initial-policy difference. The initial all-tensor publication equality gate remains enabled. Missing or changed historical files fail preparation; there is no fallback to a similarly named checkpoint or a newly selected evaluation set.

| Setting | Historical Megatron | New Core arm |
|---|---|---|
| Updates, prompts/samples | 200, 8×4, 7473 prepared train rows | Same, exact row order |
| Training sampling | Seed1, temperature1, top-p1/top-k−1 | Same |
| Response/context | 512 response, 1024 rollout context | Same |
| Native eval | Exact128 official-test subset, temperature1, every10 including0 | Same inputs and schedule |
| Optimizer | LR1e−6 constant, betas .9/.95, eps1e−8, WD0, clip1 | Same values, Core optimizer implementation |
| GRPO | Response average, clip .2/.28, std normalization off, old LP rescored | Same contract |
| Auxiliary terms | .01 load balance, 1e−5 z, sequence weighting | Same coefficients; Core instance/token weighting differs |
| Trainer | EP2, dense DP2, BF16, recompute, FlashAttention4 | Core EP2, BF16, activation checkpointing, FlashAttention4 |
| Serving | Two TP1 replicas colocated on trainer's two B300s | Two TP1 replicas on two additional B300s; four total |
| SG sampling/cache | FlashInfer, radix extra_buffer, mamba52, admission8/client4 per engine | Explicitly same settings |
| SG capacity | Engine8192, resolved KV771285, chunk/prefill16384, static .18 | Explicitly same values; current allocator must accept them |
| Graphs | Decode full through8, prefill off | Same |
| Sync/replay | Synchronous refresh every update, 1GiB IPC, replay off | Synchronous flattened publication; disaggregated transport differs |
| Checkpoints | Async NVRx every50 | Core durable checkpoints every50, synchronous boundary |

Core's exact runtime lock is retained by the image/source build. Historical SGLang/MILES/FLA pins differ; numerical/scheduling differences and the distinct auxiliary reduction remain confounds. No exact backend-equivalence claim is made.

Two evaluation series stay separate:

* **Native curve:** 26/128 (20.31%) at step0, 77/128 (60.16%) at200, best83/128 (64.84%) at190. These use the native instruction template and sampling temperature1.
* **Full official test:** [before](https://beaker.org/ex/01M13A2NB29VNFFPG5GC82Z1FG) 230/1319 (17.44%) and [after](https://beaker.org/ex/01M13B0CFKKHRRGNKQFQ3GT8SG) 274/1319 (20.77%), +44 answers/+3.34 percentage points. Requests use raw `Question: {query}\nAnswer:`, no chat template, greedy temperature0, maximum512, and stops `Question:` / double-newline. Their exact archived requests and IDs are mounted from the historical result. The new evaluation extension uses MILES `/generate` against the same live weights, proving identical raw prompt token IDs; the historical harness used `/completions`. Stop handling and sampling are explicit, and an independent signed-last-number scorer checks every registered reward. Re-scoring all1319 retained historical after-predictions reproduced274 correct with zero per-question disagreements. Full responses, tokens, versions and cap flags are retained in `core/offline-{0,200}.json`. These additional before/after evaluations are timed separately and add work absent from the historical training job.

Historical job runtime was 3h20m48s on two allocated GPUs (~6.69 GPU-hours, excluding queue). Its stored steady measurement window is rollouts40–49: mean train19.40s, scoring3.16s, rollout5.97s, publication15.04s. Batch wall mean68.17s versus median45.61s includes one199.48s checkpoint and one24.10s evaluation. Future comparisons should report cold startup, matching warm windows, generation tokens, checkpoint/evaluation overhead and allocated GPU-hours separately. Four-GPU Core wall time alone is not a compute-efficiency comparison.

Preparation (CPU, Saturn, read-only historical sources; writes only the new campaign root):

```bash
MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  ./scripts/train/build_image_and_launch.sh --miles \
  scripts/train/debug/miles_light_sft_gsm8k.sh --stage prepare
```

After successful preparation, the same committed wrapper accepts `--stage core` (four B300s, urgent, eight-hour minimum/timeout) and `--stage audit` (Saturn). The campaign directory is `/weka/oe-training-default/robertb/open-instruct/light-sft1000-gsm8k/20260911-v1`. The driver refuses to overwrite an existing `core` arm. W&B uses entity `ai2-llm`, project `olmo-rl-comparison`, group `gsm8k-light-sft1000-core-20260911-v1-core`.
