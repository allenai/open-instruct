# Router precision in the completed and current GSM8K runs

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

The specialized Core trainer preserves the intended contract: **BF16 router parameter storage and FP32 routing computation**. A missing Core upcast does not explain the observed learning difference. This conclusion concerns the actual trainer path; it does not establish that serving and training choose identical experts for every token.

## Frozen implementations

| Arm | Core or adapter revision | Router storage | Actual gating computation |
|---|---|---|---|
| Core100 | Core `b7f1e5296704779e7deb06de4c4242be9729d1a9` | BF16 | FP32 |
| Core500 | Core `290d2ca4521373bef0bf7fe4244673cc79dcc004` | BF16 | FP32 |
| Repaired Megatron100/500 | olmo-megatron `ba5615df741a24ba4aee678d0e853306b3502282` | BF16 | FP32 |

Core's MoE factory requests BF16 model storage. `olmo_core/nn/moe/v2/router.py::get_expert_logits` explicitly casts input and router weight to FP32 before `F.linear`. Router logits, softmax/top-k probabilities, normalized selected weights, and auxiliary scalar computations remain FP32. `OLMoDDPTrainModule` rejects a non-null `autocast_precision`; its model-forward context deliberately uses no autocast. Both frozen Core revisions have this contract.

The repaired Megatron provider sets `moe_router_dtype="fp32"` and `moe_router_use_torch_mm=True`. Actual `Router.gating` calls `RouterGatingLinearFunction`, which casts BF16 inputs and stored weights to FP32 before matrix multiplication; routing and auxiliary computations operate on FP32 logits. Its backward computes in FP32 and casts parameter gradients to the parameter dtype; subsequent gradient accumulation is separately configured in FP32. The repaired constructor stores router weights in BF16. The removed older patch had forced FP32 parameter storage, which violated the checkpoint/import contract.

This was already the intent in olmo-miles history: `083fae1` (“Match OLMo Core router precision contract”), followed by the HF autocast checks in `acd43e3`, the B300 gate in `1cfe2f1`, and promotion in `db867df`. The r3 repair `7d56b23` restored the adapter's constructor and compute checks. It aligned the Megatron/HF representations with Core's existing routing contract.

## Direct numerical check

A local RTX 4090 probe used seed 17, fixed BF16 activations of shape `[1, 256, 640]`, BF16 weights `[512, 640]`, 16 selected experts, and TF32 disabled. It invoked the actual Core router and actual Megatron gating implementation in each of the frozen Core100, Core500, and repaired Megatron images. This is a bounded operator check, not a full-checkpoint equivalence test.

With the actual specialized trainer's no-autocast mode, every image produced logits **exactly equal to the explicit FP32 reference**: maximum error 0, no changed selected expert sets. Core's selected probabilities and auxiliary scalar were FP32. Raw records are in [the evidence directory](router-precision-20260911/core100.json), [Core500](router-precision-20260911/core500.json), and [repaired Megatron](router-precision-20260911/megatron-repaired.json).

An intentionally imposed outer BF16 autocast was a negative control. It rounded the matrix-multiplication output despite FP32 input casts, producing maximum logit error `0.008053302764892578` and changed expert sets for 28 of 256 tokens. This happened in both implementations. **That outer autocast is not the specialized Core trainer configuration.** It explains why an HF implementation run under autocast needs its explicit local autocast exclusion; it is not evidence that Core100 ran with low-precision gating.

Initial publication's full weight comparison checks stored tensors, not activation or expert-assignment equivalence. The next zero-update diagnostic therefore captures actual SGLang activations and routing before and after each backend's initial publication. No active learning configuration or precision policy was changed for this audit.
