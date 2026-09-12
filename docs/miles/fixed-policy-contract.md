# Fixed-input Core/Megatron policy contract

`tests/miles/core_policy_contract.py` captures one real Core actor update on the immutable tiny KDA/latent-MoE fixture produced by `olmo_miles.evaluation.policy_contract`. The comparison checkout's `src` directory must be on `PYTHONPATH`: both arms import the same fixture schema and clipping/Adam validator. Fixture and checkpoint hashes are checked before model construction.

Inside the qualified Core runtime, with one available GPU:

```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
  --master_addr=127.0.0.1 --master_port=29513 \
  tests/miles/core_policy_contract.py \
  /path/to/megatron-policy-contract-02 /path/to/new-core-capture
```

Use a new output directory. The fixture determines token IDs, response lengths, binary masks, per-token advantages, old log probabilities, reduction, ratio clipping, and Adam settings. It deliberately stresses clipping. Auxiliary/z losses, dropout, replay, KL, advantage normalization, and loss recomputation are disabled. Core's normal current-policy scoring pass still runs; the policy loss explicitly uses the fixture's rollout log probabilities as its old-policy anchor.

Only the rollout provider and advantage producer are replaced by fixture data. The actual actor, policy loss, model forward/backward, gradient reduction, clipping, Adam optimizer, and model/master copies run normally. Instrumentation asserts the exact fixture tokens, masks, advantages, old probabilities and sample order at the loss boundary. Production response log probabilities are also checked against explicit `logits[t-1, target_t]` extraction. No online reward or advantage generation is being qualified.

`capture.pt` contains canonical HF tensor names for initial/final model weights, initial/final FP32 masters, gradients immediately before/after clipping, and first/second Adam moments. Core's native layout converter splits fused QKV and expert slabs without casting FP32 gradient or optimizer evidence to serving precision. The independently captured production optimizer norm checks the full canonical gradient norm. The shared validator checks clipping, first-step Adam equations, finite complete tensor inventory, dtypes, and exact master-to-model copies. `report.json` records consumed samples, shifted token scores, configuration and source hashes.

Compare both gradients and **updates relative to the initial weights**, including category-wise errors. Parameter-relative error alone can hide a large update discrepancy, and strong clipping can hide uniform preclip gradient scaling. A successful per-arm optimizer validation does not establish cross-backend numerical parity. This first experiment covers EP1/world1, one fixed update and no replay; EP/DP topology, resume and learning remain separate qualifications.

The first actual Core run passed on the local RTX4090 with fixture02, matching the fixture hash used by Megatron. All 56 canonical tensors passed strict clipping/moment checks (maximum absolute error zero); the Adam master check differed by at most 7.45e-9. The independently reconstructed preclip norm was 68.82677088 versus production 68.82676697, and the policy-only router gradient norm was 0.00208595. Eight focused CPU tests also passed. Evidence and source hashes are in `measurements/core-fixed-policy-20260910.json`; raw arrays remain in the recorded local artifact directory. Cross-backend deltas are a separate descriptive comparison.

The matched backend captures start from exactly identical BF16 weights and FP32 masters. The descriptive [comparison](measurements/backend-fixed-policy-20260910.json) finds preclip gradient relative L2 delta 1.132% with cosine 0.999937, FP32 master-update delta 7.022% with cosine 0.997534, and BF16 model-update delta 8.367%. Both backends independently satisfy strict clipping and Adam equations. This is a tiny single-update policy-only result; auxiliary semantics, EP2 and full-model training remain separate.
