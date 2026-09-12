# Native auxiliary and policy router gradients: tiny EP1 qualification

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

Both native trainers completed three backward passes on the same unequal-length tiny fixture: policy only, auxiliary only, and combined. Capture occurs at finalized optimizer input before clipping or stepping. Model parameters/buffers, persistent optimizer tensors and scheduler state remained unchanged; optimizer calls were zero. Both final-source executions exited 0. [Exact results and provenance](router-gradient-decomposition-20260911.json).

| Logical HF router 1 | Core | Megatron |
|---|---:|---:|
| Policy gradient L2 | 0.00208595 | 0.00208407 |
| Auxiliary gradient L2 | 0.00411471 | 0.00703956 |
| Combined gradient L2 | 0.00491318 | 0.00745142 |
| Policy/auxiliary cosine | 0.16609 | 0.05508 |
| Combined minus component sum, relative L2 | 0.001880 | 0.001339 |

The auxiliary gradient is larger than the policy gradient in both fixtures and about 1.71 times as large in Megatron. Native padding and normalization are preserved. This supports measuring auxiliary behavior on real batches; it does not attribute the learning gap or establish relative cumulative router drift. Norm agreement alone says nothing about cross-backend direction agreement. The small superposition residual is measured, not rounded away.

Scope: EP1 only, small model, unit loss scale, fixed serving policy anchors (`use_rollout_logprobs=True`), auxiliary balancing coefficient 0.01 and z coefficient 1e-5. The original 100-update runs used a different anchor mode. The diagnostic rejects EP2 until ownership and gradient reduction are qualified. Core logical block 1 corresponds to Megatron physical MLP layer 3; interpreting those raw indices as different layers would be wrong.

Implementation: `scripts/miles/update_zero_gradient_capture.py`; actual native smoke: `tests/miles/gradient_decomposition_smoke.py`. Six focused tests cover decomposition and state/optimizer guards. This diagnostic is separate from the queued full-checkpoint forward-only routing probes.

Cross-backend direction was subsequently checked on the saved gradient tensors after asserting identical initial router parameter SHA256s. Policy-gradient cosine is 0.9994408 (relative L2 difference 3.344% against Core); auxiliary-gradient cosine is 0.6180197; combined-gradient cosine is 0.6683437. Thus this fixture's auxiliary discrepancy includes direction, not just scale. These remain one tiny EP1 batch's measurements and do not establish accumulated optimizer behavior. Raw tensor hashes are in the companion JSON.
