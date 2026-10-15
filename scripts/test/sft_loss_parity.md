# Legacy SFT GPU loss parity

Launch from a clean, committed checkout:

```bash
./scripts/train/build_image_and_launch.sh --cuda-version 13 scripts/test/launch_sft_loss_parity.sh
```

The launcher requests one node with four B300 GPUs on Holmes, high priority,
a five-minute minimum runtime and a 45-minute timeout. It uses no model downloads,
external evaluations, WEKA mounts or credentials. Results go to the Beaker result dataset.

The tiny random Llama checks real Accelerate DDP, DeepSpeed ZeRO-2/3,
Ulysses SP=2/DP=2 and Liger fused linear cross-entropy. Twenty deterministic rows
have unequal supervised causal-token counts, including empty local batches/shards.
GAS=3 gives full and partial windows. Compare raw gradients before clipping and
SGD-with-momentum parameter deltas to a concatenated full-batch token-mean
reference with FP32 optimizer masters. Fresh processes reload checkpoints after
three SP microbatches, inside an adapter refill, and finish the same reference trajectory.

Limits fixed before GPU results: FP32 relative L2 error <= 1e-4 and maximum
absolute error <= 1e-5; BF16 <= 0.03 and <= 0.002 respectively. Both limits must
hold. The isolated old mean-of-microbatch-means negative control must fail the
gradient comparison; an unrelated launch failure does not count as sensitivity.
All 16 positive invocations must emit their pass marker. Source hashes, package
versions and per-window errors are persisted. No skipped test counts as validation.

This checks numerical correctness and checkpoint continuation, not historical
Tulu 3 recipe reproduction or downstream quality. It is separate from the full
GPU pytest suite and must not be cited as `GPU_TESTS` evidence.
