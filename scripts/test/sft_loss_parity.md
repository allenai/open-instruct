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
gradient comparison. A separate doubled-gradient ZeRO-2/BF16 control must also
fail before clipping; an unrelated launch failure does not count as sensitivity.
Both ZeRO stages run with BF16, SP on/off and fused loss on/off. Checkpoint
continuation covers both stages without SP as well as the SP refill cases.
All 38 positive invocations must emit their pass marker. Source hashes, package
versions and per-window errors are persisted. No skipped test counts as validation.

This checks numerical correctness and checkpoint continuation, not historical
Tulu 3 recipe reproduction or downstream quality. It is separate from the full
GPU pytest suite and must not be cited as `GPU_TESTS` evidence.

## Diagnosing BF16 differences

The first GPU attempt passed six non-SP cases and rejected both negative controls,
then failed ZeRO-2/SP=2 BF16 at relative L2 0.003247 and max absolute 0.00390625.
The original limits above remain unchanged for diagnosis. Run only FP32 ZeRO-2/3
SP controls and the failing BF16 case by appending `diagnostic` to the launch
command. Diagnostic mode has a 15-minute timeout and a separate success marker;
it cannot establish full-matrix parity. Failures persist the actual/reference
raw gradients, norms, maximum-error coordinates and BF16 spacing for analysis.

The initial run also exposed a real loop bug: the Ulysses adapter resets its
underlying loader when it raises `StopIteration`. After a partial window, the
buffering helper called it again and silently started another epoch. The helper
now latches exhaustion; a CPU regression executes the installed adapter across
two epochs with only its single-rank collective transport replaced. The first
focused run was stopped during startup to include this correction.

The corrected full matrix retains the original precision limits. Tight FP32
checks cover SP, fused loss, and the same five save/reload configurations as
BF16. A per-update epoch-step assertion fails immediately if iteration restarts.
