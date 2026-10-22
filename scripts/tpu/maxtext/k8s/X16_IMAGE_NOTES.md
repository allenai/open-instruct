# Notes on the drop-count image used by the -x16 runs (tpu-worker, 2026-10-22)

Image `maxtext-posttrain:51f8cc4b-p907d86f3-rfb46e215e-bs3ec936c3-dc88c6c7ea`. Fix both items in the next build.

1. **The MOE-DROPPED line never prints.** With `log_moe_dropped_tokens=true`, the MoE layer packs
   [required factor, dropped rows, routed rows] per layer into `moe_required_rbf`. The trainer's branch that unpacks
   them and prints `MOE-DROPPED step=... dropped=... routed=...` (dropcount.patch, train_sft.py) is not taken.
   - The likely cause is that the trainer reads `log_moe_dropped_tokens` from a config object that lacks it. Check
     `getattr(self.config, ...)` in `MaxTextPeftTrainer`.
   - So `REQUIRED_RBF ... max=` reports the routed count (16,777,216 = 16 x 65,536 x top-16), not a factor, and
     `per_layer` holds the triples.
   - Until it is fixed, read drops from the triples. `/root/tpu-watch/x16prod/xw.py` on Hammond does this. Never use a
     check that reads `max=` on this image.
2. **moe_lb depends on the placement.** `load_balance_loss` runs inside the expert shard_map. Under
   `olmo35-ep-as-cp` each shard holds a 65,536/ep slice of each row, so the "sequence-wise" density and probability
   are per slice, with no padding mask.
   - The value, and its gradient at `load_balance_loss_weight` 0.01, change with ep: steps 1-6 at ep64 against ep8
     differ by up to 0.03 on about 0.23, and by 0.765 against 0.232 at step 2.
   - Fixing this needs either a whole-row reduction or a padding mask. Changing it is a recipe change, so it needs a
     registration.
