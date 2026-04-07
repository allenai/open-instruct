# Manufactoria training scripts

Run from the **repository root** (so `mason.py`, `configs/`, and `open_instruct/` resolve).

- `qwen3_4b_phase1_has_8gpu.sh` — Beaker GRPO launch with Manufactoria API sidecar.
- `qwen3_4b_has_phase2.sh` — Continuation run; delegates to phase 1 with a new base checkpoint.
- `debug/2gpu_grpo_fast_manufactoria.sh` — Local API + short GRPO smoke test.

Example:

```bash
./scripts/train/build_image_and_launch.sh examples/manufactoria/scripts/qwen3_4b_phase1_has_8gpu.sh
```
