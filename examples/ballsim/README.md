# Ballsim (BounceSim) example

Verifier and API live in this package (`verifier.py`, `api.py`, `register.py`).

## CLI flags

### Recommended: `examples.grpo_fast`

Run **`python -m examples.grpo_fast`** from the repo root. It uses **`ExamplesGRPOStreamingConfig`** (`examples/grpo_streaming_config.py`): **`StreamingDataLoaderConfig`** with **`ballsim_*`** and **`manufactoria_*`** fields, so **`--ballsim_*`** are typed and listed in **`--help`**. The entrypoint imports the example register modules.

### Alternative: stock `open_instruct.grpo_fast`

Pass **`--ballsim_*`** as trailing argv; they are parsed by `parse_extra_verifier_cli_args` in `open_instruct/ground_truth_utils.py` and do **not** appear in stock **`--help`**. Import **`examples.ballsim.register`** in the same process before training (e.g. `python -c "import examples.ballsim.register" && python -m open_instruct.grpo_fast ...`), or use **`python -m examples.grpo_fast`**.

Defaults when flags are omitted are defined in `examples/ballsim/verifier.py` (env-based).

## Training scripts

See [`scripts/`](scripts/) (run from repo root):

- `qwen3_4b_instruct_ballsim.sh` — Beaker GRPO (expects `DATASETS`, `ballsim_api_setup`, etc.).
- `debug/grpo_fast_ballsim.sh` — Local API + GRPO.
- `debug/2gpu_grpo_fast_ballsim.sh` — Multi-GPU style run with cluster API setup.
