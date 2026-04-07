# Manufactoria example (custom task + verifier)

This directory shows how to add a **custom verifiable task** to Open Instruct:

1. **Verifier** — Subclass `VerifierFunction` in `open_instruct/ground_truth_utils.py` and implement `async_call` / `__call__`. Register a matching `VerifierConfig` subclass with `from_args` that reads training args and environment variables.
2. **Dataset** — Use the usual RLVR columns; the `dataset` / verifier source column must match your verifier’s `name` (here: `manufactoria`).
3. **Registration** — Decorate your verifier with **`@register_verifier`** (see `open_instruct.ground_truth_utils.register_verifier`) so it is included in `build_all_verifiers` alongside subclass discovery. The verifier module must still be imported before training. **`python -m examples.grpo_fast`** imports the example register modules for you. For stock **`python -m open_instruct.grpo_fast`**, import your module first (e.g. `python -c "import examples.manufactoria.register"`), or pass **`extra_verifier_functions`** from Python (see `open_instruct.grpo_fast.main`).
4. **Optional API** — [`api.py`](api.py) is a FastAPI server that evaluates DSL programs; the verifier calls it over HTTP.

## Configuration (CLI flags)

### Recommended: `examples.grpo_fast`

Run **`python -m examples.grpo_fast`** (repo root). It uses **`ExamplesGRPOStreamingConfig`** (see [`examples/grpo_streaming_config.py`](../grpo_streaming_config.py)): core `StreamingDataLoaderConfig` plus `manufactoria_*` and `ballsim_*` fields, so those flags are **typed** and appear in **`--help`**—the same internal merge path as `code_*` on `streaming_config` (no separate `verifier_extra_sources`).

| Flag | Purpose |
|------|---------|
| `--manufactoria_api_url` | Full URL to `POST .../test_solution`. |
| `--manufactoria_max_execution_time` | Float (e.g. `1.0`). |
| `--manufactoria_scoring_mode` | `all_pass` or `pass_rate`. |

Defaults for omitted values still come from environment variables inside `ManufactoriaVerifierConfig` (see `examples/manufactoria/verifier.py`).

### Alternative: stock `open_instruct.grpo_fast`

Core `StreamingDataLoaderConfig` has no Manufactoria fields. You can pass **`--manufactoria_*`** as **trailing** arguments; they are parsed by `parse_extra_verifier_cli_args` in `open_instruct/ground_truth_utils.py`. Those flags **do not** appear in stock **`--help`**. You must still **register** the verifier class (subclass of `VerifierFunction`) before `build_all_verifiers` runs—for example import `examples.manufactoria.register` in the same process (see one-liner below).

## Local API

```bash
uv run uvicorn examples.manufactoria.api:app --host 0.0.0.0 --port 1235
```

## Training (minimal)

**Typed help (recommended):**

```bash
uv run python -m examples.grpo_fast \
  --manufactoria_api_url http://localhost:1235/test_solution \
  --manufactoria_scoring_mode pass_rate \
  # ... other core GRPO args (dataset, model, etc.) ...
```

**Stock `grpo_fast` with trailing Manufactoria flags** — ensure the register module is imported first, e.g.:

```bash
uv run python -c "import examples.manufactoria.register" && uv run python -m open_instruct.grpo_fast \
  --manufactoria_api_url http://localhost:1235/test_solution \
  --manufactoria_scoring_mode pass_rate \
  # ... other core GRPO args ...
```

See also [`scripts/`](scripts/) for Beaker launch scripts (`qwen3_4b_phase1_has_8gpu.sh`, phase 2, and `debug/` smoke runs). Run them from the repo root so paths like `configs/` and `mason.py` resolve.
