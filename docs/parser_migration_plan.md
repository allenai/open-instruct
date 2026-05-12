# Migrating off `HfArgumentParser` / `ArgumentParserPlus`

## Why

`transformers.HfArgumentParser` declares its inputs/outputs via:

```python
DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)
```

`NewType("X", Any)` is invalid (ty raises `invalid-newtype`). Every call site therefore needs a `cast` or a `# ty: ignore` for the constructor call and for the returned dataclass to be usable. This is the largest single source of friction blocking files from being type-checked (e.g. `dpo.py`, `dpo_tune_cache.py`, `finetune.py`, `grpo_fast.py`, `reward_modeling.py`, `utils.py`, all `rejection_sampling/*`, `sample_logits_vllm.py`).

## Current API surface

`open_instruct.utils.ArgumentParserPlus` extends `HfArgumentParser` and adds:

1. **CLI only** — `parse_args_into_dataclasses()` (inherited).
2. **YAML only** — single arg `script.py config.yaml`.
3. **YAML + CLI overrides** — `script.py config.yaml --foo=bar`.
4. **Auto-unwrap** — returns a single dataclass if only one was passed, else a tuple.

Call sites pass either a single dataclass (most common) or a tuple of nested dataclasses (e.g. `(Args, TokenizerConfig)`).

## Candidate replacements

| Library | Nested dataclasses | YAML config | Type-checks cleanly | Notes |
|---|---|---|---|---|
| `tyro` | ✅ | ❌ (need adapter) | ✅ | Most ergonomic; `tyro.cli(Args)` |
| `simple-parsing` | ✅ | Partial (via plugin) | ✅ | Closest 1:1 swap for HfArgumentParser |
| `jsonargparse` | ✅ | ✅ (native) | ✅ | Best YAML story; supports `--config`, env vars, subclasses |
| stdlib `argparse` | ❌ (manual) | ❌ | ✅ | Tedious; ~30-50 LOC per dataclass |

**Recommended: `jsonargparse`** — it's the only option that preserves the YAML-config feature without a custom adapter, and its types are clean. Second choice: `tyro` + a small YAML loader.

## Migration steps

1. **Add dependency** in `pyproject.toml`: `jsonargparse[signatures]>=4.30`.
2. **Replace `ArgumentParserPlus` in `open_instruct/utils.py`** with a thin wrapper:
   ```python
   def parse_args(cls: type[T] | tuple[type, ...]) -> T | tuple:
       parser = jsonargparse.ArgumentParser()
       # add_dataclass_arguments per cls; configure --config for YAML
       ...
   ```
   Keep the function name `ArgumentParserPlus` or expose `parse_args` and update callers.
3. **Update call sites** (~20 files). Pattern is uniform:
   ```python
   # before
   parser = ArgumentParserPlus((Args, TokenizerConfig))
   args, tc = parser.parse()
   # after
   args, tc = parse_args((Args, TokenizerConfig))
   ```
4. **Remove `cast` workarounds** in `synthetic_preference_dataset.py` and similar files.
5. **Remove `# ty: ignore[invalid-argument-type]` and `[unknown-argument]`** comments on parser call sites.
6. **Remove files from `[tool.ty.src].exclude`** in `pyproject.toml` that were excluded purely due to parser errors. Re-run ty and verify.
7. **Re-test YAML configs** — every training script invocation in `scripts/train/debug/*.sh` and `configs/train_configs/*.yaml` should still work. Critical: verify CLI overrides on top of YAML still parse the same way (`script.py config.yaml --learning_rate=1e-5`).

## Risk

- Behaviour parity around bool/list parsing — `ArgumentParserPlus.parse_yaml_and_args` has hand-rolled type coercion (str→int/float/bool/list). `jsonargparse` does this via dataclass field types directly; subtle differences possible (e.g. `--foo=true` vs `--foo true`).
- `parse_args_into_dataclasses(return_remaining_strings=True)` callers (if any) — search before migrating.

## Files touched (estimate)

- `open_instruct/utils.py` (define new wrapper, delete old class)
- ~20 call sites under `open_instruct/` and `scripts/`
- `pyproject.toml` (deps + reduced exclude list)
