# vLLM Hybrid Model Dtype Serialization Bug

## Summary

vLLM's v1 engine crashes when serving `allenai/Olmo-Hybrid-Instruct-DPO-7B` (and likely any `OlmoHybridForCausalLM` model) due to a type annotation bug in `MambaSpec.dtypes`. The Mamba/SSM cache spec sends 2 dtypes (one per state tensor), but the field is typed as `tuple[torch.dtype]` (exactly 1 element), causing `msgspec` serialization to fail.

## Error

```
msgspec.ValidationError: Expected `array` of length 1, got 2 - at `$.dtypes`
```

Full traceback:
```
AsyncLLM output_handler failed.
  File "vllm/v1/engine/async_llm.py", line 663, in output_handler
    outputs = await engine_core.get_output_async()
  File "vllm/v1/engine/core_client.py", line 1022, in get_output_async
    raise self._format_exception(outputs) from None
  ...
  File "vllm/v1/serial_utils.py", line 342, in _convert_result
    return msgspec.convert(result, result_type, dec_hook=self.dec_hook)
msgspec.ValidationError: Expected `array` of length 1, got 2 - at `$.dtypes`
```

## Root Cause

In `vllm/v1/kv_cache_interface.py`, the `MambaSpec` dataclass has:

```python
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]       # <-- BUG: fixed-length tuple (exactly 1 element)
```

The `dtypes` field is `tuple[torch.dtype]`, which in Python's type system means a tuple of exactly one `torch.dtype`. For standard Mamba models with a single state tensor, this works. But `OlmoHybridForCausalLM` has Mamba layers with 2 state tensors of different dtypes, so the model returns `dtypes=(torch.float32, torch.bfloat16)` (length 2).

When vLLM's v1 engine serializes the `EngineCoreOutputs` via `msgspec`, the strict type checking rejects the length-2 tuple.

## Fix

Change line 276 to use a variable-length tuple:

```python
dtypes: tuple[torch.dtype, ...]   # variable-length tuple
```

This matches `shapes` on the line above which already uses `tuple[tuple[int, ...], ...]`.

## Affected Versions

- **vLLM 0.18.0** (PyPI release): affected
- **vLLM main** (as of 2026-03-24): affected
- **yanhong-lbh/vllm@3677c274d** (custom fork): affected

The bug exists in all versions because `MambaSpec` was written for single-dtype Mamba models and was never updated for hybrid architectures.

## Affected Models

Any model with `model_type: olmo_hybrid` that has Mamba layers with multiple state tensors:
- `allenai/Olmo-Hybrid-Instruct-DPO-7B`

## Minimal Reproduction

See `scripts/debug/repro_vllm_hybrid_dtype.py` — a standalone script that reproduces the error with just vLLM and the hybrid model.

## Workarounds

1. **Patch vLLM at install time** (in Dockerfile):
   ```bash
   sed -i 's/dtypes: tuple\[torch\.dtype\]/dtypes: tuple[torch.dtype, ...]/' \
       .venv/lib/python3.12/site-packages/vllm/v1/kv_cache_interface.py
   ```

2. **Use vLLM's v0 engine** (if available) which doesn't use msgspec serialization.

## Environment

- torch >= 2.10.0
- transformers >= 5.3.0 (first version with upstream `olmo_hybrid` support)
- vllm 0.18.0 or custom fork at 3677c274d
