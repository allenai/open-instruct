"""Patch ring-flash-attn 0.1.8 to be importable under transformers>=5.x.

ring_flash_attn/__init__.py unconditionally imports its `adapters` subpackage,
which in turn imports `is_flash_attn_greater_or_equal_2_10` from
`transformers.modeling_flash_attention_utils`. That symbol was removed in
transformers 5.x, so `import ring_flash_attn` raises ImportError, which makes
olmo-core's context-parallel codepath (ring CP) unavailable.

olmo-core only uses the top-level ring kernels, not the HF adapter, so we wrap
the adapters import in try/except.
"""

import importlib.util
import pathlib

spec = importlib.util.find_spec("ring_flash_attn")
if spec is None or spec.origin is None:
    raise SystemExit("ring_flash_attn not installed")

init_path = pathlib.Path(spec.origin)
src = init_path.read_text()

needle = "from .adapters import (\n    substitute_hf_flash_attn,\n    update_ring_flash_attn_params,\n)\n"
if needle not in src:
    raise SystemExit(f"Expected adapters import block not found in {init_path}")

replacement = (
    "try:\n"
    "    from .adapters import (\n"
    "        substitute_hf_flash_attn,\n"
    "        update_ring_flash_attn_params,\n"
    "    )\n"
    "except ImportError:\n"
    "    substitute_hf_flash_attn = None\n"
    "    update_ring_flash_attn_params = None\n"
)

init_path.write_text(src.replace(needle, replacement))
print(f"Patched {init_path}")
