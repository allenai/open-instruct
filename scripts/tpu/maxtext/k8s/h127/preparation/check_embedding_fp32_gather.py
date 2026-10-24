"""Actual pinned embedding lookup: exact forward and repeated-ID gradient oracle."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import embedding_fp32_gather_patch as repair
import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from maxtext.common.common_types import ShardMode

source = repair.SOURCE_PATH.read_bytes()
patched = repair.patch(source)
namespaces = [{"__name__": "h127_embed_original"}, {"__name__": "h127_embed_repaired"}]
for content, namespace in zip((source, patched), namespaces, strict=True):
    exec(compile(content, str(repair.SOURCE_PATH), "exec"), namespace)

mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",))
config = SimpleNamespace(
    parameter_memory_host_offload=False,
    shard_mode=ShardMode.AUTO,
    use_iota_embed=False,
    logical_axis_rules=(
        ("activation_embed_and_logits_batch", None),
        ("activation_length", None),
        ("activation_embed", None),
    ),
)
rng = np.random.default_rng(491)
table = jnp.asarray(rng.normal(size=(16, 32)).astype(np.float32))
ids = jnp.asarray(np.tile([0, 1, 2, 0, 1, 0, 3, 0], 128)[None])
cotangent = jnp.asarray(rng.normal(size=(1, 1024, 32)).astype(np.float32) * 100, dtype=jnp.bfloat16)
reference = np.zeros(table.shape, dtype=np.float64)
np.add.at(reference, np.asarray(ids).ravel(), np.asarray(cotangent, dtype=np.float64).reshape(-1, 32))
results = []
with jax.set_mesh(mesh):
    for namespace in namespaces:

        def lookup(weights, namespace=namespace):
            dummy = SimpleNamespace(
                config=config,
                cast_input_dtype=None,
                dtype=jnp.bfloat16,
                mesh=mesh,
                embedding=nnx.Param(weights),
                num_embeddings=16,
            )
            return namespace["Embed"].__call__(dummy, ids)

        values, pullback = jax.vjp(jax.jit(lookup), table)
        gradient = np.asarray(pullback(cotangent)[0], dtype=np.float64)
        results.append((np.asarray(values, dtype=np.float32), gradient))
assert np.array_equal(results[0][0], results[1][0])
errors = [float(np.linalg.norm(gradient - reference) / np.linalg.norm(reference)) for _, gradient in results]
assert errors[1] < 1e-6 and errors[1] < errors[0] / 100
result = {
    "scope": "CPU repeated-token gradient fixture using actual pinned Embed.__call__; not full-model or TPU qualification",
    "forward_exact": True,
    "original_gradient_relative_error": errors[0],
    "repaired_gradient_relative_error": errors[1],
    "source_sha256": hashlib.sha256(source).hexdigest(),
    "patched_source_sha256": hashlib.sha256(patched).hexdigest(),
}
print("EMBED-FP32-GATHER-CHECK", json.dumps(result), flush=True)
Path("/work/embedding-fp32-gather-local-check.json").write_text(json.dumps(result, indent=2) + "\n")
