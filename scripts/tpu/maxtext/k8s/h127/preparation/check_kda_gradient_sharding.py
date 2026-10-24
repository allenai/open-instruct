"""Small CPU structural check; not a TPU parity gate or full-model test."""

import json
from pathlib import Path

import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from jax import sharding
from maxtext.models import olmoe3

mesh = sharding.Mesh(np.array(jax.devices()), ("split",))
rng = np.random.default_rng(731)
shape = (1, 128, 4, 16)
q = rng.normal(size=shape).astype("float32") * 0.1
k = rng.normal(size=shape).astype("float32") * 0.1
v = rng.normal(size=(1, 128, 4, 24)).astype("float32") * 0.2
ld = -rng.uniform(0.001, 0.03, size=shape).astype("float32")
beta = rng.uniform(0.1, 0.9, size=shape[:-1]).astype("float32")
resets = np.zeros((1, 128), dtype=bool)
resets[:, [0, 37, 91]] = True
q[:, 0] = 0
k[:, 0] = 0
v[:, 0] = 0
arrays = (q, k, v, ld, beta)
weight = jnp.asarray(rng.normal(size=v.shape).astype("float32"))

with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
    results = {}
    for mode in ["head", "value", "unsharded_scan"]:
        rules = (
            ("activation_batch", None),
            ("activation_kda_heads", "split" if mode == "head" else None),
            ("activation_kda_value", "split" if mode == "value" else None),
        )

        def loss(*args):
            q, k, v, ld, b = args
            if mode == "unsharded_scan":
                out = olmoe3._delta_rule_scan(q, k, v, jnp.exp(ld), b, jnp.asarray(resets))
            else:
                with nn.logical_axis_rules(rules):
                    out = olmoe3._delta_rule_chunked_subblock(q, k, v, ld, b, jnp.asarray(resets), 64, "float32", 16)
            return jnp.sum(out * weight)

        specs = (
            [sharding.PartitionSpec(None, None, "split", None)] * 4 + [sharding.PartitionSpec(None, None, "split")]
            if mode == "head"
            else [sharding.PartitionSpec()] * 5
        )
        if mode == "value":
            specs[2] = sharding.PartitionSpec(None, None, None, "split")
        placed = tuple(jax.device_put(x, sharding.NamedSharding(mesh, s)) for x, s in zip(arrays, specs, strict=True))
        val, grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4)))(*placed)
        results[mode] = (float(val), [np.asarray(x) for x in grads])
    output = {
        "scope": "2 CPU devices, small synthetic KDA, FP32 state, document resets and zero first input; not TPU/full-model qualification",
        "losses": {k: v[0] for k, v in results.items()},
        "gradient_relative_l2": {},
    }
    for mode in ["head", "value"]:
        output["gradient_relative_l2"][mode] = {
            name: float(np.linalg.norm(a - b) / np.linalg.norm(b))
            for name, a, b in zip(
                ["q", "k", "v", "log_decay", "beta"], results[mode][1], results["unsharded_scan"][1], strict=True
            )
        }
    print("KDA-GRADIENT-CHECK", json.dumps(output), flush=True)
    Path("/work/numerical-investigation/kda-cpu-gradient-check.json").write_text(json.dumps(output, indent=2) + "\n")
