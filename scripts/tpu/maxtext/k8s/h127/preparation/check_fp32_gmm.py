"""Check the pinned FP32 grouped-matmul VJP in CPU interpretation, not TPU parity."""

import json
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp
from maxtext.kernels import megablox
from qwix import pallas as qpl


def main():
    assert jax.config.jax_default_matmul_precision == "highest"
    rng = np.random.default_rng(61024)
    lhs = jnp.asarray(rng.normal(size=(512, 128)), dtype=jnp.float32)
    rhs = jnp.asarray(rng.normal(size=(4, 128, 128)), dtype=jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=(512, 128)), dtype=jnp.float32)
    sizes = jnp.asarray([128, 256, 128, 0], dtype=jnp.int32)

    def kernel(a, b):
        return megablox.gmm(a, b, sizes, preferred_element_type=jnp.float32, tiling=(128,) * 9, interpret=True)

    def reference(a, b):
        return jnp.concatenate([a[:128] @ b[0], a[128:384] @ b[1], a[384:] @ b[2]])

    output, pullback = jax.vjp(kernel, lhs, rhs)
    expected, ref_pullback = jax.vjp(reference, lhs, rhs)
    actual_grads, expected_grads = pullback(cotangent), ref_pullback(cotangent)
    errors = {}
    for name, actual, ref in zip(
        ["forward", "lhs_gradient", "rhs_gradient"], [output, *actual_grads], [expected, *expected_grads]
    ):
        actual, ref = np.asarray(actual), np.asarray(ref)
        assert actual.dtype == ref.dtype == np.float32
        np.testing.assert_allclose(actual, ref, rtol=2e-4, atol=2e-4)
        errors[name] = float(np.linalg.norm(actual.astype(np.float64) - ref) / np.linalg.norm(ref.astype(np.float64)))

    # The backend calls qpl.dot_general without explicit precision. Verify the
    # process-wide setting reaches its unquantized lax.dot_general operation.
    def dot(a, b):
        return qpl.dot_general(a, b, (((1,), (0,)), ((), ())))

    graph = jax.make_jaxpr(dot)(lhs, rhs[0])
    dots = [eq for eq in graph.jaxpr.eqns if eq.primitive.name == "dot_general"]
    assert dots and all(eq.params["precision"] == (jax.lax.Precision.HIGHEST,) * 2 for eq in dots)
    result = {
        "passed": True,
        "relative_l2": errors,
        "unquantized_qpl_dot_precision": "HIGHEST,HIGHEST",
        "scope": "Pinned GMM forward/custom VJP, CPU Pallas interpretation; excludes TPU lowering and full-model parity.",
    }
    Path("/work/fp32-gmm-local-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
