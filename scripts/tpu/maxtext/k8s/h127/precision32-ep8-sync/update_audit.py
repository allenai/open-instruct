"""Read-only first-update diagnostics; not part of H127's scientific recipe."""

import base64
import hashlib
import io
import json

import jax
import numpy as np
from jax import numpy as jnp
from jax import sharding

VOCAB = 100352
TOKEN_IDS = (29, 198, 524, 771, 14023, 27963, 33413, 36539, 100256, 100257, 100264, 100265, 100266, 100267, 100270)


def flatten(state):
    return {jax.tree_util.keystr(path): value for path, value in jax.tree_util.tree_flatten_with_path(state)[0]}


def replicate(value):
    return jax.lax.with_sharding_constraint(value, sharding.PartitionSpec())


def selected_rows(state):
    result = {}
    for name, value in flatten(state).items():
        if value.ndim != 2:
            continue
        if value.shape[0] == VOCAB:
            rows = value[jnp.asarray(TOKEN_IDS)]
        elif value.shape[1] == VOCAB:
            rows = value[:, jnp.asarray(TOKEN_IDS)].T
        else:
            continue
        result[name] = replicate(rows.astype(jnp.float32))
    assert len(result) == 2, f"Expected input embedding and output head, found {tuple(result)}"
    return result


def before_update(params, grads):
    p, g = flatten(params), flatten(grads)
    assert p.keys() == g.keys()
    # Every leaf is reduced globally; replicated scalars and selected rows are
    # the only extra outputs. No parameter, gradient, RNG, or optimizer is edited.
    return {
        "parameter_norm2": {name: replicate(jnp.sum(value.astype(jnp.float32) ** 2)) for name, value in p.items()},
        "gradient_norm2": {name: replicate(jnp.sum(value.astype(jnp.float32) ** 2)) for name, value in g.items()},
        "rows_before": selected_rows(params),
        "row_gradients": selected_rows(grads),
    }


def emit(step, audit):
    arrays = {}
    for group, entries in audit.items():
        for name, value in entries.items():
            local = value.addressable_data(0) if isinstance(value, jax.Array) else value
            array = np.asarray(local)
            assert array.shape == value.shape, f"Audit output was not replicated: {group}/{name}"
            assert np.isfinite(array).all(), f"Nonfinite diagnostic: {group}/{name}"
            arrays[group + "/" + name] = array
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    raw = stream.getvalue()
    record = {
        "step": step,
        "token_ids": TOKEN_IDS,
        "array_count": len(arrays),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "npz_base64": base64.b64encode(raw).decode(),
    }
    print("H127-UPDATE-AUDIT " + json.dumps(record, separators=(",", ":")), flush=True)
