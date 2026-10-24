import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import optax
import routing_log_fix
import update_audit
import update_audit_patch
from flax import nnx
from jax import numpy as jnp

source = routing_log_fix.patch_source(Path("/deps/src/maxtext/trainers/post_train/sft/train_sft.py").read_bytes())
patched = update_audit_patch.patch(source)
base = {"__name__": "h127_base_test"}
audit = {"__name__": "h127_audit_test"}
exec(compile(source, "<base>", "exec"), base)
exec(compile(patched, "<audit>", "exec"), audit)


class Toy(nnx.Module):
    def __init__(self):
        self.embedding = nnx.Param(jnp.zeros((100352, 8), dtype=jnp.float32))
        self.head = nnx.Param(jnp.full((8, 100352), 0.001, dtype=jnp.float32))


def loss(model, ids):
    x = model.embedding[ids] + model.head[:, ids].T
    value = jnp.mean((x - 0.5) ** 2)
    return value, {
        "xent_sum": value,
        "total_weights": jnp.float32(1),
        "z_loss": jnp.float32(0),
        "moe_lb_loss": jnp.float32(0),
    }


def setup(namespace):
    model = Toy()
    optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3)), wrt=nnx.Param)
    dummy = SimpleNamespace(
        model=model,
        loss_fn=loss,
        _has_aux=True,
        gen_model_input_fn=lambda x: x,
        _lora_enabled=False,
        config=SimpleNamespace(get_with_default=lambda key, default: 1, max_seq_token_per_tpu=None),
    )
    fn = namespace["MaxTextPeftTrainer"].create_train_step_fn(dummy)
    return model, optimizer, nnx.jit(fn)


mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",))
with jax.set_mesh(mesh):
    m1, o1, f1 = setup(base)
    m2, o2, f2 = setup(audit)
    for step in range(1, 4):
        inputs = {"ids": jnp.array([29, 100264, 100265])}
        x1 = f1(m1, o1, inputs=inputs)
        x2 = f2(m2, o2, inputs=inputs)
        assert np.array_equal(np.asarray(x1[0]), np.asarray(x2[0]))
        assert np.array_equal(np.asarray(x1[2]), np.asarray(x2[2]))
        for left, right in [(nnx.state(m1), nnx.state(m2)), (nnx.state(o1), nnx.state(o2))]:
            a, ta = jax.tree_util.tree_flatten(left)
            b, tb = jax.tree_util.tree_flatten(right)
            assert ta == tb
            assert all(np.array_equal(np.asarray(v), np.asarray(w)) for v, w in zip(a, b, strict=True))
        record = x2[1]["h127_update_audit"]
        expected = update_audit.selected_rows(nnx.state(m2, nnx.Param))
        assert all(np.array_equal(np.asarray(record["rows_after"][k]), np.asarray(v)) for k, v in expected.items())
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            update_audit.emit(step, record)
        encoded = json.loads(stream.getvalue().split("H127-UPDATE-AUDIT ", 1)[1])
        raw = base64.b64decode(encoded["npz_base64"])
        assert hashlib.sha256(raw).hexdigest() == encoded["sha256"]
        archive = np.load(io.BytesIO(raw), allow_pickle=False)
        assert len(archive.files) == encoded["array_count"]
print(
    "UPDATE-AUDIT-CHECK",
    json.dumps(
        {
            "steps": 3,
            "devices": len(jax.devices()),
            "model_optimizer_and_loss_exactly_unchanged": True,
            "selected_rows_and_npz_verified": True,
            "patch_sha256": hashlib.sha256(patched).hexdigest(),
        }
    ),
)
