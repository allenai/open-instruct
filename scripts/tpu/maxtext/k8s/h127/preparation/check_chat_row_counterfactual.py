"""Actual pinned trainer: extra forwards leave parameters, RNG state and Adam unchanged."""

import json
from pathlib import Path
from types import SimpleNamespace

import chat_row_counterfactual as cf
import chat_row_counterfactual_patch as cf_patch
import jax
import numpy as np
import optax
import routing_log_fix
import update_audit_patch
from flax import nnx
from jax import numpy as jnp

source = routing_log_fix.patch_source(Path("/deps/src/maxtext/trainers/post_train/sft/train_sft.py").read_bytes())
source = update_audit_patch.patch(source)
patched = cf_patch.patch(source)
base, changed = {"__name__": "base"}, {"__name__": "changed"}
exec(compile(source, "<base>", "exec"), base)
exec(compile(patched, "<changed>", "exec"), changed)
assert cf.load_donor(cf.DONOR_PATH).shape == (2, 1024)
cf.load_donor = lambda _: np.array([[0.2] * 8, [-0.1] * 8], dtype=np.float32)


class Embedding(nnx.Module):
    def __init__(self):
        self.embedding = nnx.Param(jnp.zeros((100352, 8), dtype=jnp.float32))


class Toy(nnx.Module):
    def __init__(self):
        self.token_embedder = Embedding()
        self.head = nnx.Param(jnp.full((8, 100352), 0.001, dtype=jnp.float32))
        self.counter = nnx.Variable(jnp.int32(0))


def loss(model, ids):
    model.counter[...] += 1
    x = model.token_embedder.embedding[ids] + model.head[:, ids].T
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
    return model, optimizer, nnx.jit(namespace["MaxTextPeftTrainer"].create_train_step_fn(dummy))


mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",))
with jax.set_mesh(mesh):
    a, ao, af = setup(base)
    b, bo, bf = setup(changed)
    for step in (1, 2):
        inputs = {"ids": jnp.array([29, 100264, 100265])}
        ax, bx = af(a, ao, inputs=inputs), bf(b, bo, inputs=inputs)
        assert np.array_equal(np.asarray(ax[0]), np.asarray(bx[0]))
        for left, right in ((nnx.state(a), nnx.state(b)), (nnx.state(ao), nnx.state(bo))):
            av, at = jax.tree_util.tree_flatten(left)
            bv, bt = jax.tree_util.tree_flatten(right)
            assert at == bt and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(av, bv, strict=True))
        values = np.asarray(bx[1]["h127_chat_counterfactual"])
        if step == 1:
            assert not values.any()
        else:
            assert values[0, 0] == float(ax[0])
            assert len(set(values[:, 0])) == 4
            cf.emit(step, bx[1]["h127_chat_counterfactual"])
print(
    "CHAT-ROW-CF-CHECK",
    json.dumps({"steps": 2, "loss_model_optimizer_nonparam_state_exact": True, "four_forwards_distinct": True}),
)
