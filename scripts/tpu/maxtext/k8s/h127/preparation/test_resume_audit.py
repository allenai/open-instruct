"""Exercise the resume audit on a real tiny Flax/Optax/Tunix checkpoint."""

import tempfile

import jax
import optax
from flax import nnx
from resume_audit import install
from tunix.sft import checkpoint_manager

root = tempfile.mkdtemp(prefix="h127-resume-")
model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
model.rngs = nnx.Rngs(42)
optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)
for _ in range(5):
    grads = jax.tree.map(lambda x: x * 0 + 0.1, nnx.state(model, nnx.Param))
    optimizer.update(model, grads)
manager = checkpoint_manager.CheckpointManager(root_directory=root)
assert manager.save(5, model, optimizer, force=True)
manager.close()
model2 = nnx.Linear(2, 3, rngs=nnx.Rngs(123))
model2.rngs = nnx.Rngs(456)
optimizer2 = nnx.Optimizer(model2, optax.adamw(1e-3), wrt=nnx.Param)
manager2 = checkpoint_manager.CheckpointManager(root_directory=root)
install(5)
assert manager2.maybe_restore(model2, optimizer2)[0] == 5
manager2.close()
# A weights-only checkpoint must not silently pass as a full training resume.
root3 = tempfile.mkdtemp(prefix="h127-resume-no-optimizer-")
manager3 = checkpoint_manager.CheckpointManager(root_directory=root3)
assert manager3.save(5, model, force=True)
manager3.close()
manager4 = checkpoint_manager.CheckpointManager(root_directory=root3)
try:
    manager4.maybe_restore(model2, optimizer2)
except AssertionError as error:
    assert "Optimizer omitted" in str(error), error
else:
    raise AssertionError("Weights-only checkpoint accepted as an optimizer resume")
manager4.close()
print("RESUME-AUDIT-TEST-PASSED", flush=True)
