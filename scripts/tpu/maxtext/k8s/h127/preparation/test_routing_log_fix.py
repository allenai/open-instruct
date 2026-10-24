"""Exercise the real pinned image's logger before and after its config repair."""

import ast
import hashlib
import importlib
from types import SimpleNamespace
from unittest import mock

import numpy as np
from maxtext.trainers.post_train.sft import train_sft
from monitor_tpu import assess
from routing_log_fix import REPLACEMENTS, SOURCE_PATH, SOURCE_SHA256, patch_source

source = SOURCE_PATH.read_bytes()
assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
triples = np.asarray([[1.5, 0, 16777216]] * 15, dtype=np.float32)


def capture(module, values, enabled=True):
    # This reproduces the real separation: Tunix config has no MaxText flag.
    trainer = SimpleNamespace(_train_steps=0, config=SimpleNamespace(), _h127_log_moe_dropped_tokens=enabled)
    aux = {"xent_sum": 100.0, "total_weights": 100.0, "z_loss": 0.0, "moe_lb_loss": 0.1}
    aux["moe_required_rbf"] = values
    lines = []
    with mock.patch.object(module, "_log_hbm"), mock.patch.object(module.max_logging, "log", side_effect=lines.append):
        module.MaxTextPeftTrainer._post_process_train_step(trainer, aux)
    return "\n".join(lines)


config = {"steps": 1, "reference_ce": 1.0, "reference_tokens": 100, "expected_batches": []}
before = capture(train_sft, triples)
assert "MOE-DROPPED" not in before
assert "Invalid per-layer capacity evidence" in assess(config, before, True)["failures"]

patched = patch_source(source)
# Reversing just these two edits restores the complete original program AST.
restored = patched.decode()
for old, new in reversed(REPLACEMENTS):
    assert restored.count(new) == 1
    restored = restored.replace(new, old)
assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(source))
try:
    patch_source(source + b"\n")
except ValueError:
    pass
else:
    raise AssertionError("A different image source was accepted")

SOURCE_PATH.write_bytes(patched)
importlib.reload(train_sft)
after = capture(train_sft, triples)
result = assess(config, after, True)
assert result["passed"], result
assert result["required_capacity"][1]["maximum"] == 1.5
assert result["drops"][1]["routed"] == 15 * 16777216
assert result["drops"][1]["dropped"] == 0

triples[:, 1] = np.arange(15)
nonzero = assess(config, capture(train_sft, triples), True)
assert nonzero["drops"][1]["dropped"] == 105
assert nonzero["drops"][1]["per_layer"] == list(range(15))
disabled = capture(train_sft, triples[:, 0], enabled=False)
assert "MOE-DROPPED" not in disabled
assert "max=1.5000" in disabled
print("ROUTING-LOGGER-TEST-PASSED", "patched_sha256=" + hashlib.sha256(patched).hexdigest())
