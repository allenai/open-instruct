"""Check that missing, duplicated and incorrect TPU evidence cannot pass."""

import copy

from monitor_tpu import assess

config = {
    "steps": 2,
    "reference_ce": 1.0,
    "reference_tokens": 100,
    "expected_batches": [{"host": 0, "step": 1, "sha256": "a" * 64}],
}
text = (
    "LOSS-PARTS step=1 ce=1.000001 moe_lb=0.1 z=0.1 tokens=100\n"
    "LOSS-PARTS step=2 ce=0.99 moe_lb=0.1 z=0.1 tokens=100\n"
    "MOE-DROPPED step=1 dropped=0 routed=10000\n"
    "MOE-DROPPED step=2 dropped=0 routed=10000\n" + "BATCH-HASH n=1 host=0 sha256=" + "a" * 64 + "\n"
)
assert assess(config, text, True)["passed"]
assert assess(config, text + text, True)["passed"]
assert not assess(config, text)["passed"]
assert not assess(config, text.replace("step=2", "step=3"), True)["passed"]
assert not assess(config, text.replace("1.000001", "1.01"), True)["passed"]
assert not assess(config, text.replace("tokens=100", "tokens=99"), True)["passed"]
assert not assess(config, text.replace("a" * 64, "b" * 64), True)["passed"]
assert not assess(config, text.replace("dropped=0", "dropped=200"), True)["passed"]
assert not assess(config, text.replace("MOE-DROPPED", "MISSING-DROPS"), True)["passed"]
assert not assess(config, text.replace("1.000001", "nan"), True)["passed"]
try:
    assess(config, text + text.replace("1.000001", "1.002"), True)
except AssertionError:
    pass
else:
    raise AssertionError("Conflicting duplicates accepted")
resume = copy.deepcopy(config)
resume.update(first_step=51, steps=52, resume_step=50)
assert not assess(resume, text, True)["passed"]
print("MONITOR-TEST-PASSED")
