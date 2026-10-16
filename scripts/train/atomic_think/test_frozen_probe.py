import pytest
from scripts.train.atomic_think import frozen_probe


def test_wrapper_difference_is_allowed_but_state_differences_are_not():
    frozen_probe.check_hashes({"module.weight": "hash"}, {"weight": "hash"})
    with pytest.raises(ValueError, match="differs from training"):
        frozen_probe.check_hashes({"module.weight": "before"}, {"weight": "after"})
    with pytest.raises(ValueError, match="keys differ"):
        frozen_probe.check_hashes({"module.weight": "hash"}, {})
    with pytest.raises(ValueError, match="duplicate"):
        frozen_probe.normalize_hashes({"module.weight": "hash", "weight": "hash"})
