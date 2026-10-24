"""Test path containment and the checksum used to qualify checkpoint transfers."""

import pytest
from download_checkpoint import checksum, safe_relative


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../outside", "a//b", ""])
def test_reject_unsafe_or_noncanonical_path(name):
    with pytest.raises(AssertionError):
        safe_relative(name)


def test_valid_relative_path():
    assert str(safe_relative("ocdbt.process_0/d/abc")) == "ocdbt.process_0/d/abc"


def test_known_crc32c(tmp_path):
    path = tmp_path / "bytes"
    path.write_bytes(b"123456789")
    assert checksum(path) == {"size": 9, "crc32c": "4waSgw=="}
