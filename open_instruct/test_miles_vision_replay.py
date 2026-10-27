"""Multi-image pooling offsets and pixels survive tensor serialization exactly."""

import io
from types import SimpleNamespace

import pytest
import torch

from open_instruct.miles.datasets import vision_replay


def test_multi_image_pooling_and_round_trip():
    first = SimpleNamespace(
        patches=torch.arange(12).float().reshape(1, 1, 4, 3),
        pooling=torch.tensor([[[0, -1], [2, 3]]]),
        offsets=[(1, 2)],
    )
    second = SimpleNamespace(
        patches=torch.arange(24).float().reshape(1, 2, 4, 3),
        pooling=torch.tensor([[[1, 4], [7, -1]]]),
        offsets=[(5, 6)],
    )
    values = vision_replay.training_inputs([first, second], [0, 3, 3, 4, 2, 3, 3, 4], 3)
    assert values["pooled_patches_idx"].tolist() == [[[0, -1], [2, 3], [5, 8], [11, -1]]]
    assert values["image_positions"].tolist() == [1, 2, 5, 6]
    assert values["images"].shape == (1, 3, 4, 3)
    stream = io.BytesIO()
    torch.save(values, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    assert all(torch.equal(tensor, restored[key]) for key, tensor in values.items())
    assert first.pooling.tolist() == [[[0, -1], [2, 3]]]
    assert second.pooling.tolist() == [[[1, 4], [7, -1]]]


@pytest.mark.parametrize("positions,pools", [([(0, 1)], [[[0, 1]]]), ([(1, 1)], [[[0, 9]]]), ([(1, 1)], [[[-1, -1]]])])
def test_bad_alignment_rejected(positions, pools):
    image = SimpleNamespace(patches=torch.zeros(1, 1, 4, 3), pooling=torch.tensor(pools), offsets=positions)
    with pytest.raises(ValueError):
        vision_replay.training_inputs([image], [0, 3, 2], 3)
