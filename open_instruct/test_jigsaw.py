"""CPU invariants of the actual generator, pixels, splits and exact reward."""

import asyncio
import dataclasses
import json

import numpy as np
import pytest
from PIL import Image

from open_instruct.miles.datasets import jigsaw
from open_instruct.miles.rewards import jigsaw as reward


@pytest.fixture
def photos(tmp_path):
    source = tmp_path / "photos"
    source.mkdir()
    for index in range(8):
        pixels = np.random.default_rng(index).integers(0, 256, (80, 80, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(source / f"{index}.png")
    return jigsaw.source_photos(source)


def test_exact_reconstruction_and_swaps(photos, tmp_path):
    config = jigsaw.Config(canvas=64, piece=24, train_photos=4, eval_photos=4)
    images, info = jigsaw.make_puzzle(photos[0], photos[1], config, "same_photo", 3)
    reconstructed = images[0].copy()
    reconstructed.paste(images[1], info["hole_xy"])
    assert reconstructed.tobytes() == jigsaw.canvas(photos[0], config.canvas).tobytes()
    rows = jigsaw.paired_rows(images, tmp_path, "pair", info)
    assert rows[0]["input"] == rows[1]["input"]
    assert rows[0]["images"][0] == rows[1]["images"][0]
    assert rows[0]["images"][1:] == rows[1]["images"][:0:-1]
    assert [r["label"] for r in rows] == ["A", "B"]
    for row in rows:
        for descriptor in row["metadata"]["vision"]["images"]:
            with Image.open(descriptor["path"]) as image:
                assert jigsaw.pixels_sha256(image) == descriptor["pixels_sha256"]


@pytest.mark.parametrize("position", range(5))
def test_shift_is_distinct_at_every_hole_position(photos, position):
    config = jigsaw.Config(canvas=64, piece=24, train_photos=4, eval_photos=4)
    images, info = jigsaw.make_puzzle(photos[0], photos[1], config, "shift", position)
    assert info["hole_xy"] != info["distractor_xy"]
    assert images[1].tobytes() != images[2].tobytes()


def test_blank_candidates_rejected(tmp_path):
    path = tmp_path / "flat.png"
    Image.new("RGB", (64, 64), "white").save(path)
    photo = jigsaw.source_photos(tmp_path)[0]
    config = jigsaw.Config(canvas=64, piece=24, train_photos=2, eval_photos=2)
    with pytest.raises(ValueError, match="distinct candidates"):
        jigsaw.make_puzzle(photo, photo, config, "same_photo", 3)


def test_disjoint_sources_and_reproducibility(photos, tmp_path):
    config = jigsaw.Config(canvas=64, piece=24, train_photos=4, eval_photos=4)
    manifests = [jigsaw.generate(photos, tmp_path / name, config) for name in ("one", "two")]
    identities = {}
    for split in ("train", "heldout"):
        content = [
            (tmp_path / name / f"{split}.jsonl").read_text().replace(str(tmp_path / name), "ROOT")
            for name in ("one", "two")
        ]
        assert content[0] == content[1]
        rows = [json.loads(line) for line in content[0].splitlines()]
        assert len(rows) == 16
        assert sum(row["label"] == "A" for row in rows) == 8
        identities[split] = {x for row in rows for x in row["metadata"]["source_image_sha256"]}
    assert not identities["train"] & identities["heldout"]
    assert manifests[0]["sources"] == manifests[1]["sources"]
    assert dataclasses.asdict(config) == manifests[0]["config"]


def test_duplicate_pixels_do_not_leak_across_splits(tmp_path):
    im = Image.new("RGB", (64, 64), "red")
    im.save(tmp_path / "one.png")
    im.save(tmp_path / "two.png", compress_level=0)
    assert len(jigsaw.source_photos(tmp_path)) == 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A", "A"),
        ("(B)", "B"),
        (" Answer: A.\n", "A"),
        ("A or B", None),
        ("A B", None),
        ("The answer is A", None),
        ("(A", None),
        ("", None),
    ],
)
def test_strict_choice_reward(text, expected):
    assert reward.answer(text) == expected
    verifier = reward.JigsawVerifier()
    for gold in ("A", "B"):
        assert asyncio.run(verifier.async_call([], text, gold)).score == float(expected == gold)
    with pytest.raises(ValueError, match="target"):
        asyncio.run(verifier.async_call([], text, "C"))
