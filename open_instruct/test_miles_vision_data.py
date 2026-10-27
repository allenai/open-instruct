"""Image conditioning is immutable and length/split checks include pixels."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from open_instruct.miles.datasets import jigsaw, run_data, vision
from open_instruct.miles.errors import InputError


class Processor:
    """Small deterministic processor to exercise preparation without a checkpoint."""

    chat_template = None
    eos_token_id = 0
    config = {"processor": "test"}

    def __init__(self):
        self.tokenizer = self

    def encode(self, text, *, add_special_tokens):
        return [1] * text.count("<|image|>") + [2]

    def prepare(self, ids, images):
        if ids.count(1) != len(images):
            raise ValueError("placeholder mismatch")
        prepared = []
        for index, image in enumerate(images):
            prepared.append(
                SimpleNamespace(
                    patches=torch.from_numpy(np.array(image)).float().reshape(1, 1, -1, 3),
                    pooling=torch.tensor([[[0], [1]]]),
                    token_ids=[3, 3],
                    offsets=[(1 + index * 2, 2 + index * 2)],
                )
            )
        return [0] + [3] * (2 * len(images)) + [2], prepared


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    processor = Processor()
    monkeypatch.setattr(vision, "load_processor", lambda path: processor)
    monkeypatch.setattr(vision, "processor_files", lambda p: [Path(__file__)])
    monkeypatch.setattr(run_data, "_tokenizer", lambda path: processor)
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text('{"model_type":"olmo3moe_vision"}')
    data = tmp_path / "data"
    data.mkdir()
    for name, colors in (("train", ["red", "green", "blue"]), ("eval", ["black", "white", "yellow"])):
        images = [Image.new("RGB", (8, 8), color) for color in colors]
        rows = jigsaw.paired_rows(images, data, name, {"source_image_sha256": [jigsaw.pixels_sha256(images[0])]})
        (data / f"{name}.jsonl").write_bytes(b"".join(jigsaw.encoded(row) for row in rows))
    (data / "verifiers.json").write_bytes(jigsaw.encoded(jigsaw.VERIFIERS))
    spec = {
        "prompt_data": str(data / "train.jsonl"),
        "eval_prompt_data": ["heldout", str(data / "eval.jsonl")],
        "reward_config": str(data / "verifiers.json"),
    }
    return spec, hf, tmp_path / "prepared"


def prepare(dataset, limit=100):
    spec, hf, output = dataset
    return run_data.prepare_data(spec, hf, output, max_prompt_length=limit, seed=17)


def test_shared_text_different_images_and_resume(dataset):
    result = prepare(dataset)
    row = json.loads(Path(result["prompt_data"]).read_text().splitlines()[0])
    assert row["metadata"]["run_prompt_tokens"] == 8
    assert row["metadata"]["vision"]["prepared"]["placeholder_token_ids"] == [0, 1, 1, 1, 2]
    assert prepare(dataset) == result
    image = Path(row["images"][0])
    image.write_bytes(image.read_bytes() + b"modified")
    with pytest.raises(InputError, match="source changed"):
        prepare(dataset)


def test_expanded_length_is_not_tokenizer_length(dataset):
    with pytest.raises(InputError, match="8 tokens"):
        prepare(dataset, limit=6)
    assert not dataset[2].exists()


@pytest.mark.parametrize("damage", ["bytes", "order", "missing", "pixels", "format"])
def test_image_contract_rejects_damage(dataset, damage):
    spec = dataset[0]
    path = Path(spec["prompt_data"])
    row = json.loads(path.read_text().splitlines()[0])
    descriptor = row["metadata"]["vision"]["images"][0]
    if damage == "bytes":
        descriptor["sha256"] = "0" * 64
    elif damage == "pixels":
        descriptor["pixels_sha256"] = "0" * 64
    elif damage == "order":
        row["images"].reverse()
    elif damage == "format":
        row["metadata"]["vision"]["prompt_format"] = "chat"
    else:
        row.pop("images")
    path.write_bytes(jigsaw.encoded(row))
    with pytest.raises(InputError):
        prepare(dataset)


def test_shared_source_photo_rejected_even_with_different_crops(dataset):
    spec = dataset[0]
    training = json.loads(Path(spec["prompt_data"]).read_text().splitlines()[0])
    evaluation_path = Path(spec["eval_prompt_data"][1])
    evaluation = json.loads(evaluation_path.read_text().splitlines()[0])
    evaluation["metadata"]["source_image_sha256"] = training["metadata"]["source_image_sha256"]
    evaluation_path.write_bytes(jigsaw.encoded(evaluation))
    with pytest.raises(InputError, match="overlap"):
        prepare(dataset)


def test_changed_prepared_image_tensor_rejected(dataset):
    result = prepare(dataset)
    row = json.loads(Path(result["prompt_data"]).read_text().splitlines()[0])
    altered = copy.deepcopy(row)
    altered["metadata"]["vision"]["prepared"]["images"][0]["pooling_sha256"] = "0" * 64
    with pytest.raises(InputError, match="layout/features changed"):
        vision.prepare_row(altered, Processor(), {})
