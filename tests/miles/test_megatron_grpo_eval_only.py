"""Megatron verifier GRPO's eval-only switch: validation and run identity."""

import tomllib
from pathlib import Path

import pytest

from open_instruct.miles.configuration import specs
from open_instruct.miles.errors import InputError
from open_instruct.miles.execution import workflow

FIXTURE = Path(__file__).parent / "fixtures/megatron_grpo/qwen35-2b-verifier-dppo-async-short.toml"


def load(**training):
    document = tomllib.loads(FIXTURE.read_text())
    document["training"].update(training)
    return document


def test_eval_only_defaults_off_without_changing_existing_run_identity():
    spec = specs.from_dict(load(), config_path=FIXTURE)
    document = spec.to_dict()
    assert document["training"]["eval_only"] is False
    recorded = {**document, "training": {k: v for k, v in document["training"].items() if k != "eval_only"}}
    assert workflow.fingerprint(document) == workflow.fingerprint(recorded)


def test_eval_only_is_part_of_run_identity():
    trained = specs.from_dict(load(), config_path=FIXTURE).to_dict()
    evaluated = specs.from_dict(load(eval_only=True), config_path=FIXTURE).to_dict()
    assert evaluated["training"]["eval_only"] is True
    assert workflow.fingerprint(evaluated) != workflow.fingerprint(trained)


def test_eval_only_requires_the_async_producer():
    document = load(eval_only=True)
    document["miles"] = {}
    with pytest.raises(InputError, match="miles.fully_async=true"):
        specs.from_dict(document, config_path=FIXTURE)


def test_eval_only_must_be_boolean():
    with pytest.raises(InputError, match="training.eval_only"):
        specs.from_dict(load(eval_only="yes"), config_path=FIXTURE)
