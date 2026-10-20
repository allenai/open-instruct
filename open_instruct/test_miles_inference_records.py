"""Record configuration remains available without installing MILES."""

import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError

ASYNC = {"async": {"fully_async": True}}


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"records_root": "relative/store"}, "absolute"),
        ({"records_responses": "some"}, "records_responses"),
        ({"records_responses": "sample", "records_response_sample_rate": 1.5}, "records_response_sample_rate"),
    ],
)
def test_core_rejects_invalid_record_settings(fields, message):
    with pytest.raises(InputError, match=message):
        CoreConfig(**fields)


def document(tmp_path, **sections):
    return {
        "schema_version": 1,
        "name": "records-trial",
        "model": {"source": "model", "format": "hf"},
        "output": {"root": str(tmp_path / "run")},
        "data": {"tasks": [{"task": "gsm8k", "train_count": 32, "eval_count": 16}]},
        **sections,
    }


@pytest.mark.parametrize(
    "records",
    [{"responses": "sample"}, {"responses": "off", "response_sample_rate": 0.5}, {"response_sample_rate": 0.5}],
)
def test_sample_rate_is_required_only_for_sampled_responses(tmp_path, records):
    section = {"enabled": True, "root": "/weka/store", **records}
    with pytest.raises(InputError, match="required with, and only with"):
        RunSpec.from_dict(document(tmp_path, records=section, **ASYNC), config_path=tmp_path / "run.toml")


def test_recording_requires_fully_async(tmp_path):
    with pytest.raises(InputError, match="requires fully_async=true"):
        RunSpec.from_dict(
            document(tmp_path, records={"enabled": True, "root": "/weka/store"}), config_path=tmp_path / "run.toml"
        )


def test_records_section_compiles_to_core_and_round_trips(tmp_path):
    section = {"enabled": True, "root": "store", "responses": "sample", "response_sample_rate": 0.25}
    spec = RunSpec.from_dict(document(tmp_path, records=section, **ASYNC), config_path=tmp_path / "run.toml")
    core = spec.compile().core
    assert core.records_root == str(tmp_path / "store")
    assert (core.records_responses, core.records_response_sample_rate) == ("sample", 0.25)
    assert spec.plan()["records"]["root"] == str(tmp_path / "store")
    again = RunSpec.from_dict(spec.to_dict(), config_path=tmp_path / "elsewhere" / "run.toml")
    assert again.compile().core.records_root == core.records_root


def test_records_are_off_unless_enabled(tmp_path):
    assert RunSpec.from_dict(document(tmp_path), config_path=tmp_path / "run.toml").compile().core.records_root is None
    disabled = document(tmp_path, records={"enabled": False, "root": "/weka/store"})
    assert RunSpec.from_dict(disabled, config_path=tmp_path / "run.toml").compile().core.records_root is None
    with pytest.raises(InputError, match="records.root is required"):
        RunSpec.from_dict(document(tmp_path, records={"enabled": True}), config_path=tmp_path / "run.toml")
    with pytest.raises(InputError, match="records"):
        RunSpec.from_dict(
            document(tmp_path, records={"enabled": True, "path": "/x"}), config_path=tmp_path / "run.toml"
        )
