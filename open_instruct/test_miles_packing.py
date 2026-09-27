"""CPU-only packing configuration checks."""

import pytest

from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.configuration.run_spec import RunSpec


def test_options_compile_to_concatenated_loss_layout():
    config = RunConfig(
        CoreConfig(sequence_packing=True, max_sequence_length=16), {"hf_checkpoint": "/hf", "global_batch_size": 4}
    )
    argv = config.arguments()
    assert argv[argv.index("--qkv-format") + 1] == "thd"
    with pytest.raises(ValueError, match="requires qkv_format"):
        RunConfig(config.core, {**config.miles, "qkv_format": "bshd"}).validate()
    with pytest.raises(ValueError, match="requires sequence_packing"):
        RunConfig(CoreConfig(packing_max_tokens=8192), config.miles).validate()
    with pytest.raises(ValueError, match="must cover"):
        RunConfig(
            CoreConfig(sequence_packing=True, packing_max_tokens=8, max_sequence_length=16), config.miles
        ).validate()


def test_researcher_trainer_section_accepts_packing(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text("""schema_version = 1
name = "packing"
[model]
source = "/model"
format = "hf"
[output]
root = "/output"
[data]
prompt_data = "/data/train.jsonl"
reward_config = "/data/rewards.json"
[trainer]
sequence_packing = true
packing_max_tokens = 8192
""")
    spec = RunSpec.load(path)
    # plan is intentionally CPU safe; the compiled core fields preserve the knobs.
    core = spec.compile().core
    assert core.sequence_packing and core.packing_max_tokens == 8192
