"""CPU checks for the Olmo 3 experiment inputs and source-preserving staging."""

import hashlib
import json
from pathlib import Path

import pytest
from scripts.miles import launch_olmo3_preparation, stage_olmo3

from open_instruct.miles import topology
from open_instruct.miles.errors import InputError
from open_instruct.miles.run_spec import RunSpec

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "filename,batch,trainer,engines,replicas",
    [("qualification/olmo3-think-gsm8k.toml", 8, 2, 1, 1), ("proposals/olmo3-think-dolci-200.toml", 512, 8, 7, 2)],
)
def test_olmo3_configs_use_dense_recipe_controls(filename, batch, trainer, engines, replicas):
    spec = RunSpec.load(ROOT / "configs/miles" / filename)
    config = spec.compile()
    assert config.core.expert_parallel_size == 1
    assert not config.miles["use_rollout_routing_replay"]
    assert config.miles["actor_num_gpus_per_node"] == trainer
    assert config.miles["rollout_num_gpus"] == engines
    assert config.miles["global_batch_size"] == batch
    assert config.miles["adam_beta2"] == 0.999
    assert config.miles["eps_clip_high"] == 0.272
    assert not config.miles["grpo_std_normalization"]
    assert config.miles["calculate_per_token_loss"]
    assert config.miles["rollout_max_context_len"] == 2048 + 32768
    assert config.miles["save_interval"] <= config.miles["num_rollout"]
    assert topology.plan(spec)["replicas"] == replicas


def test_staging_preserves_snapshot_and_pins_original_prompt(tmp_path):
    source, target = tmp_path / "source", tmp_path / "staged"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"olmo3"}')
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer_config.json").write_text('{"chat_template":"old"}')
    (source / "chat_template.jinja").write_text("old")
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    stage_olmo3.stage(source, target)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original
    assert (target / "model.safetensors").is_symlink()
    assert not (target / "tokenizer_config.json").is_symlink()
    template = (target / "chat_template.jinja").read_text()
    assert (
        hashlib.sha256(template.encode()).hexdigest()
        == "eba6e269f669706e5c788e370160dad953137f3fa7014fa69d03c7b3ad9f0e72"
    )
    assert json.loads((target / "tokenizer_config.json").read_text())["chat_template"] == template
    with pytest.raises(InputError, match="already exists"):
        stage_olmo3.stage(source, target)


def test_checkpoint_preparation_is_cpu_only_on_saturn():
    spec = RunSpec.load(ROOT / "configs/miles/qualification/olmo3-think-gsm8k-robertb-20260912.toml")
    task = launch_olmo3_preparation.specification("test-image", spec)["tasks"][0]
    assert task["constraints"] == {"cluster": ["ai2/saturn"]}
    assert "gpuCount" not in task["resources"]
    assert not task["context"]["autoResume"]
    assert "preflight_attention" not in task["arguments"][0]
    assert "scripts.miles.prepare_olmo3_checkpoint" in task["arguments"][0]
    assert "open_instruct.miles train" not in task["arguments"][0]
