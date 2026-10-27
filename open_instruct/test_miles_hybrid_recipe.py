"""The temporary hybrid recipe must preserve the requested 1k-step topology."""

from pathlib import Path

from open_instruct.miles.configuration import validation
from open_instruct.miles.configuration.run_spec import RunSpec

ROOT = Path(__file__).resolve().parents[1]


def test_hybrid_recipe_precision_topology_and_checkpointing():
    path = ROOT / "configs/miles/benchmarks/hybrid_275m_gsm8k.toml"
    spec = RunSpec.from_dict(validation.read_document(path), config_path=path)
    config = spec.compile()
    assert config.miles["num_rollout"] == 1000
    assert config.miles["rollout_batch_size"] == config.miles["n_samples_per_prompt"] == 4
    assert config.miles["global_batch_size"] == 16
    assert config.miles["save_interval"] == 100
    assert config.miles["eval_interval"] == 100
    assert config.miles["sglang_disable_cuda_graph"] is True
    assert config.miles["sglang_chunked_prefill_size"] == -1
    assert config.core.attention_backend == "flash_2"
    assert config.core.fp32_lm_head is False
    assert "/olmo-core/v1/275m/midtrain/step0038147/" in config.core.model_config
    plan = spec.plan()
    assert plan["launch"]["gpus_per_replica"] == 2
    assert plan["launch"]["auto_resume"] is True
    assert plan["launch"]["max_retries"] == 3
    assert plan["launch"]["env"]["MILES_HYBRID_BRIDGE"] == "1"
    assert plan["launch"]["env"]["MILES_HYBRID_REFERENCE_BACKBONE"] == "1"
