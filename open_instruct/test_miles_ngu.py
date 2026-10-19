"""CPU compilation of the public NGU recipes into the pinned runtime's CLI."""

from pathlib import Path

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec


@pytest.mark.parametrize(
    "recipe,option,value",
    [
        ("qwen3_4b_base_deepscaler_ngu", "ngu-requeue-probability", "0.875"),
        ("qwen3_4b_base_deepscaler_ngu_budget4_k8n16", "ngu-max-budget", "4"),
    ],
)
def test_ngu_recipes_compile_to_the_runtime_handler(recipe, option, value):
    path = Path(__file__).resolve().parents[1] / "configs/ngu" / f"{recipe}.toml"
    config = RunSpec.load(path).compile()
    assert config.core.publication_mode == "barrier"
    assert config.miles["fully_async"] is True
    assert config.miles["async_unused_samples_handler"] == "never_give_up"
    argv = config.arguments()
    assert argv[argv.index("--async-unused-samples-handler") + 1] == "never_give_up"
    assert argv[argv.index(f"--{option}") + 1] == value
    assert argv[argv.index("--ngu-solved-reward") + 1] == "1.0"
