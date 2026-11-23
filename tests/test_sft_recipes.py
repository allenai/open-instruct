"""scripts/train/sft.py: every recipe plans a launch, and the A4B recipe keeps what makes it run."""

import importlib.util
import json
import pathlib
import shlex
import sys

import pytest

SFT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train" / "sft.py"
spec = importlib.util.spec_from_file_location("sft_recipes", SFT_PATH)
assert spec is not None and spec.loader is not None
sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft)


def plan(monkeypatch, capsys, *args):
    """Run sft.py --dry_run without Beaker and return the mason command it prints, split into arguments."""

    def no_beaker(*_args, **_kwargs):
        raise OSError("no beaker in tests")

    monkeypatch.setattr(sft.subprocess, "run", no_beaker)
    monkeypatch.setattr(sys, "argv", ["sft.py", *args, "--dry_run"])
    sft.main()
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("Output: /weka/oe-adapt-default/allennlp/deletable_checkpoint/<user>/sft/")
    return shlex.split(lines[-1])


def value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


@pytest.mark.parametrize("name", sorted(sft.RECIPES))
def test_recipe_plans_a_launch(monkeypatch, capsys, name):
    recipe = sft.RECIPES[name]
    cmd = plan(monkeypatch, capsys, "--recipe", name)
    assert value(cmd, "--image") == recipe["image"]
    assert value(cmd, "--num_nodes") == str(recipe["nodes"])
    assert value(cmd, "--cluster") == recipe["cluster"]
    assert value(cmd, "--max_train_steps") == str(recipe["steps"])
    assert value(cmd, "--data_loader_seed") == str(recipe["seed"])
    assert value(cmd, "--run_name") == f"{name}-s{recipe['seed']}"
    assert f"EXPECTED_NUMPY_CACHE={recipe['cache']}" in cmd
    for env in recipe.get("env", []):
        assert cmd[cmd.index(env) - 1] == "--env"
    # The training command follows mason's "--" and starts torchrun on the recipe's node count.
    train = cmd[cmd.index("--") + 1 :]
    assert train[:2] == ["torchrun", f"--nnodes={recipe['nodes']}"]


def test_a4b_recipe(monkeypatch, capsys):
    cmd = plan(monkeypatch, capsys, "--recipe", "a4b")
    # main's image has no transformer-engine, so A4B must run on its own image.
    assert value(cmd, "--image") != sft.RECIPES["c12r"]["image"]
    assert value(cmd, "--config_name") == "configs/olmo_core/olmo35_small_a4b_sft.json"
    # The image runs the config from its copy of this repo, so the file must be here, with whole-block
    # recompute and the nested checkpoints off (#2291: -16% s/step at the same loss).
    config = json.loads((SFT_PATH.parents[2] / value(cmd, "--config_name")).read_text())
    assert config["model"]["recompute_each_block"] is True
    blocks = [config["model"]["block"], *config["model"]["block_overrides"].values()]
    for block in blocks:
        for flag in ("checkpoint_attn", "checkpoint_permute_moe_unpermute", "checkpoint_second_unpermute"):
            assert block.get(flag) is False, flag
    assert value(cmd, "--model_name_or_path").startswith(
        "/weka/olmo-3p5-post-checkpoints/bases/olmo35-4b-active-4t-decay159k-causal/"
    )
    assert value(cmd, "--expert_parallel_degree") == "8"
    assert value(cmd, "--loss_implementation") == "fused_linear"
    assert value(cmd, "--learning_rate") == "7.3e-5"
    assert "--no_compile_model" in cmd
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in cmd
    # c12r's data and length.
    c12r = plan(monkeypatch, capsys, "--recipe", "c12r")
    for flag in ("--mixer_list", "--extra_numpy_dirs", "--seed", "--max_train_steps", "--max_seq_length"):
        assert value(cmd, flag) == value(c12r, flag)
    mixer = cmd[cmd.index("--mixer_list") + 1 : cmd.index("--local_cache_dir")]
    assert mixer == c12r[c12r.index("--mixer_list") + 1 : c12r.index("--local_cache_dir")]
