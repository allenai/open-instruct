"""scripts/train/sft_mm.py: every multimodal recipe plans a launch of the pinned OLMo-core recipe."""

import importlib.util
import json
import pathlib
import shlex
import subprocess
import sys

import pytest

SFT_MM_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train" / "sft_mm.py"
spec = importlib.util.spec_from_file_location("sft_mm_recipes", SFT_MM_PATH)
assert spec is not None and spec.loader is not None
sft_mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft_mm)
# plan() stubs subprocess.run out for the script, which shares the module.
RUN = subprocess.run


def plan(monkeypatch, capsys, *args):
    """Run sft_mm.py --dry_run without Beaker; return the output line and the Beaker spec it prints."""

    def no_beaker(*_args, **_kwargs):
        raise OSError("no beaker in tests")

    monkeypatch.setattr(sft_mm.subprocess, "run", no_beaker)
    monkeypatch.setattr(sys, "argv", ["sft_mm.py", *args, "--dry_run"])
    sft_mm.main()
    out = capsys.readouterr().out
    first, rest = out.split("\n", 1)
    return first, json.loads(rest)


def torchrun_args(spec):
    """The arguments of the job's torchrun line, from the recipe script on."""
    line = spec["tasks"][0]["arguments"][0].strip().splitlines()[-1]
    args = shlex.split(line)
    assert args[:2] == ["exec", "torchrun"]
    return args[args.index("--nproc_per_node=8") + 1 :]


@pytest.mark.parametrize("name", sorted(sft_mm.RECIPES))
def test_recipe_plans_a_launch(monkeypatch, capsys, name):
    recipe = sft_mm.RECIPES[name]
    output, spec = plan(monkeypatch, capsys, "--recipe", name)
    run = f"olmo35-mmsft-{name}-s{sft_mm.DEFAULT_SEED}"
    assert output == f"Output: /weka/oe-adapt-default/allennlp/deletable_checkpoint/<user>/sft/{run}"
    task = spec["tasks"][0]
    assert task["image"] == {"beaker": sft_mm.IMAGE}
    assert task["replicas"] == sft_mm.NODES
    assert task["constraints"] == {"cluster": [sft_mm.CLUSTER]}
    assert task["context"]["autoResume"] is True
    assert task["timeout"] == recipe["timeout"]
    script = task["arguments"][0]
    # The job trains the pinned OLMo-core commit, and checks it got it.
    assert f"fetch -q --depth 1 {sft_mm.OLMO_CORE_REPO} {sft_mm.OLMO_CORE_SHA}" in script
    assert f'test "$(git -C /tmp/olmo-core rev-parse HEAD)" = {sft_mm.OLMO_CORE_SHA}' in script
    args = torchrun_args(spec)
    assert args[:4] == ["/tmp/olmo-core/src/scripts/train/Olmo35-Multimodal-SFT.py", "train", run, sft_mm.CLUSTER]
    overrides = args[4:]
    for override in [*recipe["overrides"], *sft_mm.COMMON_OVERRIDES]:
        assert override in overrides
    assert f"--recipe.data_seed={sft_mm.DEFAULT_SEED}" in overrides
    assert overrides[-1] == f"--recipe.num_nodes={sft_mm.NODES}"
    assert not any(o.startswith("--recipe.stop_at_step=") for o in overrides)


def test_seed_steps_and_checkpoint(monkeypatch, capsys):
    output, spec = plan(monkeypatch, capsys, "--recipe", "4n-emo256k", "--seed", "34522", "--steps", "30")
    assert output.endswith("/sft/olmo35-mmsft-4n-emo256k-s34522-30st")
    overrides = torchrun_args(spec)[4:]
    assert "--recipe.data_seed=34522" in overrides
    assert "--recipe.stop_at_step=30" in overrides

    _, spec = plan(monkeypatch, capsys, "--recipe", "4n-emo256k", "--checkpoint", "/weka/x/step1", "--name", "mine")
    overrides = torchrun_args(spec)[4:]
    assert [o for o in overrides if o.startswith("--recipe.base=")] == ["--recipe.base=/weka/x/step1"]


def test_gate_reference(monkeypatch, capsys):
    output, spec = plan(monkeypatch, capsys, "--recipe", "4n-emo256k", "--steps", "30", "--gate_reference")
    overrides = torchrun_args(spec)[4:]
    assert f"--recipe.batch_hash_dir={output}/gates" in overrides
    assert "--recipe.step1_reference=true" in overrides


def test_checkpoint_needs_name(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sft_mm.py", "--recipe", "4n-emo256k", "--checkpoint", "/weka/x", "--dry_run"])
    with pytest.raises(SystemExit):
        sft_mm.main()


def test_job_script_is_valid_bash(monkeypatch, capsys, tmp_path):
    _, spec = plan(monkeypatch, capsys, "--recipe", "4n-hires")
    script = tmp_path / "job.sh"
    script.write_text(spec["tasks"][0]["arguments"][0])
    RUN(["bash", "-n", str(script)], check=True)
