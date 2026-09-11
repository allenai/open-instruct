"""Prepare an update-zero diagnostic from the exact submitted Megatron100 command."""

import argparse
import copy
import json
from pathlib import Path

IMAGE = "01M26TT1EQTB84RV0MM650W0YE"
SOURCE = "0e648108c70c5d5256a9b93a88b4f8d610e44ea0"
ORIGINAL_OUTPUT = "/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1/megatron-r3"


def replace_value(argv, flag, value):
    if argv.count(flag) != 1:
        raise ValueError(f"Expected exactly one original {flag}")
    index = argv.index(flag)
    if index + 1 == len(argv) or argv[index + 1].startswith("--"):
        raise ValueError(f"Expected a value for {flag}")
    argv[index + 1] = str(value)


def prepare(manifest, *, driver, output, probe_dir):
    """Return driver argv and Ray runtime environment without starting either."""
    if manifest.get("image") != IMAGE or manifest.get("olmo_miles_source") != SOURCE:
        raise ValueError("Use the frozen r3 image and source manifest")
    output = Path(output)
    if not output.is_absolute() or output == Path(ORIGINAL_OUTPUT) or Path(ORIGINAL_OUTPUT) in output.parents:
        raise ValueError("Diagnostic output must be an absolute directory outside the original r3 outputs")
    driver, probe_dir = Path(driver), Path(probe_dir)
    if not driver.is_absolute() or not probe_dir.is_absolute() or driver.parent != probe_dir:
        raise ValueError("The diagnostic driver must be directly inside its absolute probe directory")
    argv = list(manifest["argv"])
    expected_prefix = ["/usr/bin/python", "-m", "olmo_miles.compat.miles_entrypoint", "/root/miles/train.py"]
    if argv[:4] != expected_prefix:
        raise ValueError("The original command is not the supported MILES compatibility entrypoint")
    if "--check-weight-update-equal" not in argv or "--no-load-optim" not in argv or "--no-load-rng" not in argv:
        raise ValueError("Original initial-weight load and full comparison guards are required")
    argv[3] = str(driver)
    replace_value(argv, "--num-rollout", "0")
    replace_value(argv, "--save-debug-rollout-data", output / "rollout_data/{rollout_id}.pt")
    replace_value(argv, "--save-debug-train-data", output / "train_data/{rollout_id}_{rank}.pt")
    replace_value(argv, "--wandb-dir", output / "wandb")
    replace_value(argv, "--wandb-mode", "disabled")
    if argv.count("--use-wandb") != 1:
        raise ValueError("Expected the original tracking switch")
    argv.remove("--use-wandb")
    # Native Megatron rejects a zero LR horizon even when no optimizer update is requested.
    # Preserve the original 100-update constant-LR schedule; the driver still performs zero updates.
    if "--lr-decay-iters" in argv:
        raise ValueError("Expected the original implicit scheduler horizon")
    argv.extend(["--lr-decay-iters", "100"])
    runtime_env = copy.deepcopy(manifest["runtime_env"])
    if runtime_env.get("worker_process_setup_hook") != "olmo_miles.compat.sglang_worker_setup.setup_worker":
        raise ValueError("The original Megatron/SGLang worker setup hook is required")
    env = runtime_env["env_vars"]
    env["PYTHONPATH"] = str(probe_dir) + ":" + env["PYTHONPATH"]
    env["WANDB_MODE"] = "disabled"
    env["OLMO_MILES_PROCESS_ATTEMPT_ID"] = "update-zero-megatron"
    return {
        "image": IMAGE,
        "olmo_miles_source": SOURCE,
        "olmo_megatron_source": manifest["olmo_megatron_source"],
        "command": argv,
        "bootstrap_script": f"export OLMO_MILES_EXPECTED_REVISION={SOURCE}\n" + manifest["bootstrap_script"],
        "bootstrap_requires": "GITHUB_TOKEN from original Beaker secret; git; network to github.com",
        "runtime_env": runtime_env,
        "output": str(output),
        "num_optimizer_updates": 0,
        "added_arguments": ["--lr-decay-iters", "100"],
        "scheduler_horizon": "Original100 horizon retained for native scheduler construction only",
        "scope": "Actual trainer initialization and initial publication only; common diagnostic driver must not train",
        "changed_flags": [
            "--num-rollout",
            "--save-debug-rollout-data",
            "--save-debug-train-data",
            "--wandb-dir",
            "--wandb-mode",
            "--use-wandb",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-dir", required=True)
    args = parser.parse_args()
    result = prepare(
        json.loads(args.manifest.read_text()), driver=args.driver, output=args.output, probe_dir=args.probe_dir
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
