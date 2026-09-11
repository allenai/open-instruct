"""Launch immutable preparation, Core500, or paired CPU audit on Beaker."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from scripts.miles import extended_gsm8k, launch_gsm8k_parity


def specification(image, stage):
    base_stage = "core" if stage == "core" else "prepare"
    document = launch_gsm8k_parity.specification(image, base_stage)
    task = document["tasks"][0]
    task["name"] = "gsm8k-500-" + stage
    document["description"] = "Fresh SFT GSM8K Core/Megatron500: same400 ordered prompts, five passes,128 held-out"
    command = task["arguments"][0].replace(launch_gsm8k_parity.ROOT, str(extended_gsm8k.ROOT))
    options = "--updates 500 --eval-interval 20"
    if stage == "prepare":
        command = command.replace(
            'python scripts/miles/prepare_gsm8k_parity.py "$RUN_ROOT"', "python scripts/miles/extended_gsm8k.py"
        )
        command = command.replace(
            "--validate-only",
            options
            + " --save-interval 100 --chunked-prefill-size 16384 --campaign "
            + extended_gsm8k.CAMPAIGN
            + " --validate-only",
        )
        command += 'cp "$RUN_ROOT/extension.json" /output/\n'
    elif stage == "core":
        command = command.replace(
            'python scripts/miles/gsm8k_parity.py "$RUN_ROOT"',
            'python scripts/miles/gsm8k_parity.py "$RUN_ROOT" '
            + options
            + " --save-interval 100 --chunked-prefill-size 16384 --campaign "
            + extended_gsm8k.CAMPAIGN,
        )
        task["timeout"] = "18h"
        task["context"]["minRuntime"] = "12h"
    elif stage == "audit":
        command = command[: command.index("python scripts/miles/prepare_gsm8k_parity.py")]
        command += (
            "\n".join(
                [
                    f'python scripts/miles/analyze_gsm8k_parity.py audit "$RUN_ROOT" --backend {backend} {options}'
                    for backend in ("core", "megatron")
                ]
            )
            + "\n"
        )
        command += f'python scripts/miles/analyze_gsm8k_parity.py compare "$RUN_ROOT" {options}\n'
        command += 'cp "$RUN_ROOT/core/audit.json" /output/core-audit.json\ncp "$RUN_ROOT/megatron/audit.json" /output/megatron-audit.json\ncp "$RUN_ROOT/comparison.json" /output/\n'
    else:
        raise ValueError("Unknown extension stage")
    task["arguments"] = [command]
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--stage", choices=("prepare", "core", "audit"), required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    document = json.dumps(specification(args.image, args.stage), indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="miles-gsm8k-500-") as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
