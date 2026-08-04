"""Check every --flag in a launcher script against grpo_fast's dataclasses.

A wrong flag name only shows up when the job reaches the front of the queue and
argparse rejects it, which on a busy partition can cost an hour to learn. Usage:

    python projects/pedagogy_rm/scripts/check_flags.py projects/pedagogy_rm/scripts/train.sh

Comment lines are dropped before scanning, which is what keeps #SBATCH directives and
prose mentions of Slurm options out of the comparison. The flags themselves are read
from the whole of what is left rather than from the launch command alone: the mode
branches build their flags into an array well above the python call, and those are the
ones most likely to be wrong.
"""

import dataclasses
import re
import sys

from open_instruct import grpo_fast


def known_fields() -> set[str]:
    configs = [
        grpo_fast.grpo_utils.GRPOExperimentConfig,
        grpo_fast.TokenizerConfig,
        grpo_fast.ModelConfig,
        grpo_fast.data_loader_lib.StreamingDataLoaderConfig,
        grpo_fast.data_loader_lib.VLLMConfig,
        grpo_fast.EnvsConfig,
    ]
    return {field.name for config in configs for field in dataclasses.fields(config)}


def flags_in(path: str) -> set[str]:
    with open(path) as handle:
        code = [line for line in handle if not line.lstrip().startswith("#")]
    return set(re.findall(r"(?<!-)--([a-zA-Z][a-zA-Z0-9_]*)", "".join(code)))


def main() -> None:
    fields = known_fields()
    unknown_total = 0
    for path in sys.argv[1:]:
        used = flags_in(path)
        unknown = sorted(flag for flag in used if flag not in fields)
        print(f"{path}: {len(used)} flags")
        for flag in unknown:
            print(f"  UNKNOWN  --{flag}")
        print("  all accepted" if not unknown else "")
        unknown_total += len(unknown)
    if unknown_total:
        raise SystemExit(f"{unknown_total} flag(s) no dataclass accepts")


if __name__ == "__main__":
    main()
