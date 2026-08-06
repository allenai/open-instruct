"""Checks that launch scripts only pass flags their entrypoint's parser accepts.

Launch scripts under scripts/train/ tend to outlive config refactors (e.g.
--gradient_checkpointing was replaced by --activation_memory_budget), and a
stale flag makes HfArgumentParser raise at job startup. This test extracts the
flags each committed script passes to a known entrypoint and validates them
against that entrypoint's actual ArgumentParserPlus.
"""

import re
from pathlib import Path

import pytest

from open_instruct import dpo_utils
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.utils import ArgumentParserPlus

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts" / "train"

ENTRYPOINT_DATACLASSES = {
    "open_instruct/dpo_tune_cache.py": (dpo_utils.DPOExperimentConfig, TokenizerConfig),
    "open_instruct/dpo.py": (dpo_utils.DPOExperimentConfig, TokenizerConfig),
}

FLAG_RE = re.compile(r"(?<=\s)--([a-zA-Z_][a-zA-Z0-9_]*)")
ARRAY_EXPANSION_RE = re.compile(r"\$\{(\w+)\[@\]\}")


def extract_command_text(text: str, start: int) -> str:
    """Return the shell command starting at `start`, following backslash line continuations."""
    command_lines = []
    for line in text[start:].split("\n"):
        command_lines.append(line)
        if not line.rstrip().endswith("\\"):
            break
    return "\n".join(command_lines)


def resolve_array_expansions(text: str, command: str) -> str:
    """Append the contents of bash arrays expanded inside `command` (e.g. "${COMMON_ARGS[@]}")."""
    for name in ARRAY_EXPANSION_RE.findall(command):
        definition = re.search(rf"^{name}=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)
        if definition:
            command += "\n" + definition.group(1)
    return command


def iter_entrypoint_flags(script: Path):
    text = script.read_text()
    for entrypoint in ENTRYPOINT_DATACLASSES:
        for match in re.finditer(re.escape(entrypoint) + r"\b", text):
            command = resolve_array_expansions(text, extract_command_text(text, match.end()))
            yield entrypoint, ["--" + name for name in FLAG_RE.findall(command)]


def scripts_with_known_entrypoints() -> list[Path]:
    return [
        script
        for script in sorted(SCRIPTS_DIR.rglob("*.sh"))
        if any(entrypoint in script.read_text() for entrypoint in ENTRYPOINT_DATACLASSES)
    ]


@pytest.mark.parametrize(
    "script", scripts_with_known_entrypoints(), ids=lambda script: str(script.relative_to(REPO_ROOT))
)
def test_launch_script_flags_parse(script: Path):
    valid_flags_cache: dict[str, set[str]] = {}
    for entrypoint, flags in iter_entrypoint_flags(script):
        if entrypoint not in valid_flags_cache:
            parser = ArgumentParserPlus(ENTRYPOINT_DATACLASSES[entrypoint])
            valid_flags_cache[entrypoint] = set(parser._option_string_actions)
        unknown = [flag for flag in flags if flag not in valid_flags_cache[entrypoint]]
        assert not unknown, (
            f"{script.relative_to(REPO_ROOT)} passes flags {unknown} that {entrypoint} does not accept. "
            f"The entrypoint's config classes have likely changed since this script was written."
        )
