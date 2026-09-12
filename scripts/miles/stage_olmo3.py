"""Stage a local Olmo 3 snapshot with the original RL chat template; never alter the source."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from open_instruct.miles import workflow
from open_instruct.miles.errors import InputError

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs/miles/templates/olmo-thinker.jinja"


def stage(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists():
        raise InputError(f"Staging output already exists: {output}; choose a new path")
    if output.resolve().is_relative_to(source):
        raise InputError("Staging output must be outside the original snapshot")
    identity = workflow.model_identity(source)
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "olmo3" or not list(source.glob("*.safetensors")):
        raise InputError("Expected an Olmo 3 HF snapshot with safetensors weights")
    template = TEMPLATE.read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for path in source.iterdir():
            target = temporary / path.name
            if path.is_file() and path.suffix == ".safetensors":
                target.symlink_to(path.resolve())
            elif path.name.startswith("."):
                continue
            elif path.is_dir():
                shutil.copytree(path, target)
            else:
                shutil.copy2(path, target)
        tokenizer_path = temporary / "tokenizer_config.json"
        tokenizer_config = json.loads(tokenizer_path.read_text())
        tokenizer_config["chat_template"] = template
        tokenizer_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n")
        (temporary / "chat_template.jinja").write_text(template)
        if workflow.model_identity(source) != identity:
            raise InputError("Source checkpoint changed during staging")
        workflow.write_json(
            temporary / "olmo3-staging.json",
            {
                "source": identity,
                "template": "olmo_thinker",
                "template_identity": workflow.fingerprint(template),
                "note": "Weights reference the local source; retain it. Download a pinned HF revision before staging.",
            },
        )
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(stage(args.source, args.output))


if __name__ == "__main__":
    main()
