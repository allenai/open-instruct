"""Install a template in an HF export without reserializing its tokenizer or weights.

Existing exports can be updated with ``python -m open_instruct.export_chat_template
--checkpoint-dir PATH --export-chat-template TEMPLATE.jinja``.
"""

import argparse
import json
import pathlib
import tempfile

from transformers import AutoTokenizer
from transformers.utils import chat_template_utils

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def read_export_chat_template(path: str | pathlib.Path | None) -> str | None:
    """Read before converting weights so a missing or empty template fails early."""
    if path is None:
        return None
    template = pathlib.Path(path).read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"Export chat template is empty: {path}")
    # Match apply_chat_template's environment, including generation blocks.
    # This Transformers-internal compiler is also exercised by our render tests.
    chat_template_utils._compile_jinja_template(template)
    return template


def read_special_token_state(tokenizer_dir: str | pathlib.Path) -> dict:
    """Snapshot loaded special-token roles and IDs, including metadata overrides."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    return {
        name: (tokens, tokenizer.convert_tokens_to_ids(tokens))
        for name, tokens in tokenizer.special_tokens_map.items()
    }


def _replace_text(path: pathlib.Path, text: str) -> None:
    # Replace the directory entry rather than writing through a symlink/hardlink
    # into the original SFT export when preparing an RL sibling.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        try:
            handle.write(text)
            handle.close()
            temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def install_export_chat_template(checkpoint_dir: str | pathlib.Path, template: str | None) -> None:
    """Replace the single chat template; None leaves every export file untouched.

    Only chat_template.jinja and an existing embedded tokenizer_config.json template
    are changed. In particular, never load/save a tokenizer: doing so can change its
    pre/post-processing under a different Transformers version.
    """
    if template is None:
        return
    if not template.strip():
        raise ValueError("Export chat template is empty")
    chat_template_utils._compile_jinja_template(template)
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    config_path = checkpoint_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object in {config_path}")
    if config.get("chat_template") is not None and not isinstance(config["chat_template"], str):
        raise ValueError("Template-only export requires a single template; found embedded named templates")
    if any((checkpoint_dir / "additional_chat_templates").glob("*.jinja")):
        raise ValueError("Template-only export requires a single template; found additional_chat_templates/*.jinja")

    # Read and validate metadata before modifying either file.
    if "chat_template" in config:
        config["chat_template"] = template
        _replace_text(config_path, json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    _replace_text(checkpoint_dir / "chat_template.jinja", template)
    logger.info("Installed export chat template in %s", checkpoint_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=pathlib.Path)
    parser.add_argument("--export-chat-template", required=True, type=pathlib.Path)
    args = parser.parse_args()
    install_export_chat_template(args.checkpoint_dir, read_export_chat_template(args.export_chat_template))


if __name__ == "__main__":
    main()
