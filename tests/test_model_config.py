"""Unit tests for open_instruct.model_utils.ModelConfig.

``ModelConfig`` is parsed straight onto the command line by ``ArgumentParserPlus``
(see ``reward_modeling.py``), so every field on it becomes a user-visible ``--flag``.
A field that nothing reads is therefore worse than dead code: the flag is accepted,
it is logged to wandb via ``asdict(model_config)``, and it silently does nothing.

That is exactly what happened to the PEFT/LoRA and bitsandbytes blocks -- ``--use_peft``
and ``--load_in_4bit`` parsed fine on ``reward_modeling.py`` while the run did plain
full finetuning. These tests keep that from coming back.
"""

import ast
import dataclasses
import pathlib
import re
import unittest

from open_instruct.model_utils import ModelConfig

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Fields that are intentionally never read via attribute access. Add an entry here
# (with a reason) rather than deleting the check if a field is genuinely write-only.
ALLOWED_UNREAD_FIELDS: dict[str, str] = {}


def _python_sources() -> list[pathlib.Path]:
    """Every tracked Python file, skipping virtualenvs and vendored trees."""
    skip = (".venv", "venv", "oe-eval-internal", "site-packages", "build", ".git")
    return [
        path
        for path in REPO_ROOT.rglob("*.py")
        if not any(part in skip for part in path.parts)
    ]


def _attribute_reads(field_name: str, blob: str) -> int:
    """Count ``.field_name`` attribute accesses.

    Matching on the leading dot is what makes this meaningful: several scripts declare
    their *own* ``lora_alpha``/``lora_dropout`` on ``FlatArguments`` and read them as
    ``args.lora_alpha``. A bare-name search would count those and mask a dead field, so
    we deliberately accept that this also counts reads off unrelated objects -- it makes
    the check conservative (it can only under-report deadness, never over-report it).
    """
    return len(re.findall(rf"\.{re.escape(field_name)}\b", blob))


class TestModelConfigFieldsAreUsed(unittest.TestCase):
    def setUp(self) -> None:
        sources = _python_sources()
        self.assertGreater(len(sources), 20, "source discovery looks broken")
        self.blob = "\n".join(p.read_text(errors="ignore") for p in sources)

    def test_every_field_is_read_somewhere(self) -> None:
        dead = []
        for field in dataclasses.fields(ModelConfig):
            if field.name in ALLOWED_UNREAD_FIELDS:
                continue
            if _attribute_reads(field.name, self.blob) == 0:
                dead.append(field.name)

        self.assertEqual(
            dead,
            [],
            "ModelConfig fields are exposed as CLI flags, so a field nothing reads is a "
            "flag that silently does nothing. Either wire these up or remove them: "
            f"{dead}",
        )

    def test_detects_a_deliberately_dead_field(self) -> None:
        """Negative control: the check above must actually be able to fail.

        Without this, `test_every_field_is_read_somewhere` would keep passing even if
        `_attribute_reads` were broken into always returning a positive count.
        """
        self.assertEqual(_attribute_reads("field_no_source_file_mentions", self.blob), 0)
        self.assertGreater(_attribute_reads("model_name_or_path", self.blob), 0)

    def test_bare_name_matches_do_not_count_as_reads(self) -> None:
        """A declaration alone must not look like a read."""
        self.assertEqual(_attribute_reads("lora_r", "lora_r: int | None = 16"), 0)
        self.assertEqual(_attribute_reads("lora_r", "peft_config.lora_r"), 1)


class TestModelConfigHasNoUnwiredPeftFlags(unittest.TestCase):
    """Pin the specific regression: PEFT/quantization flags with no implementation.

    ``reward_modeling.py`` is the entrypoint that puts ``ModelConfig`` on the CLI and it
    has no LoRA support, so re-adding these without also adding the code that reads them
    would resurrect the silent no-op.
    """

    REMOVED_FLAGS = (
        "use_peft",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "lora_modules_to_save",
        "lora_task_type",
        "load_in_8bit",
        "load_in_4bit",
        "bnb_4bit_quant_type",
        "use_bnb_nested_quant",
    )

    def test_peft_flags_absent_until_implemented(self) -> None:
        present = {f.name for f in dataclasses.fields(ModelConfig)}
        resurrected = sorted(present.intersection(self.REMOVED_FLAGS))
        self.assertEqual(
            resurrected,
            [],
            "These flags were removed because nothing consumed them. If you are adding "
            "real PEFT support, also add the code that reads them (and drop them from "
            f"this list): {resurrected}",
        )

    def test_reward_modeling_still_exposes_model_config(self) -> None:
        """Guard the premise: if this stops being true, the test above loses its point."""
        source = (REPO_ROOT / "open_instruct" / "reward_modeling.py").read_text()
        tree = ast.parse(source)
        exposes = any(
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "ArgumentParserPlus"
            and "ModelConfig" in ast.unparse(node)
            for node in ast.walk(tree)
        )
        self.assertTrue(
            exposes,
            "reward_modeling.py no longer parses ModelConfig from the CLI; revisit "
            "whether these tests still describe the right invariant.",
        )


if __name__ == "__main__":
    unittest.main()
