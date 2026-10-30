"""MILES bindings for ToolRL's offline reference-call reward."""

import dataclasses

from open_instruct import toolrl_utils


@dataclasses.dataclass
class ToolRLConfig:
    think_prefix_in_prompt: bool = False


@dataclasses.dataclass
class ToolRLResult:
    score: float
    cost: float = 0.0
    diagnostics: dict = dataclasses.field(default_factory=dict)


class ToolRLVerifier:
    def __init__(self, verifier_config=None):
        self.config = verifier_config or ToolRLConfig()

    @classmethod
    def get_config_class(cls):
        return ToolRLConfig

    async def async_call(self, tokens, prediction, label, **kwargs):
        result = toolrl_utils.score_response(
            prediction, label, think_prefix_in_prompt=self.config.think_prefix_in_prompt
        )
        return ToolRLResult(
            result.score,
            diagnostics={
                "kind": "toolrl",
                "status": "completed",
                "format_score": result.format_score,
                "correctness_score": result.correctness_score,
                "reference_has_calls": bool(toolrl_utils.validate_target(label)),
            },
        )


@dataclasses.dataclass
class ToolRLThinkPrefixConfig(ToolRLConfig):
    think_prefix_in_prompt: bool = True


class ToolRLThinkPrefixVerifier(ToolRLVerifier):
    @classmethod
    def get_config_class(cls):
        return ToolRLThinkPrefixConfig

    def __init__(self, verifier_config=None):
        super().__init__(verifier_config or ToolRLThinkPrefixConfig())
