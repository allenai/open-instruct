"""MILES bindings for ToolRL's offline reference-call reward."""

import dataclasses

from open_instruct import toolrl_utils


@dataclasses.dataclass
class ToolRLConfig:
    think_prefix_in_prompt: bool = False
    all_calls_required: bool = False


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
        has_calls = bool(toolrl_utils.validate_target(label))
        full_success = has_calls and result.score == 4.0
        # Keep the original [-3, 4] range; no-call format grading is unchanged.
        binary_score = (4.0 if full_success else -3.0) if has_calls else result.score
        return ToolRLResult(
            binary_score if self.config.all_calls_required else result.score,
            diagnostics={
                "kind": "toolrl",
                "status": "completed",
                "format_score": result.format_score,
                "correctness_score": result.correctness_score,
                "reference_has_calls": has_calls,
                "required_calls": len(label["calls"]),
                "full_success": full_success,
                "partial_score": result.score,
                "binary_score": binary_score,
                "all_calls_required": self.config.all_calls_required,
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
