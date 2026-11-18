"""MILES bindings for ToolRL's offline reference-call reward.

ToolRL's published reward runs from -3 to 4 on rows that expect calls and from 0
to 1 on rows that expect none. MILES rewards are otherwise in [0, 1]. Without
``grpo_std_normalization``, a group's advantages scale with its reward range, so
this binding maps both row kinds onto [0, 1] by default. With group std
normalization, this affine rescaling leaves advantages unchanged apart from
numerical stabilization. Within one prompt group this is a positive rescaling:
it keeps the ordering and relative spacing of the responses. ``unit_scale=False`` keeps the
published scale; ``score_floor`` then tells the reward gates that the worst score
on a call row is -3, not 0.
"""

import dataclasses

from open_instruct import toolrl_utils


@dataclasses.dataclass
class ToolRLConfig:
    think_prefix_in_prompt: bool = False
    all_calls_required: bool = False
    unit_scale: bool = True


@dataclasses.dataclass
class ToolRLResult:
    score: float
    cost: float = 0.0
    diagnostics: dict = dataclasses.field(default_factory=dict)


class ToolRLVerifier:
    grading_objective = "composite"
    grading_scopes = ("response",)
    grading_components = {
        "correctness": {"objective": "answer", "scope": "response"},
        "format": {"objective": "format", "scope": "full"},
    }
    grading_requires_reasoning = True

    def __init__(self, verifier_config=None):
        self.config = verifier_config or ToolRLConfig()

    @classmethod
    def get_config_class(cls):
        return ToolRLConfig

    def score_floor(self, label):
        """The lowest score this verifier can return for a target."""
        return 0.0 if self.config.unit_scale or not toolrl_utils.validate_target(label) else -3.0

    async def async_call(self, tokens, prediction, label, **kwargs):
        return self._grade(
            prediction,
            label,
            think_prefix_in_prompt=label.get("think_prefix_in_prompt", self.config.think_prefix_in_prompt),
        )

    async def async_call_parts(self, tokens, parts, label, *, scope, component_inputs, **kwargs):
        # Format examines the original boundary; correctness sees the same
        # delivered action selected for other graders. The shared dispatcher
        # resolves each declared sub-objective; this method never splits text.
        return self._grade(
            component_inputs["format"],
            label,
            think_prefix_in_prompt=parts.prompt_opens_reasoning,
            action=component_inputs["correctness"],
        )

    def _grade(self, prediction, label, *, think_prefix_in_prompt, action=None):
        result = toolrl_utils.score_response(
            prediction, label, think_prefix_in_prompt=think_prefix_in_prompt, action=action
        )
        has_calls = bool(toolrl_utils.validate_target(label))
        full_success = has_calls and result.score == 4.0
        # Diagnostics keep the published [-3, 4] range; no-call format grading is unchanged.
        binary_score = (4.0 if full_success else -3.0) if has_calls else result.score
        score = binary_score if self.config.all_calls_required else result.score
        if self.config.unit_scale and has_calls:
            score = (score + 3.0) / 7.0
        return ToolRLResult(
            score,
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
                "unit_scale": self.config.unit_scale,
            },
        )
