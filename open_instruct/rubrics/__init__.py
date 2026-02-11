"""
Evolving rubrics utilities for GRPO training.

Environment Variables:
    RUBRIC_JUDGE_MODEL: LLM model for scoring responses against rubrics.
        Default: "gpt-4.1"
    RUBRIC_GENERATION_MODEL: LLM model for generating new rubrics dynamically.
        Default: "gpt-4.1"
    LITELLM_MAX_CONCURRENT_CALLS: Maximum concurrent LiteLLM API calls.
        Default: "256"
    LITELLM_DEFAULT_TIMEOUT: Default timeout for LiteLLM calls in seconds.
        Default: "600"
"""

from open_instruct.rubrics.prompts import INSTANCE_WISE_RUBRIC_GENERATION_PROMPT, RUBRIC_SCORING_PROMPT

__all__ = ["RUBRIC_SCORING_PROMPT", "INSTANCE_WISE_RUBRIC_GENERATION_PROMPT"]
