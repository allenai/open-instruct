"""Unit tests for grpo_fast_genvalue helpers (no GPU required)."""
import math

import pytest

from open_instruct.grpo_fast_genvalue import GenValueExperimentConfig, segment_rollout


# ── segment_rollout ────────────────────────────────────────────────────────────

def test_segment_rollout_fixed_basic():
    tokens = list(range(10))
    result = segment_rollout(tokens, None, mode="fixed", fixed_chunk_size=3)
    # boundaries at 3, 6, and the final token (9)
    assert result == [3, 6, 9]


def test_segment_rollout_fixed_exact_multiple():
    tokens = list(range(6))
    result = segment_rollout(tokens, None, mode="fixed", fixed_chunk_size=3)
    assert result == [3, 5]


def test_segment_rollout_fixed_terminal_appended():
    tokens = list(range(5))
    result = segment_rollout(tokens, None, mode="fixed", fixed_chunk_size=10)
    assert result == [4], "should append terminal index when no boundary falls before it"


def test_segment_rollout_empty():
    assert segment_rollout([], None, mode="fixed") == []
    assert segment_rollout([], None, mode="sae", sae_threshold=0.5) == []


def test_segment_rollout_sae_basic():
    tokens = [0, 1, 2, 3, 4]
    # logprob < log(0.5) ≈ -0.693 → tokens 1 and 3 are boundaries
    logprobs = [0.0, -1.0, -0.1, -1.5, -0.2]
    result = segment_rollout(tokens, logprobs, mode="sae", sae_threshold=0.5)
    assert 1 in result
    assert 3 in result
    assert result[-1] == 4, "terminal should always be included"


def test_segment_rollout_sae_no_low_prob():
    tokens = [0, 1, 2]
    logprobs = [0.0, -0.1, -0.2]  # all above threshold
    result = segment_rollout(tokens, logprobs, mode="sae", sae_threshold=0.01)
    # all probs > 0.01, so only terminal boundary
    assert result == [2]


def test_segment_rollout_sae_missing_logprobs():
    with pytest.raises(ValueError, match="SAE segmentation requires response_logprobs"):
        segment_rollout([1, 2, 3], None, mode="sae")


def test_segment_rollout_terminal_not_duplicated():
    tokens = list(range(3))
    logprobs = [0.0, -1.0, -1.0]  # last two are boundaries
    result = segment_rollout(tokens, logprobs, mode="sae", sae_threshold=0.5)
    # should not have 2 appearing twice
    assert result.count(2) == 1


def test_segment_rollout_fixed_sorted():
    tokens = list(range(20))
    result = segment_rollout(tokens, None, mode="fixed", fixed_chunk_size=4)
    assert result == sorted(result)


# ── GenValueExperimentConfig validation ───────────────────────────────────────

def _base_kwargs():
    """Minimal valid kwargs for GenValueExperimentConfig (all fields have defaults)."""
    return dict(
        use_generative_value_model=True,
        gen_value_segmentation="fixed",
        gen_value_chunk_size=256,
        gen_value_score_min=0.0,
        gen_value_score_max=10.0,
        gen_value_conditioning="none",
    )


def test_genvalue_config_valid():
    kwargs = _base_kwargs()
    cfg = GenValueExperimentConfig(**kwargs)
    assert cfg.use_generative_value_model is True
    assert cfg.gen_value_segmentation == "fixed"


def test_genvalue_config_requires_flag():
    kwargs = _base_kwargs()
    kwargs["use_generative_value_model"] = False
    with pytest.raises(ValueError, match="requires --use_generative_value_model"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_bad_segmentation():
    kwargs = _base_kwargs()
    kwargs["gen_value_segmentation"] = "wavelet"
    with pytest.raises(ValueError, match="must be 'sae' or 'fixed'"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_sae_requires_use_sae():
    kwargs = _base_kwargs()
    kwargs["gen_value_segmentation"] = "sae"
    kwargs["use_sae"] = False
    with pytest.raises(ValueError, match="requires --use_sae"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_sae_with_use_sae():
    kwargs = _base_kwargs()
    kwargs["gen_value_segmentation"] = "sae"
    kwargs["use_sae"] = True
    kwargs["use_value_model"] = True  # use_sae requires use_value_model
    cfg = GenValueExperimentConfig(**kwargs)
    assert cfg.gen_value_segmentation == "sae"


def test_genvalue_config_bad_chunk_size():
    kwargs = _base_kwargs()
    kwargs["gen_value_chunk_size"] = 0
    with pytest.raises(ValueError, match="must be > 0"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_bad_score_range():
    kwargs = _base_kwargs()
    kwargs["gen_value_score_max"] = kwargs["gen_value_score_min"]
    with pytest.raises(ValueError, match="score_max must be greater"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_bad_conditioning():
    kwargs = _base_kwargs()
    kwargs["gen_value_conditioning"] = "oracle"
    with pytest.raises(ValueError, match="must be one of"):
        GenValueExperimentConfig(**kwargs)


def test_genvalue_config_valid_conditionings():
    for cond in ("none", "gt", "correct_demo", "rollout_context"):
        kwargs = _base_kwargs()
        kwargs["gen_value_conditioning"] = cond
        cfg = GenValueExperimentConfig(**kwargs)
        assert cfg.gen_value_conditioning == cond


# ── _build_sample_scoring_prompts (pure-Python, no GPU) ───────────────────────

def test_build_sample_scoring_prompts_length():
    from unittest.mock import MagicMock
    from open_instruct.grpo_fast_genvalue import _build_sample_scoring_prompts
    from open_instruct.dataset_transformation import INPUT_IDS_PROMPT_KEY

    # Build a tiny fake dataset
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "What is 2+2?"
    dataset = [
        {INPUT_IDS_PROMPT_KEY: [1, 2, 3], "ground_truth": "4"},
        {INPUT_IDS_PROMPT_KEY: [4, 5, 6], "ground_truth": "5"},
        {INPUT_IDS_PROMPT_KEY: [7, 8, 9], "ground_truth": "6"},
    ]

    kwargs = _base_kwargs()
    cfg = GenValueExperimentConfig(**kwargs)

    prompts = _build_sample_scoring_prompts(cfg, tokenizer, dataset, n=2, ground_truths_key="ground_truth")
    assert len(prompts) == 2
    for p in prompts:
        assert isinstance(p, str)
        assert len(p) > 0
