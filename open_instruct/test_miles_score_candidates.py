"""Score extraction options must survive RunSpec compilation without changing sampling."""

import pytest

from open_instruct.miles.configuration import options
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError


def compile_spec(tmp_path, miles, asynchronous=False):
    return RunSpec.from_dict(
        {
            "schema_version": 1,
            "name": "score-candidates",
            "model": {"source": "model", "format": "hf"},
            "output": {"root": str(tmp_path / "run")},
            "data": {"tasks": [{"task": "gsm8k", "train_count": 32, "eval_count": 16}]},
            "async": {"fully_async": asynchronous},
            "miles": {"use_miles_router": True, **miles},
        },
        config_path=tmp_path / "run.toml",
    ).compile()


@pytest.mark.parametrize("k", [32, 64, 128])
@pytest.mark.parametrize("mode,transport", [("topk", "nested"), ("topk", "flat"), ("topk", "array"), ("mc", "array")])
def test_candidate_options_compile(tmp_path, k, mode, transport):
    requested = {"rollout_top_logprobs_num": k, "rollout_score_mode": mode, "rollout_score_transport": transport}
    config = compile_spec(tmp_path, requested)
    for key, value in requested.items():
        assert config.miles[key] == value
        encoded = options.encode_options({key: value})
        argv = config.arguments()
        assert argv[argv.index(encoded[0]) + 1] == encoded[1]
    assert config.miles.get("rollout_top_k", -1) == -1


@pytest.mark.parametrize("transport", ["nested", "flat", "array"])
@pytest.mark.parametrize("correction", ["none", "tis", "mis"])
def test_sct_loss_options_forwarded(tmp_path, transport, correction):
    requested = {
        "loss_type": "score_centering",
        "rollout_top_logprobs_num": 32,
        "rollout_score_transport": transport,
        "score_centering_is": correction,
        "score_centering_tis_clip": 2.5,
        "score_centering_mis_low": 0.4,
        "score_centering_mis_high": 4.0,
    }
    config = compile_spec(tmp_path, requested)
    for key, value in requested.items():
        assert config.miles[key] == value
    assert not config.miles["use_tis"]


@pytest.mark.parametrize("correction", [None, "tis", "mis"])
def test_async_score_centering_uses_its_own_correction(tmp_path, correction):
    requested = {"loss_type": "score_centering", "rollout_top_logprobs_num": 32}
    if correction is not None:
        requested["score_centering_is"] = correction
    config = compile_spec(tmp_path, requested, asynchronous=True)
    assert not config.miles["use_tis"]
    assert config.miles["score_centering_is"] == (correction or "tis")


@pytest.mark.parametrize(
    "requested",
    [
        {"rollout_score_transport": "array"},
        {"rollout_score_mode": "mc", "rollout_top_logprobs_num": 32},
        {"rollout_score_mode": "mc", "rollout_top_logprobs_num": 16, "rollout_score_transport": "array"},
        {"rollout_top_logprobs_num": 32, "rollout_score_transport": "array", "rollout_top_k": 32},
        {"loss_type": "score_centering"},
        {"loss_type": "score_centering", "rollout_top_logprobs_num": 32, "use_tis": True},
    ],
)
def test_invalid_collection_rejected_before_launch(tmp_path, requested):
    with pytest.raises(InputError):
        compile_spec(tmp_path, requested)


@pytest.mark.parametrize("transport", ["flat", "array"])
def test_packed_scores_require_transparent_router(tmp_path, transport):
    with pytest.raises(InputError, match="use_miles_router"):
        compile_spec(
            tmp_path, {"rollout_top_logprobs_num": 32, "rollout_score_transport": transport, "use_miles_router": False}
        )
