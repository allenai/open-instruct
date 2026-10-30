"""Refresh opener seeds become scored, trained response tokens; unseeded samples are unchanged."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from miles.backends.core_utils.publication import policy_refresh
from miles.backends.core_utils.rollout import refreshing_rollout, response_seed
from miles.backends.core_utils.rollout.errors import GenerationRequestTimeout
from miles.backends.core_utils.rollout.refreshing_rollout import RefreshingRolloutFn
from miles.utils.types import Sample

from open_instruct.miles.configuration import config as run_config
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.rollout.data_source import DashboardDrainingRolloutDataSource

VOCAB = ["<think>", ">\n", " Okay", "Okay", "Q", ">", "\n", " ", "!"]


class Tokenizer:
    """Greedy longest-match tokenizer; ``>\\n`` merges like a byte-level BPE."""

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            token = max((t for t in VOCAB if text.startswith(t)), key=len)
            ids.append(VOCAB.index(token))
            text = text[len(token) :]
        return ids


def ids(*tokens):
    return [VOCAB.index(t) for t in tokens]


def producer():
    p = RefreshingRolloutFn.__new__(RefreshingRolloutFn)
    p.args = SimpleNamespace(
        olmo_core=SimpleNamespace(refresh_request_timeout=5),
        sglang_router_ip="localhost",
        sglang_router_port=1234,
        rollout_max_response_len=8,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        rollout_top_logprobs_num=0,
        sglang_speculative_algorithm=None,
    )
    return p


def request(p, prompt="Q<think>", seed=None):
    value = Sample(group_index=0, index=0, prompt=prompt)
    if seed is not None:
        value.metadata = {response_seed.REQUEST_KEY: {"text": seed}}
    state = SimpleNamespace(tokenizer=Tokenizer(), processor=None)
    return SimpleNamespace(args=p.args, state=state, sample=value, sampling_params={"max_new_tokens": 8})


def serve(monkeypatch, posts, *, seed_version="3", spans=(("3", 0, 1), ("4", 1, 2))):
    """Fake SGLang: prefill-only scoring and a two-token continuation across one publication."""

    async def post(url, payload, **kwargs):
        posts.append(copy.deepcopy(payload))
        if payload["sampling_params"]["max_new_tokens"] == 0:
            tail = payload["input_ids"][payload["logprob_start_len"] :]
            items = [[None, tail[0], None]] + [[-0.25 * (i + 1), t, None] for i, t in enumerate(tail[1:])]
            return {"text": "", "meta_info": {"input_token_logprobs": items, "weight_version": seed_version}}
        output = {
            "text": "!!",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.5, VOCAB.index("!"), None], [-0.6, VOCAB.index("!"), None]],
                "weight_version": spans[-1][0],
                "weight_versions": [dict(version=v, start=s, end=e) for v, s, e in spans],
            },
        }
        return json.dumps(output).encode() if kwargs.get("return_bytes") else output

    monkeypatch.setattr(refreshing_rollout, "post", post)


def test_unseeded_sample_sends_one_unchanged_request(monkeypatch):
    posts = []
    serve(monkeypatch, posts, spans=(("4", 0, 2),))
    p = producer()
    value = asyncio.run(p._generate_response_impl(request(p))).samples
    assert len(posts) == 1
    assert posts[0]["input_ids"] == ids("Q", "<think>")
    assert posts[0]["sampling_params"]["max_new_tokens"] == 8
    assert "logprob_start_len" not in posts[0]
    assert value.response == "!!" and value.response_length == 2
    assert value.rollout_log_probs == [-0.5, -0.6]
    assert response_seed.RECORD_KEY not in (value.metadata or {})


def test_seed_is_scored_trained_and_versioned_ahead_of_the_continuation(monkeypatch):
    posts = []
    serve(monkeypatch, posts)
    p = producer()
    value = asyncio.run(p._generate_response_impl(request(p, seed=" Okay"))).samples
    scoring, generation = posts
    assert scoring["input_ids"] == ids("Q", "<think>", " Okay")
    assert scoring["logprob_start_len"] == 1 and scoring["return_logprob"] is True
    assert generation["input_ids"] == ids("Q", "<think>", " Okay")
    assert generation["sampling_params"]["max_new_tokens"] == 7  # the seed is charged to the budget
    assert value.tokens == ids("Q", "<think>", " Okay", "!", "!")
    assert value.response == " Okay!!" and value.response_length == 3
    assert value.rollout_log_probs == [-0.25, -0.5, -0.6]
    value.validate()
    # The seed carries the scoring version; continuation spans shift behind it.
    assert value.metadata["policy_refresh"]["spans"] == [
        dict(version=3, start=0, end=2),
        dict(version=4, start=2, end=3),
    ]
    assert value.metadata["policy_refresh"]["replay_version"] == 4
    assert value.metadata[response_seed.RECORD_KEY] == {
        "token_ids": ids(" Okay"),
        "log_probs": [-0.25],
        "weight_version": "3",
    }
    assert value.metadata[response_seed.REQUEST_KEY] == {"text": " Okay"}


@pytest.mark.parametrize("seed", [None, " Okay"])
def test_stalled_request_is_a_recoverable_generation_timeout(monkeypatch, seed):
    """Seed scoring times out the same way as generation, so the group is requeued."""

    async def stall(url, payload, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(refreshing_rollout, "post", stall)
    p = producer()
    p.args.olmo_core.refresh_request_timeout = 0.01
    with pytest.raises(GenerationRequestTimeout):
        asyncio.run(p._generate_response_impl(request(p, seed=seed)))


def test_seed_that_merges_with_the_prompt_is_rejected_before_any_request(monkeypatch):
    posts = []
    serve(monkeypatch, posts)
    p = producer()
    with pytest.raises(ValueError, match="tokenizes differently after the prompt"):
        asyncio.run(p._generate_response_impl(request(p, prompt="Q>", seed="\n")))
    assert posts == []


@pytest.mark.parametrize("bad", [{"text": ""}, {"text": 3}, {"text": "a", "extra": 1}, "a"])
def test_malformed_seed_request_fails_closed(bad):
    value = Sample(metadata={response_seed.REQUEST_KEY: bad})
    with pytest.raises(ValueError):
        response_seed.requested(value)


def test_seed_must_leave_response_budget():
    with pytest.raises(ValueError, match="no response-token budget"):
        response_seed.remaining_sampling_params(SimpleNamespace(rollout_max_response_len=2), {}, 2)


def test_misaligned_seed_scores_fail_closed():
    output = {"meta_info": {"input_token_logprobs": [[None, 1, None], [-0.1, 9, None]], "weight_version": "1"}}
    with pytest.raises(ValueError, match="not aligned"):
        response_seed.parse_scores(output, [2])


def test_seeded_record_survives_refresh_provenance_checks():
    value = Sample(tokens=ids("Q", " Okay", "!"), response_length=2, rollout_log_probs=[-0.2, -0.5])
    value.metadata = {
        response_seed.RECORD_KEY: {"token_ids": ids(" Okay"), "log_probs": [-0.2], "weight_version": "2"}
    }
    meta = response_seed.merged_meta_info(
        value,
        {
            "weight_version": "2",
            "weight_versions": [dict(version="2", start=0, end=1)],
            "output_token_logprobs": [[-0.5, VOCAB.index("!"), None]],
        },
    )
    record = policy_refresh.record_response(value, meta)
    assert record["spans"] == [dict(version=2, start=0, end=2)]


def data_source(openers=(), count=0):
    source = DashboardDrainingRolloutDataSource.__new__(DashboardDrainingRolloutDataSource)
    source._opener_seeds, source._opener_seeds_per_group = tuple(openers), count
    return source


def groups(n_groups, size):
    return [[Sample(group_index=g, index=g * size + i) for i in range(size)] for g in range(n_groups)]


def test_openers_rotate_across_groups_and_leave_other_slots_unseeded():
    batch = groups(3, 4)
    data_source(("\n", " Okay", "Okay"), 2)._seed_openers(batch)
    texts = [[(s.metadata or {}).get(response_seed.REQUEST_KEY, {}).get("text") for s in g] for g in batch]
    assert texts == [["\n", " Okay", None, None], ["Okay", "\n", None, None], [" Okay", "Okay", None, None]]
    again = groups(3, 4)
    data_source(("\n", " Okay", "Okay"), 2)._seed_openers(again)
    assert [s.metadata for g in again for s in g] == [s.metadata for g in batch for s in g]


def test_disabled_seeding_leaves_samples_untouched():
    batch = groups(2, 3)
    data_source()._seed_openers(batch)
    assert all(s.metadata == {} for g in batch for s in g)


def test_opener_fields_must_be_set_together():
    with pytest.raises(InputError, match="set together"):
        compiled('core.opener_seeds=["Okay"]').arguments()
    with pytest.raises(InputError, match="set together"):
        compiled("core.opener_seeds_per_group=1").arguments()
    with pytest.raises(InputError, match="repeat"):
        run_config.CoreConfig(opener_seeds=["a", "a"], opener_seeds_per_group=1)
    assert run_config.CoreConfig(opener_seeds=["\n", "Okay"], opener_seeds_per_group=1).opener_seeds == ("\n", "Okay")


def compiled(*overrides):
    return RunSpec.load("configs/miles/examples/node.toml", list(overrides)).compile()


def core_argument(config):
    argv = config.arguments()
    return json.loads(argv[argv.index("--olmo-core-config") + 1])


SEEDS = ['core.opener_seeds=["\\n", " Okay"]', "core.opener_seeds_per_group=2"]


def test_disabled_seeding_keeps_the_native_core_argument_unchanged():
    assert not {"opener_seeds", "opener_seeds_per_group"} & set(core_argument(compiled()))


def test_refresh_run_carries_opener_seeds_to_the_runtime():
    core = core_argument(compiled(*SEEDS))
    assert core["opener_seeds"] == ["\n", " Okay"] and core["opener_seeds_per_group"] == 2
    assert run_config.CoreConfig(**core).opener_seeds == ("\n", " Okay")


def test_opener_seeds_require_refresh_publication():
    with pytest.raises(InputError, match="publication_mode"):
        compiled(*SEEDS, 'core.publication_mode="barrier"').arguments()


def test_opener_seeds_must_leave_an_unseeded_sample():
    with pytest.raises(InputError, match="unseeded sample"):
        compiled('core.opener_seeds=["Okay"]', "core.opener_seeds_per_group=4").arguments()
