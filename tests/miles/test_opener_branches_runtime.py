"""Refresh opener branches: per-prompt candidate probe, uniform draw, even split into seeded samples."""

import asyncio
import copy
import json
import math
from types import SimpleNamespace

import pytest
from miles.backends.core_utils.rollout import opener_branches, refreshing_rollout, response_seed
from miles.backends.core_utils.rollout.errors import GenerationRequestTimeout
from miles.backends.core_utils.rollout.refreshing_rollout import RefreshingRolloutFn
from miles.utils.types import Sample

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError

# Token 7 is half of a multi-byte character; 8 completes it. Token 9 is special.
VOCAB = ["Q", "<think>", "\n", " Okay", "Okay", "I", "The", "\xe2", "\x80\x94", "</think>", "!"]
SPECIAL = [VOCAB.index("</think>")]


class Tokenizer:
    all_special_ids = SPECIAL

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            token = max((t for t in VOCAB if text.startswith(t)), key=len)
            ids.append(VOCAB.index(token))
            text = text[len(token) :]
        return ids

    def decode(self, ids):
        raw = "".join(VOCAB[i] for i in ids).encode("latin-1")
        return raw.decode("utf-8", errors="replace")


def ids(*tokens):
    return [VOCAB.index(t) for t in tokens]


PROMPT = ids("Q", "<think>")
# Next-token distributions after the prompt, keyed by the response prefix.
DIST = {
    (): {"\n": 0.5, "Okay": 0.3, " Okay": 0.1, "\xe2": 0.05, "</think>": 0.05},
    ("\n",): {"I": 0.7, "The": 0.3},
    ("\xe2",): {"\x80\x94": 1.0},
}


def serve(monkeypatch, posts, *, stall_probe=False):
    """Fake SGLang: opener probes, prefill-only seed scoring and a one-token continuation."""

    async def post(url, payload, **kwargs):
        posts.append(copy.deepcopy(payload))
        if "top_logprobs_num" in payload:
            if stall_probe:
                await asyncio.sleep(10)
            prefix = tuple(VOCAB[t] for t in payload["input_ids"][len(PROMPT) :])
            top = [[math.log(p), VOCAB.index(t), None] for t, p in DIST[prefix].items()]
            return {"text": "", "meta_info": {"output_top_logprobs": [top], "weight_version": "5"}}
        if payload["sampling_params"]["max_new_tokens"] == 0:
            tail = payload["input_ids"][payload["logprob_start_len"] :]
            items = [[None, tail[0], None]] + [[-0.1, t, None] for t in tail[1:]]
            return {"text": "", "meta_info": {"input_token_logprobs": items, "weight_version": "5"}}
        return {
            "text": "!",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.5, VOCAB.index("!"), None]],
                "weight_version": "5",
                "weight_versions": [dict(version="5", start=0, end=1)],
            },
        }

    monkeypatch.setattr(refreshing_rollout, "post", post)


def producer(branches=2, pool=20):
    p = RefreshingRolloutFn.__new__(RefreshingRolloutFn)
    p.args = SimpleNamespace(
        olmo_core=SimpleNamespace(refresh_request_timeout=5, opener_branches=branches, opener_pool=pool),
        sglang_router_ip="localhost",
        sglang_router_port=1234,
        rollout_seed=17,
        rollout_max_response_len=8,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
    )
    p.state = SimpleNamespace(tokenizer=Tokenizer(), processor=None)
    return p


def group(size):
    return [Sample(group_index=3, index=30 + i, prompt="Q<think>") for i in range(size)]


def texts(found):
    return [Tokenizer().decode(path) for path, _ in found]


def probe(posts, monkeypatch, pool=20):
    serve(monkeypatch, posts)
    p = producer(pool=pool)
    score = lambda x: p._prefill(opener_branches.probe_payload(x), "opener probe")  # noqa: E731
    return asyncio.run(opener_branches.candidates(score, Tokenizer(), PROMPT, pool))


def test_probe_keeps_whitespace_distinct_and_expands_unfinished_prefixes(monkeypatch):
    found, version = probe([], monkeypatch)
    assert version == "5"
    # Newline paths are expanded to their first word; the split character is completed;
    # the special closing tag is never an opener.
    assert texts(found) == ["\nI", "Okay", "\nThe", " Okay", "—"]
    assert [round(math.exp(lp), 4) for _, lp in found] == [0.35, 0.3, 0.15, 0.1, 0.05]


def test_pool_keeps_only_the_most_likely_openers(monkeypatch):
    found, _ = probe([], monkeypatch, pool=3)
    assert texts(found) == ["\nI", "Okay", "\nThe"]


def test_draw_is_uniform_without_replacement_and_reproducible():
    pool = [([i], -float(i)) for i in range(20)]
    first = opener_branches.choose(pool, 4, "17:3:30")
    assert len({tuple(p) for p, _ in first}) == 4
    assert first == opener_branches.choose(pool, 4, "17:3:30")
    # Every candidate, including the least likely, is drawn about equally often.
    hits = [0] * 20
    for k in range(4000):
        for path, _ in opener_branches.choose(pool, 4, f"key:{k}"):
            hits[path[0]] += 1
    assert min(hits) > 0.8 * 800 and max(hits) < 1.2 * 800
    with pytest.raises(ValueError, match="branches are required"):
        opener_branches.choose(pool[:3], 4, "k")


def test_group_splits_evenly_and_seeds_by_token_ids(monkeypatch):
    posts = []
    serve(monkeypatch, posts)
    p = producer(branches=2)
    samples = group(8)
    asyncio.run(p._branch_openers(samples))
    seeds = [s.metadata[response_seed.REQUEST_KEY]["token_ids"] for s in samples]
    assert seeds[:4] == [seeds[0]] * 4 and seeds[4:] == [seeds[4]] * 4 and seeds[0] != seeds[4]
    record = samples[5].metadata[opener_branches.RECORD_KEY]
    assert record["branch"] == 1 and record["probe_weight_version"] == "5" and len(record["pool"]) == 5
    assert record["chosen"] == [seeds[0], seeds[4]]


def test_seeded_branch_generates_from_its_opener_tokens(monkeypatch):
    posts = []
    serve(monkeypatch, posts)
    p = producer(branches=2)
    samples = group(2)
    asyncio.run(p._branch_openers(samples))
    seed = samples[0].metadata[response_seed.REQUEST_KEY]["token_ids"]
    posts.clear()
    request = SimpleNamespace(args=p.args, state=p.state, sample=samples[0], sampling_params={"max_new_tokens": 8})
    value = asyncio.run(p._generate_response_impl(request)).samples
    scoring, generation = posts
    assert scoring["input_ids"] == PROMPT + seed and generation["input_ids"] == PROMPT + seed
    assert value.tokens == PROMPT + seed + ids("!")
    assert value.rollout_log_probs == [-0.1] * len(seed) + [-0.5]
    value.validate()


def test_stalled_probe_is_a_recoverable_generation_timeout(monkeypatch):
    serve(monkeypatch, [], stall_probe=True)
    p = producer()
    p.args.olmo_core.refresh_request_timeout = 0.01
    with pytest.raises(GenerationRequestTimeout):
        asyncio.run(p._branch_openers(group(2)))


def test_branching_is_off_without_the_setting():
    assert opener_branches.settings(SimpleNamespace(olmo_core=SimpleNamespace())) is None
    assert opener_branches.settings(SimpleNamespace(olmo_core=SimpleNamespace(opener_branches=0))) is None
    assert opener_branches.settings(producer(branches=4, pool=20).args) == (4, 20)


@pytest.mark.parametrize(
    "bad", [{"token_ids": []}, {"token_ids": [1, -1]}, {"token_ids": "1"}, {"text": "a", "token_ids": [1]}]
)
def test_malformed_token_seed_fails_closed(bad):
    with pytest.raises(ValueError):
        response_seed.requested(Sample(metadata={response_seed.REQUEST_KEY: bad}))


def compiled(*overrides):
    return RunSpec.load("configs/miles/examples/medium.toml", list(overrides)).compile()


def core_argument(config):
    argv = config.arguments()
    return json.loads(argv[argv.index("--olmo-core-config") + 1])


BRANCHES = ["core.opener_branches=4", "inference.samples_per_prompt=16", "inference.global_batch_size=1024"]


def test_disabled_branching_keeps_the_native_core_argument_unchanged():
    assert not {"opener_branches", "opener_pool"} & set(core_argument(compiled()))


def test_refresh_run_carries_opener_branches_to_the_runtime():
    core = core_argument(compiled(*BRANCHES))
    assert core["opener_branches"] == 4 and core["opener_pool"] == 20


def test_opener_branches_require_refresh_publication():
    with pytest.raises(InputError, match="publication_mode"):
        compiled(*BRANCHES, 'core.publication_mode="barrier"').arguments()


def test_group_must_split_evenly_across_branches():
    with pytest.raises(InputError, match="multiple of core.opener_branches"):
        compiled("core.opener_branches=3", *BRANCHES[1:]).arguments()


def test_pool_must_hold_every_branch():
    with pytest.raises(InputError, match="opener_pool"):
        compiled(*BRANCHES, "core.opener_pool=3").arguments()


def test_branches_and_fixed_seeds_cannot_be_combined():
    with pytest.raises(InputError, match="cannot be combined"):
        compiled(*BRANCHES, 'core.opener_seeds=["Okay"]', "core.opener_seeds_per_group=1").arguments()
