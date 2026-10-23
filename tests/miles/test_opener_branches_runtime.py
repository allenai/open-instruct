"""Refresh opener branches: exact per-prompt opener pool, uniform draw, even split into seeded samples."""

import asyncio
import copy
import json
import math
import threading
from types import SimpleNamespace

import httpx
import pytest
from miles.backends.core_utils.rollout import opener_branches, refreshing_rollout, response_seed
from miles.backends.core_utils.rollout.errors import GenerationRequestTimeout
from miles.backends.core_utils.rollout.generation_admission import GenerationAdmission
from miles.backends.core_utils.rollout.refreshing_rollout import RefreshingRolloutFn
from miles.rollout.submission_scheduler import SampleBackfillSubmission
from miles.utils.types import Sample

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.rollout.data_source import DashboardDrainingRolloutDataSource

# "\xe2" is half of a multi-byte character that "\x80\x94" completes; "</think>" is special.
WORDS = [f" w{i}" for i in range(40)]
VOCAB = ["Q", "<think>", "\n", " Okay", "Okay", "I", "The", "\xe2", "\x80\x94", "</think>", "!", *WORDS]
VOCAB += ["<functions>", "</functions>", "<function_calls>", "</function_calls>", "<|extra_id_2|>"]
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
SMALL = {
    (): {"\n": 0.5, "Okay": 0.3, " Okay": 0.1, "\xe2": 0.05, "</think>": 0.05},
    ("\n",): {"I": 0.7, "The": 0.3},
    ("\xe2",): {"\x80\x94": 1.0},
}
# More than TOP likely words at the root: an exact pool of 25 must look past the root's top 20.
WIDE = {
    (): {"\n": 0.3, **{w: (0.7 / 30) * (1 - 0.01 * i) for i, w in enumerate(WORDS[:30])}},
    ("\n",): {w: 0.5 * 0.5**j for j, w in enumerate(WORDS)},
}


def serve(monkeypatch, posts, dist=SMALL, *, versions=lambda n: "5", stall=None, fail_at=None):
    """Fake SGLang: opener probes (top-k honoured), prefill-only seed scoring, one-token continuation."""
    probes = []
    active = [0, 0]  # current, peak concurrent probes

    async def post(url, payload, **kwargs):
        posts.append(copy.deepcopy(payload))
        if "top_logprobs_num" in payload:
            probes.append(payload)
            active[0] += 1
            active[1] = max(active)
            try:
                if stall is not None:
                    await stall.wait()
                if fail_at is not None and len(probes) == fail_at:
                    raise httpx.TransportError("probe failed")
                prefix = tuple(VOCAB[t] for t in payload["input_ids"][len(PROMPT) :])
                ranked = sorted(dist[prefix].items(), key=lambda kv: -kv[1])[: payload["top_logprobs_num"]]
                top = [[math.log(p), VOCAB.index(t), None] for t, p in ranked]
                meta = {"output_top_logprobs": [top], "weight_version": versions(len(probes))}
                return {"text": "", "meta_info": meta}
            finally:
                active[0] -= 1
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
    return probes, active


class Ledger:
    def __init__(self):
        self.persisted = []

    def persist_group_metadata(self, group, keys):
        self.persisted.append(([copy.deepcopy(s.metadata) for s in group], keys))


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
        use_sampling_support_replay=False,
        rollout_top_logprobs_num=0,
        sglang_speculative_algorithm=None,
    )
    p.state = SimpleNamespace(tokenizer=Tokenizer(), processor=None)
    p._generation_admission = GenerationAdmission(8)
    p.data_source = Ledger()
    return p


def group(size):
    return [Sample(group_index=3, index=30 + i, prompt="Q<think>") for i in range(size)]


def texts(found):
    return [Tokenizer().decode(path) for path, _ in found]


def search(monkeypatch, dist=SMALL, pool=20, mode="visible", **serve_options):
    posts = []
    serve(monkeypatch, posts, dist, **serve_options)
    p = producer(pool=pool)

    async def probe(x, top):
        return await p._prefill(opener_branches.probe_payload(x, top), "opener probe")

    return asyncio.run(opener_branches.candidates(probe, Tokenizer(), PROMPT, pool, mode=mode)), posts


def test_literal_first_token_keeps_whitespace_and_low_probability_candidates(monkeypatch):
    distribution = {(): {"\n": 0.8, "<think>": 0.1, "</think>": 0.05, "Okay": 0.049, " Okay": 0.001}}
    result, posts = search(monkeypatch, distribution, pool=3, mode="first_token")
    assert texts(result["found"]) == ["\n", "Okay", " Okay"]
    assert all(len(path) == 1 for path, _ in result["found"])
    assert result["found"][-1][1] == pytest.approx(math.log(0.001))
    assert len(posts) == 1 and posts[0]["input_ids"] == PROMPT
    assert posts[0]["sampling_params"]["temperature"] == 1.0
    assert result["exact"] and result["mode"] == "first_token"


def test_literal_first_token_widens_at_same_prompt_and_retries_version_change(monkeypatch):
    distribution = {(): {**WIDE[()], "<think>": 0.5, "</think>": 0.4}}
    result, posts = search(
        monkeypatch, distribution, pool=25, mode="first_token", versions=lambda n: "4" if n == 1 else "5"
    )
    assert result["version"] == "5" and result["exact"]
    assert len(posts) == 4 and all(p["input_ids"] == PROMPT for p in posts)
    assert all(len(path) == 1 for path, _ in result["found"])


def test_literal_first_token_refuses_short_pool(monkeypatch):
    with pytest.raises(ValueError, match="pool of 8 required"):
        search(monkeypatch, pool=8, mode="first_token")


@pytest.mark.parametrize("marker", VOCAB[-5:])
def test_literal_first_token_excludes_unflagged_protocol_markers(monkeypatch, marker):
    result, _ = search(monkeypatch, {(): {marker: 0.9, "\n": 0.1}}, pool=1, mode="first_token")
    assert texts(result["found"]) == ["\n"]


def test_literal_first_token_is_wired_and_persisted_with_even_branches(monkeypatch):
    posts = []
    serve(monkeypatch, posts, WIDE)
    p = producer(branches=4, pool=8)
    p.args.olmo_core.opener_mode = "first_token"
    samples = group(16)
    asyncio.run(p._branch_openers(samples))
    assert len(p.data_source.persisted) == 1
    for sample in samples:
        assert sample.metadata["opener_branches"]["mode"] == "first_token"
        assert len(sample.metadata[response_seed.REQUEST_KEY]["token_ids"]) == 1
    branches = [s.metadata["opener_branches"]["branch"] for s in samples]
    assert branches == [b for b in range(4) for _ in range(4)]
    before = copy.deepcopy([s.metadata for s in samples])
    asyncio.run(p._branch_openers(samples))
    assert [s.metadata for s in samples] == before and len(posts) == 1


def brute_force(dist, pool):
    """Every opener path and its probability, by full enumeration."""
    paths, stack = [], [((), 0.0)]
    while stack:
        prefix, lp = stack.pop()
        for token, p in dist[prefix].items():
            if token in ("</think>",):
                continue
            child = prefix + (token,)
            text = Tokenizer().decode([VOCAB.index(t) for t in child])
            if text.strip() and "\ufffd" not in text:
                paths.append((text, lp + math.log(p)))
            elif child in dist:
                stack.append((child, lp + math.log(p)))
    return sorted(paths, key=lambda item: -item[1])[:pool]


def test_search_keeps_whitespace_distinct_and_expands_unfinished_prefixes(monkeypatch):
    result, _ = search(monkeypatch)
    assert result["version"] == "5" and result["exact"]
    # Newline paths run to their first word; the split character is completed;
    # the special closing tag is never an opener.
    assert texts(result["found"]) == ["\nI", "Okay", "\nThe", " Okay", "\u2014"]
    assert [round(math.exp(lp), 4) for _, lp in result["found"]] == [0.35, 0.3, 0.15, 0.1, 0.05]


def test_search_returns_the_exact_pool_beyond_the_first_probe_width(monkeypatch):
    result, posts = search(monkeypatch, WIDE, pool=25)
    expected = brute_force(WIDE, 25)
    assert texts(result["found"]) == [t for t, _ in expected]
    assert result["exact"]
    # The root was read again, wider, to admit root words ranked past its first 20.
    assert [p["top_logprobs_num"] for p in posts if p["input_ids"] == PROMPT] == [20, 200]


def test_search_cut_short_by_its_budget_is_inexact(monkeypatch):
    monkeypatch.setattr(opener_branches, "MAX_REQUESTS", 1)
    result, _ = search(monkeypatch)
    assert not result["exact"] and math.isclose(result["missed"], 0.5)


def test_depth_limit_is_an_omission(monkeypatch):
    monkeypatch.setattr(opener_branches, "MAX_DEPTH", 1)
    result, _ = search(monkeypatch)
    assert not result["exact"] and math.isclose(result["missed"], 0.5)
    assert "\nI" not in texts(result["found"])


def test_search_spanning_a_publication_is_repeated_under_one_version(monkeypatch):
    # The first search's second probe is served by version 6: the search is repeated.
    result, posts = search(monkeypatch, versions=lambda n: "5" if n == 1 else "6")
    assert result["version"] == "6" and result["versions"] == {"6"}
    assert len(posts) > 3


def test_search_that_never_settles_on_one_version_fails(monkeypatch):
    with pytest.raises(RuntimeError, match="spanned policy versions"):
        search(monkeypatch, versions=lambda n: str(n))


def test_probes_run_one_at_a_time_and_stop_at_a_failure(monkeypatch):
    posts = []
    probes, active = serve(monkeypatch, posts, fail_at=2)
    p = producer()
    with pytest.raises(httpx.TransportError):
        asyncio.run(p._branch_openers(group(2)))
    assert len(probes) == 2 and active == [0, 1]


def test_probes_wait_for_admission_and_hold_a_slot_while_running(monkeypatch):
    async def run():
        stall = asyncio.Event()
        probes, _ = serve(monkeypatch, [], stall=stall)
        p = producer()
        p._generation_admission.pause()
        task = asyncio.create_task(p._branch_openers(group(2)))
        await asyncio.sleep(0.05)
        assert probes == []  # admission is paused
        p._generation_admission.resume()
        await asyncio.sleep(0.05)
        assert len(probes) == 1 and p._generation_admission.active == 1
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(p._generation_admission.wait_idle(), 0.05)  # a drain waits for the probe
        stall.set()
        await task
        assert p._generation_admission.active == 0

    asyncio.run(run())


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


def test_assignment_is_persisted_and_reused_on_retry(monkeypatch):
    posts = []
    serve(monkeypatch, posts)
    p = producer(branches=2)
    samples = group(4)
    asyncio.run(p._branch_openers(samples))
    [(persisted, keys)] = p.data_source.persisted
    assert keys == (response_seed.REQUEST_KEY, opener_branches.RECORD_KEY)
    # A requeued group arrives with the persisted metadata and is not probed again.
    retried = [Sample(group_index=3, index=30 + i, prompt="Q<think>", metadata=m) for i, m in enumerate(persisted)]
    posts.clear()
    asyncio.run(p._branch_openers(retried))
    assert posts == [] and [s.metadata for s in retried] == [s.metadata for s in samples]


def test_data_source_writes_the_assignment_into_its_pristine_records():
    source = DashboardDrainingRolloutDataSource.__new__(DashboardDrainingRolloutDataSource)
    source._cursor_lock = threading.RLock()
    source._fully_async = True
    source._recovery_groups = None
    source._delegate = SimpleNamespace(buffer=[])
    pristine = group(2)
    source._pending_groups = {3: copy.deepcopy(pristine)}
    produced = copy.deepcopy(pristine)
    for s in produced:
        s.metadata = {"a": 1, "keep": "private"}
    source.persist_group_metadata(produced, ("a",))
    assert source.requeue_pending_groups([3]) == 1
    assert [s.metadata for s in source._delegate.buffer[0]] == [{"a": 1}, {"a": 1}]
    with pytest.raises(RuntimeError, match="no pristine record"):
        source.persist_group_metadata([Sample(group_index=9, index=0)], ("a",))


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
    assert record["pool_exact"] and record["pool_missed_probability"] == 0.0
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
    serve(monkeypatch, [], stall=asyncio.Event())
    p = producer()
    p.args.olmo_core.refresh_request_timeout = 0.01
    with pytest.raises(GenerationRequestTimeout):
        asyncio.run(p._branch_openers(group(2)))


def test_failed_probe_frees_its_group_slots_while_another_group_runs(monkeypatch):
    # Both groups are charged at submission; the running one keeps its slots, and the
    # one whose probe failed gives its slots back at once instead of at the next idle.
    serve(monkeypatch, [], fail_at=1)
    p = producer()
    p._scheduler = SampleBackfillSubmission(4)
    running, failing = group(4), group(4)
    p._scheduler.on_submit([running, failing])
    assert not p._scheduler.has_capacity(pending_groups=2, group_budget=2)
    with pytest.raises(httpx.TransportError):
        asyncio.run(p._generate_group(failing))
    assert p._scheduler.samples_in_flight == len(running)
    assert p._scheduler.has_capacity(pending_groups=1, group_budget=2)


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
    return RunSpec.load("configs/miles/examples/node.toml", list(overrides)).compile()


def core_argument(config):
    argv = config.arguments()
    return json.loads(argv[argv.index("--olmo-core-config") + 1])


BRANCHES = ["core.opener_branches=4", "inference.samples_per_prompt=16", "inference.global_batch_size=1024"]


def test_disabled_branching_keeps_the_native_core_argument_unchanged():
    assert not {"opener_branches", "opener_pool", "opener_mode"} & set(core_argument(compiled()))


def test_first_token_mode_reaches_runtime_and_invalid_mode_is_rejected():
    assert core_argument(compiled(*BRANCHES, 'core.opener_mode="first_token"'))["opener_mode"] == "first_token"
    with pytest.raises(InputError, match="opener_mode"):
        compiled(*BRANCHES, 'core.opener_mode="typo"')


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
