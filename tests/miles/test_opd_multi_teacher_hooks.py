"""Multi-teacher OPD hooks against the pinned Miles router: per-teacher admission and score records."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from miles.utils.types import Sample

from open_instruct.miles.distillation import opd_hooks, opd_prepare

URLS = {"math": "http://127.0.0.1:1/generate", "ifeval": "http://127.0.0.1:2/generate"}
NAMES = {url: name for name, url in URLS.items()}


def routed_args(**urls):
    urls = urls or URLS
    return SimpleNamespace(
        opd_teacher_urls=[f"{name}={url}" for name, url in urls.items()],
        opd_teacher_key="domain",
        rm_url=None,
        sglang_router_request_timeout_secs=7,
    )


def sample(index, domain, tokens=(1, 2, 3)):
    item = Sample(index=index, tokens=list(tokens), response_length=2, response="ab")
    item.metadata = {} if domain is None else {"domain": domain}
    return item


@pytest.fixture
def hooks(monkeypatch, tmp_path):
    monkeypatch.delenv("OI_OPD_EOPD_TOP_K", raising=False)
    monkeypatch.delenv(opd_hooks.EOS_REMAP_ENV, raising=False)
    monkeypatch.setenv("OI_OPD_OUTPUT", str(tmp_path))
    monkeypatch.setattr(opd_hooks, "_LIMITS", {})
    return tmp_path


def test_teacher_name_follows_the_miles_router():
    args = routed_args()
    assert opd_hooks.teacher_name(args, sample(0, "math")) == "math"
    assert opd_hooks.teacher_name(args, sample(0, "ifeval")) == "ifeval"
    with pytest.raises(ValueError, match="matches no --opd-teacher-urls name"):
        opd_hooks.teacher_name(args, sample(0, "code"))
    with pytest.raises(ValueError, match="missing teacher key 'domain'"):
        opd_hooks.teacher_name(args, sample(0, None))
    fallback = routed_args(math=URLS["math"], default=URLS["ifeval"])
    assert opd_hooks.teacher_name(fallback, sample(0, "code")) == "default"
    assert opd_hooks.teacher_name(fallback, sample(0, None)) == "default"
    assert opd_hooks.teacher_name(SimpleNamespace(rm_url="http://t"), sample(0, "math")) is None
    assert opd_hooks.teacher_name(SimpleNamespace(opd_teacher_urls=None), sample(0, "math")) is None


def test_teacher_concurrency_reads_per_teacher_limits():
    environment = {opd_prepare.TEACHER_CONCURRENCIES_ENV: "math:32,ifeval:8", "OI_OPD_TEACHER_CONCURRENCY": "5"}
    assert opd_hooks.teacher_concurrency("math", environment) == 32
    assert opd_hooks.teacher_concurrency("ifeval", environment) == 8
    assert opd_hooks.teacher_concurrency(None, environment) == 5
    assert opd_hooks.teacher_concurrency("default", {}) == 4


def test_each_teacher_admits_requests_against_its_own_limit(monkeypatch, hooks):
    """Math (limit 1) and IFEval (limit 2) each saturate their own semaphore; a busy math teacher
    does not hold back IFEval requests."""
    monkeypatch.setenv(opd_prepare.TEACHER_CONCURRENCIES_ENV, "math:1,ifeval:2")
    active = dict.fromkeys(URLS, 0)
    peak = dict.fromkeys(URLS, 0)
    seen = []

    async def scores(args, item, **kwargs):
        name = NAMES[opd_hooks.on_policy_distillation._teacher_url_for_sample(args, item)]
        seen.append((item.index, name))
        active[name] += 1
        peak[name] = max(peak[name], active[name])
        await asyncio.sleep(0.01)
        active[name] -= 1
        return {"meta_info": {"input_token_logprobs": [[None, 1], [-1.0, 2], [-2.0, 3]]}}

    monkeypatch.setattr(opd_hooks.on_policy_distillation, "reward_func", scores)
    args = routed_args()
    samples = [sample(index, "math" if index % 2 else "ifeval") for index in range(8)]

    async def run():
        return await asyncio.gather(*(opd_hooks.reward(args, item) for item in samples))

    assert asyncio.run(run()) == [0.0] * 8
    assert peak == {"math": 1, "ifeval": 2}
    assert sorted(seen) == [(index, "math" if index % 2 else "ifeval") for index in range(8)]
    assert set(opd_hooks._LIMITS) == {"math", "ifeval"}
    for item in samples:
        assert item.metadata[opd_hooks.TEACHER_RESPONSE_KEY][opd_hooks.TEACHER_NAME_KEY] == item.metadata["domain"]


def test_unroutable_samples_fail_before_any_teacher_request(monkeypatch, hooks):
    async def scores(*args, **kwargs):
        raise AssertionError("no teacher should be asked")

    monkeypatch.setattr(opd_hooks.on_policy_distillation, "reward_func", scores)
    with pytest.raises(ValueError, match="matches no --opd-teacher-urls name"):
        asyncio.run(opd_hooks.reward(routed_args(), sample(0, "code")))
    assert opd_hooks._LIMITS == {}


def test_eopd_scoring_posts_to_the_routed_teacher(monkeypatch, hooks):
    monkeypatch.setenv("OI_OPD_EOPD_TOP_K", "2")
    monkeypatch.setenv("OI_OPD_EOPD_TAU", "0.5")
    urls = []

    async def post(url, payload, timeout_secs=None):
        urls.append((url, timeout_secs))
        return {
            "meta_info": {
                "input_token_logprobs": [[None, 1], [-1.0, 2], [-2.0, 3]],
                "input_top_logprobs": [None, [[-0.1, 2], [-2.0, 7]], [[-0.5, 3], [-0.9, 8]]],
            }
        }

    monkeypatch.setattr(opd_hooks.on_policy_distillation, "_post_json", post)
    asyncio.run(opd_hooks.reward(routed_args(), sample(0, "ifeval")))
    asyncio.run(opd_hooks.reward(routed_args(), sample(1, "math")))
    assert urls == [(URLS["ifeval"], 7), (URLS["math"], 7)]


def test_score_records_name_the_teacher_route_domain_and_reverse_kl(monkeypatch, hooks):
    async def scores(args, item, **kwargs):
        return {"meta_info": {"input_token_logprobs": [[None, 1], [-1.0, 2], [-2.0, 3]]}}

    monkeypatch.setattr(opd_hooks.on_policy_distillation, "reward_func", scores)
    args = routed_args()
    items = [sample(0, "math"), sample(1, "ifeval")]
    items[0].rollout_log_probs = [-0.5, -1.0]
    for item in items:
        asyncio.run(opd_hooks.reward(args, item))
    opd_hooks.post_process(args, items)
    records = [json.loads(line) for line in (hooks / "teacher-scores.jsonl").read_text().splitlines()]
    assert [(r["teacher"], r["route"], r["domain"]) for r in records] == [
        ("math", "math", "math"),
        ("ifeval", "ifeval", "ifeval"),
    ]
    # (-0.5 - -1.0 + -1.0 - -2.0) / 2: the student's sampled-token log-ratio against the teacher.
    assert records[0]["reverse_kl"] == pytest.approx(0.75)
    assert "reverse_kl" not in records[1]


def test_single_teacher_records_carry_no_teacher_name(monkeypatch, hooks):
    async def scores(args, item, **kwargs):
        return {"meta_info": {"input_token_logprobs": [[None, 1], [-1.0, 2], [-2.0, 3]]}}

    monkeypatch.setattr(opd_hooks.on_policy_distillation, "reward_func", scores)
    args = SimpleNamespace(rm_url="http://teacher", opd_teacher_urls=None)
    item = sample(0, "math")
    asyncio.run(opd_hooks.reward(args, item))
    opd_hooks.post_process(args, [item])
    record = json.loads((hooks / "teacher-scores.jsonl").read_text())
    assert "teacher" not in record and "route" not in record and record["domain"] == "math"
    assert set(opd_hooks._LIMITS) == {None}
