"""CPU checks for multi-teacher OPD: ``[teachers.<name>]`` parsing, validation, allocation and routing."""

import copy
import json
import re
from pathlib import Path

import pytest

from open_instruct.miles.configuration import specs, validation
from open_instruct.miles.distillation import opd_audit, opd_config, opd_prepare, opd_runtime
from open_instruct.miles.errors import InputError
from open_instruct.miles.execution import launch, workflow

CONFIG = Path(__file__).resolve().parents[1] / "tests/miles/fixtures/opd/qwen35-4b-tiny.toml"
MATH = "/weka/teachers/math/hf-99"
IFEVAL = "/weka/teachers/ifeval/hf-99"
PREPARED = {
    "model": "/assets/Qwen3.5-4B-851bf6e806ef",
    "teachers": {"math": "/assets/teacher-math-hf-99-local-a", "ifeval": "/assets/teacher-ifeval-hf-99-local-b"},
    "data": {"prompt_data": "/assets/data/train.jsonl", "eval_prompt_data": ["gsm8k", "/assets/data/eval.jsonl"]},
}
URLS = {"math": "http://127.0.0.1:1/generate", "ifeval": "http://127.0.0.1:2/generate"}


def document(**teachers):
    """The tiny OPD fixture with ``[teachers]`` in place of ``[teacher]``: math and ifeval share one GPU."""
    doc = validation.read_document(CONFIG)
    doc["teachers"] = teachers or {
        "math": {"source": MATH, "mem_fraction_static": 0.4},
        "ifeval": {"source": IFEVAL, "mem_fraction_static": 0.6},
    }
    return doc


def load(doc, overrides=None):
    return specs.from_dict(doc, config_path=CONFIG, overrides=overrides)


def arguments(spec, prepared=PREPARED, urls=URLS):
    return opd_runtime.native_arguments(spec, prepared, "/assets/ckpt", urls, [])


def test_teachers_parse_with_defaults_and_shared_gpu_allocation():
    spec = load(document())
    assert spec.routed
    assert spec.document["teacher_pool"] == opd_config.TEACHER_POOL_DEFAULTS
    assert "teacher" not in spec.document
    assert spec.document["teachers"]["math"] == {
        **opd_config.ROUTED_TEACHER_DEFAULTS,
        "source": MATH,
        "mem_fraction_static": 0.4,
    }
    assert spec.allocation() == {
        "replicas": 1,
        "gpus_per_replica": 4,
        "ray_gpus": 3,
        "roles": {"trainer": [0, 1], "student": [2], "teacher": [3]},
        "teachers": {"math": [3], "ifeval": [3]},
    }
    teachers = spec.teachers()
    assert list(teachers) == ["math", "ifeval"]
    assert teachers["ifeval"]["gpus"] == [3]
    assert teachers["ifeval"]["request_timeout"] == opd_config.TEACHER_POOL_DEFAULTS["request_timeout"]
    assert any("Multi-teacher" in warning for warning in spec.plan()["warnings"])
    assert launch.specification("test-image", spec)["tasks"][0]["resources"]["gpuCount"] == 4
    assert specs.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


def test_one_gpu_per_teacher_and_tensor_parallel_teachers():
    doc = document(math={"source": MATH}, ifeval={"source": IFEVAL, "devices": [1, 2]})
    spec = load(doc, ["trainer.gpus=4", "inference.gpus=1"])
    assert spec.allocation()["roles"]["teacher"] == [5, 6, 7]
    assert spec.allocation()["teachers"] == {"math": [5], "ifeval": [6, 7]}
    command = opd_runtime.teacher_command(spec, PREPARED, 9000, "ifeval")
    assert command[command.index("--tp") + 1] == "2"
    assert command[command.index("--mem-fraction-static") + 1] == "0.6"


def test_single_teacher_documents_keep_their_shape_and_identity():
    spec = specs.load(CONFIG)
    assert not spec.routed
    document = spec.to_dict()
    assert "teachers" not in document and "teacher_pool" not in document
    assert set(document["teacher"]) == set(opd_config.DEFAULTS["teacher"])
    assert list(spec.teachers()) == ["teacher"]
    assert spec.teachers()["teacher"]["gpus"] == [3]
    assert "teachers" not in spec.allocation()
    values = opd_runtime.native_arguments(spec, {**PREPARED, "teacher": "/t"}, "/ckpt", URLS["math"], [])
    assert values[values.index("--rm-url") + 1] == URLS["math"]
    assert "--opd-teacher-urls" not in values and "--opd-teacher-key" not in values
    assert opd_runtime.teacher_concurrency_environment(spec) == {"OI_OPD_TEACHER_CONCURRENCY": "4"}
    command = opd_runtime.teacher_command(spec, {"teacher": "/frozen-teacher"}, 12345)
    assert command[command.index("--mem-fraction-static") + 1] == "0.6"
    assert command[command.index("--tp") + 1] == "1"


def test_native_arguments_route_by_metadata_without_rm_url():
    doc = document()
    doc["teacher_pool"] = {"route_key": "domain", "request_timeout": 900}
    args = arguments(load(doc))
    index = args.index("--opd-teacher-urls")
    assert args[index + 1 : index + 3] == [f"math={URLS['math']}", f"ifeval={URLS['ifeval']}"]
    assert args[args.index("--opd-teacher-key") + 1] == "domain"
    assert args[args.index("--sglang-router-request-timeout-secs") + 1] == "900"
    assert args[args.index("--opd-type") + 1] == "sglang"
    assert "--rm-url" not in args


def test_routed_teacher_serving_and_concurrency_settings():
    doc = document()
    doc["teachers"]["math"] |= {"concurrency": 32, "enable_fp32_lm_head": True}
    spec = load(doc, ["inference.max_context_length=4096"])
    command = opd_runtime.teacher_command(spec, PREPARED, 9000, "math")
    assert command[command.index("--model-path") + 1] == PREPARED["teachers"]["math"]
    assert command[command.index("--max-running-requests") + 1] == "32"
    assert command[command.index("--max-total-tokens") + 1] == str(4096 * 32)
    assert command[command.index("--mem-fraction-static") + 1] == "0.4"
    assert "--enable-fp32-lm-head" in command
    assert "--enable-fp32-lm-head" not in opd_runtime.teacher_command(spec, PREPARED, 9001, "ifeval")
    assert opd_runtime.teacher_concurrency_environment(spec) == {
        opd_prepare.TEACHER_CONCURRENCIES_ENV: "math:32,ifeval:4"
    }
    assert opd_runtime.teacher_log_name(spec, "math") == "teacher-math.log"


def _mutate(path, value):
    def apply(doc):
        target = doc
        for key in path[:-1]:
            target = target[key]
        if value is _DELETE:
            target.pop(path[-1])
        else:
            target[path[-1]] = value

    return apply


_DELETE = object()


@pytest.mark.parametrize(
    "change, message",
    [
        (_mutate(("teacher",), {"source": MATH}), "not both"),
        (_mutate(("teachers",), {"math": {"source": MATH}}), "at least two teachers"),
        (_mutate(("teachers", "default"), {"source": MATH, "devices": [1]}), "allow_default_teacher"),
        (_mutate(("teachers", "Math"), {"source": MATH, "devices": [1]}), "differ only by case"),
        (_mutate(("teachers", "if eval"), {"source": IFEVAL, "devices": [1]}), "names may contain only"),
        (_mutate(("teachers", "math", "mem_fraction_static"), _DELETE), "set mem_fraction_static for teachers.math"),
        (_mutate(("teachers", "math", "devices"), [0, 1]), "identical teachers.<name>.devices"),
        (_mutate(("teachers", "math", "devices"), [2]), "without gaps"),
        (_mutate(("teachers", "math", "devices"), []), "nonempty list"),
        (_mutate(("teachers", "math", "devices"), [0, 0]), "repeats a GPU"),
        (_mutate(("teachers", "math", "source"), ""), "teachers.math.source"),
        (_mutate(("teachers", "math", "source"), "Qwen/Qwen3.5-9B-Unpinned"), "teachers.math.revision"),
        (_mutate(("teachers", "math", "gpus"), 1), "Unknown teachers.math fields"),
        (_mutate(("teachers", "math", "concurrency"), 0), "teachers.math.concurrency"),
        (_mutate(("teachers", "math", "mem_fraction_static"), 1.0), "teachers.math.mem_fraction_static"),
        (_mutate(("teacher_pool",), {"route_key": ""}), "teacher_pool.route_key"),
        (_mutate(("teacher_pool",), {"route_key": "a b"}), "teacher_pool.route_key"),
        (_mutate(("teacher_pool",), {"request_timeout": 0}), "teacher_pool.request_timeout"),
        (_mutate(("teacher_pool",), {"key": "domain"}), "Unknown teacher_pool fields"),
        (_mutate(("launch", "gpus_per_replica"), 5), "launch.gpus_per_replica disagrees"),
        (_mutate(("miles",), {"opd_teacher_urls": ["math=http://x"]}), "teachers.<name>"),
        (_mutate(("miles",), {"opd_teacher_key": "domain"}), "teacher_pool.route_key"),
    ],
)
def test_ambiguous_or_invalid_multi_teacher_configs_are_rejected(change, message):
    doc = document()
    change(doc)
    with pytest.raises(InputError, match=re.escape(message)):
        load(doc)


def test_teacher_pool_requires_teachers_and_default_teacher_needs_opt_in():
    doc = validation.read_document(CONFIG)
    doc["teacher_pool"] = {"route_key": "domain"}
    with pytest.raises(InputError, match="teacher_pool configures"):
        load(doc)
    doc = document()
    doc["teachers"]["default"] = {"source": MATH, "devices": [1]}
    doc["teacher_pool"] = {"allow_default_teacher": True}
    assert "default" in load(doc).teachers()


def test_teacher_gpus_count_toward_the_one_node_limit():
    doc = document(math={"source": MATH}, ifeval={"source": IFEVAL, "devices": [1]})
    with pytest.raises(InputError, match=re.escape("the teachers' devices = 9 must be <= 8")):
        load(doc, ["trainer.gpus=4", "inference.gpus=3"])
    assert load(doc, ["trainer.gpus=4", "inference.gpus=2"]).allocation()["gpus_per_replica"] == 8


def test_duplicate_teacher_tables_fail_in_the_run_file(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(
        CONFIG.read_text() + f'\n[teachers.math]\nsource = "{MATH}"\n\n[teachers.math]\nsource = "{IFEVAL}"\n'
    )
    with pytest.raises(InputError):
        specs.load(path)


def test_local_teacher_sources_resolve_against_the_run_file():
    doc = document()
    doc["teachers"]["math"]["source"] = "./teachers/math"
    spec = load(doc)
    assert spec.document["teachers"]["math"]["source"] == str((CONFIG.parent / "teachers/math").resolve())
    assert spec.document["teachers"]["math"]["revision"] == ""


def test_multi_teacher_composes_with_the_domain_balanced_loss():
    overrides = [
        'training.loss_aggregation="domain"',
        'distillation.policy_loss="dppo"',
        "distillation.use_rollout_logprobs=true",
    ]
    args = arguments(load(document(), overrides))
    assert "--calculate-per-token-loss" in args
    assert args[args.index("--opd-teacher-key") + 1] == "domain"


def test_multi_teacher_run_identity_tracks_the_teachers():
    first = load(document()).to_dict()
    changed = copy.deepcopy(document())
    changed["teachers"]["ifeval"]["concurrency"] = 8
    assert workflow.fingerprint(first) != workflow.fingerprint(load(changed).to_dict())


def _identity(vocab="v", special=None):
    return {"vocab_sha256": vocab, "special_tokens": special or {"eos_token": "<|im_end|>"}}


def test_every_teacher_must_share_the_learner_vocabulary():
    opd_prepare.check_teachers(_identity(), {"math": _identity(), "ifeval": _identity()})
    with pytest.raises(InputError, match="Teacher 'ifeval' and learner token IDs differ"):
        opd_prepare.check_teachers(_identity(), {"math": _identity(), "ifeval": _identity("w")})
    with pytest.raises(InputError, match="Teacher 'math' and learner special tokens differ"):
        opd_prepare.check_teachers(_identity(), {"math": _identity(special={"eos_token": "<eos>"})})
    with pytest.raises(InputError, match="^Teacher and learner token IDs differ"):
        opd_prepare.check_teachers(_identity(), {"teacher": _identity("w")})


def test_eos_alignment_requires_teachers_to_agree():
    assert opd_prepare.shared_eos_token({"math": "<|im_end|>", "ifeval": "<|im_end|>"}) == "<|im_end|>"
    with pytest.raises(InputError, match="same token"):
        opd_prepare.shared_eos_token({"math": "<|im_end|>", "ifeval": "<|endoftext|>"})


def test_prepared_paths_and_identities_cover_both_forms():
    assert opd_prepare.teacher_paths(PREPARED) == PREPARED["teachers"]
    assert opd_prepare.teacher_paths({"teacher": "/t"}) == {"teacher": "/t"}
    assert opd_prepare.teacher_identities({"identities": {"teacher": {"a": 1}}}) == {"teacher": {"a": 1}}


def _prompts(path, domains):
    rows = [{"input": "x", "metadata": {} if domain is None else {"domain": domain}} for domain in domains]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_prepared_prompts_must_route_to_configured_teachers(tmp_path):
    teachers = {"math": {}, "ifeval": {}}
    path = _prompts(tmp_path / "train.jsonl", ["math", "ifeval", "math"])
    assert opd_prepare.check_routes(path, "domain", teachers) == {"math": 2, "ifeval": 1}
    with pytest.raises(InputError, match=re.escape("{'code': 1, None: 1}")):
        opd_prepare.check_routes(_prompts(path, ["math", "ifeval", "code", None]), "domain", teachers)
    with pytest.raises(InputError, match=re.escape("No training prompt routes to teachers ['ifeval']")):
        opd_prepare.check_routes(_prompts(path, ["math"]), "domain", teachers)
    with_default = {"math": {}, "default": {}}
    assert opd_prepare.check_routes(_prompts(path, ["math", "code", None]), "domain", with_default) == {
        "math": 1,
        "default": 2,
    }


def _scores(path, records):
    lines = []
    for index, (teacher, route, domain, kl) in enumerate(records):
        record = {
            "sample_index": index,
            "response_length": 2,
            "teacher_log_probs": [-1.0, -3.0],
            "teacher": teacher,
            "route": route,
            "domain": domain,
        }
        if kl is not None:
            record["reverse_kl"] = kl
        lines.append(json.dumps(record) + "\n")
    path.write_text("".join(lines))
    return path


def test_audit_summarizes_each_teacher_and_domain(tmp_path):
    path = _scores(
        tmp_path / "teacher-scores.jsonl",
        [("math", "math", "math", 0.5), ("math", "math", "math", 1.5), ("ifeval", "ifeval", "ifeval", 0.2)],
    )
    result = opd_audit.audit_teachers(path, {"math": {}, "ifeval": {}})
    assert result["teachers"]["math"] == {
        "samples": 2,
        "tokens": 4,
        "mean_teacher_log_prob": -2.0,
        "mean_reverse_kl": 1.0,
    }
    assert result["teachers"]["ifeval"]["samples"] == 1
    assert result["domains"] == {
        "math": {"samples": 2, "mean_reverse_kl": 1.0},
        "ifeval": {"samples": 1, "mean_reverse_kl": 0.2},
    }


@pytest.mark.parametrize(
    "records, message",
    [
        ([("math", "ifeval", "ifeval", None), ("ifeval", "ifeval", "ifeval", None)], "was scored by 'math'"),
        ([("math", "math", "math", None), ("code", "code", "code", None)], "unknown teacher 'code'"),
        ([("math", "math", "math", None), (None, None, "x", None)], "unknown teacher None"),
        ([("math", "math", "math", None)], "Teachers ['ifeval'] scored no sample"),
    ],
)
def test_audit_rejects_misrouted_or_idle_teachers(tmp_path, records, message):
    path = _scores(tmp_path / "teacher-scores.jsonl", records)
    with pytest.raises(ValueError, match=re.escape(message)):
        opd_audit.audit_teachers(path, {"math": {}, "ifeval": {}})


def test_audit_accepts_default_fallback(tmp_path):
    path = _scores(
        tmp_path / "teacher-scores.jsonl", [("math", "math", "math", None), ("default", "code", "code", None)]
    )
    result = opd_audit.audit_teachers(path, {"math": {}, "default": {}})
    assert result["teachers"]["default"]["samples"] == 1
