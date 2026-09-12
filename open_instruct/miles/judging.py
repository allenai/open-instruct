"""Named services and rubric bindings, following the olmo-miles run-file contract."""

import copy
import hashlib
import json
import re
from urllib.parse import urlsplit

from open_instruct.miles import validation
from open_instruct.miles.errors import InputError

REGISTRY_ENV = "OI_MILES_JUDGE_REGISTRY"
PROFILES = {
    "open-instruct/" + name: name
    for name in ("general-quality", "general-quality_ref", "general-web_instruct_general_verifier")
}


def parse(document):
    judges = copy.deepcopy(document.get("judges", {}))
    rubrics = copy.deepcopy(document.get("rubrics", {}))
    routing = copy.deepcopy(document.get("judging", {}))
    for name, value in (("judges", judges), ("rubrics", rubrics), ("judging", routing)):
        validation.mapping(value, name)
    validation.fields(routing, "judging", {"bindings"})
    bindings = routing.get("bindings", {})
    validation.mapping(bindings, "judging.bindings")
    for name, service in judges.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise InputError("judge names must be lowercase identifiers")
        validation.mapping(service, f"judges.{name}")
        validation.fields(
            service,
            f"judges.{name}",
            {
                "mode",
                "backend",
                "model",
                "revision",
                "gpus",
                "tensor_parallel_size",
                "prepared_dir",
                "chat_template",
                "endpoint",
                "max_context_length",
                "max_concurrent_calls",
                "timeout",
            },
        )
        validation.text(service.get("model"), f"judges.{name}.model")
        mode = service.get("mode")
        if mode not in ("managed", "external"):
            raise InputError("judge mode must be managed or external")
        for key, default in (("max_context_length", 40960), ("max_concurrent_calls", 16), ("timeout", 120)):
            service.setdefault(key, default)
            validation.integer(service[key], f"judges.{name}.{key}")
        if mode == "managed":
            service.setdefault("backend", "sglang")
            service.setdefault("gpus", 1)
            service.setdefault("tensor_parallel_size", 1)
            service.setdefault("chat_template", "qwen3-no-thinking")
            if service["backend"] != "sglang" or service["chat_template"] != "qwen3-no-thinking":
                raise InputError("Managed judges currently support SGLang with qwen3-no-thinking")
            for key in ("gpus", "tensor_parallel_size"):
                validation.integer(service[key], f"judges.{name}.{key}")
            if service["gpus"] != service["tensor_parallel_size"]:
                raise InputError("judge gpus must equal tensor_parallel_size")
            if not re.fullmatch(r"[a-f0-9]{40}", str(service.get("revision", ""))):
                raise InputError("managed judge requires an immutable 40-character revision")
            if not isinstance(service.get("prepared_dir"), str) or not service["prepared_dir"].startswith("/"):
                raise InputError("managed judge requires an absolute prepared_dir cached before GPU allocation")
            if "endpoint" in service:
                raise InputError("The launcher resolves managed judge endpoints")
        else:
            endpoint = urlsplit(service.get("endpoint", ""))
            if (
                endpoint.scheme not in ("http", "https")
                or not endpoint.hostname
                or endpoint.username
                or endpoint.password
            ):
                raise InputError("external judge requires an HTTP(S) endpoint without credentials")
            if set(service) & {"gpus", "tensor_parallel_size", "prepared_dir", "revision", "chat_template", "backend"}:
                raise InputError("external judge cannot specify managed placement fields")
    for name, rubric in rubrics.items():
        validation.mapping(rubric, f"rubrics.{name}")
        validation.fields(rubric, f"rubrics.{name}", {"profile", "max_response_tokens", "temperature"})
        if rubric.get("profile") not in PROFILES:
            raise InputError("rubric requires a supported open-instruct/general-* profile")
        rubric.setdefault("max_response_tokens", 2048)
        rubric.setdefault("temperature", 1.0)
        validation.integer(rubric["max_response_tokens"], f"rubrics.{name}.max_response_tokens")
        validation.number(rubric["temperature"], f"rubrics.{name}.temperature")
        if rubric["temperature"] < 0:
            raise InputError("rubric temperature must be nonnegative")
    for name, binding in bindings.items():
        validation.mapping(binding, f"binding {name}")
        validation.fields(binding, f"binding {name}", {"judge", "rubric"})
        if binding.get("judge") not in judges or binding.get("rubric") not in rubrics:
            raise InputError(f"binding {name} references an unknown judge or rubric")
        if rubrics[binding["rubric"]]["max_response_tokens"] >= judges[binding["judge"]]["max_context_length"]:
            raise InputError("judge output reservation must be smaller than its context")
    if judges and not bindings:
        raise InputError("named judges require explicit verifier bindings")
    return {"judges": judges, "rubrics": rubrics, "judging": {"bindings": bindings}}


def registry(sections):
    active = {binding["judge"] for binding in sections["judging"]["bindings"].values()}
    return copy.deepcopy(
        {
            "schema_version": 1,
            "judges": {name: sections["judges"][name] for name in sorted(active)},
            "rubrics": sections["rubrics"],
            "bindings": sections["judging"]["bindings"],
        }
    )


def registry_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
