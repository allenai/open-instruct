"""Pinned native BFCL worker for the H196 Olmo 3.5 tiny tokenizer/tool contract.

The worker owns a vLLM environment independently of ordinary SGLang evaluation.
Checkpoint weights stay selected by MILES; this profile fixes serving/data semantics.
"""

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

WORKER = "bfcl-native-h196-v1"
PROFILE: dict[str, Any] = {
    "protocol_version": "bfcl-native-v1",
    "image": "01M7CK4QK3MJ7Q9MZYRQTMTPYW",
    "evaluator_revision": "56dc89212fc040a0b7fd957901c3b326b8a5cd11",
    "dataset_repo": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
    "dataset_revision": "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
    "context_length": 65536,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": -1,
    "num_samples": 1,
    "tool_call_parser": "qwen3_xml",
    "reasoning_parser": "olmo3",
    "plugin_archive": "/weka/oe-adapt-default/abhishekr/handoff/termination-2x2/ladders-3542d99.tar",
    "plugin_archive_sha256": "12117833da3cc502df5be6bef511165e7a4536090e7956bf4414a88b1a5a822f",
    "core_revision": "0c85d71c9ff2b8bf9aa34b279639d6514e1715a8",
    "tokenizer_assets": {
        "tokenizer.json": "799d80b76c4c6770c13b5958f4ec2550cb4c640c7004c1bd6a190fd37646efc1",
        "chat_template.jinja": "f43922cd53394a3b36b3369cf8d61e1f88bdd966c657bbd60fff20e53b9ebacd",
    },
    "panels": {
        "single": {
            "tasks": [
                "bfcl_simple",
                "bfcl_multiple",
                "bfcl_parallel",
                "bfcl_parallel_multiple",
                "bfcl_java",
                "bfcl_javascript",
                "bfcl_irrelevance",
                "bfcl_live_simple",
                "bfcl_live_multiple",
                "bfcl_live_parallel",
                "bfcl_live_parallel_multiple",
                "bfcl_live_irrelevance",
                "bfcl_live_relevance",
            ],
            "native_suite_equivalent": "bfcl:categories",
            "harness": "default",
            "max_tokens": 4096,
            "expected_instances": 3641,
            "category_counts": {
                "simple": 400,
                "multiple": 200,
                "parallel": 200,
                "parallel_multiple": 200,
                "java": 100,
                "javascript": 50,
                "irrelevance": 240,
                "live_simple": 258,
                "live_multiple": 1053,
                "live_parallel": 16,
                "live_parallel_multiple": 24,
                "live_irrelevance": 882,
                "live_relevance": 18,
            },
        },
        "multi": {
            "tasks": [
                "bfcl_multi_turn_base",
                "bfcl_multi_turn_miss_func",
                "bfcl_multi_turn_miss_param",
                "bfcl_multi_turn_long_context",
            ],
            "native_suite_equivalent": "bfcl:multi_turn",
            "harness": "bfcl_multi_turn",
            "max_tokens": 16384,
            "expected_instances": 800,
            "category_counts": {"base": 200, "miss_func": 200, "miss_param": 200, "long_context": 200},
        },
    },
    "thinking": {
        "server_reasoning_parser": "olmo3",
        "task_strip_thinking": True,
        "task_strip_unclosed_thinking": True,
        "task_thinking_prefilled": False,
        "reason": "Native server already splits reasoning from final text/tool_calls; do not "
        "double-treat final text as inside a prefilled block.",
    },
}
PROFILE["model_config"] = {
    "model_type": "olmo3moe",
    "architectures": ["Olmo3MoeForCausalLM"],
    "hidden_size": 1024,
    "num_hidden_layers": 16,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "num_experts_per_tok": 16,
    "max_position_embeddings": 65536,
}
EVALUATOR_REVISION = PROFILE["evaluator_revision"]


def resolved_profile(config):
    options = config.get("bfcl", {})
    if not isinstance(options, dict) or set(options) - {"plugin_archive"}:
        raise ValueError("evaluation.bfcl accepts only the checksum-pinned plugin_archive locator")
    archive = options.get("plugin_archive", PROFILE["plugin_archive"])
    if not isinstance(archive, str) or not Path(archive).is_absolute() or ".." in Path(archive).parts:
        raise ValueError("evaluation.bfcl.plugin_archive must be an absolute shared path")
    return {**PROFILE, "plugin_archive": archive}


def identity(config):
    profile = resolved_profile(config)
    return {
        "name": WORKER,
        "image": profile["image"],
        "revision": profile["evaluator_revision"],
        "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def validate_checkpoint(checkpoint, update, profile):
    checkpoint = Path(checkpoint)
    if update > 0 and not (checkpoint / ".complete").is_file():
        raise ValueError("Native BFCL checkpoint export has no completion marker")
    model = json.loads((checkpoint / "config.json").read_text())
    if any(model.get(key) != value for key, value in profile["model_config"].items()):
        raise ValueError("Native BFCL worker is supported only for the H196 Olmo 3.5 tiny model family")
    for name, expected in profile["tokenizer_assets"].items():
        if hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Native BFCL tokenizer/tool template differs from the qualified profile")


def panel_for(task):
    for panel, settings in PROFILE["panels"].items():
        if task in settings["tasks"]:
            return panel
    raise ValueError(f"Unsupported native BFCL task: {task}")


def generation(panel):
    return {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": -1,
        "num_samples": 1,
        "max_tokens": PROFILE["panels"][panel]["max_tokens"],
        "fit_max_tokens_to_context": True,
        "seed": 42,
        "num_fewshot": 0,
        "strip_thinking": True,
        "strip_unclosed_thinking": True,
        "thinking_prefilled": False,
    }


def normalize_task(task, config):
    panel = panel_for(task["task"])
    expected_identity = identity(config)
    if task.get("worker_profile", expected_identity) != expected_identity:
        raise ValueError("Native BFCL task worker profile differs from the resolved runtime")
    task["worker_profile"] = expected_identity
    expected = generation(panel)
    if any(key not in expected or value != expected[key] for key, value in task.get("generation", {}).items()):
        raise ValueError("Native BFCL profile requires its explicit T1/top-p0.95/context-fit sampling contract")
    if (
        task.get("kind", "task") != "task"
        or task.get("scoring")
        or task.get("harness")
        or task.get("auxiliary_gpus", 0)
    ):
        raise ValueError("Native BFCL worker uses its pinned native task/scorer without auxiliary providers")
    preset = PROFILE["panels"][panel]["harness"]
    if task.get("harness_preset", preset) != preset:
        raise ValueError("Native BFCL task has an incompatible harness")
    task["generation"] = expected
    task["harness_preset"] = preset


def validate_panels(tasks):
    selected = {task["task"] for task in tasks if task.get("worker") == WORKER}
    for settings in PROFILE["panels"].values():
        expected = set(settings["tasks"])
        if selected & expected and selected & expected != expected:
            raise ValueError("Native BFCL worker requires each selected category panel in full")
        intervals = {task.get("interval") for task in tasks if task["task"] in expected}
        if len(intervals) > 1:
            raise ValueError("Native BFCL categories in a panel require the same evaluation interval")


def settings(config):
    profile = resolved_profile(config)
    return {
        **config,
        "image": profile["image"],
        "revision": profile["evaluator_revision"],
        "gpus": 1,
        "provider_kwargs": {},
        "external_server_args": [],
        "worker": {"name": WORKER, "profile": profile, "identity": identity(config)},
    }


def launch_command(receipt, output):
    tasks = receipt["tasks"]
    for task in tasks:
        normalized = json.loads(json.dumps(task))
        normalize_task(normalized, receipt["evaluation"])
        if normalized != task:
            raise ValueError("Native BFCL task differs from its resolved sampling/runtime contract")
    panel = panel_for(tasks[0]["task"])
    if {task["task"] for task in tasks} != set(PROFILE["panels"][panel]["tasks"]):
        raise ValueError("Native BFCL group must contain exactly one complete native category panel")
    if (
        receipt["evaluation"]["image"] != PROFILE["image"]
        or receipt["evaluation"]["revision"] != EVALUATOR_REVISION
        or receipt["evaluation"]["gpus"] != 1
        or receipt["evaluation"].get("worker", {}).get("profile") != resolved_profile(receipt["evaluation"])
    ):
        raise ValueError("Native BFCL worker profile differs from the frozen receipt")
    identity = {
        "arm": receipt["training"]["name"],
        "update": receipt["update"],
        "panel": panel,
        "group_id": receipt["group_id"],
        "protocol_hash": receipt["evaluation"]["worker"]["identity"]["profile_sha256"],
        "expected_instances": PROFILE["panels"][panel]["expected_instances"],
    }
    cache = Path(receipt["evaluation"]["root"]) / "evaluation" / "runtime-cache" / WORKER
    return _command(
        resolved_profile(receipt["evaluation"]), receipt["checkpoint"], str(output), panel, identity, cache
    )


def validate_counts(output, tasks):
    expected = {
        task["task"]: PROFILE["panels"][panel_for(task["task"])]["category_counts"][
            task["task"].removeprefix("bfcl_multi_turn_")
            if task["task"].startswith("bfcl_multi_turn_")
            else task["task"].removeprefix("bfcl_")
        ]
        for task in tasks
    }
    observed = {}
    for path in Path(output).rglob("metrics.json"):
        value = json.loads(path.read_text())
        for result in value.get("tasks", []):
            name = result.get("task")
            if isinstance(name, str) and name.startswith("bfcl") and name not in expected:
                raise ValueError("Unexpected native BFCL category metrics")
            if name in expected:
                if type(result.get("instances_saved")) is not int:
                    raise ValueError("Native BFCL category has invalid instance count")
                if name in observed:
                    raise ValueError("Duplicate native BFCL task metrics")
                observed[name] = result.get("instances_saved")
    if observed != expected:
        raise ValueError(f"Native BFCL category populations differ: expected{expected}, observed{observed}")


def _command(p, checkpoint, output, panel, identity, cache_root):
    q = shlex.quote
    config = {"protocol": p, "checkpoint": checkpoint, "output_root": output, "identity": identity}
    encoded = base64.b64encode(json.dumps(config, sort_keys=True).encode()).decode()
    package_pins = [
        "vllm==0.19.1",
        "transformers==5.16.1",
        "huggingface_hub==1.16.1",
        "flash-linear-attention==0.5.2",
        "datasets==4.8.4",
        "torch==2.10.0",
        "opencv-python-headless==4.13.0.92",
        "torch-c-dlpack-ext==0.1.5",
        "tree-sitter-java==0.23.5",
        "tree-sitter-javascript==0.23.1",
        "openai==2.45.0",
    ]
    native = [
        "/tmp/bfcl-native-env/bin/olmo-eval",
        "run",
        "--model",
        checkpoint,
        "--harness",
        p["panels"][panel]["harness"],
    ]
    overrides = {
        "provider.kind": "vllm_server",
        "provider.kwargs.startup_timeout": 840,
        "provider.kwargs.enable_auto_tool_choice": True,
        "provider.kwargs.tool_call_parser": "qwen3_xml",
        "provider.kwargs.reasoning_parser": "olmo3",
        "provider.kwargs.dtype": "bfloat16",
        "provider.kwargs.max_model_len": 65536,
        "provider.kwargs.enforce_eager": True,
        "provider.kwargs.mamba_ssm_cache_dtype": "float32",
        "provider.kwargs.trust_remote_code": True,
        "provider.kwargs.enable_prefix_caching": False,
        "provider.kwargs.timeout": 1800,
        "provider.kwargs.seed": 42,
        "max_concurrency": 32,
    }
    for key, value in overrides.items():
        native.extend(["-o", key + "=" + (value if isinstance(value, str) else json.dumps(value))])
    for task in p["panels"][panel]["tasks"]:
        native.extend(["--task", task])
        for key, value in {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": -1,
            "num_samples": 1,
            "max_tokens": p["panels"][panel]["max_tokens"],
            "fit_max_tokens_to_context": True,
            "seed": 42,
            "num_fewshot": 0,
            "strip_thinking": True,
            "strip_unclosed_thinking": True,
            "thinking_prefilled": False,
        }.items():
            native.extend(["-o", key + "=" + json.dumps(value)])
    native.extend(
        ["--output-dir", output, "--save-predictions", "--save-requests", "--resume", "--no-upload", "--num-gpus", "1"]
    )
    lines = [
        "set -euo pipefail",
        "date +%s > /tmp/bfcl-bootstrap-start",
        'test "$(git -C /opt/olmo-eval rev-parse HEAD)" = ' + q(EVALUATOR_REVISION),
        "mkdir -p " + q(str(cache_root / "uv")) + " " + q(str(cache_root / "hf")),
        "export UV_CACHE_DIR=" + q(str(cache_root / "uv")),
        'export UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-1800}"',
        "export HF_HOME=" + q(str(cache_root / "hf")),
        "export HF_HUB_CACHE=" + q(str(cache_root / "hf/hub")),
        "export HF_DATASETS_CACHE=" + q(str(cache_root / "hf/datasets")),
        "export OLMO_EVAL_UPLOAD=0 TOKENIZERS_PARALLELISM=false",
        "echo " + q(p["plugin_archive_sha256"] + "  " + p["plugin_archive"]) + " | sha256sum -c -",
        "mkdir -p /tmp/bfcl-native-ladder",
        "tar -xf " + q(p["plugin_archive"]) + " -C /tmp/bfcl-native-ladder",
        "test -d /tmp/bfcl-native-ladder/ladders/olmoe3/vllm_plugin",
        "test -d /tmp/bfcl-native-ladder/ladders/olmoe3/transformers_plugin",
        "git init --quiet /tmp/bfcl-native-core",
        "git -C /tmp/bfcl-native-core remote add origin https://github.com/allenai/OLMo-core.git",
        "git -C /tmp/bfcl-native-core fetch --quiet --depth=1 origin " + q(p["core_revision"]),
        "git -C /tmp/bfcl-native-core checkout --quiet " + q(p["core_revision"]),
        "uv venv --python 3.12 /tmp/bfcl-native-env",
        "uv pip install --python /tmp/bfcl-native-env/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0",
        "printf '%s\\n' 'apache-tvm-ffi==0.1.15.post1' > /tmp/bfcl-native-constraints.txt",
        "export UV_CONSTRAINT=/tmp/bfcl-native-constraints.txt",
        "uv pip install --python /tmp/bfcl-native-env/bin/python "
        + shlex.join(package_pins)
        + " -e /tmp/bfcl-native-core -e /tmp/bfcl-native-ladder/ladders/olmoe3/vllm_plugin -e /tmp/bfcl-native-ladder/ladders/olmoe3/transformers_plugin -e /opt/olmo-eval",
        "export PATH=/tmp/bfcl-native-env/bin:$PATH",
        "export PYTHONPATH=/opt/olmo-eval/src",
        "export TRITON_PTXAS_PATH=/tmp/bfcl-native-env/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas",
        "export VLLM_PYTHON=/tmp/bfcl-native-env/bin/python",
        shlex.join(["/tmp/bfcl-native-env/bin/python", str(Path(__file__).resolve()), "record-runtime", encoded]),
        shlex.join(native),
    ]
    return "\n".join(lines) + "\n"


def record_runtime(config):
    """Preserve each attempt while refusing reuse across model/runtime identity changes."""
    profile = config["protocol"]
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(config["checkpoint"])
    validate_checkpoint(checkpoint, config["identity"]["update"], profile)
    revision = subprocess.check_output(["git", "-C", "/opt/olmo-eval", "rev-parse", "HEAD"], text=True).strip()
    if revision != profile["evaluator_revision"]:
        raise ValueError("Native BFCL evaluator revision differs")
    versions = {
        name: importlib_metadata.version(name)
        for name in [
            "vllm",
            "torch",
            "transformers",
            "huggingface-hub",
            "datasets",
            "flash-linear-attention",
            "tree-sitter-java",
            "tree-sitter-javascript",
            "openai",
        ]
    }
    record = {
        "protocol_hash": config["identity"]["protocol_hash"],
        "identity": config["identity"],
        "checkpoint": str(checkpoint),
        "weight_bytes_sha256": None,
        "weight_hash_status": "Not independently hashed",
        "tokenizer_asset_hashes": profile["tokenizer_assets"],
        "runtime_versions": versions,
        "evaluator_revision": revision,
        "native_inference_started_unix": time.time(),
        "bootstrap_started_unix": int(Path("/tmp/bfcl-bootstrap-start").read_text()),
    }
    canonical = root / "bfcl-protocol-runtime.json"
    if canonical.exists():
        previous = json.loads(canonical.read_text())
        keys = ["protocol_hash", "checkpoint", "tokenizer_asset_hashes", "runtime_versions", "evaluator_revision"]
        if any(previous.get(key) != record[key] for key in keys):
            raise ValueError("Native BFCL resume runtime identity differs; retain progress for review")
    else:
        canonical.write_text(json.dumps(record, indent=2) + "\n")
    attempts = root / "runtime-attempts"
    attempts.mkdir(exist_ok=True)
    (attempts / f"{time.time_ns()}.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record the registered native BFCL runtime")
    parser.add_argument("command", choices=("record-runtime",))
    parser.add_argument("payload")
    args = parser.parse_args()
    record_runtime(json.loads(base64.b64decode(args.payload)))
