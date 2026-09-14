"""Managed teacher and native Megatron execution for the initial OPD exercise."""

import importlib.util
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from open_instruct import logger_utils
from open_instruct.miles import opd_prepare, workflow
from open_instruct.miles.errors import InputError

logger = logger_utils.setup_logger(__name__)


def request(url, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def model_args(miles_root, environment):
    return shlex.split(
        subprocess.check_output(
            [sys.executable, str(miles_root / "miles/utils/external_utils/model_args_utils.py"), "qwen3.5-4B"],
            env=environment,
            text=True,
        )
    )


def native_arguments(spec, prepared, checkpoint, teacher_url, architecture):
    doc, root = spec.document, Path(spec.output["root"])
    inf, training = doc["inference"], doc["training"]
    batch = inf["rollout_batch_size"] * inf["samples_per_prompt"]
    values = {
        "train-backend": "megatron",
        "hf-checkpoint": prepared["model"],
        "ref-load": str(checkpoint),
        "save": str(root / "checkpoints"),
        "save-interval": training["save_interval"],
        "save-hf": str(root / "hf-{rollout_id}"),
        "num-rollout": training["num_rollouts"],
        "actor-num-nodes": 1,
        "actor-num-gpus-per-node": 2,
        "num-gpus-per-node": 3,
        "rollout-num-gpus": 1,
        "rollout-num-gpus-per-engine": 1,
        "tensor-model-parallel-size": 2,
        "pipeline-model-parallel-size": 1,
        "context-parallel-size": 1,
        "expert-model-parallel-size": 1,
        "expert-tensor-parallel-size": 1,
        "micro-batch-size": 1,
        "global-batch-size": batch,
        "rollout-batch-size": inf["rollout_batch_size"],
        "n-samples-per-prompt": inf["samples_per_prompt"],
        "prompt-data": prepared["data"]["prompt_data"],
        "input-key": "input",
        "label-key": "label",
        "metadata-key": "metadata",
        "rollout-max-response-len": inf["max_response_length"],
        "rollout-temperature": inf["temperature"],
        "rollout-seed": doc["data"]["seed"],
        "seed": doc["data"]["seed"],
        "sglang-context-length": inf["max_context_length"],
        "seq-length": inf["max_context_length"],
        "sglang-mem-fraction-static": 0.6,
        "sglang-max-running-requests": 8,
        "sglang-max-total-tokens": inf["max_context_length"] * 8,
        "sglang-attention-backend": "triton",
        "sglang-sampling-backend": "pytorch",
        "sglang-router-request-timeout-secs": doc["teacher"]["request_timeout"],
        "advantage-estimator": "grpo",
        "opd-type": "sglang",
        "opd-kl-coef": doc["distillation"]["kl_coef"],
        "opd-log-prob-top-k": 0,
        "custom-rm-path": "open_instruct.miles.opd_hooks.reward",
        "custom-reward-post-process-path": "open_instruct.miles.opd_hooks.post_process",
        "rm-url": teacher_url,
        "eval-function-path": "open_instruct.miles.opd_hooks.evaluate",
        "eval-interval": training["num_rollouts"],
        "n-samples-per-eval-prompt": 1,
        "eval-max-response-len": inf["max_response_length"],
        "eval-temperature": 0.0,
        "optimizer": "adam",
        "lr": doc["optimizer"]["learning_rate"],
        "lr-decay-style": "constant",
        "lr-warmup-iters": 0,
        "weight-decay": 0.0,
        "adam-beta1": 0.9,
        "adam-beta2": 0.98,
        "clip-grad": 1.0,
        "attention-dropout": 0.0,
        "attention-backend": "flash",
        "qkv-format": "bshd",
        "hidden-dropout": 0.0,
        "recompute-granularity": "full",
        "recompute-method": "uniform",
        "recompute-num-layers": 1,
        "megatron-to-hf-mode": "raw",
        "dump-details": str(root / "debug"),
        "wandb-project": "open-instruct-opd",
        "wandb-group": spec.name,
        "wandb-dir": str(root / "wandb"),
    }
    if training["resume"]:
        values["load"] = str(root / "checkpoints")
    args = list(architecture)
    for key, value in values.items():
        args += [f"--{key}", str(value)]
    args += [
        "--use-opd",
        "--rollout-shuffle",
        "--sequence-parallel",
        "--disable-grpo-std-normalization",
        "--sglang-disable-flashinfer-autotune",
        "--sglang-disable-cuda-graph",
        "--sglang-disable-radix-cache",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--use-wandb",
    ]
    args += ["--eval-prompt-data", *prepared["data"]["eval_prompt_data"]]
    return args


def execute(spec):
    root = Path(spec.output["root"])
    root.mkdir(parents=True, exist_ok=True)
    with workflow.run_directory(spec) as (_, state):
        prepared = opd_prepare.prepare(spec)
        workflow.write_json(root / "prepared.json", prepared)
        if spec.document["training"]["phase"] == "prepare":
            workflow.write_json(root / "result.json", {"status": "prepared", "assets": spec.output["assets"]})
            state.update(status="complete", finished_unix=time.time())
            workflow.write_json(root / "workflow.json", state)
            return
        native = importlib.util.find_spec("miles")
        if native is None or native.origin is None:
            raise InputError("Use the candidate Miles OPD image")
        miles_root = Path(native.origin).resolve().parents[1]
        environment = dict(os.environ)
        environment.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
        environment.update(
            {
                "PYTHONPATH": "/src/Megatron-LM:" + environment.get("PYTHONPATH", ""),
                "MILES_USE_LEGACY_ROLLOUT_V1": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "WANDB_MODE": "offline",
                "OI_OPD_OUTPUT": str(root),
                "OI_OPD_TEACHER_CONCURRENCY": str(spec.document["teacher"]["concurrency"]),
                "CONVERT_KEEP_PP1": "1",
            }
        )
        with (root / "runtime-tests.log").open("w") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    "/opt/core-rl/tests/miles/test_opd_hooks.py",
                    "/opt/core-rl/tests/miles/test_opd_audit.py",
                    "/opt/core-rl/tests/miles/test_opd_attention.py",
                ],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        visible = environment.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
        if len(visible) != 4:
            raise InputError(f"Expected four visible GPU devices; got {visible}")
        architecture = model_args(miles_root, environment)
        checkpoint = Path(spec.output["assets"]) / (
            Path(prepared["model"]).name + "-tp2-" + workflow.fingerprint(architecture)[:12]
        )
        conversion_env = environment | {"CUDA_VISIBLE_DEVICES": ",".join(visible[:2])}
        if not (checkpoint / "latest_checkpointed_iteration.txt").exists():
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                str(miles_root / "tools/convert_hf_to_torch_dist.py"),
                *architecture,
                "--hf-checkpoint",
                prepared["model"],
                "--save",
                str(checkpoint),
                "--tensor-model-parallel-size",
                "2",
            ]
            logger.info("Converting the learner to Megatron")
            with (root / "conversion.log").open("w") as stream:
                subprocess.run(
                    command, env=conversion_env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1800
                )
        teacher_port = port()
        url = f"http://127.0.0.1:{teacher_port}"
        teacher_env = environment | {"CUDA_VISIBLE_DEVICES": visible[3]}
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            prepared["teacher"],
            "--host",
            "127.0.0.1",
            "--port",
            str(teacher_port),
            "--tp",
            "1",
            "--mem-fraction-static",
            "0.6",
            "--context-length",
            str(spec.document["inference"]["max_context_length"]),
            "--max-running-requests",
            str(spec.document["teacher"]["concurrency"]),
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
            "--max-total-tokens",
            str(spec.document["inference"]["max_context_length"] * spec.document["teacher"]["concurrency"]),
            "--disable-flashinfer-autotune",
            "--disable-cuda-graph",
            "--disable-radix-cache",
        ]
        arguments = native_arguments(spec, prepared, checkpoint, url + "/generate", architecture)
        with (root / "native-preflight.log").open("w") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from miles.utils.arguments import parse_args; "
                    "from miles.rollout.data_source import RolloutDataSourceWithBuffer; "
                    "RolloutDataSourceWithBuffer(parse_args())",
                    *arguments,
                ],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        teacher = learner = None
        try:
            with (root / "teacher.log").open("w") as teacher_log, (root / "training.log").open("a") as train_log:
                teacher = subprocess.Popen(
                    command, env=teacher_env, stdout=teacher_log, stderr=subprocess.STDOUT, start_new_session=True
                )
                deadline = time.monotonic() + spec.document["teacher"]["startup_timeout"]
                while True:
                    if teacher.poll() is not None:
                        raise RuntimeError("Teacher exited during startup; see teacher.log")
                    try:
                        info = request(url + "/get_model_info")
                        if info.get("model_path") != prepared["teacher"]:
                            raise RuntimeError(f"Teacher identity differs: {info}")
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Teacher readiness deadline exceeded") from None
                        time.sleep(2)
                probe = request(
                    url + "/generate",
                    {
                        "input_ids": [1, 2, 3, 4],
                        "sampling_params": {"max_new_tokens": 0},
                        "return_logprob": True,
                        "logprob_start_len": 0,
                    },
                    timeout=180,
                )
                workflow.write_json(root / "teacher-preflight.json", {"identity": info, "score_probe": probe})
                arguments = native_arguments(spec, prepared, checkpoint, url + "/generate", architecture)
                workflow.write_json(root / "native-arguments.json", arguments)
                learner_env = environment | {"CUDA_VISIBLE_DEVICES": ",".join(visible[:3])}
                learner = subprocess.Popen(
                    [sys.executable, "-m", "open_instruct.miles.opd_train", *arguments],
                    env=learner_env,
                    stdout=train_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while learner.poll() is None:
                    if teacher.poll() is not None:
                        raise RuntimeError("Teacher failed during training")
                    time.sleep(2)
                if learner.returncode:
                    raise RuntimeError(f"Native Miles failed ({learner.returncode}); see training.log")
            marker = root / "checkpoints/latest_checkpointed_iteration.txt"
            if not marker.exists() or not (root / "teacher-scores.jsonl").exists():
                raise RuntimeError("Training exited without checkpoint or teacher-score evidence")
            stop(teacher)
            teacher = None
            subprocess.run(
                [sys.executable, "-m", "open_instruct.miles.opd_audit", str(root)],
                env=environment,
                check=True,
                timeout=900,
            )
            audit = json.loads((root / "audit.json").read_text())
            export_path = audit["export"]["path"]
            command[command.index("--model-path") + 1] = export_path
            with (root / "export-reload.log").open("w") as stream:
                teacher = subprocess.Popen(
                    command, env=teacher_env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                )
                deadline = time.monotonic() + spec.document["teacher"]["startup_timeout"]
                while True:
                    if teacher.poll() is not None:
                        raise RuntimeError("Export reload failed; see export-reload.log")
                    try:
                        info = request(url + "/get_model_info")
                        if info.get("model_path") != export_path:
                            raise RuntimeError("Export server loaded an unexpected model")
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Export reload deadline exceeded") from None
                        time.sleep(2)
                first = json.loads((root / "teacher-scores.jsonl").read_text().splitlines()[0])
                probe = request(
                    url + "/generate",
                    {
                        "input_ids": first["tokens"][: -first["response_length"]],
                        "sampling_params": {"max_new_tokens": 32, "temperature": 0},
                        "return_logprob": True,
                    },
                    timeout=180,
                )
                workflow.write_json(root / "export-reload.json", {"model": info, "generation": probe})
            workflow.write_json(
                root / "result.json",
                {
                    "status": "trained",
                    "checkpoint_iteration": marker.read_text().strip(),
                    "learner": prepared["identities"]["model"],
                    "teacher": prepared["identities"]["teacher"],
                    "note": "Tiny OPD mechanics and export reload passed; no learning-quality claim.",
                },
            )
            state.update(status="complete", finished_unix=time.time())
            workflow.write_json(root / "workflow.json", state)
        finally:
            stop(learner)
            stop(teacher)
