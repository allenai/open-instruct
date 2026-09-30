"""Managed teacher-free Qwen Megatron verifier-GRPO mechanics workflow."""

import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

from open_instruct import logger_utils
from open_instruct.miles import (
    megatron_grpo_args,
    megatron_grpo_audit,
    megatron_grpo_convert,
    opd_runtime,
    run_data,
    workflow,
)
from open_instruct.miles.errors import InputError

logger = logger_utils.setup_logger(__name__)


def execute(spec):
    root = Path(spec.output["root"])
    with workflow.run_directory(spec) as (_, state):
        model = workflow.prepare_model(spec)
        descriptor = json.loads((Path(model) / "config.json").read_text())
        if descriptor.get("model_type") != "qwen3":
            raise InputError("Megatron GRPO qualification requires the dense Qwen3 model family")
        inf, trainer, training = (spec.document[k] for k in ("inference", "trainer", "training"))
        data_key = workflow.fingerprint(
            {
                "data": spec.document["data"],
                "model": workflow.model_identity(model),
                "prompt_limit": inf["max_prompt_length"],
            }
        )[:16]
        prepared = {
            "model": model,
            "identities": {"model": workflow.model_identity(model)},
            "data": run_data.prepare_data(
                spec.document["data"],
                Path(model),
                Path(spec.output["assets"]) / f"data-{data_key}",
                max_prompt_length=inf["max_prompt_length"],
                seed=spec.document["data"]["seed"],
            ),
        }
        workflow.write_json(root / "prepared.json", prepared)
        if training["phase"] == "prepare":
            workflow.write_json(root / "result.json", {"status": "prepared", "algorithm": "grpo"})
            state.update(status="complete", finished_unix=time.time())
            workflow.write_json(root / "workflow.json", state)
            return
        native = importlib.util.find_spec("miles")
        if native is None or native.origin is None:
            raise InputError("Use the qualified native Megatron/SGLang image with committed code overlay")
        miles_root = Path(native.origin).resolve().parents[1]
        environment = dict(os.environ)
        environment.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
        environment.update(
            {
                "PYTHONPATH": "/src/Megatron-LM:" + environment.get("PYTHONPATH", ""),
                "MILES_USE_LEGACY_ROLLOUT_V1": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "WANDB_MODE": spec.document["tracking"]["wandb_mode"],
                "OI_GRPO_OUTPUT": str(root),
                "OI_GRPO_REWARD_CONFIG": prepared["data"]["reward_config"],
            }
        )
        converter_path = miles_root / "tools/convert_hf_to_torch_dist.py"
        converter_source = converter_path.read_text()
        (root / "native-converter.py").write_text(converter_source)
        converter_sha = megatron_grpo_convert.verify_keep_pp1(converter_source, str(converter_path))
        workflow.write_json(
            root / "conversion-control.json",
            {
                "native_source_sha256": converter_sha,
                "CONVERT_KEEP_PP1": "1",
                "tensor_parallel_size": trainer["tensor_parallel_size"],
                "pipeline_parallel_size": 1,
            },
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
                    "/opt/core-rl/tests/miles/test_megatron_grpo.py",
                ],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        layout = spec.allocation()
        visible = environment.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(layout["gpus_per_replica"])))).split(
            ","
        )
        if len(visible) != layout["gpus_per_replica"]:
            raise InputError("Visible GPU count differs from the audited teacher-free allocation")
        environment["OI_GRPO_RAY_GPUS"] = str(layout["ray_gpus"])
        architecture = opd_runtime.model_args(miles_root, environment, spec.document["model"]["architecture"])
        tp = trainer["tensor_parallel_size"]
        checkpoint = Path(spec.output["assets"]) / (
            "learner-tp" + str(tp) + "-" + workflow.fingerprint(architecture)[:12]
        )
        if not (checkpoint / "latest_checkpointed_iteration.txt").exists():
            conversion_env = environment | {
                "CONVERT_KEEP_PP1": "1",
                "CUDA_VISIBLE_DEVICES": ",".join(visible[i] for i in layout["roles"]["trainer"][:tp]),
            }
            command = megatron_grpo_args.conversion_command(
                sys.executable, miles_root, architecture, model, checkpoint, tp
            )
            with (root / "conversion.log").open("w") as stream:
                logger.info("Starting TP%s/PP1 learner conversion; retained log: %s", tp, root / "conversion.log")
                megatron_grpo_convert.run_conversion(command, conversion_env, stream)
                logger.info("Learner conversion completed")
        arguments = megatron_grpo_args.native_arguments(spec, prepared, checkpoint, architecture)
        workflow.write_json(root / "native-arguments.json", arguments)
        preflight = "from miles.utils.arguments import parse_args; from miles.rollout.data_source import RolloutDataSourceWithBuffer; a=parse_args(); assert not a.use_opd and a.opd_kl_coef == 0 and not a.fully_async and a.kl_coef == 0 and not a.use_kl_loss and not a.use_rollout_logprobs and not a.normalize_advantages and a.rewards_normalization and a.loss_type == 'policy_loss' and a.advantage_estimator == 'grpo'; RolloutDataSourceWithBuffer(a)"
        with (root / "native-preflight.log").open("w") as stream:
            subprocess.run(
                [sys.executable, "-c", preflight, *arguments],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        process = None
        try:
            with (root / "training.log").open("a") as stream:
                process = subprocess.Popen(
                    [sys.executable, "-m", "open_instruct.miles.megatron_grpo_train", *arguments],
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                if process.wait():
                    raise RuntimeError("Native verifier GRPO failed; inspect training.log")
            marker = root / "checkpoints/latest_checkpointed_iteration.txt"
            if not marker.exists():
                raise RuntimeError("Missing recoverable Megatron checkpoint")
            audit = megatron_grpo_audit.audit(spec, prepared)
            export = audit["export"]["path"]
            reload_env = environment | {
                "CUDA_VISIBLE_DEVICES": ",".join(
                    visible[i] for i in layout["roles"]["student"][: inf["tensor_parallel_size"]]
                )
            }
            port = opd_runtime.port()
            url = f"http://127.0.0.1:{port}"
            command = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                export,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--tp",
                str(inf["tensor_parallel_size"]),
                "--mem-fraction-static",
                "0.3",
                "--context-length",
                str(inf["max_context_length"]),
                "--max-running-requests",
                "4",
                "--max-total-tokens",
                str(inf["max_context_length"] * 4),
                "--attention-backend",
                "triton",
                "--sampling-backend",
                "pytorch",
                "--disable-flashinfer-autotune",
                "--disable-cuda-graph",
                "--disable-radix-cache",
            ]
            with (root / "export-reload.log").open("w") as stream:
                process = subprocess.Popen(
                    command, env=reload_env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                )
                deadline = time.monotonic() + 900
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Fresh export server failed")
                    try:
                        info = opd_runtime.request(url + "/get_model_info")
                        if info.get("model_path") != export:
                            raise RuntimeError("Fresh serving process loaded a different model")
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Fresh export reload deadline exceeded") from None
                        time.sleep(2)
                first = json.loads((root / "verifier-rewards.jsonl").read_text().splitlines()[0])
                probe = opd_runtime.request(
                    url + "/generate",
                    {
                        "input_ids": first["tokens"][: -first["response_length"]],
                        "sampling_params": {"max_new_tokens": 32, "temperature": 0},
                        "return_logprob": True,
                    },
                    timeout=180,
                )
                entries = probe.get("meta_info", {}).get("output_token_logprobs", [])
                if not entries or not all(math.isfinite(float(item[0])) for item in entries):
                    raise RuntimeError("Fresh export produced missing or nonfinite token log probabilities")
                workflow.write_json(
                    root / "export-reload.json",
                    {
                        "passed": True,
                        "model": info,
                        "generation": probe,
                        "note": "Fresh-process reload, not numerical equivalence.",
                    },
                )
            workflow.write_json(
                root / "result.json",
                {
                    "status": "trained",
                    "algorithm": "grpo",
                    "teacher": None,
                    "checkpoint_iteration": marker.read_text().strip(),
                    "optimizer_updates": training["num_rollouts"],
                    "note": "Reward-learning mechanics, audit and fresh export reload; no throughput or learning-quality claim.",
                },
            )
            state.update(status="complete", finished_unix=time.time())
            workflow.write_json(root / "workflow.json", state)
        finally:
            opd_runtime.stop(process)
