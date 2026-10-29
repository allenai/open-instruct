"""Managed teacher and native Megatron execution for the initial OPD exercise."""

import contextlib
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from open_instruct import logger_utils
from open_instruct.miles.configuration import options
from open_instruct.miles.distillation import eopd_math, opd_prepare, opd_retention, opd_selection_audit, opd_timing
from open_instruct.miles.errors import InputError
from open_instruct.miles.execution import cluster, workflow
from open_instruct.miles.megatron_grpo import domain_balance, dppo_settings

logger = logger_utils.setup_logger(__name__)


def with_native_overrides(arguments, overrides):
    """Apply the run file's ``[miles]`` passthrough (normalized by ``opd_config``) to the native
    argument list: an option the wrapper already emits is replaced in place of its single
    occurrence (a switch turned off simply disappears), anything else is appended."""
    if not overrides:
        return list(arguments)
    index = options.option_index()
    kept, skipping = [], False
    for token in arguments:
        if token.startswith("--"):
            record = index.get(token[2:].split("=", 1)[0].replace("-", "_"))
            replaced = record is not None and record["dest"] in overrides
            skipping = replaced and "=" not in token
            if replaced:
                continue
        elif skipping:
            continue
        kept.append(token)
    return kept + options.encode_options(overrides)


def request(url, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_group(command, *, timeout, **kwargs):
    """Like ``subprocess.run(check=True)``, in a new session whose whole group is terminated afterwards.

    ``subprocess.run`` kills only the immediate child on timeout, so torchrun workers could keep their
    GPUs; terminating the group also reaps workers that outlive a finished leader."""
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        returncode = process.wait(timeout=timeout)
    finally:
        cluster.terminate(process)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


PROFILES = Path(__file__).resolve().parent / "model_profiles"


def model_args(miles_root, environment, architecture):
    """Render the Megatron architecture profile; repository profiles shadow the Miles copies."""
    directory = PROFILES if (PROFILES / f"{architecture}.py").exists() else miles_root / "scripts/models"
    code = (
        "import sys; from pathlib import Path; "
        "from miles.utils.external_utils.model_args_utils import load_model_args; "
        "print(load_model_args(sys.argv[1], model_script_dir=Path(sys.argv[2])))"
    )
    return shlex.split(
        subprocess.check_output([sys.executable, "-c", code, architecture, str(directory)], env=environment, text=True)
    )


def eopd_settings(spec):
    """EOPD settings for the hooks and the custom loss, forwarded as OI_OPD_EOPD_* environment."""
    return eopd_math.Settings.from_distillation(spec.document["distillation"])


def teacher_concurrency_environment(spec):
    """Per-teacher admission limits for ``opd_hooks.reward``: one ``OI_OPD_TEACHER_CONCURRENCY`` for
    the single teacher, ``OI_OPD_TEACHER_CONCURRENCIES="math:32,ifeval:32"`` for routed teachers."""
    teachers = spec.teachers()
    if not spec.routed:
        return {"OI_OPD_TEACHER_CONCURRENCY": str(teachers["teacher"]["concurrency"])}
    value = ",".join(f"{name}:{settings['concurrency']}" for name, settings in teachers.items())
    return {opd_prepare.TEACHER_CONCURRENCIES_ENV: value}


def native_arguments(spec, prepared, checkpoint, teacher_url, architecture):
    """``teacher_url`` is the single teacher's scoring URL, or ``{name: url}`` for routed teachers."""
    doc, root = spec.document, Path(spec.output["root"])
    inf, training, trainer, tracking = doc["inference"], doc["training"], doc["trainer"], doc["tracking"]
    batch = inf["rollout_batch_size"] * inf["samples_per_prompt"]
    optimizer_steps = training["optimizer_steps_per_rollout"]
    optimizer = doc["optimizer"]
    values = {
        "train-backend": "megatron",
        "hf-checkpoint": prepared["model"],
        "ref-load": str(checkpoint),
        "save": str(root / "checkpoints"),
        "save-interval": training["save_interval"],
        "save-hf": str(root / "hf-{rollout_id}"),
        "num-rollout": training["num_rollouts"],
        "actor-num-nodes": 1,
        "actor-num-gpus-per-node": trainer["gpus"],
        "num-gpus-per-node": trainer["gpus"] + inf["gpus"],
        "rollout-num-gpus": inf["gpus"],
        "rollout-num-gpus-per-engine": inf["tensor_parallel_size"],
        "tensor-model-parallel-size": trainer["tensor_parallel_size"],
        "pipeline-model-parallel-size": 1,
        "context-parallel-size": 1,
        "expert-model-parallel-size": 1,
        "expert-tensor-parallel-size": 1,
        "micro-batch-size": 1,
        "global-batch-size": batch // optimizer_steps,
        "rollout-batch-size": inf["rollout_batch_size"],
        "n-samples-per-prompt": inf["samples_per_prompt"],
        "prompt-data": prepared["data"]["prompt_data"],
        "input-key": "input",
        "label-key": "label",
        "metadata-key": "metadata",
        "rollout-max-response-len": inf["max_response_length"],
        "rollout-temperature": inf["temperature"],
        "rollout-top-p": inf["top_p"],
        "rollout-seed": doc["data"]["seed"],
        "seed": doc["data"]["seed"],
        "sglang-context-length": inf["max_context_length"],
        "seq-length": inf["max_context_length"],
        "sglang-mem-fraction-static": 0.6,
        "sglang-max-running-requests": inf["max_running_requests"],
        "sglang-max-total-tokens": inf["max_context_length"] * inf["max_running_requests"],
        "sglang-attention-backend": "triton",
        "sglang-sampling-backend": "pytorch",
        "sglang-router-request-timeout-secs": (doc["teacher_pool"] if spec.routed else doc["teacher"])[
            "request_timeout"
        ],
        "advantage-estimator": "grpo",
        "opd-type": "sglang",
        "opd-kl-coef": doc["distillation"]["kl_coef"],
        "opd-log-prob-top-k": 0,
        "custom-rm-path": "open_instruct.miles.distillation.opd_hooks.reward",
        "custom-reward-post-process-path": "open_instruct.miles.distillation.opd_hooks.post_process",
        "eval-interval": training["eval_interval"] or training["num_rollouts"],
        "n-samples-per-eval-prompt": inf["eval_samples_per_prompt"],
        "eval-max-response-len": inf["eval_max_response_length"] or inf["max_response_length"],
        "eval-temperature": inf["eval_temperature"],
        "eval-top-p": inf["eval_top_p"],
        "optimizer": "adam",
        "lr": optimizer["learning_rate"],
        "lr-decay-style": optimizer["lr_decay_style"],
        "lr-warmup-iters": optimizer["lr_warmup_iters"],
        "min-lr": optimizer["min_lr"],
        "weight-decay": optimizer["weight_decay"],
        "adam-beta1": optimizer["adam_beta1"],
        "adam-beta2": optimizer["adam_beta2"],
        "clip-grad": 1.0,
        "attention-dropout": 0.0,
        "attention-backend": "flash",
        "qkv-format": "thd",
        "hidden-dropout": 0.0,
        "recompute-granularity": "full",
        "recompute-method": "uniform",
        "recompute-num-layers": 1,
        "megatron-to-hf-mode": "raw",
        "dump-details": str(root / "debug"),
        "wandb-project": tracking["wandb_project"],
        "wandb-group": spec.name,
        "wandb-dir": str(root / "wandb"),
    }
    if not spec.routed:
        values["rm-url"] = teacher_url
    if not doc["miles"].get("fully_async", False):
        values["eval-function-path"] = "open_instruct.miles.distillation.opd_hooks.evaluate"
    if optimizer["lr_decay_style"] != "constant":
        # Megatron schedules over optimizer iterations, not rollouts.
        values["lr-decay-iters"] = training["num_rollouts"] * optimizer_steps
    if training["keep_checkpoints"]:
        # training.keep_checkpoints: Miles calls the hook on rank 0 after each save and its HF
        # export; it retains the newest OI_OPD_KEEP_CHECKPOINTS Megatron saves (opd_retention).
        values["custom-megatron-post-save-hook-path"] = opd_retention.POST_SAVE_HOOK
    if training["resume"] and (root / "checkpoints" / "latest_checkpointed_iteration.txt").exists():
        # Megatron restores weights, optimizer and RNG from the newest iteration and Miles
        # restores the data cursor from checkpoints/rollout, then continues at the next
        # rollout id. A first launch has no marker and starts from the converted learner.
        values["load"] = str(root / "checkpoints")
        # Megatron's load_checkpoint restores the LR scheduler's step count from the
        # checkpoint; Miles then steps it again by `iteration * global_batch_size` unless
        # this switch is set (miles/backends/megatron_utils/model.py,
        # initialize_model_and_optimizer). Without it every resume jumps the cosine
        # schedule ahead by one optimizer step per completed rollout and a late resume
        # trains at LR 0 (arm 2 attempts 2 and 3, 2026-09-19/20). The switch also makes
        # the scheduler take max_lr/min_lr/decay from the checkpoint, so a run file may
        # not change the LR schedule across a resume. Defined by Megatron's dataclass
        # config, so it is not in options.json and cannot come through [miles].
        resume_switches = ["--use-checkpoint-opt-param-scheduler"]
    else:
        resume_switches = []
    args = list(architecture)
    for key, value in values.items():
        args += [f"--{key}", str(value)]
    args += resume_switches
    args += [
        "--use-opd",
        "--rollout-shuffle",
        "--sequence-parallel",
        "--disable-grpo-std-normalization",
        "--sglang-disable-flashinfer-autotune",
        "--sglang-disable-cuda-graph",
        "--sglang-disable-radix-cache",
        # The student rollout router fronts a fixed set of engines on one node. Its
        # circuit breaker opens for 60s after ten failed requests (aborts at the end
        # of a rollout count), and Miles retries a request at most 60 times one second
        # apart, so an open circuit can exhaust the retries and fail the rollout.
        "--router-disable-circuit-breaker",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
    ]
    if training["loss_aggregation"] in ("token", "domain"):
        # "domain" also reweights tokens by domain in the DPPO loss (domain_balance).
        # Token-mean over the mini-batch (verl's loss_agg_mode=token-mean) instead of Miles'
        # per-response mean.
        args.append("--calculate-per-token-loss")
    if doc["distillation"]["use_rollout_logprobs"]:
        # Score the student side of the reverse KL with the rollout engine's log-probs,
        # matching Open Instruct's --use_vllm_logprobs behaviour instead of the trainer's
        # pre-update forward pass.
        args.append("--use-rollout-logprobs")
    if doc["distillation"]["eopd"]:
        # EOPD: upstream policy loss plus the entropy-gated forward KL over the teacher's top-k.
        # The hooks request the top-k and the loss reads OI_OPD_EOPD_* (see eopd_settings).
        args += [
            "--loss-type",
            "custom_loss",
            "--custom-loss-function-path",
            "open_instruct.miles.distillation.eopd_loss.policy_loss",
        ]
    if doc["distillation"]["policy_loss"] == "dppo":
        # DPPO's trust-region mask replaces the clipped surrogate on the reverse-KL advantages.
        args += ["--loss-type", "custom_loss", "--custom-loss-function-path", dppo_settings.LOSS_FUNCTION]
    if tracking["wandb_mode"] != "disabled":
        args += ["--use-wandb", "--wandb-mode", tracking["wandb_mode"]]
        if tracking["wandb_entity"]:
            args += ["--wandb-team", tracking["wandb_entity"]]
    if spec.routed:
        # Miles scores each sample with the teacher named by metadata[route_key] and leaves --rm-url
        # unused, so it is not set; an unknown name fails the rollout unless a "default" teacher exists.
        args += ["--opd-teacher-urls", *(f"{name}={url}" for name, url in teacher_url.items())]
        args += ["--opd-teacher-key", doc["teacher_pool"]["route_key"]]
    args += ["--eval-prompt-data", *prepared["data"]["eval_prompt_data"]]
    return with_native_overrides(args, doc["miles"])


def teacher_command(spec, prepared, teacher_port, name="teacher"):
    """Build teacher and export-reload serving arguments independently of student flags."""
    settings = spec.teachers()[name]
    context = spec.document["inference"]["max_context_length"]
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        opd_prepare.teacher_paths(prepared)[name],
        "--host",
        "127.0.0.1",
        "--port",
        str(teacher_port),
        "--tp",
        str(len(settings["gpus"])),
        "--mem-fraction-static",
        str(settings["mem_fraction_static"]),
        "--context-length",
        str(context),
        "--max-running-requests",
        str(settings["concurrency"]),
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
        "--max-total-tokens",
        str(context * settings["concurrency"]),
        "--disable-flashinfer-autotune",
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]
    if settings["enable_fp32_lm_head"]:
        command.append("--enable-fp32-lm-head")
    if settings["enable_deterministic_inference"]:
        command.append("--enable-deterministic-inference")
    return command


def wait_for_server(process, url, model_path, timeout, failure):
    """Poll ``/get_model_info`` until the server reports ``model_path``; returns that info."""
    deadline = time.monotonic() + timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError(failure)
        try:
            info = request(url + "/get_model_info")
            if info.get("model_path") != model_path:
                raise RuntimeError(f"Server at {url} loaded an unexpected model: {info}")
            return info
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Readiness deadline exceeded for {url}") from None
            time.sleep(2)


def teacher_log_name(spec, name):
    return f"teacher-{name}.log" if spec.routed else "teacher.log"


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
                "MILES_USE_LEGACY_ROLLOUT_V1": "0" if spec.document["miles"].get("fully_async") else "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "WANDB_MODE": spec.document["tracking"]["wandb_mode"],
                "OI_OPD_OUTPUT": str(root),
                "OI_OPD_REWARD_CONFIG": prepared["data"]["reward_config"],
                **teacher_concurrency_environment(spec),
                opd_retention.KEEP_ENV: str(spec.document["training"]["keep_checkpoints"]),
                "CONVERT_KEEP_PP1": "1",
                **eopd_settings(spec).environment(),
                **(spec.dppo_settings().environment() if spec.dppo_settings() else {}),
                **domain_balance.environment(spec.document["training"]["loss_aggregation"]),
                **(
                    {
                        opd_prepare.EOS_REMAP_ENV: opd_prepare.eos_remap_environment(
                            opd_prepare.teacher_eos_remap(prepared["model"])
                        )
                    }
                    if spec.document["model"]["align_eos_with_teacher"]
                    else {}
                ),
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
                    *(["/opt/core-rl/tests/miles/test_opd_multi_teacher_hooks.py"] if spec.routed else []),
                    *(
                        ["/opt/core-rl/tests/miles/test_opd_async.py"]
                        if spec.document["miles"].get("fully_async")
                        else []
                    ),
                ],
                env=domain_balance.test_environment(opd_selection_audit.preflight_environment(environment)),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        allocation = spec.allocation()
        total, roles = allocation["gpus_per_replica"], allocation["roles"]
        visible = environment.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(total)))).split(",")
        if len(visible) != total:
            raise InputError(f"Expected {total} visible GPU devices; got {visible}")
        devices = {role: ",".join(visible[index] for index in indices) for role, indices in roles.items()}
        tensor_parallel = spec.document["trainer"]["tensor_parallel_size"]
        architecture = model_args(miles_root, environment, spec.document["model"]["architecture"])
        checkpoint = Path(spec.output["assets"]) / (
            Path(prepared["model"]).name + f"-tp{tensor_parallel}-" + workflow.fingerprint(architecture)[:12]
        )
        conversion_env = environment | {
            "CUDA_VISIBLE_DEVICES": ",".join(visible[index] for index in roles["trainer"][:tensor_parallel])
        }
        if not (checkpoint / "latest_checkpointed_iteration.txt").exists():
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={tensor_parallel}",
                str(miles_root / "tools/convert_hf_to_torch_dist.py"),
                *architecture,
                "--hf-checkpoint",
                prepared["model"],
                "--save",
                str(checkpoint),
                "--tensor-model-parallel-size",
                str(tensor_parallel),
            ]
            logger.info("Converting the learner to Megatron")
            with (root / "conversion.log").open("w") as stream:
                run_group(command, env=conversion_env, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
        teachers = spec.teachers()
        paths = opd_prepare.teacher_paths(prepared)
        ports = {}
        for name in teachers:
            ports[name] = port()
            while ports[name] in list(ports.values())[:-1]:
                ports[name] = port()
        urls = {name: f"http://127.0.0.1:{number}" for name, number in ports.items()}
        commands = {name: teacher_command(spec, prepared, number, name) for name, number in ports.items()}
        teacher_envs = {
            name: environment | {"CUDA_VISIBLE_DEVICES": ",".join(visible[index] for index in settings["gpus"])}
            for name, settings in teachers.items()
        }
        scoring = {name: url + "/generate" for name, url in urls.items()}
        arguments = native_arguments(
            spec, prepared, checkpoint, scoring if spec.routed else scoring["teacher"], architecture
        )
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
        servers = {}
        learner = monitor = None
        try:
            if opd_timing.enabled():
                monitor = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "open_instruct.miles.distillation.opd_gpu_monitor",
                        "--output",
                        str(root / "gpu-telemetry.jsonl"),
                        "--roles",
                        json.dumps(allocation["roles"]),
                    ],
                    env=environment,
                    start_new_session=True,
                )
            with contextlib.ExitStack() as logs:
                train_log = logs.enter_context((root / "training.log").open("a"))
                preflight = {}
                # One server at a time: teachers sharing a GPU size their memory pools from what the
                # previous ones left free, so each must finish loading before the next starts.
                for name, settings in teachers.items():
                    log_name = teacher_log_name(spec, name)
                    teacher_log = logs.enter_context((root / log_name).open("w"))
                    servers[name] = subprocess.Popen(
                        commands[name],
                        env=teacher_envs[name],
                        stdout=teacher_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    info = wait_for_server(
                        servers[name],
                        urls[name],
                        paths[name],
                        settings["startup_timeout"],
                        f"Teacher {name} exited during startup; see {log_name}",
                    )
                    probe = request(
                        urls[name] + "/generate",
                        {
                            "input_ids": [1, 2, 3, 4],
                            "sampling_params": {"max_new_tokens": 0},
                            "return_logprob": True,
                            "logprob_start_len": 0,
                        },
                        timeout=180,
                    )
                    preflight[name] = {"identity": info, "score_probe": probe}
                workflow.write_json(
                    root / "teacher-preflight.json", preflight if spec.routed else preflight["teacher"]
                )
                workflow.write_json(root / "native-arguments.json", arguments)
                learner_env = environment | {
                    "CUDA_VISIBLE_DEVICES": devices["trainer"] + "," + devices["student"],
                    "OI_OPD_RAY_GPUS": str(allocation["ray_gpus"]),
                }
                learner = subprocess.Popen(
                    [sys.executable, "-m", "open_instruct.miles.distillation.opd_train", *arguments],
                    env=learner_env,
                    stdout=train_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while learner.poll() is None:
                    failed = [name for name, server in servers.items() if server.poll() is not None]
                    if failed:
                        raise RuntimeError(f"Teacher {', '.join(failed)} failed during training")
                    time.sleep(2)
                if learner.returncode:
                    raise RuntimeError(f"Native Miles failed ({learner.returncode}); see training.log")
            marker = root / "checkpoints/latest_checkpointed_iteration.txt"
            if not marker.exists() or not (root / "teacher-scores.jsonl").exists():
                raise RuntimeError("Training exited without checkpoint or teacher-score evidence")
            for name in list(servers):
                cluster.terminate(servers.pop(name))
            subprocess.run(
                [sys.executable, "-m", "open_instruct.miles.distillation.opd_audit", str(root)],
                env=environment,
                check=True,
                timeout=900,
            )
            audit = json.loads((root / "audit.json").read_text())
            export_path = audit["export"]["path"]
            # The export reloads on the first teacher's GPUs and serving settings.
            reload = next(iter(teachers))
            command = commands[reload]
            command[command.index("--model-path") + 1] = export_path
            with (root / "export-reload.log").open("w") as stream:
                servers["export"] = subprocess.Popen(
                    command, env=teacher_envs[reload], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                )
                info = wait_for_server(
                    servers["export"],
                    urls[reload],
                    export_path,
                    teachers[reload]["startup_timeout"],
                    "Export reload failed; see export-reload.log",
                )
                record = json.loads((root / "teacher-scores.jsonl").read_text().splitlines()[0])
                probe = request(
                    urls[reload] + "/generate",
                    {
                        "input_ids": record["tokens"][: -record["response_length"]],
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
                    **(
                        {"teachers": prepared["identities"]["teachers"]}
                        if spec.routed
                        else {"teacher": prepared["identities"]["teacher"]}
                    ),
                    "note": "Tiny OPD mechanics and export reload passed; no learning-quality claim.",
                },
            )
            state.update(status="complete", finished_unix=time.time())
            workflow.write_json(root / "workflow.json", state)
        finally:
            cluster.terminate(learner)
            for server in servers.values():
                cluster.terminate(server)
            cluster.terminate(monitor)
