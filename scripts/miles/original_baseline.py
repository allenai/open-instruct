"""Frozen-data adapter for historical Open Instruct comparisons.

Run inside the original Olmo 3 image. The original trainer and verifiers remain
in that image; Adam beta2 is aligned and the initial evaluation is explicitly scheduled.
Two profiles exist: the single-verifier GSM8K control and the four-domain Dolci
basket, which also hosts the original recipe's Qwen3-32B judge inside the job.
"""

import argparse
import collections
import fcntl
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from open_instruct import dataset_transformation

ORIGINAL_TRAINER_SHA256 = "6367478b4c957cfe595745dd1549ad1884585fc7bc4b21a589cbe8ed46cee3b1"
PASSTHROUGH_TEMPLATE = "{{ messages[0]['content'] }}"
# The released Olmo 3 Think recipe's shared code-execution endpoint; the MILES
# basket grades against the same service with the same pass-rate threshold.
CODE_API_URL = "https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program"
JUDGE_PORT = 8001
JUDGE_MODEL = "Qwen/Qwen3-32B"
# Two H100s host the judge at the MILES arms' 131,072-token YaRN context: the
# bf16 weights alone fill most of one device. The trainer sees the other six.
JUDGE_GPUS = "6,7"
JUDGE_TENSOR_PARALLEL = 2
JUDGE_CONTEXT = 131072
TRAINER_GPUS = "0,1,2,3,4,5"
PROFILES = {
    "gsm8k": {
        "verifiers": {"gsm8k"},
        "eval_files": ("eval.jsonl",),
        "train_count": 6000,
        "eval_count": 512,
        "prompts": 16,
        "engines": 4,
        "steps": 200,
        "judge": False,
        "wandb_group": "olmo3-sft-learning-confidence-20260914",
    },
    "basket": {
        # Original Open Instruct verifier names, as carried by the frozen Dolci basket rows.
        "verifiers": {"math", "ifeval", "code", "code_stdio", "general-quality", "general-quality_ref"},
        "eval_files": ("math.jsonl", "ifeval.jsonl", "code.jsonl", "general.jsonl"),
        "train_count": None,
        "eval_count": 512,
        "prompts": 64,
        "engines": 2,
        "steps": 100,
        "judge": True,
        "wandb_group": "dolci-basket-32k-zero-20260914",
    },
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def ground_truth(name, target):
    """Serialize a frozen MILES verifier target as the original verifiers read labels.

    The historical verifiers take one string label: scalar answers stay strings,
    IFEval constraints and code tests are JSON strings (the original parses them).
    """
    if name == "gsm8k":
        return str(target)
    if isinstance(target, str):
        return target
    if name in ("ifeval", "code", "code_stdio") and isinstance(target, (dict, list)):
        return json.dumps(target, sort_keys=True, ensure_ascii=False)
    raise ValueError(f"Unsupported frozen target type for {name}: {type(target).__name__}")


def convert_row(row, tokenizer, *, profile="gsm8k"):
    specs = row["metadata"]["verifiers"]
    names = PROFILES[profile]["verifiers"]
    if len(specs) != 1 or specs[0]["name"] not in names or specs[0].get("weight", 1) != 1:
        if profile == "gsm8k":
            raise ValueError("Original GSM8K control requires exactly one unit-weight GSM8K verifier")
        raise ValueError(f"Original basket rows require exactly one unit-weight verifier from {sorted(names)}")
    result = {
        "messages": [{"role": "user", "content": row["input"]}],
        "ground_truth": ground_truth(specs[0]["name"], specs[0]["target"]),
        "dataset": specs[0]["name"],
        "prepared_sample_id": row["metadata"]["prepared_sample_id"],
    }
    transformed = dataset_transformation.rlvr_tokenize_v2(dict(result), tokenizer)
    ids = transformed[dataset_transformation.INPUT_IDS_PROMPT_KEY]
    if sha(encoded(ids)) != row["metadata"]["run_prompt_token_ids_sha256"]:
        raise ValueError(f"Original tokenizer changed frozen prompt tokens: {result['prepared_sample_id']}")
    if not dataset_transformation.rlvr_filter_v1(
        transformed, tokenizer, max_prompt_token_length=2048, max_token_length=34816
    ):
        raise ValueError("Original filter would remove a frozen row")
    return result


def read_jsonl(path):
    # str.splitlines() also splits valid Unicode separators inside JSON strings.
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def legacy_model(model, output):
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "olmo3" or config.get("architectures") != ["Olmo3ForCausalLM"]:
        raise ValueError("Historical alias requires a public Olmo3ForCausalLM checkpoint")
    shapes = {}
    for shard in model.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 -- safe_open is not an iterable mapping
                if name.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight")):
                    if name in shapes:
                        raise ValueError(f"Duplicate checkpoint tensor: {name}")
                    shapes[name] = handle.get_slice(name).get_shape()
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    expected = {
        f"model.layers.{layer}.self_attn.{kind}_norm.weight": [heads * head_dim]
        for layer in range(config["num_hidden_layers"])
        for kind, heads in (("q", config["num_attention_heads"]), ("k", config["num_key_value_heads"]))
    }
    if shapes != expected:
        raise ValueError("Checkpoint Q/K norms do not match the historical global-normalization implementation")
    output.mkdir()
    links = {}
    for path in model.iterdir():
        if path.is_file() and path.name != "config.json":
            (output / path.name).symlink_to(path.resolve())
            links[path.name] = {"target": str(path.resolve()), "size": path.stat().st_size}
    config.update(model_type="olmo2-retrofit", architectures=["Olmo2RetrofitForCausalLM"])
    (output / "config.json").write_bytes(encoded(config))
    return {"source_config_sha256": sha((model / "config.json").read_bytes()), "norm_shapes": shapes, "links": links}


def restore_public_exports(output, source):
    restored = []
    for path in (output / "model").rglob("config.json"):
        config = json.loads(path.read_text())
        if config.get("model_type") != "olmo2-retrofit":
            continue
        if not list(path.parent.glob("*.safetensors")) and not list(path.parent.glob("pytorch_model*.bin")):
            continue
        config.update(model_type="olmo3", architectures=["Olmo3ForCausalLM"])
        path.write_bytes(encoded(config))
        for name in (
            "tokenizer_config.json",
            "chat_template.jinja",
            "special_tokens_map.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        ):
            if (source / name).is_file():
                shutil.copyfile(source / name, path.parent / name)
        restored.append(str(path.parent))
    if not restored:
        raise ValueError("Original trainer exited without a public HF model export")
    return restored


def frozen_splits(source, profile):
    """Read the frozen MILES splits; basket held-out domains merge into one eval file."""
    settings = PROFILES[profile]
    train = read_jsonl(source / "train.jsonl")
    evaluation = []
    for filename in settings["eval_files"]:
        rows = read_jsonl(source / filename)
        if profile == "basket" and len(rows) != settings["eval_count"] // len(settings["eval_files"]):
            raise ValueError(
                f"Expected {settings['eval_count'] // len(settings['eval_files'])} frozen rows in {filename}"
            )
        evaluation.extend(rows)
    if settings["train_count"] is not None and len(train) != settings["train_count"]:
        raise ValueError(f"Expected {settings['train_count']} frozen train rows")
    if not train:
        raise ValueError("Frozen training split is empty")
    if len(evaluation) != settings["eval_count"]:
        raise ValueError(f"Expected {settings['eval_count']} frozen eval rows")
    return {"train": train, "eval": evaluation}


def prepare(model, source, output, *, profile="gsm8k"):
    if output.exists():
        raise ValueError("Prepared comparison directory already exists; verify or use a fresh path")
    manifest = json.loads((source / "manifest.json").read_text())
    for filename, expected in manifest["outputs"].items():
        if sha((source / filename).read_bytes()) != expected:
            raise ValueError(f"Core prepared artifact changed: {filename}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".original-gsm8k-", dir=output.parent))
    try:
        alias = legacy_model(model, staging / "legacy-model")
        token_root = staging / "tokenizer"
        token_root.mkdir()
        tokenizer = dataset_transformation.TokenizerConfig(tokenizer_name_or_path=str(model)).tokenizer
        original_template = tokenizer.chat_template
        tokenizer.chat_template = PASSTHROUGH_TEMPLATE
        tokenizer.save_pretrained(token_root)
        shutil.copyfile(model / "config.json", token_root / "config.json")
        # Check the exact loader that training uses, including its pad-token rules.
        tokenizer = dataset_transformation.TokenizerConfig(tokenizer_name_or_path=str(token_root)).tokenizer
        receipt = {
            "model": str(model),
            "source": str(source),
            "profile": profile,
            "model_alias": alias,
            "splits": {},
            "original_chat_template_sha256": sha(original_template.encode()),
            "adapter_chat_template": PASSTHROUGH_TEMPLATE,
            "source_manifest_sha256": sha((source / "manifest.json").read_bytes()),
        }
        identities = []
        for split, rows in frozen_splits(source, profile).items():
            converted = [convert_row(row, tokenizer, profile=profile) for row in rows]
            ids = {row["prepared_sample_id"] for row in converted}
            if profile == "gsm8k" and len(ids) != len(converted):
                raise ValueError("Duplicate source identities")
            identities.append(ids)
            raw = b"".join(encoded(row) for row in converted)
            (staging / f"{split}.jsonl").write_bytes(raw)
            smoke_rows = converted[: (256 if profile == "basket" else 64) if split == "train" else 8]
            (staging / f"smoke-{split}.jsonl").write_bytes(b"".join(encoded(row) for row in smoke_rows))
            receipt["splits"][split] = {
                "rows": len(converted),
                "sha256": sha(raw),
                "all_prompt_tokens_equal": True,
                "datasets": dict(sorted(collections.Counter(row["dataset"] for row in converted).items())),
            }
        if identities[0] & identities[1]:
            raise ValueError("Training and heldout identities overlap")
        receipt["files"] = {
            str(p.relative_to(staging)): sha(p.read_bytes())
            for p in staging.rglob("*")
            if p.is_file() and not p.is_symlink()
        }
        (staging / "preparation.json").write_bytes(encoded(receipt))
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print("ORIGINAL_BASELINE_PREPARATION_PASSED", json.dumps(receipt), flush=True)


def patch_trainer(text, *, keep_zero_advantage_groups=False):
    changes = {
        "adam_alignment": {
            "before": "torch.optim.AdamW(optim_params, lr=args.learning_rate, fused=args.fused_optimizer)",
            "after": "torch.optim.AdamW(optim_params, lr=args.learning_rate, fused=args.fused_optimizer, betas=(0.9, 0.95), eps=1e-8)",
        },
        "update_accounting": {
            "before": "        if (\n            args.checkpoint_state_freq > 0\n",
            "after": (
                "        with open(os.environ['OI_ORIGINAL_BASELINE_UPDATE_LEDGER'], 'a') as update_log:\n"
                "            update_log.write(json.dumps({'driver_step': training_step}) + '\\n')\n"
                "        if (\n            args.checkpoint_state_freq > 0\n"
            ),
        },
        "export_generation_metadata": {
            "before": "                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict)\n",
            "after": (
                "                model_to_save.generation_config.do_sample = True\n"
                "                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict)\n"
            ),
        },
        "retain_partial_evaluation": {
            "before": "        # Accumulate evaluation results from all vLLM engines\n",
            "after": (
                "        # The evaluator is the only consumer. Wait for a whole round before\n"
                "        # removing results; a short poll must not discard a partial round.\n"
                "        eval_deadline = time.monotonic() + timeout\n"
                "        while evaluation_inference_results_Q.qsize() < num_eval_prompts:\n"
                "            remaining = eval_deadline - time.monotonic()\n"
                "            if remaining <= 0:\n"
                "                raise Empty\n"
                "            time.sleep(min(0.01, remaining))\n"
                "        # Accumulate evaluation results from all vLLM engines\n"
            ),
        },
        "initial_evaluation": {
            "before": "            training_step % args.local_eval_every == 0\n",
            "after": "            (training_step % args.local_eval_every == 0 or (training_step == 1 and args.eval_on_step_0))\n",
        },
    }
    if keep_zero_advantage_groups:
        changes["retain_zero_advantage_groups"] = {
            "before": "            non_zero_gradient_index = np.where(expanded_mask)[0]\n",
            "after": "            non_zero_gradient_index = np.arange(len(scores))  # Core comparison: retain zero-advantage groups\n",
        }
    for name, change in changes.items():
        if text.count(change["before"]) != 1:
            raise ValueError(f"Cannot unambiguously apply original benchmark adjustment: {name}")
        text = text.replace(change["before"], change["after"])
    return text, changes


def completion_record(steps, updates, exports, invocation):
    """Do not label an export-only or entirely filtered run a training success."""
    driver_steps = [row["driver_step"] for row in updates]
    if not driver_steps:
        raise RuntimeError("Original baseline exported weights but completed no optimizer updates")
    if any(type(step) is not int or not 1 <= step <= steps for step in driver_steps):
        raise ValueError("Optimizer update ledger contains an invalid driver step")
    if driver_steps != sorted(set(driver_steps)):
        raise ValueError("Optimizer update ledger must contain increasing, unique driver steps")
    return {
        "driver_steps": steps,
        "completed_updates": len(updates),
        "optimizer_driver_steps": updates,
        "public_exports": exports,
        "invocation_sha256": sha(encoded(invocation)),
    }


def prepare_resume(output, record, *, steps=200):
    """Check the unchanged recipe and native rank metadata before resuming."""
    previous = json.loads((output / "invocation.json").read_text())
    for key in ("command", "original_source_sha256", "model_alias"):
        if previous[key] != record[key]:
            raise ValueError(f"Resume changes the original run's {key}")
    if (output / "completion.json").exists():
        raise ValueError("Original run is already complete")
    root = output / "checkpoints"
    tag = (root / "latest").read_text().strip()
    if not tag.startswith("global_step") or not tag[11:].isdigit():
        raise ValueError("Invalid native checkpoint tag")
    checkpoint = root / tag
    newer = [p.name for p in root.glob("global_step*") if p.name[11:].isdigit() and int(p.name[11:]) > int(tag[11:])]
    if newer:
        raise ValueError(f"Inspect incomplete newer checkpoints before resuming: {newer}")
    clocks = []
    for rank in range(4):
        state = torch.load(
            checkpoint / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt", map_location="cpu", weights_only=False
        )
        if "rng_states" not in state or type(state.get("training_step")) is not int:
            raise ValueError("Checkpoint lacks optimizer update clock or RNG state")
        clocks.append(state["training_step"])
        shard = checkpoint / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        if not shard.is_file() or shard.stat().st_size == 0:
            raise ValueError(f"Missing optimizer shard for rank {rank}")
    if len(set(clocks)) != 1 or not 0 < clocks[0] < steps:
        raise ValueError(f"Invalid checkpoint update clocks: {clocks}")
    ledger = output / "optimizer-updates.jsonl"
    updates = read_jsonl(ledger)
    retained = [u for u in updates if u["driver_step"] <= clocks[0]]
    if not retained or retained[-1]["driver_step"] != clocks[0]:
        raise ValueError("Update ledger does not cover the saved checkpoint")
    attempt = uuid.uuid4().hex
    shutil.copyfile(ledger, output / f"optimizer-updates-before-resume-{attempt}.jsonl")
    temporary = ledger.with_suffix(".tmp")
    temporary.write_bytes(b"".join(encoded(u) for u in retained))
    temporary.replace(ledger)
    result = {
        "checkpoint": str(checkpoint),
        "completed_steps": clocks[0],
        "discarded_unsaved_updates": len(updates) - len(retained),
        "invocation": record,
    }
    (output / f"resume-{attempt}.json").write_bytes(encoded(result))
    print("ORIGINAL_BASELINE_RESUME", json.dumps(result), flush=True)
    return result


def train(
    model,
    prepared,
    output,
    *,
    smoke,
    keep_zero_advantage_groups=False,
    resume=False,
    profile="gsm8k",
    steps=None,
    judge_prepared=None,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process owns this original-framework run") from error
        return _train(
            model,
            prepared,
            output,
            smoke=smoke,
            keep_zero_advantage_groups=keep_zero_advantage_groups,
            resume=resume,
            profile=profile,
            steps=steps,
            judge_prepared=judge_prepared,
        )


def judge_service(judge_prepared):
    """The vLLM OpenAI server hosting the recipe's Qwen3-32B judge on two reserved GPUs.

    The prepared judge directory is the one MILES uses: same immutable snapshot,
    the same no-thinking chat template and the same YaRN 128K rope override, so
    both frameworks grade with the identical judge model, prompt, template and
    context capacity; no judge-side truncation applies in either arm.
    """
    prepared = json.loads((Path(judge_prepared) / "prepared.json").read_text())
    if prepared.get("verdict") != "passed" or prepared.get("model") != JUDGE_MODEL:
        raise ValueError("Prepared judge identity differs from the recipe's Qwen3-32B judge")
    if sha(Path(prepared["template"]).read_bytes()) != prepared["template_sha256"]:
        raise ValueError("Prepared judge template hash mismatch")
    if "<think>\n\n</think>" not in prepared.get("rendered_canary", ""):
        raise ValueError("Prepared judge template does not close its thinking block")
    model_config = json.loads((Path(prepared["snapshot"]) / "config.json").read_text())
    if model_config.get("model_type") != "qwen3" or model_config.get("rope_scaling"):
        raise ValueError("The YaRN 128K judge override requires an unscaled Qwen3 checkpoint")
    overrides = {
        "max_position_embeddings": JUDGE_CONTEXT,
        "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
    }
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        prepared["snapshot"],
        "--served-model-name",
        JUDGE_MODEL,
        "--chat-template",
        prepared["template"],
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(JUDGE_TENSOR_PARALLEL),
        "--max-model-len",
        str(JUDGE_CONTEXT),
        "--hf-overrides",
        json.dumps(overrides, sort_keys=True),
        "--gpu-memory-utilization",
        "0.9",
        # Judge traffic is a few dozen requests per collection; skip graph
        # capture and compilation so the server is up in minutes, like the
        # MILES judge, which also runs without CUDA graphs.
        "--enforce-eager",
        "--host",
        "127.0.0.1",
        "--port",
        str(JUDGE_PORT),
        "--disable-log-requests",
    ]
    return {
        "command": command,
        "api_base": f"http://127.0.0.1:{JUDGE_PORT}/v1",
        "snapshot": prepared["snapshot"],
        "template_sha256": prepared["template_sha256"],
        "revision": prepared.get("revision"),
        "context_length": JUDGE_CONTEXT,
        "rope_overrides": overrides,
    }


def start_judge(service, log_path, *, timeout=1800):
    """Start the judge on the reserved GPUs and wait until it serves /health."""
    started = time.monotonic()
    # The handle must outlive this function: the server writes to it for the whole run.
    log = open(log_path, "ab")  # noqa: SIM115
    process = subprocess.Popen(
        service["command"],
        stdout=log,
        stderr=subprocess.STDOUT,
        # Unbuffered so the judge log is complete if the job is interrupted.
        env={**os.environ, "CUDA_VISIBLE_DEVICES": JUDGE_GPUS, "PYTHONUNBUFFERED": "1"},
    )
    print("ORIGINAL_BASELINE_JUDGE_STARTING", json.dumps({"log": str(log_path), "gpus": JUDGE_GPUS}), flush=True)
    deadline = time.monotonic() + timeout
    health = service["api_base"].removesuffix("/v1") + "/health"

    def tail():
        try:
            return Path(log_path).read_text(errors="replace").splitlines()[-40:]
        except OSError:
            return []

    while time.monotonic() < deadline:
        if process.poll() is not None:
            print("ORIGINAL_BASELINE_JUDGE_LOG_TAIL", json.dumps(tail()), flush=True)
            raise RuntimeError(
                f"Judge server exited with {process.returncode} before becoming healthy; see {log_path}"
            )
        try:
            with urllib.request.urlopen(health, timeout=5) as response:
                if response.status == 200:
                    print(
                        "ORIGINAL_BASELINE_JUDGE_READY",
                        json.dumps({"seconds": round(time.monotonic() - started, 1), "api_base": service["api_base"]}),
                        flush=True,
                    )
                    return process
        except Exception:
            pass
        time.sleep(5)
    print("ORIGINAL_BASELINE_JUDGE_LOG_TAIL", json.dumps(tail()), flush=True)
    process.terminate()
    raise RuntimeError(f"Judge server did not become healthy within {timeout}s; see {log_path}")


def training_options(prepared, output, *, profile, steps, smoke, keep_zero_advantage_groups=False):
    """The historical trainer arguments for one profile; pure so tests can check them."""
    settings = PROFILES[profile]
    prefix = "smoke-" if smoke else ""
    # Exercise the historical filtering/packing loop with enough distinct prompts
    # to fill four H100 ranks. Full comparisons retain their profile's batch size.
    if profile == "gsm8k":
        prompts_per_collection = 128 if smoke and not keep_zero_advantage_groups else settings["prompts"]
    else:
        prompts_per_collection = settings["prompts"]
    options = {
        "exp_name": output.name,
        "model_name_or_path": str(prepared / "legacy-model"),
        "tokenizer_name_or_path": str(prepared / "tokenizer"),
        "attn_implementation": "flash_attention_2",
        "torch_dtype": "bfloat16",
        "dataset_mixer_list": [str(prepared / "train.jsonl"), "1.0"],
        "dataset_mixer_eval_list": [str(prepared / f"{prefix}eval.jsonl"), "1.0"],
        "dataset_mixer_list_splits": "train",
        "dataset_mixer_eval_list_splits": "train",
        "max_token_length": 34816,
        "max_prompt_token_length": 2048,
        "response_length": 32768,
        "pack_length": 34816,
        "num_learners_per_node": 4,
        "vllm_num_engines": settings["engines"],
        "vllm_tensor_parallel_size": 1,
        # The GSM8K control kept eager vLLM engines; the basket follows the
        # released recipe's defaults (CUDA graphs on) and gives its two
        # engines more KV budget, since generation bounds its update cadence.
        "vllm_enforce_eager": settings["judge"] is False,
        "vllm_gpu_memory_utilization": 0.85 if settings["judge"] else 0.7,
        "vllm_enable_prefix_caching": False,
        "deepspeed_stage": 3,
        "gradient_checkpointing": True,
        "per_device_train_batch_size": 1,
        "num_unique_prompts_rollout": prompts_per_collection,
        "num_samples_per_prompt_rollout": 4,
        "total_episodes": steps * prompts_per_collection * 4,
        "num_mini_batches": 1,
        "num_epochs": 1,
        "learning_rate": 1e-6,
        "lr_scheduler_type": "constant",
        "warm_up_steps": 0,
        "weight_decay": 0.0,
        "beta": 0.0,
        "clip_lower": 0.2,
        "clip_higher": 0.28,
        "advantage_normalization_type": "centered",
        "verification_reward": 1.0,
        "temperature": 1.0,
        "seed": 17,
        "async_steps": 0,
        "inflight_updates": False,
        "eval_on_step_0": True,
        "local_eval_every": 1 if smoke else 50,
        "save_freq": steps,
        "checkpoint_state_freq": steps if smoke else 25,
        "checkpoint_state_dir": str(output / "checkpoints"),
        "output_dir": str(output / "model"),
        "push_to_hub": False,
        "try_auto_save_to_beaker": False,
        "try_launch_beaker_eval_jobs_on_weka": False,
        "save_traces": True,
        "with_tracking": True,
        "wandb_entity": "ai2-llm",
        "wandb_project_name": "olmo-rl-comparison",
        "backend_timeout": 120,
    }
    if settings["judge"]:
        # The released recipe's judge and code settings; the code service and its
        # pass-rate threshold are the ones the MILES basket grades against.
        options.update(
            {
                "llm_judge_model": f"hosted_vllm/{JUDGE_MODEL}",
                "llm_judge_timeout": 600,
                "llm_judge_max_tokens": 2048,
                "llm_judge_max_context_length": JUDGE_CONTEXT,
                "llm_judge_temperature": 1.0,
                "code_api_url": CODE_API_URL,
                "code_pass_rate_reward_threshold": 0.99,
                "code_max_execution_time": 1.0,
                "backend_timeout": 1200,
            }
        )
    return options


def _train(
    model,
    prepared,
    output,
    *,
    smoke,
    keep_zero_advantage_groups=False,
    resume=False,
    profile="gsm8k",
    steps=None,
    judge_prepared=None,
):
    receipt = json.loads((prepared / "preparation.json").read_text())
    if receipt["model"] != str(model):
        raise ValueError("Prepared model identity differs")
    if receipt.get("profile", "gsm8k") != profile:
        raise ValueError("Prepared data belongs to a different comparison profile")
    for name, digest in receipt["files"].items():
        if sha((prepared / name).read_bytes()) != digest:
            raise ValueError(f"Prepared artifact changed: {name}")
    for name, link in receipt["model_alias"]["links"].items():
        path = prepared / "legacy-model" / name
        if not path.is_symlink() or str(path.resolve()) != link["target"] or path.stat().st_size != link["size"]:
            raise ValueError(f"Original model alias changed: {name}")
    if output.exists() and not resume:
        raise ValueError("Use a fresh original-framework run directory, or explicitly resume its native checkpoint")
    if resume and not (output / "invocation.json").is_file():
        raise ValueError("Resume requires an existing original-framework invocation")
    output.mkdir(parents=True, exist_ok=resume)
    trainer = Path("/stage/open_instruct/grpo_fast.py")
    original = trainer.read_bytes()
    if sha(original) != ORIGINAL_TRAINER_SHA256:
        raise ValueError("Original trainer source differs from the audited image")
    patched, changes = patch_trainer(original.decode(), keep_zero_advantage_groups=keep_zero_advantage_groups)
    trainer.write_text(patched)
    settings = PROFILES[profile]
    steps = 3 if smoke else (steps or settings["steps"])
    options = training_options(
        prepared,
        output,
        profile=profile,
        steps=steps,
        smoke=smoke,
        keep_zero_advantage_groups=keep_zero_advantage_groups,
    )
    service = None
    if settings["judge"]:
        if not judge_prepared:
            raise ValueError("The basket profile requires --judge-prepared (the MILES prepared judge directory)")
        service = judge_service(judge_prepared)
    command = [sys.executable, str(trainer)]
    for key, value in options.items():
        command.append("--" + key)
        command.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    differences = [
        "vLLM versus SGLang",
        "DeepSpeed/HF versus OLMo-core",
        "Original token-mean packed loss versus Core response reduction",
        (
            "Zero-advantage groups retained to match Core; historical pruning disabled explicitly"
            if keep_zero_advantage_groups
            else "Original zero-advantage filtering can skip driver steps; completed optimizer calls are counted separately"
        ),
    ]
    if profile == "gsm8k":
        differences += [
            "Original historical GSM8K verifier versus current verifier",
            "Four H100 trainers/four inference GPUs versus the two-B300/four-inference Core control",
        ]
    else:
        differences += [
            "Judge served by the original image's vLLM (TP2, YaRN 131,072 context) versus the MILES SGLang judge "
            "(TP1, same snapshot, template and YaRN context); identical prompts, no truncation in either arm",
            "Four H100 trainers/two vLLM engines/two judge GPUs versus two B300 trainers/five SGLang engines/one SGLang judge",
        ]
    record = {
        "command": command,
        "profile": profile,
        "driver_steps": steps,
        "model_alias": receipt["model_alias"],
        "original_source_sha256": sha(original),
        "patched_source_sha256": sha(trainer.read_bytes()),
        "source_adjustments": changes,
        "eval_driver_step_offsets": sorted({0, *range((1 if smoke else 50) - 1, steps, 1 if smoke else 50)}),
        "remaining_differences": differences,
        "judge": {k: v for k, v in service.items() if k != "command"} if service else None,
    }
    if resume:
        prepare_resume(output, record, steps=steps)
    else:
        (output / "invocation.json").write_bytes(encoded(record))
    print("ORIGINAL_BASELINE_COMMAND", json.dumps(record), flush=True)
    env = {
        **os.environ,
        "WANDB_RUN_GROUP": settings["wandb_group"],
        "OI_ORIGINAL_BASELINE_UPDATE_LEDGER": str(output / "optimizer-updates.jsonl"),
    }
    judge = None
    if service:
        (output / "judge-command.json").write_bytes(encoded(service))
        judge = start_judge(service, output / "judge-server.log")
        env.update(HOSTED_VLLM_API_BASE=service["api_base"], CUDA_VISIBLE_DEVICES=TRAINER_GPUS)
    try:
        subprocess.run(command, check=True, env=env)
    finally:
        if judge is not None and judge.poll() is None:
            judge.terminate()
            try:
                judge.wait(timeout=60)
            except subprocess.TimeoutExpired:
                judge.kill()

    exports = restore_public_exports(output, model)
    update_path = output / "optimizer-updates.jsonl"
    updates = read_jsonl(update_path) if update_path.exists() else []
    completion = completion_record(steps, updates, exports, record)
    (output / "completion.json").write_bytes(encoded(completion))
    print("ORIGINAL_BASELINE_COMPLETED", json.dumps(completion), flush=True)


def load_zero_converter():
    """Load DeepSpeed's CPU converter without initializing its Triton GPU ops.

    This historical DeepSpeed version treats an installed Triton as usable even
    on CPU-only workers. Suppress that optional dependency during its import;
    restore module visibility afterward. Training processes do not use this.
    """
    if torch.cuda.is_available() or "deepspeed" in sys.modules:
        return importlib.import_module("deepspeed.utils.zero_to_fp32")
    previous = sys.modules.get("triton")
    sys.modules["triton"] = None
    try:
        importlib.import_module("deepspeed")
        return importlib.import_module("deepspeed.utils.zero_to_fp32")
    finally:
        if previous is None:
            sys.modules.pop("triton", None)
        else:
            sys.modules["triton"] = previous


def export_checkpoint(model, checkpoint_root, tag, output):
    """Export one immutable DeepSpeed tag for independent held-out evaluation."""
    if not tag.startswith("global_step") or not tag[len("global_step") :].isdigit():
        raise ValueError("Use an explicit global_stepN checkpoint tag")
    checkpoint = checkpoint_root / tag
    files = sorted(checkpoint.glob("*.pt"))
    if not files:
        raise ValueError(f"No checkpoint shards in {checkpoint}")
    inventory = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    state_file = checkpoint / "zero_pp_rank_0_mp_rank_00_model_states.pt"
    if not state_file.exists():
        state_file = checkpoint / "mp_rank_00_model_states.pt"
    model_state = torch.load(state_file, map_location="cpu", weights_only=False)
    training_step = model_state.get("training_step")
    if type(training_step) is not int or training_step < 1:
        raise ValueError("Checkpoint lacks its completed driver step")
    del model_state
    expected = {}
    for shard in sorted(model.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            for name in list(tensors.keys()):
                expected[name] = tuple(tensors.get_slice(name).get_shape())
    if not expected:
        raise ValueError("Reference HF model has no safetensors")
    state = load_zero_converter().get_fp32_state_dict_from_zero_checkpoint(str(checkpoint_root), tag=tag)
    if set(state) != set(expected):
        raise ValueError(
            f"Checkpoint tensor names differ: missing={set(expected) - set(state)}, extra={set(state) - set(expected)}"
        )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "hf"
    if destination.exists():
        raise ValueError(f"Export already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output) as temporary:
        staging = Path(temporary)
        weight_map, chunk, size, total, index = {}, {}, 0, 0, 0
        for name, tensor in state.items():
            if tuple(tensor.shape) != expected[name] or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid checkpoint tensor: {name}")
            value = tensor.to(torch.bfloat16).contiguous()
            chunk[name] = value
            size += value.numel() * value.element_size()
            total += value.numel() * value.element_size()
            if size >= 2_000_000_000:
                filename = f"model-{index:05d}.safetensors"
                save_file(chunk, staging / filename, metadata={"format": "pt"})
                weight_map.update({key: filename for key in chunk})
                chunk, size, index = {}, 0, index + 1
        if chunk:
            filename = f"model-{index:05d}.safetensors"
            save_file(chunk, staging / filename, metadata={"format": "pt"})
            weight_map.update({key: filename for key in chunk})
        for path in model.iterdir():
            if (
                path.is_file()
                and (path.suffix in {".json", ".jinja", ".txt", ".model"})
                and "safetensors" not in path.name
            ):
                shutil.copyfile(path, staging / path.name)
        (staging / "model.safetensors.index.json").write_bytes(
            encoded({"metadata": {"total_size": total}, "weight_map": weight_map})
        )
        after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(checkpoint.glob("*.pt"))}
        if after != inventory:
            raise ValueError("Checkpoint changed during export")
        receipt = {
            "checkpoint": str(checkpoint),
            "training_step": training_step,
            "tag": tag,
            "tensors": len(weight_map),
            "bytes": total,
            "dtype": "bfloat16",
            "reference_model": str(model),
            "source_inventory": inventory,
            "output": str(destination),
        }
        (staging / "export-receipt.json").write_bytes(encoded(receipt))
        staging.rename(destination)
    (output / "export.json").write_bytes(encoded(receipt))
    print("ORIGINAL_CHECKPOINT_EXPORT_PASSED", json.dumps(receipt), flush=True)


def evaluate_checkpoint(model, source, prepared, output):
    """Greedy evaluation on the exact frozen 512 prompts, retaining every output."""
    if output.exists():
        raise ValueError("Use a fresh evaluation output directory")
    receipt = json.loads((prepared / "preparation.json").read_text())
    if sha((source / "manifest.json").read_bytes()) != receipt["source_manifest_sha256"]:
        raise ValueError("Evaluation source differs from the prepared comparison")
    manifest = json.loads((source / "manifest.json").read_text())
    if sha((source / "eval.jsonl").read_bytes()) != manifest["outputs"]["eval.jsonl"]:
        raise ValueError("Frozen evaluation file changed")
    rows = read_jsonl(source / "eval.jsonl")
    if len(rows) != 512:
        raise ValueError("Expected the frozen 512-question evaluation")
    tokenizer = dataset_transformation.TokenizerConfig(tokenizer_name_or_path=str(prepared / "tokenizer")).tokenizer
    prompts = []
    for row in rows:
        converted = convert_row(row, tokenizer)
        transformed = dataset_transformation.rlvr_tokenize_v2(converted, tokenizer)
        prompts.append({"prompt_token_ids": transformed[dataset_transformation.INPUT_IDS_PROMPT_KEY]})
    output.mkdir(parents=True)
    alias = legacy_model(model, output / "legacy-model")
    vllm = importlib.import_module("vllm")
    verifiers = importlib.import_module("open_instruct.ground_truth_utils")
    llm = vllm.LLM(
        model=str(output / "legacy-model"),
        tokenizer=str(prepared / "tokenizer"),
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.7,
        max_model_len=34816,
        max_num_seqs=32,
        seed=17,
    )
    sampling = vllm.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=32768,
        n=1,
        seed=17,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        stop=[],
    )
    verifier = verifiers.GSM8KVerifier()
    results = []
    with (output / "generations.jsonl").open("w") as stream:
        for start in range(0, len(rows), 32):
            completions = llm.generate(prompts[start : start + 32], sampling, use_tqdm=True)
            if len(completions) != len(rows[start : start + 32]):
                raise ValueError("Missing evaluation responses")
            for row, prompt, completion in zip(rows[start : start + 32], prompts[start : start + 32], completions):
                if completion.prompt_token_ids != prompt["prompt_token_ids"] or len(completion.outputs) != 1:
                    raise ValueError("Evaluation output has different prompt tokens or sample count")
                response = completion.outputs[0]
                label = str(row["metadata"]["verifiers"][0]["target"])
                result = {
                    "id": row["metadata"]["prepared_sample_id"],
                    "label": label,
                    "prompt_token_ids": prompt["prompt_token_ids"],
                    "text": response.text,
                    "token_ids": list(response.token_ids),
                    "finish_reason": response.finish_reason,
                    "score": verifier(
                        tokenized_prediction=list(response.token_ids), prediction=response.text, label=label
                    ).score,
                }
                stream.write(encoded(result).decode())
                results.append(result)
            stream.flush()
            print("ORIGINAL_EVAL_PROGRESS", len(results), sum(r["score"] for r in results), flush=True)
    summary = {
        "model": str(model),
        "model_alias": alias,
        "source": str(source),
        "eval_sha256": manifest["outputs"]["eval.jsonl"],
        "questions": len(results),
        "correct": sum(r["score"] for r in results),
        "accuracy": sum(r["score"] for r in results) / len(results),
        "capped": sum(r["finish_reason"] == "length" for r in results),
        "mean_response_tokens": sum(len(r["token_ids"]) for r in results) / len(results),
        "temperature": 0.0,
        "max_new_tokens": 32768,
        "seed": 17,
        "verifier": "historical-open-instruct-GSM8KVerifier",
        "engine": "historical-vllm",
        "max_num_seqs": 32,
    }
    (output / "evaluation.json").write_bytes(encoded(summary))
    print("ORIGINAL_CHECKPOINT_EVAL_PASSED", json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "train", "export", "evaluate", "resume"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-zero-advantage-groups", action="store_true")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--checkpoint-tag")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="gsm8k")
    parser.add_argument("--steps", type=int, help="Driver steps for train/resume; defaults to the profile budget")
    parser.add_argument("--judge-prepared", type=Path, help="MILES prepared judge directory (basket profile)")
    args = parser.parse_args()
    if args.stage == "evaluate":
        if args.profile != "gsm8k":
            parser.error("Independent evaluation is only implemented for the GSM8K profile")
        evaluate_checkpoint(args.model, args.source, args.prepared, args.output)
    elif args.stage == "export":
        if not args.checkpoint_root or not args.checkpoint_tag:
            parser.error("export requires --checkpoint-root and --checkpoint-tag")
        export_checkpoint(args.model, args.checkpoint_root, args.checkpoint_tag, args.output)
    elif args.stage == "prepare":
        prepare(args.model, args.source, args.prepared, profile=args.profile)
    else:
        train(
            args.model,
            args.prepared,
            args.output,
            smoke=args.stage == "smoke",
            resume=args.stage == "resume",
            keep_zero_advantage_groups=args.keep_zero_advantage_groups,
            profile=args.profile,
            steps=args.steps,
            judge_prepared=args.judge_prepared,
        )


if __name__ == "__main__":
    main()
