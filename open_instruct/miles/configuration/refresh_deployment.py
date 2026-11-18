"""Admit a serialized actor checkpoint without widening refresh topology."""

from pathlib import Path

import yaml

from open_instruct.miles.errors import InputError


def validate(options):
    message = "engine_drain requires the managed single-turn TP1 deployment; sglang_config may only change the actor checkpoint"
    try:
        raw = yaml.safe_load(Path(options["sglang_config"]).read_text())
        if not isinstance(raw, dict) or set(raw) != {"sglang"}:
            raise ValueError(message)
        models = raw["sglang"]
        if not isinstance(models, list) or len(models) != 1:
            raise ValueError(message)
        model = models[0]
        if set(model) != {"name", "model_path", "update_weights", "server_groups"}:
            raise ValueError(message)
        if model["name"] != "default" or model["update_weights"] is not True:
            raise ValueError(message)
        if not isinstance(model["model_path"], str) or not model["model_path"]:
            raise ValueError(message)
        groups = model["server_groups"]
        if not isinstance(groups, list) or len(groups) != 1:
            raise ValueError(message)
        group = groups[0]
        if set(group) != {"worker_type", "num_gpus", "num_gpus_per_engine"}:
            raise ValueError(message)
        if (
            group["worker_type"] != "regular"
            or type(group["num_gpus_per_engine"]) is not int
            or group["num_gpus_per_engine"] != 1
        ):
            raise ValueError(message)
        count = group["num_gpus"]
        if type(count) is not int or count <= 0 or count != options.get("rollout_num_gpus"):
            raise ValueError(message)
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        raise InputError(message) from error
