"""Select a backend-specific run contract without importing a training stack."""

import copy

from open_instruct.miles.configuration import run_spec, validation
from open_instruct.miles.distillation import opd_config
from open_instruct.miles.megatron_grpo import grpo_config

# Single-node Megatron routes: one Beaker task, the managed native driver, plan()["warnings"].
MEGATRON_SPECS = (opd_config.OPDRunSpec, grpo_config.MegatronGRPORunSpec)


def from_dict(document, *, config_path=None, overrides=None):
    document = copy.deepcopy(document)
    run_spec._apply_overrides(document, overrides)
    algorithm = document.get("training", {}).get("algorithm")
    backend = document.get("trainer", {}).get("backend")
    if algorithm == "opd" and backend in (None, "megatron"):
        kind = opd_config.OPDRunSpec
    elif algorithm == "grpo" and backend == "megatron":
        kind = grpo_config.MegatronGRPORunSpec
    else:
        kind = run_spec.RunSpec
    return kind.from_dict(document, config_path=config_path or "run.toml")


def load(path, overrides=None):
    return from_dict(validation.read_document(path), config_path=path, overrides=overrides)
