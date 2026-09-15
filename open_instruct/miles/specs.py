"""Select a backend-specific run contract without importing a training stack."""

import copy

from open_instruct.miles import opd_config, run_spec, validation


def from_dict(document, *, config_path=None, overrides=None):
    document = copy.deepcopy(document)
    run_spec._apply_overrides(document, overrides)
    kind = (
        opd_config.OPDRunSpec
        if document.get("training", {}).get("algorithm") == "opd"
        and document.get("trainer", {}).get("backend", "megatron") == "megatron"
        else run_spec.RunSpec
    )
    return kind.from_dict(document, config_path=config_path or "run.toml")


def load(path, overrides=None):
    return from_dict(validation.read_document(path), config_path=path, overrides=overrides)
