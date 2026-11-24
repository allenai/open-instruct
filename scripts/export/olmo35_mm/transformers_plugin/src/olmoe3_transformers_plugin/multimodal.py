# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/transformers_plugin/src/olmoe3_transformers_plugin/multimodal.py
"""Export a multimodal (``MultimodalLMConfig``) OLMoDDP checkpoint to one HF directory.

The language model goes through the text-only export path unchanged, so its tensors and its
``config.json`` are exactly what a text-only export of the same LM produces. The export then
becomes one ``Olmo3MoeVisionForConditionalGeneration`` directory holding the whole model:

* ``model.safetensors``: ``language_model.<text export names>``, ``vision_tower.<olmo-core
  VisionTransformer names>`` and ``connector.<olmo-core VisionConnector names>``;
* ``config.json``: ``text_config`` (the text export's config, verbatim), ``vision_config``
  and ``connector_config`` (olmo-core config dicts), ``image_patch_token_id``, ``vit_layers``
  and ``olmo_core_config`` (the checkpoint's full ``config.json``);
* the text team's custom code files, as the text export writes them, plus
  ``configuration_olmo3moe_vision.py`` / ``modeling_olmo3moe_vision.py``;
* the tokenizer files, as the text export writes them.

Source checkpoints are written by ``MultimodalOLMoDDPTrainModule``: no ``model.*`` entries;
trainable parameters as FP32 masters ``module.<name>.main``, frozen parameters as
``frozen_model.<name>``, persistent buffers as ``model_buffer.<name>``. The key resolution
mirrors olmo-core's ``src/scripts/export_multimodal_olmoddp.py`` (not an installed module).
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Collection, Dict, Tuple

import torch
import torch.distributed.checkpoint as dcp
from olmo_core.config import DType
from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader

from .hf_vision.configuration_olmo3moe_vision import ARCHITECTURE, MODEL_TYPE

log = logging.getLogger(__name__)

MULTIMODAL_CONFIG_CLASS = "MultimodalLMConfig"
MULTIMODAL_SUBMODULES = ("lm", "vision", "connector")
LANGUAGE_MODEL_PREFIX = "language_model."
VISION_TOWER_PREFIX = "vision_tower."
CONNECTOR_PREFIX = "connector."
VISION_CODE_FILES = ("configuration_olmo3moe_vision.py", "modeling_olmo3moe_vision.py")
_VISION_CODE_DIR = Path(__file__).parent / "hf_vision"


def is_multimodal_model_config(model_config: Dict[str, Any]) -> bool:
    """Whether an experiment config's ``model`` section is a ``MultimodalLMConfig``."""
    class_name = str(model_config.get("_CLASS_", ""))
    if class_name.rsplit(".", 1)[-1] == MULTIMODAL_CONFIG_CLASS:
        return True
    return all(isinstance(model_config.get(name), dict) for name in MULTIMODAL_SUBMODULES)


def has_multimodal_checkpoint_keys(checkpoint_keys: Collection[str]) -> bool:
    """Whether a distributed checkpoint holds a multimodal model (``lm``/``vision``/``connector``)."""
    prefixes = tuple(
        f"{kind}.{name}."
        for kind in ("module", "frozen_model", "model_buffer", "model")
        for name in MULTIMODAL_SUBMODULES
    )
    return any(key.startswith(prefixes) for key in checkpoint_keys)


def checkpoint_key(name: str, checkpoint_keys: Collection[str]) -> str:
    """
    Find the single checkpoint entry holding model state ``name``.

    :param name: A key of the multimodal model's state dict (``lm.…``, ``vision.…``, ``connector.…``).
    :param checkpoint_keys: All keys of the distributed checkpoint.

    :raises KeyError: If no entry, or more than one entry, matches.
    """
    candidates = [
        f"frozen_model.{name}",
        f"module.{name}.main",
        f"{name}.main",
        f"model.{name}",
        f"model_buffer.{name}",
    ]
    found = [key for key in candidates if key in checkpoint_keys]
    if len(found) != 1:
        raise KeyError(f"Expected exactly one checkpoint entry for '{name}', found {found}")
    return found[0]


def load_multimodal_state_dict(
    checkpoint_input_path: str, model_config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """
    Rebuild the ``MultimodalLM`` state dict from a multimodal OLMoDDP checkpoint.

    The model is built on the meta device and every parameter/buffer is loaded in place from
    its checkpoint entry, exactly as the text path loads ``module.<name>.main`` masters: a
    flattened master fills the parameter's own storage (so it takes the model's dtype), a
    ``frozen_model.<name>`` entry fills the full parameter, ``model_buffer.<name>`` a buffer.

    :param checkpoint_input_path: The checkpoint step directory.
    :param model_config: The ``MultimodalLMConfig`` dict (attention backends already set).

    :returns: The state dict keyed as ``MultimodalLM.state_dict()`` (``lm.*``, ``vision.*``,
        ``connector.*``).
    """
    from olmo_core.nn.vision.multimodal import MultimodalLMConfig

    model_and_optim = Path(checkpoint_input_path) / "model_and_optim"
    reader = RemoteFileSystemReader(str(model_and_optim))
    checkpoint_keys = set(reader.read_metadata().state_dict_metadata)
    if not has_multimodal_checkpoint_keys(checkpoint_keys):
        raise RuntimeError(f"{model_and_optim} holds no lm/vision/connector entries")

    config = MultimodalLMConfig.from_dict(copy.deepcopy(model_config))
    model = config.build(init_device="meta")
    model.to_empty(device=torch.device("cpu"))

    to_load: Dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        key = checkpoint_key(name, checkpoint_keys)
        # DCP loads in place; a flattened master fills the parameter through its own view.
        to_load[key] = tensor.view(-1) if key.endswith(".main") else tensor
    log.info(
        "Loading %d tensors (%d fp32 masters, %d frozen parameters, %d buffers) from '%s'",
        len(to_load),
        sum(key.endswith(".main") for key in to_load),
        sum(key.startswith("frozen_model.") for key in to_load),
        sum(key.startswith("model_buffer.") for key in to_load),
        model_and_optim,
    )
    dcp.load(to_load, storage_reader=reader)
    return {key: value.detach() for key, value in model.state_dict().items()}


def split_multimodal_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a ``MultimodalLM`` state dict into LM (prefix stripped), vision and connector parts."""
    lm: Dict[str, torch.Tensor] = {}
    vision: Dict[str, torch.Tensor] = {}
    connector: Dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name.startswith("lm."):
            lm[name[len("lm.") :]] = tensor
        elif name.startswith("vision."):
            vision[name[len("vision.") :]] = tensor
        elif name.startswith("connector."):
            connector[name[len("connector.") :]] = tensor
        else:
            raise RuntimeError(f"Unexpected multimodal state entry '{name}'")
    if not lm or not vision or not connector:
        raise RuntimeError("Multimodal state must hold lm, vision and connector tensors")
    return lm, vision, connector


def _read_text_export_tensors(output_path: Path) -> Dict[str, torch.Tensor]:
    """Read every tensor of a (possibly sharded) safetensors export and drop its weight files."""
    from safetensors.torch import load_file

    index_path = output_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
        index_path.unlink()
    else:
        shard_names = ["model.safetensors"]
    tensors: Dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard = output_path / shard_name
        tensors.update(load_file(str(shard)))
        shard.unlink()
    return tensors


def write_multimodal_export(
    output_path: str | Path,
    *,
    vision_state_dict: Dict[str, torch.Tensor],
    connector_state_dict: Dict[str, torch.Tensor],
    model_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    dtype: DType | None,
) -> Dict[str, Any]:
    """
    Turn a finished text export in ``output_path`` into the whole-model multimodal layout.

    :param output_path: The directory the text path exported the language model to.
    :param vision_state_dict: The olmo-core vision encoder state dict (olmo-core names).
    :param connector_state_dict: The olmo-core connector state dict (olmo-core names).
    :param model_config: The checkpoint's ``MultimodalLMConfig`` dict.
    :param experiment_config: The checkpoint's full ``config.json``.
    :param dtype: Export dtype of the vision/connector tensors; ``None`` keeps theirs.

    :returns: The combined ``config.json`` contents.
    """
    from safetensors.torch import save_file

    output_path = Path(output_path)
    text_config = json.loads((output_path / "config.json").read_text())
    if text_config.get("model_type") != "olmo3moe":
        raise RuntimeError(f"Expected an olmo3moe text export, got {text_config.get('model_type')}")

    tensors = {
        f"{LANGUAGE_MODEL_PREFIX}{name}": tensor
        for name, tensor in _read_text_export_tensors(output_path).items()
    }
    pt_dtype = dtype.as_pt() if dtype is not None else None
    for prefix, state in (
        (VISION_TOWER_PREFIX, vision_state_dict),
        (CONNECTOR_PREFIX, connector_state_dict),
    ):
        for name, tensor in state.items():
            if pt_dtype is not None:
                tensor = tensor.to(pt_dtype)
            tensors[f"{prefix}{name}"] = tensor.contiguous()
    save_file(tensors, str(output_path / "model.safetensors"), metadata={"format": "pt"})

    config: Dict[str, Any] = {
        "architectures": [ARCHITECTURE],
        "model_type": MODEL_TYPE,
        "auto_map": {
            "AutoConfig": "configuration_olmo3moe_vision.Olmo3MoeVisionConfig",
            "AutoModel": f"modeling_olmo3moe_vision.{ARCHITECTURE}",
            "AutoModelForCausalLM": f"modeling_olmo3moe_vision.{ARCHITECTURE}",
        },
        "text_config": text_config,
        "vision_config": copy.deepcopy(model_config["vision"]),
        "connector_config": copy.deepcopy(model_config["connector"]),
        "image_patch_token_id": model_config.get("image_patch_token_id", 0),
        "vit_layers": list(model_config.get("vit_layers", [-1])),
        "olmo_core_config": copy.deepcopy(experiment_config),
        "tie_word_embeddings": False,
    }
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "dtype", "transformers_version"):
        if name in text_config:
            config[name] = text_config[name]
    (output_path / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    for file_name in VISION_CODE_FILES:
        shutil.copyfile(_VISION_CODE_DIR / file_name, output_path / file_name)
    return config


def check_multimodal_export_tensors(output_path: str | Path) -> None:
    """
    Check that the export's tensors are exactly the wrapper model's state dict.

    The wrapper is built on the meta device from the export's ``config.json`` and its state
    dict names and shapes are compared with the safetensors file.

    :raises RuntimeError: On any missing, unexpected or misshaped tensor.
    """
    from safetensors import safe_open

    from .hf_vision.configuration_olmo3moe_vision import Olmo3MoeVisionConfig
    from .hf_vision.modeling_olmo3moe_vision import Olmo3MoeVisionForConditionalGeneration

    output_path = Path(output_path)
    config = Olmo3MoeVisionConfig.from_pretrained(output_path)
    with torch.device("meta"):
        model = Olmo3MoeVisionForConditionalGeneration(config)
    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    with safe_open(str(output_path / "model.safetensors"), framework="pt") as weights:
        actual = {name: tuple(weights.get_slice(name).get_shape()) for name in weights.keys()}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    misshaped = sorted(
        name for name in expected if name in actual and expected[name] != actual[name]
    )
    if missing or unexpected or misshaped:
        raise RuntimeError(
            "Multimodal export does not match the wrapper model: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}, misshaped={misshaped[:10]}"
        )
    for prefix in (LANGUAGE_MODEL_PREFIX, VISION_TOWER_PREFIX, CONNECTOR_PREFIX):
        if not any(name.startswith(prefix) for name in actual):
            raise RuntimeError(f"Multimodal export holds no {prefix}* tensors")
