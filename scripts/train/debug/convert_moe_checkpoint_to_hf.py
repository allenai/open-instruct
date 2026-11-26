"""Convert an olmo-core MoE v2 checkpoint (KDA / OLMoE3) to HuggingFace format.

Thin wrapper around olmo-core's own converter, mirroring
`ladders/olmoe3/transformers_plugin/convert_checkpoint.py` in scaling-ladders.
Requires olmo-core on `akshitab/emo_modularity` or later: earlier revisions raise
NotImplementedError for peri-LN checkpoints, which these are.

The model config comes from a file rather than the checkpoint's own config.json,
because the training config is the one whose blocks match the saved weights. Pass
the olmo-core training config JSON the run was launched with (it needs the `model`
and `dataset.tokenizer` sections); none ships in this repo.

    uv run python scripts/train/debug/convert_moe_checkpoint_to_hf.py \
        -i <ckpt>/step172 -o <ckpt>/hf_step172 \
        -c <run's training config>.json

For KDA checkpoints add `--skip-validation --device cpu` (see the caveat below).
Input and output must be local or WEKA paths: `-i` is joined with pathlib and the
EOS step writes to a local directory, so remote (`gs://`) paths are not supported.

Validation caveat: strict logit validation FAILS (~92-99% mismatch, abs diff
~1.5) on every OLMoE3 latent-KDA 1.2B checkpoint, including a known-good
midtrain checkpoint -- the two forward implementations disagree at this scale regardless
of checkpoint, config, or the OLMO_HF_* parity env vars (bisected 2026-08-20).
The scaling-ladders production workload passes --skip-validation --device cpu;
strict validation only ever passed on the 275M/480M ladder models. Use
--skip-validation for KDA and validate functionally with an eval instead
(CPU and CUDA conversions were verified bit-identical over the full 35 GB).

generation_config.json gets eos_token_id [100265, 100257] (<|im_end|>, <|endoftext|>) by default: the
chat stop list for the default Olmo 3 chat tokenizer, as chat SFT exports need. A tokenizer with other ids, or a
base/midtrain export that should stop only at <|endoftext|>, passes --eos-token-ids explicitly (e.g. 100257).

KDA weights and `--kda-decay`: an export must store the value the training forward used. Five tensors per KDA layer
can be float32 in a forward: the decay weights `A_log` and `dt_bias`, and the `q/k/v_conv1d` weights.

- `bf16-grid` (default): every value is the float32 master rounded to the bfloat16 grid, as in every export this
  script wrote before the option. Right for a checkpoint whose forward saw bfloat16-rounded weights: OLMo-core
  training with bfloat16 model weights and float32 masters, and MaxText SFT with MAXTEXT_FORWARD_BF16_ALL_PARAMS=1.
  `A_log` and `dt_bias` are stored as float32 (the dtype olmo-core's KDA and its HF modeling declare) holding the
  rounded value; the convolution weights stay bfloat16, so nothing a loader sees of them changes.
- `fp32`: all five tensors are stored as float32 masters, unrounded. Right for a checkpoint whose forward used them
  in float32: MaxText pre_train.train (pretraining, midtraining, the Small joint SFT), which leaves small tensors
  in float32, with `kda_conv_in_compute_dtype: false`. With `kda_conv_in_compute_dtype: true` the forward used
  bfloat16 convolution weights beside float32 decay weights, and neither mode stores that exactly: `bf16-grid`
  rounds the decay weights and `fp32` leaves the convolution weights unrounded. It refuses a model with no KDA
  layers, and a model config that builds KDA weights in another dtype (the loader would round them before the
  save). Check the result with check_kda_export.py. Serve such an export with a loader that keeps the five in
  float32, as olmo-sglang's KDA layer allocates them; casting the whole model to bfloat16 rounds them again. It
  needs `--skip-validation`: olmo-core's validation casts its reference model to one dtype, so it would compare
  rounded weights with unrounded ones.

Note for GDN-based MoE hybrids (not KDA): olmo-core's `is_olmo_hybrid_model`
returns True for them and routes to the dense-hybrid exporter, which crashes on
the missing `feed_forward` attribute. KDA is unaffected because
`KimiDeltaAttention` is not a `GatedDeltaNet` subclass.
"""

import argparse
import contextlib
import json
import pathlib
import tempfile

import torch
from olmo_core.config import DType
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
from olmo_core.distributed.utils import get_full_tensor
from olmo_core.nn.attention import KimiDeltaAttention
from olmo_core.nn.hf import convert_checkpoint as core_convert
from olmo_core.nn.hf import convert_checkpoint_to_hf
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.utils import prepare_cli_environment
from torch.distributed.checkpoint import state_dict as dist_cp_sd

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def checkpoint_key_for(name: str, checkpoint_keys: set[str], master_first: bool = False) -> str | None:
    """The checkpoint key holding parameter ``name``: the model's own tensor first, as olmo-core's converter reads it.

    ``master_first`` puts the optimizer's float32 master (``<name>.main``) ahead instead. ``--kda-decay fp32`` asks
    that for the KDA weights it exports in float32: where a checkpoint also holds the model's copy in the compute
    dtype (``model.<name>`` in bfloat16), that copy read into a float32 parameter would be rounded values under a
    float32 dtype. Every other weight, and every weight in the other modes, keeps the original order.
    """
    model_copies = (f"model.{name}", f"model.module.{name}", name, f"module.{name}")
    masters = (f"{name}.main", f"module.{name}.main")
    candidates = (*masters, *model_copies) if master_first else (*model_copies, *masters)
    return next((key for key in candidates if key in checkpoint_keys), None)


def load_ddp_main_params(
    model_and_optim_dir: str, model_config: dict, work_dir: str, kda_decay: str = "bf16-grid"
) -> dict | None:
    """Gather model weights from an OLMoDDP checkpoint, or return None for a conventional one.

    ``OLMoDDPTrainModule`` stores the authoritative weights as the fused optimizer's fp32
    ``module.<parameter>.main`` tensors, not under ``model.<parameter>``. olmo-core on
    ``akshitab/emo_modularity`` (f2cf93839) taught ``convert_checkpoint_to_hf`` that layout
    (``_load_ddp_optimizer_model_state``); the Olmo 3.5 hero lineage (89e7dcb7) never got it,
    so every hero SFT conversion died with "Missing key in checkpoint state_dict:
    model.embeddings.weight" (2026-09-16). This is that function,
    ported: the same key candidates in the same order (checkpoint_key_for); the converter then takes the gathered
    state dict. With ``kda_decay`` ``fp32``, the KDA weights exported in float32 are read from their float32
    masters, and a stored KDA tensor that is not float32 is refused, in either layout.
    """
    stored = get_checkpoint_metadata(model_and_optim_dir).state_dict_metadata
    checkpoint_keys = set(stored)
    if not any(key.endswith(".main") for key in checkpoint_keys):
        if kda_decay == "fp32":
            # olmo-core loads this layout itself, into parameters of the config's dtype: a KDA weight stored in
            # bfloat16 would arrive widened to float32 and pass the save's dtype check. Refuse it from the metadata.
            model = TransformerConfig.from_dict(model_config).build(init_device="meta")
            for name in sorted(set().union(*kda_weight_keys(model))):
                key = checkpoint_key_for(name, checkpoint_keys)
                dtype = None if key is None else stored[key].properties.dtype
                if dtype != torch.float32:
                    raise RuntimeError(
                        f"--kda-decay fp32: the checkpoint stores '{key or name}' as {dtype}, so there is no float32 "
                        "value of this KDA weight to export"
                    )
        return None
    logger.info("OLMoDDP checkpoint layout (<param>.main); gathering weights through the fused optimizer keys")
    model = TransformerConfig.from_dict(model_config).build(init_device="meta")
    model = model.to_empty(device="cpu")
    kda_masters = set().union(*kda_weight_keys(model)) if kda_decay == "fp32" else set()
    keys_to_load: list[str] = []
    destinations: list[tuple[str, torch.Tensor]] = []
    missing: list[str] = []
    for name, param in model.named_parameters():
        key = checkpoint_key_for(name, checkpoint_keys, master_first=name in kda_masters)
        if key is None:
            missing.append(name)
            continue
        keys_to_load.append(key)
        destinations.append((name, param))
    if missing:
        raise RuntimeError("DDP checkpoint is missing weights for model parameters: " + ", ".join(missing))
    loaded = load_keys(model_and_optim_dir, keys_to_load, work_dir=work_dir)
    with torch.no_grad():
        for key, (name, destination), value in zip(keys_to_load, destinations, loaded, strict=True):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Checkpoint value '{key}' is not a tensor")
            if name in kda_masters and value.dtype != torch.float32:
                raise RuntimeError(
                    f"--kda-decay fp32: the checkpoint stores '{key}' as {value.dtype}, so there is no float32 "
                    "value of this KDA weight to export"
                )
            if value.numel() != destination.numel():
                raise RuntimeError(
                    f"Checkpoint value '{key}' has {value.numel()} elements, model parameter '{name}' has {destination.numel()}"
                )
            destination.copy_(value.reshape(destination.shape).to(destination.dtype))
        for name, buffer in model.named_buffers():
            key = f"model_buffer.{name}"
            if key in checkpoint_keys:
                value = next(load_keys(model_and_optim_dir, [key], work_dir=work_dir))
                buffer.copy_(value.reshape(buffer.shape).to(buffer.dtype))
    options = dist_cp_sd.StateDictOptions(full_state_dict=True, cpu_offload=True)
    return dist_cp_sd.get_model_state_dict(model, options=options)


KDA_DECAY_MODES = ("bf16-grid", "fp32")


KDA_CONVS = ("q_conv1d", "k_conv1d", "v_conv1d")


def kda_weight_keys(model: torch.nn.Module) -> tuple[set[str], set[str]]:
    """State-dict keys of every KDA layer's decay weights and of its q/k/v convolution weights."""
    decay, convs = set(), set()
    for name, module in model.named_modules():
        if isinstance(module, KimiDeltaAttention):
            decay |= {f"{name}.A_log", f"{name}.dt_bias"}
            convs |= {f"{name}.{conv}.weight" for conv in KDA_CONVS}
    return decay, convs


def cast_for_export(
    model_state_dict: dict,
    decay_keys: set[str],
    conv_keys: set[str],
    kda_decay: str,
    other_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Cast a model state dict for export; ``kda_decay`` decides the KDA weights, the rest get ``other_dtype``.

    ``bf16-grid``: the decay weights are float32 holding the master rounded to the bfloat16 grid, and the
    convolution weights get ``other_dtype`` like everything else, so every value equals an all-bfloat16 export's.
    ``fp32``: the decay and convolution weights are the float32 masters.
    """
    if kda_decay not in KDA_DECAY_MODES:
        raise ValueError(f"kda_decay must be one of {KDA_DECAY_MODES}, got {kda_decay!r}")
    missing = sorted((decay_keys | conv_keys) - set(model_state_dict))
    if missing:
        raise KeyError(f"KDA weights missing from the model state dict: {missing}")
    if kda_decay == "fp32":
        if not decay_keys:
            raise ValueError("--kda-decay fp32 on a model with no KDA layers: there is nothing to keep in float32")
        # The masters must reach this point unrounded. A model config that builds KDA parameters in another dtype
        # (KimiDeltaAttentionConfig dtype: bfloat16) makes the checkpoint loader round them first; widening those
        # back to float32 would write rounded values under a float32 dtype.
        rounded = sorted(key for key in decay_keys | conv_keys if model_state_dict[key].dtype != torch.float32)
        if rounded:
            raise RuntimeError(
                "--kda-decay fp32: the checkpoint loader delivered these KDA weights in another dtype, so their "
                f"float32 values are already lost: {rounded[:4]}{' ...' if len(rounded) > 4 else ''} "
                f"({model_state_dict[rounded[0]].dtype}). Convert with a model config whose KDA layers are float32."
            )
    cast = {}
    off_grid = 0
    for key, value in model_state_dict.items():
        if key not in decay_keys and key not in conv_keys:
            cast[key] = value.to(dtype=other_dtype)
            continue
        master = value.to(dtype=torch.float32)
        on_grid = master.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        off_grid += int((on_grid != master).sum())
        if kda_decay == "fp32":
            cast[key] = master
        else:
            cast[key] = on_grid if key in decay_keys else value.to(dtype=other_dtype)
    if decay_keys:
        outcome = "rounded to the bfloat16 grid" if kda_decay == "bf16-grid" else "kept off the bfloat16 grid"
        logger.info(
            "KDA weights (%s): %d decay tensors written as float32, %d convolution tensors as %s; %d elements %s",
            kda_decay,
            len(decay_keys),
            len(conv_keys),
            "float32" if kda_decay == "fp32" else other_dtype,
            off_grid,
            outcome,
        )
    return cast


@contextlib.contextmanager
def kda_decay_export(kda_decay: str):
    """Make olmo-core's converter save KDA weights as ``kda_decay`` says while this context is open.

    ``convert_checkpoint_to_hf`` hands ``save_hf_model`` one dtype for every tensor. This wraps that call: the state
    dict is cast here, per tensor, and saved as it is. Checkpoint loading and everything else stay olmo-core's.

    The wrapper only acts if olmo-core calls ``save_hf_model`` through its module, as the pinned revision does. If the
    context closes without that call (olmo-core changed, or the model took the dense-hybrid exporter), an ``fp32``
    request was dropped and the export holds rounded KDA weights: that raises. With ``bf16-grid`` the values are
    the same either way, so it only logs.
    """
    save_hf_model = core_convert.save_hf_model
    wrapped = False

    def save(save_dir, model_state_dict, model, *args, dtype=None, **kwargs):
        nonlocal wrapped
        wrapped = True
        if dtype is None:
            # Not reached from this script, which always passes a dtype: forward the call unchanged.
            return save_hf_model(save_dir, model_state_dict, model, *args, dtype=None, **kwargs)
        full = {key: get_full_tensor(state) for key, state in model_state_dict.items()}
        cast = cast_for_export(full, *kda_weight_keys(model), kda_decay, dtype.as_pt())
        return save_hf_model(save_dir, cast, model, *args, dtype=None, **kwargs)

    core_convert.save_hf_model = save
    try:
        yield
    finally:
        core_convert.save_hf_model = save_hf_model
    if not wrapped:
        if kda_decay == "fp32":
            raise RuntimeError(
                "--kda-decay fp32 was not applied: olmo-core's converter did not save through save_hf_model, "
                "so the export holds bfloat16-rounded KDA weights"
            )
        logger.info("olmo-core's converter did not save through save_hf_model; every tensor has its one dtype")


CHAT_EOS_TOKEN_IDS = (100265, 100257)  # <|im_end|>, <|endoftext|>


def write_generation_eos(output_dir: pathlib.Path, eos_token_ids: list[int]) -> None:
    """Set generation_config.json's eos_token_id to the chat stop list; the converter derives it from config.json."""
    path = output_dir / "generation_config.json"
    generation_config = json.loads(path.read_text())
    generation_config["eos_token_id"] = list(eos_token_ids)
    path.write_text(json.dumps(generation_config, indent=2) + "\n")
    logger.info("generation_config.json eos_token_id = %s", eos_token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--checkpoint-input-path", required=True)
    parser.add_argument("-o", "--huggingface-output-dir", required=True)
    parser.add_argument("-c", "--config", required=True, help="Training config json (model + dataset.tokenizer).")
    parser.add_argument("-t", "--tokenizer", default="allenai/olmo-3-tokenizer-instruct-dev")
    parser.add_argument("-s", "--max-sequence-length", type=int, default=8192)
    parser.add_argument("--skip-validation", dest="validate", action="store_false")
    # The chat models end a turn with <|im_end|> (100265), not <|endoftext|> (100257). Without it in
    # generation_config.json, servers that take stop tokens from there (vLLM's OpenAI server) never stop the turn:
    # an A4B SFT export without it repeated its tool call up to the token budget on BFCL.
    # Delete this override once the olmo-core pin's save_hf_model writes the chat stop list itself.
    parser.add_argument(
        "--eos-token-ids",
        type=int,
        nargs="+",
        default=list(CHAT_EOS_TOKEN_IDS),
        help="generation_config.json eos_token_id list (default: <|im_end|>, <|endoftext|>)",
    )
    # Defaults to CUDA, not CPU: validation runs both implementations, and the
    # KDA/fla Triton kernels reject CPU tensors ("Pointer argument (at 0) cannot be
    # accessed from Triton"). On CPU the weights still get written, so the failure
    # looks like a converted-but-unvalidated checkpoint -- the exact state that let
    # a 38%-logit-error conversion through on Olmo-Hybrid-7B. Do not silence it.
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument(
        "--kda-decay",
        choices=KDA_DECAY_MODES,
        default="bf16-grid",
        help="KDA weights, by what the training forward used: bf16-grid (default) stores every value rounded to the "
        "bfloat16 grid, as before; fp32 stores A_log, dt_bias and the q/k/v_conv1d weights as float32 masters, for "
        "a checkpoint whose forward used them in float32.",
    )
    args = parser.parse_args()
    if args.kda_decay == "fp32" and args.validate:
        parser.error("--kda-decay fp32 needs --skip-validation: validation rounds the reference's KDA weights")

    payload = json.loads(pathlib.Path(args.config).read_text())
    model_config = payload["model"]
    tokenizer_config = payload.get("dataset", {}).get("tokenizer")
    if tokenizer_config is None:
        raise SystemExit(f"{args.config} has no dataset.tokenizer section")

    logger.info("converting %s -> %s", args.checkpoint_input_path, args.huggingface_output_dir)
    # load_keys is a generator, but load_ddp_main_params consumes it fully inside this block and returns
    # materialized CPU tensors, so work_dir can go before the conversion. Keep the consumption inside the block.
    with tempfile.TemporaryDirectory() as work_dir:
        model_state_dict = load_ddp_main_params(
            str(pathlib.Path(args.checkpoint_input_path) / "model_and_optim"), model_config, work_dir, args.kda_decay
        )
    with kda_decay_export(args.kda_decay):
        convert_checkpoint_to_hf(
            original_checkpoint_path=args.checkpoint_input_path,
            output_path=args.huggingface_output_dir,
            transformer_config_dict=model_config,
            tokenizer_config_dict=tokenizer_config,
            model_state_dict=model_state_dict,
            dtype=DType.bfloat16,
            max_sequence_length=args.max_sequence_length,
            tokenizer_id=args.tokenizer,
            validate=args.validate,
            device=args.device,
            validation_device=args.device,
        )
    write_generation_eos(pathlib.Path(args.huggingface_output_dir), args.eos_token_ids)
    logger.info("CONVERSION_OK")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
