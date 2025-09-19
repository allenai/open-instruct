from argparse import ArgumentParser
from functools import partial
import os
import re
from typing import Any, Optional

import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams


def _register_debug_hooks(
    debug_state: dict[str, tuple[int, torch.Tensor]],
    model: torch.nn.Module,
    model_type: str,
    overwrite_debug_state: dict[str, tuple[int, torch.Tensor]] | None = None,
):
    overwrite_debug_state = overwrite_debug_state or {}

    def module_hook(
        debug_state: dict[str, tuple[int, torch.Tensor]],
        name: str,
        module: torch.nn.Module,
        args,
        kwargs,
        output,
    ):
        if name == "logits_processor":
            assert len(args) >= 2 and isinstance(args[1], torch.Tensor)
            assert isinstance(output, torch.Tensor), output

            return module_hook(
                debug_state, "lm_head", module, [args[1]], kwargs, output
            )

        if "qkv_proj" in name:
            assert len(args) >= 1 and isinstance(args[0], torch.Tensor)
            assert isinstance(output, tuple), (name, output)
            assert isinstance(output[0], torch.Tensor), (name, output[0])

            output_tensor = output[0]
            # q_output, k_output, v_output = output.chunk(3, dim=-1)
            if output_tensor.shape[-1] == 6144:
                q_output, k_output, v_output = output_tensor.split(
                    [4096, 1024, 1024], dim=-1
                )
            elif output_tensor.shape[-1] == 12288:
                q_output, k_output, v_output = output_tensor.split(
                    [4096, 4096, 4096], dim=-1
                )
            else:
                q_output, k_output, v_output = output_tensor.split(
                    [128, 64, 64], dim=-1
                )

            q_output = module_hook(
                debug_state,
                name.replace("qkv_proj", "q_proj"),
                module,
                args,
                kwargs,
                q_output,
            )
            k_output = module_hook(
                debug_state,
                name.replace("qkv_proj", "k_proj"),
                module,
                args,
                kwargs,
                k_output,
            )
            v_output = module_hook(
                debug_state,
                name.replace("qkv_proj", "v_proj"),
                module,
                args,
                kwargs,
                v_output,
            )

            if (
                q_output is not None
                and k_output is not None
                and v_output is not None
            ):
                return (
                    torch.cat([q_output, k_output, v_output], dim=-1).to(
                        device=output_tensor.device, dtype=output_tensor.dtype
                    ),
                    *output[1:],
                )
            return None
        if "gate_up_proj" in name:
            assert len(args) >= 1 and isinstance(args[0], torch.Tensor)
            assert isinstance(output, tuple), (name, output)
            assert isinstance(output[0], torch.Tensor), (name, output[0])

            output_tensor = output[0]
            gate_output, up_output = output_tensor.chunk(2, dim=-1)

            gate_output = module_hook(
                debug_state,
                name.replace("gate_up_proj", "gate_proj"),
                module,
                args,
                kwargs,
                gate_output,
            )
            up_output = module_hook(
                debug_state,
                name.replace("gate_up_proj", "up_proj"),
                module,
                args,
                kwargs,
                up_output,
            )
            if gate_output is not None and up_output is not None:
                return (
                    torch.cat([gate_output, up_output], dim=-1).to(
                        device=output_tensor.device, dtype=output_tensor.dtype
                    ),
                    *output[1:],
                )
            return None

        if name.endswith("self_attn"):
            if model_type == "hf":
                assert isinstance(kwargs["hidden_states"], torch.Tensor), (
                    name,
                    kwargs["hidden_states"],
                )
                args = [kwargs["hidden_states"]]
            elif model_type == "vllm":
                assert len(args) >= 2, name
                assert isinstance(args[1], torch.Tensor)
                args = [args[1]]
        if name.endswith("self_attn.rotary_emb"):
            if model_type == "vllm":
                assert len(args) >= 3, (name, args)
                assert isinstance(args[1], torch.Tensor), (name, args[1])
                assert isinstance(args[2], torch.Tensor), (name, args[2])
                module_hook(
                    debug_state,
                    f"{name}.q",
                    module,
                    [args[1].flatten()],
                    kwargs,
                    output[0].flatten(),
                )
                module_hook(
                    debug_state,
                    f"{name}.k",
                    module,
                    [args[2].flatten()],
                    kwargs,
                    output[1].flatten(),
                )
                return
            elif model_type == "hf":
                assert len(args) >= 2, (name, args)
                assert isinstance(args[0], torch.Tensor), (name, args[0])
                assert isinstance(args[1], torch.Tensor), (name, args[1])
                module_hook(
                    debug_state,
                    f"{name}.q",
                    module,
                    [args[0].flatten()],
                    kwargs,
                    output[0].flatten(),
                )
                module_hook(
                    debug_state,
                    f"{name}.k",
                    module,
                    [args[1].flatten()],
                    kwargs,
                    output[1].flatten(),
                )
                return
        if model_type == "vllm" and re.match(r"model.layers.\d+$", name):
            assert len(args) >= 2, (name, args)
            assert isinstance(args[0], torch.Tensor), (name, args[0])
            assert isinstance(args[1], torch.Tensor), (name, args[1])
            args = [args[1]]

        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            state_name = f"{name}|input"
            inp: torch.Tensor = args[0]

            if state_name not in debug_state:
                debug_state[state_name] = (
                    len(debug_state),
                    inp.detach().cpu().clone(),
                )
            else:
                # print(f"Skipping {state_name}")
                pass

        output_tensor, other_args = None, None
        if isinstance(output, (tuple, list)) and isinstance(
            output[0], torch.Tensor
        ):
            output_tensor = output[0]
            other_args = output[1:]
        elif isinstance(output, torch.Tensor):
            output_tensor = output

        if output_tensor is not None:
            state_name = f"{name}|output"

            # if output_tensor.isinf().sum() > 0 and hasattr(module, "weight"):
            #     print(name, module.weight.abs().max())

            if state_name not in debug_state:
                # print(model_type, state_name)
                debug_state[state_name] = (
                    len(debug_state),
                    output_tensor.detach().cpu().clone(),
                )

                if state_name in overwrite_debug_state:
                    # print(
                    #     state_name,
                    #     output_tensor.shape,
                    #     overwrite_debug_state[state_name][1].shape,
                    #     output_tensor.shape
                    #     == overwrite_debug_state[state_name][1].shape,
                    # )
                    if (
                        output_tensor.shape
                        == overwrite_debug_state[state_name][1].squeeze().shape
                    ):
                        # print(f"Overwriting {state_name}")
                        override_output_tensor = (
                            overwrite_debug_state[state_name][1]
                            .clone()
                            .squeeze()
                            .to(
                                device=output_tensor.device,
                                dtype=output_tensor.dtype,
                            )
                        )
                        if other_args:
                            return override_output_tensor, *other_args
                        else:
                            return override_output_tensor
            else:
                # print(f"Skipping {state_name}")
                pass

        return output

    for name, module in model.named_modules():
        module.register_forward_hook(
            partial(module_hook, debug_state, name), with_kwargs=True
        )


def run_vllm_model(
    model_name_or_path: str,
    message: str,
    *,
    max_new_tokens: int = 16,
    max_seq_len: Optional[int] = None,
    revision: Optional[str] = None,
    tokenizer_id: Optional[str] = None,
    dtype: str = "auto",
    debug: bool = True,
    overwrite_debug_state: dict[str, tuple[int, torch.Tensor]] | None = None,
    hf_overrides: Optional[dict[str, Any]] = None,
    vllm_compilation_level: Optional[int] = None,
):
    print("Loading vLLM model")
    # Needed in order to apply debug hooks
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    compilation_config = vllm_compilation_level

    vllm_model = LLM(
        model_name_or_path,
        revision=revision,
        tokenizer=tokenizer_id,
        dtype=dtype,
        max_model_len=max_seq_len,
        hf_overrides=hf_overrides,
        compilation_config=compilation_config,
        enforce_eager=compilation_config is not None and compilation_config != 3,
        gpu_memory_utilization=0.95,
    )

    vllm_debug_state: dict[str, tuple[int, torch.Tensor]] = {}
    if debug:
        vllm_model.apply_model(
            partial(
                _register_debug_hooks,
                vllm_debug_state,
                model_type="vllm",
                overwrite_debug_state=overwrite_debug_state,
            )
        )

    print("Running vLLM model")
    vllm_response = vllm_model.generate(
        message,
        sampling_params=SamplingParams(
            temperature=0, max_tokens=max_new_tokens
        ),
    )
    print("vLLM output:", vllm_response[0].outputs[0].text)

    del vllm_model

    return vllm_response, vllm_debug_state


def run_hf_model(
    model_name_or_path: str,
    message: str,
    *,
    max_new_tokens: int = 16,
    revision: Optional[str] = None,
    tokenizer_id: Optional[str] = None,
    dtype: str = "auto",
    debug: bool = True,
    overwrite_debug_state: dict[str, tuple[int, torch.Tensor]] | None = None,
    hf_overrides: Optional[dict[str, Any]] = None,
    vllm_compilation_level: Optional[int] = None,
):
    if tokenizer_id:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, revision=revision
        )

    print("Loading hf model")
    hf_config = AutoConfig.from_pretrained(
        str(model_name_or_path), revision=revision, **hf_overrides
    )
    # hf_config._attn_implementation = "eager"
    hf_config._attn_implementation = "sdpa"
    # hf_config._attn_implementation = "flex_attention"
    # hf_config._attn_implementation = "flash_attention_2"
    hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        str(model_name_or_path),
        config=hf_config,
        revision=revision,
        device_map="auto",
        dtype=dtype,
    )

    hf_debug_state: dict[str, tuple[int, torch.Tensor]] = {}
    if debug:
        _register_debug_hooks(
            hf_debug_state,
            hf_model,
            model_type="hf",
            overwrite_debug_state=overwrite_debug_state,
        )

    print("Running hf model")
    inputs = tokenizer(
        message, return_tensors="pt", return_token_type_ids=False
    )
    inputs = {k: v.to(hf_model.device.type) for k, v in inputs.items()}
    hf_response = hf_model.generate(**inputs, max_new_tokens=max_new_tokens)

    print(
        "HF output:",
        tokenizer.batch_decode(hf_response, skip_special_tokens=True)[0],
    )

    del hf_model

    return hf_response, hf_debug_state


def main(
    model_name_or_path: str,
    *,
    max_new_tokens: int = 16,
    max_seq_len: Optional[int] = None,
    revision: Optional[str] = None,
    tokenizer_id: Optional[str] = None,
    dtype: str = "auto",
    debug: bool = True,
    sliding_window: Optional[int] = None,
    overwrite_debug_state: bool = False,
    vllm_compilation_level: Optional[int] = None,
    vllm_first: bool = False,
):
    revision = revision or "main"

    # message = "Bitcoin is a "
    # message = "San Francisco is "
    # message = "San Francisco is 1 of 5 cities in the US"
    message = "San Francisco is 1 of 5"

    # print("HF debug state:", hf_debug_state)
    # print("vLLM debug state:", vllm_debug_state)

    hf_overrides = {}
    if sliding_window is not None:
        hf_overrides["sliding_window"] = sliding_window

    hf_debug_state = {}
    vllm_debug_state = {}

    if vllm_first:
        vllm_response, vllm_debug_state = run_vllm_model(
            model_name_or_path=model_name_or_path,
            message=message,
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
            revision=revision,
            tokenizer_id=tokenizer_id,
            dtype=dtype,
            debug=debug,
            overwrite_debug_state=hf_debug_state if overwrite_debug_state else None,
            hf_overrides=hf_overrides,
            vllm_compilation_level=vllm_compilation_level,
        )
    hf_response, hf_debug_state = run_hf_model(
        model_name_or_path=model_name_or_path,
        message=message,
        max_new_tokens=max_new_tokens,
        revision=revision,
        tokenizer_id=tokenizer_id,
        dtype=dtype,
        debug=debug,
        overwrite_debug_state=vllm_debug_state
        if overwrite_debug_state
        else None,
        hf_overrides=hf_overrides,
        vllm_compilation_level=vllm_compilation_level,
    )
    if not vllm_first:
        vllm_response, vllm_debug_state = run_vllm_model(
            model_name_or_path=model_name_or_path,
            message=message,
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
            revision=revision,
            tokenizer_id=tokenizer_id,
            dtype=dtype,
            debug=debug,
            overwrite_debug_state=hf_debug_state if overwrite_debug_state else None,
            hf_overrides=hf_overrides,
            vllm_compilation_level=vllm_compilation_level,
        )

    hf_keys = hf_debug_state.keys()
    vllm_keys = vllm_debug_state.keys()

    if len(vllm_exclusive_keys := vllm_keys - hf_keys) > 0:
        print(
            "vllm exclusive keys:",
            sorted(vllm_exclusive_keys, key=lambda k: vllm_debug_state[k][0]),
        )
    if len(hf_exclusive_keys := hf_keys - vllm_keys) > 0:
        print(
            "hf exclusive keys:",
            sorted(hf_exclusive_keys, key=lambda k: hf_debug_state[k][0]),
        )

    sorted_keys = [
        key
        for key, _ in sorted(
            hf_debug_state.items(), key=lambda item: item[1][0]
        )
    ]

    for i, key in enumerate(sorted_keys):
        hf_idx, hf_state = hf_debug_state[key]

        if key not in vllm_debug_state:
            print(f"No vllm state for {key} (HF {hf_idx}), continuing")
            continue

        vllm_idx, vllm_state = vllm_debug_state[key]

        hf_state = hf_state.squeeze()
        vllm_state = vllm_state.squeeze()

        if hf_state.shape != vllm_state.shape:
            print(
                f"Shape mismatch for key {key}, hf {hf_state.shape} vllm {vllm_state.shape}"
            )

        if hf_state.dtype != vllm_state.dtype:
            print(
                f"Dtype mismatch for key {key}, hf {hf_state.dtype} vllm {vllm_state.dtype}"
            )

        hf_inf_count = hf_state.isinf().sum().item()
        vllm_inf_count = vllm_state.isinf().sum().item()
        if hf_inf_count > 0 or vllm_inf_count > 0:
            print(
                f"Inf detected for key {key}, hf count {hf_inf_count} vllm count {vllm_inf_count}"
            )
            print(
                f"Inf detected for key {key}, hf {hf_state} vllm {vllm_state}"
            )

        if len(hf_state.squeeze().shape) == len(vllm_state.squeeze().shape):
            hf_state = hf_state.squeeze()
            vllm_state = vllm_state.squeeze()

            common_shape = tuple(
                min(hf_dim, vllm_dim)
                for hf_dim, vllm_dim in zip(hf_state.shape, vllm_state.shape)
            )
            for i, dim in enumerate(common_shape):
                hf_state = hf_state.narrow(i, 0, dim)
                vllm_state = vllm_state.narrow(i, 0, dim)

            print(
                f"Coord diff abs mean for key {key} (HF/vllm {hf_idx}/{vllm_idx})",
                (hf_state - vllm_state).float().abs().mean().item(),
            )

        # if not torch.allclose(hf_state, vllm_state, atol=1e-4, rtol=1e-4):
        #     raise RuntimeError(f"State not close for {i}-th key {key}.\n hf {hf_state} \n vllm {vllm_state}")

    # max_logit_diff = torch.max(torch.abs(hf_out.logits - olmo_out.logits))
    # if max_logit_diff >= 1e-4:
    #     raise RuntimeError(max_logit_diff)

    # print("Passed with max logit diff:", max_logit_diff)
    vllm_output = vllm_response[0].outputs[0].text

    if tokenizer_id:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, revision=revision
        )

    hf_output = tokenizer.batch_decode(hf_response, skip_special_tokens=True)[
        0
    ].removeprefix(message)

    print("vLLM output:", vllm_output)
    print("HF output:", hf_output)

    if vllm_output == hf_output:
        print("vLLM and HF output are the same!")
    else:
        print("vLLM and HF output differ :(")

    if overwrite_debug_state:
        print(
            "Warning: Outputs may have been affected by debug state overwriting. Implementations might not match."
        )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--model-name-or-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-c",
        "--max-new-tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
    )
    parser.add_argument(
        "-r",
        "--revision",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
    )
    parser.add_argument(
        "--dtype",
        default="auto",
    )
    parser.add_argument("--skip-debug", action="store_false", dest="debug")
    parser.add_argument(
        "--overwrite-debug-state",
        action="store_true",
        dest="overwrite_debug_state",
    )
    parser.add_argument(
        "--vllm-compilation-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="0 disables compilation, 3 is the max (and default of vllm), 1 is the smallest that does not break debug state (or maybe 0 only...).",
    )
    parser.add_argument(
        "--vllm-first",
        action="store_true",
        help="Run on vllm first then HF. Good for reducing peak memory, bad for overwriting debug state.",
    )
    args = parser.parse_args()

    main(
        args.model_name_or_path,
        max_new_tokens=args.max_new_tokens,
        revision=args.revision,
        tokenizer_id=args.tokenizer,
        dtype=args.dtype,
        debug=args.debug,
        max_seq_len=args.seq_len,
        sliding_window=args.sliding_window,
        overwrite_debug_state=args.overwrite_debug_state,
        vllm_compilation_level=args.vllm_compilation_level,
        vllm_first=args.vllm_first,
    )
