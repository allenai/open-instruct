# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os
import time

import bitsandbytes as bnb
import torch
import yaml
from bitsandbytes.functional import dequantize_4bit
from huggingface_hub import HfApi
from peft import PeftConfig, PeftModel
from peft.utils import _get_submodules
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import maybe_use_ai2_hf_entity, retry_on_exception


def dequantize_model(model, dtype=torch.bfloat16, device="cuda"):
    """
    'model': the peftmodel you loaded with qlora.
    'dtype': dtype that the model was trained using
    'device': device to load the model to
    """
    cls = bnb.nn.Linear4bit
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                quant_state = copy.deepcopy(module.weight.quant_state)

                # quant_state changed from a list in newer version of bitsandbytes (0.41.3 onwards)
                if isinstance(quant_state, list):
                    quant_state[2] = dtype
                else:
                    quant_state.dtype = dtype

                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)

                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)

                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)
        # to save model, you have to unset this attribute
        model.is_loaded_in_4bit = False

        return model


@retry_on_exception()
def push_folder_to_hub(
    output_dir: str, hf_repo_id: str | None = None, hf_repo_revision: str | None = None, private: bool = True
):
    hf_repo_url = f"https://huggingface.co/{hf_repo_id}/tree/{hf_repo_revision}"
    api = HfApi()
    if not api.repo_exists(hf_repo_id):
        api.create_repo(hf_repo_id, exist_ok=True, private=private)
    if hf_repo_revision is not None:
        api.create_branch(repo_id=hf_repo_id, branch=hf_repo_revision, exist_ok=True)
    api.upload_folder(
        repo_id=hf_repo_id,
        revision=hf_repo_revision,
        folder_path=output_dir,
        commit_message="upload checkpoint",
        run_as_future=False,
    )
    print(f"🔥 pushed to {hf_repo_url}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_model_name_or_path", type=str, required=False)
    parser.add_argument("--base_model_name_or_path", type=str, required=False)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=False)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--qlora", action="store_true")  # qlora requires special treatment.
    parser.add_argument("--save_tokenizer", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--pad_to_multiple_of", type=int, default=8)  # if you want to pad the token embeddings
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configs = dict()
    if args.config_file:
        with open(args.config_file) as f:
            configs = yaml.safe_load(f)
        # If the config file is provided, reuse settings which are same in the training scripts
        args.base_model_name_or_path = configs["model_name_or_path"]
        args.lora_model_name_or_path = configs["output_dir"]
        args.use_fast_tokenizer = not configs["use_slow_tokenizer"]
        args.qlora = configs.get("use_qlora", False)
        args.seed = configs.get("seed", args.seed)
    if args.lora_model_name_or_path is None:
        raise ValueError("Please provide the path to the lora adapter model.")
    peft_config = PeftConfig.from_pretrained(args.lora_model_name_or_path)
    print("Loading the base model...")
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path if args.base_model_name_or_path else peft_config.base_model_name_or_path,
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map={"": 0} if torch.cuda.is_available() else None,
        )
        # base_model = dequantize_model(base_model, device=base_model.device)
        base_model = dequantize_model(base_model, device="cpu")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path if args.base_model_name_or_path else peft_config.base_model_name_or_path,
            dtype=torch.bfloat16,
        )

    # If tokenizer is specified, use it.
    # Otherwise, use the tokenizer in the lora model folder or the base model folder.
    if args.tokenizer_name_or_path:
        print(f"Loading the tokenizer from {args.tokenizer_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=args.use_fast_tokenizer)
    else:
        try:
            print("Trying to load the tokenizer in the lora model folder...")
            tokenizer = AutoTokenizer.from_pretrained(args.lora_model_name_or_path, use_fast=args.use_fast_tokenizer)
        except Exception as e:
            print(
                f"No tokenizer found in the lora model folder. Using the tokenizer in the base model folder... e:{e}"
            )
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, use_fast=args.use_fast_tokenizer)

    embedding_size = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print(
            f"The vocabulary the tokenizer contains {len(tokenizer) - embedding_size} more tokens than the base model."
        )
        print("Resizing the token embeddings of the merged model...")
        if args.pad_to_multiple_of > 0:
            base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=args.pad_to_multiple_of)
        else:
            base_model.resize_token_embeddings(len(tokenizer))

    print("Loading the lora model...")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_model_name_or_path)
    print("Merging the lora modules...")
    merged_model = lora_model.merge_and_unload()

    output_dir = args.output_dir if args.output_dir else (args.lora_model_name_or_path.rstrip("/") + "_merged/")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir)

    if args.save_tokenizer:
        print(f"Saving the tokenizer to {output_dir}...")
        tokenizer.save_pretrained(output_dir)

    if args.push_to_hub:
        if "hf_repo_id" not in configs:  # auto-generate one
            configs["hf_repo_id"] = "open_instruct_dev"
        if "hf_entity" not in configs:  # first try to use AI2 entity
            configs["hf_entity"] = maybe_use_ai2_hf_entity()
        if configs["hf_entity"] is None:  # then try to use the user's entity
            configs["hf_entity"] = HfApi().whoami()["name"]
        configs["hf_repo_id"] = f"{configs['hf_entity']}/{configs['hf_repo_id']}"
        if "hf_repo_revision" not in configs:  # auto-generate one
            if "exp_name" not in configs:
                configs["exp_name"] = os.path.basename(__file__)[: -len(".py")]
            configs["hf_repo_revision"] = (
                f"{configs['exp_name']}__{args.base_model_name_or_path.replace('/', '_')}__{args.seed}__{int(time.time())}"
            )
        push_folder_to_hub(output_dir, configs["hf_repo_id"], configs["hf_repo_revision"])
