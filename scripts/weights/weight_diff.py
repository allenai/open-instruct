#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import fire
import torch
import tqdm
import transformers


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@torch.inference_mode()
def make_diff(
    path_raw: str,
    path_tuned: str,
    path_diff: str,
    device="cpu",  # "cuda" or "cpu"
):
    """Make the weight diff.

    This function is given to present full transparency of how the weight diff was created.

    Run:
        python weight_diff.py make_diff --path_raw <your_path_raw> --path_tuned <your_path_tuned> --path_diff <your_path_diff>
    """
    model_tuned: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_tuned, device_map={"": torch.device(device)}, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model_raw: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw, device_map={"": torch.device(device)}, dtype=torch.float32, low_cpu_mem_usage=True
    )

    tokenizer_tuned: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(path_tuned)
    tokenizer_raw: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(path_raw)
    if tokenizer_raw.pad_token is None:
        tokenizer_raw.add_special_tokens(dict(pad_token="[PAD]"))
        model_raw.resize_token_embeddings(len(tokenizer_raw))

    state_dict_tuned = model_tuned.state_dict()
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_tuned):
        state_dict_tuned[key].add_(-state_dict_raw[key])

    model_tuned.save_pretrained(path_diff)
    tokenizer_tuned.save_pretrained(path_diff)


@torch.inference_mode()
def recover(
    path_raw,
    path_diff,
    path_tuned: str | None = None,
    original_model: str | None = None,
    device="cpu",
    test_inference=True,
):
    """Recover the original weights from the released weight diff.

    This function is given for you to run.

    Things to do before running this:
        1. Convert Meta's released weights into huggingface format. Follow this guide:
            https://huggingface.co/docs/transformers/main/model_doc/llama
        2. Make sure you cloned the released weight diff into your local machine. The weight diff is located at:
            https://huggingface.co/tatsu-lab/alpaca-7b/tree/main
        3. Run this function with the correct paths. E.g.,
            python weight_diff.py recover --path_raw <path_to_step_1_dir> --path_diff <path_to_step_2_dir>

    Additional notes:
        - If things run too slowly, and you have an 80G GPU lying around, let GPU go brrr by setting `--device "cuda"`.
        - If you want to save the recovered weights, set `--path_tuned <your_path_tuned>`.
            Next time you can load the recovered weights directly from `<your_path_tuned>`.
        - to run inference on a reference model (e.g. to ensure diff is correct), set `--original_model <your_model_name>`.
    """
    model_raw: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw, device_map={"": torch.device(device)}, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model_recovered: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_diff, device_map={"": torch.device(device)}, dtype=torch.float32, low_cpu_mem_usage=True
    )

    tokenizer_raw: transformers.PreTrainedTokenizer = transformers.LlamaTokenizer.from_pretrained(path_raw)
    if tokenizer_raw.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"), model=model_raw, tokenizer=tokenizer_raw
        )
    tokenizer_recovered: transformers.PreTrainedTokenizer = transformers.LlamaTokenizer.from_pretrained(path_diff)

    state_dict_recovered = model_recovered.state_dict()
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_recovered):
        state_dict_recovered[key].add_(state_dict_raw[key])

    if path_tuned is not None:
        model_recovered.save_pretrained(path_tuned)
        tokenizer_recovered.save_pretrained(path_tuned)

    if test_inference:
        input_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\r\n\r\n"
            "### Instruction:\r\nList three technologies that make life easier.\r\n\r\n### Response:"
        )
        inputs = tokenizer_recovered(input_text, return_tensors="pt")
        out = model_recovered.generate(inputs=inputs.input_ids, max_new_tokens=100)
        output_text = tokenizer_recovered.batch_decode(out, skip_special_tokens=True)[0]
        output_text = output_text[len(input_text) :]
        print("Recovered model:")
        print(f"Input: {input_text}\nCompletion: {output_text}")
        if original_model:
            og_tokenizer = transformers.AutoTokenizer.from_pretrained(original_model)
            og_model = transformers.AutoModelForCausalLM.from_pretrained(original_model)
            og_inputs = og_tokenizer(input_text, return_tensors="pt")
            og_out = og_model.generate(inputs=og_inputs.input_ids, max_new_tokens=100)
            og_output_text = og_tokenizer.batch_decode(og_out, skip_special_tokens=True)[0]
            og_output_text = og_output_text[len(input_text) :]
            print("Original model:")
            print(f"Input: {input_text}\nCompletion: {og_output_text}")

    return model_recovered, tokenizer_recovered


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
