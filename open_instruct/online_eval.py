from collections import defaultdict
import time
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizer,
)

from open_instruct.dataset_processor import (
    CHAT_TEMPLATES,
    INPUT_IDS_PROMPT_KEY,
    DatasetConfig,
    SFTDatasetProcessor,
    SimpleGenerateCollator,
)
from open_instruct.model_utils import (
    batch_generation,
    batch_generation_vllm,
    get_reward,
    print_rich_table,
    truncate_response,
    unwrap_model_for_generation,
)
from vllm import SamplingParams, LLM

api = HfApi()
INVALID_LOGPROB = 1.0


def evaluate(
    model: nn.Module,
    reward_model: nn.Module,
    accelerator: Accelerator,
    stop_token_id: int,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: PreTrainedTokenizer,
    response_length: int,
    temperature: float = (0.01 + 1e-7),
    max_sampled_texts: int = 10,
) -> Dict[str, List[str]]:
    generation_config = GenerationConfig(
        max_new_tokens=response_length,
        temperature=temperature,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        eos_token_id=stop_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    table = defaultdict(list)
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        for batch in dataloader:
            query = batch[INPUT_IDS_PROMPT_KEY]
            with torch.no_grad():
                context_length = query.shape[1]
                query_response, _ = batch_generation(
                    unwrapped_model,
                    query,
                    query.shape[0],
                    tokenizer.pad_token_id,
                    generation_config,
                )
                response = query_response[:, context_length:]
                postprocessed_response = response
                if stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(stop_token_id, tokenizer.pad_token_id, response)
                table["query"].extend(tokenizer.batch_decode(query, skip_special_tokens=True))
                table["model response"].extend(tokenizer.batch_decode(postprocessed_response))

                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                _, score, _ = get_reward(
                    reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                )
                table["score"].extend(score.float().cpu().numpy())

            if len(table["query"]) >= max_sampled_texts:
                break
    return table



def evaluate_vllm(
    llm: LLM,
    unwrapped_model: nn.Module,
    reward_model: nn.Module,
    accelerator: Accelerator,
    stop_token_id: int,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: PreTrainedTokenizer,
    response_length: int,
    device: torch.device,
    per_device_eval_batch_size: int,
    world_size: int,
    max_sampled_texts: int = 10,
) -> Dict[str, List[str]]:
    generation_config = SamplingParams(
        temperature=(0.01 + 1e-7),
        top_p=1.0,
        max_tokens=response_length,
        include_stop_str_in_output=True,
    )
    g_vllm_responses = torch.zeros(
        (per_device_eval_batch_size * world_size, response_length),
        device=device,
        dtype=torch.long,
    )
    table = defaultdict(list)
    for batch in dataloader:
        query = batch[INPUT_IDS_PROMPT_KEY].to(device)
        local_vllm_responses = batch_generation_vllm(
            llm,
            generation_config,
            accelerator,
            query,
            unwrapped_model,
            g_vllm_responses,
            tokenizer.pad_token_id,
            response_length,
            device,
        )
        query_response = torch.cat((query, local_vllm_responses), 1)
        with torch.no_grad():
            context_length = query.shape[1]
            response = query_response[:, context_length:]
            postprocessed_response = response
            if stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                postprocessed_response = truncate_response(stop_token_id, tokenizer.pad_token_id, response)
            table["query"].extend(tokenizer.batch_decode(query, skip_special_tokens=True))
            model_response = tokenizer.batch_decode(postprocessed_response)
            model_response = [item.replace(tokenizer.pad_token, "") for item in model_response]
            table["model response"].extend(model_response)

            postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
            _, score, _ = get_reward(
                reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
            )
            table["score"].extend(score.float().cpu().numpy())

        if len(table["query"]) >= max_sampled_texts:
            break
    return table


def evaluate_vllm1(
    llm: LLM,
    unwrapped_model: nn.Module,
    reward_model: nn.Module,
    accelerator: Accelerator,
    stop_token_id: int,
    dataset: Dataset,
    data_collator: SimpleGenerateCollator,
    tokenizer: PreTrainedTokenizer,
    response_length: int,
    device: torch.device,
    per_device_eval_batch_size: int,
    world_size: int,
    max_sampled_texts: int = 10,
) -> Dict[str, List[str]]:
    generation_config = SamplingParams(
        temperature=(0.01 + 1e-7),
        top_p=1.0,
        max_tokens=response_length,
        include_stop_str_in_output=True,
    )
    g_vllm_responses = torch.zeros(
        (max_sampled_texts * world_size, response_length),
        device=device,
        dtype=torch.long,
    )

    data = dataset[max_sampled_texts * world_size][INPUT_IDS_PROMPT_KEY]
    if accelerator.is_main_process:
        llmp = llm.llm_engine.model_executor.driver_worker.model_runner.model
        start_time = time.time()
        llmp.load_weights(unwrapped_model.named_parameters())
        print(f"🔥 Loading weights using shared memory: takes {time.time() - start_time:.2f} seconds")
        outputs = llm.generate(prompt_token_ids=data, sampling_params=generation_config)
        padded_response_token_ids = []
        for output in outputs:
            token_ids = list(output.outputs[0].token_ids)
            DUMMY_PAD_TOKEN = 0  # we can't use tokenizer.pad_token_id because it's outside vocab and `torch.gather(all_logprob, 2, response.unsqueeze(-1))` will error out
            padded_token_ids = token_ids + [DUMMY_PAD_TOKEN] * (response_length - len(token_ids))
            padded_response_token_ids.append(padded_token_ids)
        padded_response_token_ids = torch.tensor(padded_response_token_ids, device=device)
        g_vllm_responses[:] = padded_response_token_ids

    broadcast(g_vllm_responses, 0)

    table = defaultdict(list)
    for i in range(0, max_sampled_texts, per_device_eval_batch_size):
        query = data[i * world_size : (i + per_device_eval_batch_size) * world_size]
        query = data_collator(query).to(device)
        response = g_vllm_responses[i * world_size : (i + per_device_eval_batch_size) * world_size]
        context_length = query.shape[1]
        postprocessed_response = response
        if stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
            postprocessed_response = truncate_response(stop_token_id, tokenizer.pad_token_id, response)
        table["query"].extend(tokenizer.batch_decode(query, skip_special_tokens=True))
        model_response = tokenizer.batch_decode(postprocessed_response)
        model_response = [item.replace(tokenizer.pad_token, "") for item in model_response]
        table["model response"].extend(model_response)

        postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
        _, score, _ = get_reward(
            reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
        )
        table["score"].extend(score.float().cpu().numpy())
    return table

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m", num_labels=1)
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m", num_labels=1)
    dataset_config = DatasetConfig(
        dataset_name="trl-internal-testing/sentiment-trl-style",
        chat_template="simple_concat_with_space",
        sft_messages_key="chosen",
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-14m", num_labels=1)
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m", padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # NOTE: we do not resize the embedding
    tokenizer.chat_template = CHAT_TEMPLATES[dataset_config.chat_template]
    eval_dataset = load_dataset(dataset_config.dataset_name)["test"]
    dataset_processor = SFTDatasetProcessor(
        tokenizer=tokenizer,
        config=dataset_config,
    )
    eval_dataset = dataset_processor.tokenize(eval_dataset)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=8,
        collate_fn=SimpleGenerateCollator(tokenizer.pad_token_id),
    )
    stop_token_id = tokenizer.eos_token_id
    response_length = 53
    table = evaluate(
        model,
        reward_model,
        accelerator,
        stop_token_id,
        dataloader,
        tokenizer,
        response_length,
    )
    print_rich_table(pd.DataFrame(table))
    ...
