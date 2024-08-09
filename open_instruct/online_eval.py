from collections import defaultdict
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from accelerate import Accelerator
from datasets import load_dataset
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
    get_reward,
    print_rich_table,
    truncate_response,
    unwrap_model_for_generation,
)

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
