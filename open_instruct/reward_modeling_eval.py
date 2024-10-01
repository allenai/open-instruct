from collections import defaultdict
from typing import Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from open_instruct.dataset_processor import (
    CHAT_TEMPLATES,
    INPUT_IDS_CHOSEN_KEY,
    INPUT_IDS_REJECTED_KEY,
    INPUT_IDS_KEY,
    LABELS_KEY,
    DatasetConfig,
    PreferenceDatasetProcessor,
    SimplePreferenceCollator,
    PRMDatasetProcessor,
    SimplePRMCollator

)
from open_instruct.model_utils import get_reward, get_prm_reward, print_rich_table

api = HfApi()


def find_shared_text(chosen_text: str, rejected_text: str):
    """return shared (prompt) text between chosen and rejected text"""
    for i in range(min(len(chosen_text), len(rejected_text))):
        if chosen_text[i] != rejected_text[i]:
            break

    return chosen_text[:i]


def evaluate(
    model: PreTrainedModel, dataloader: DataLoader, tokenizer: PreTrainedTokenizer, max_sampled_texts: int = 0
) -> Tuple[dict, dict]:
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_chosen_rewards = 0
    total_rejected_rewards = 0
    total_reward_margin = 0
    total_batches = 0
    table = None
    if max_sampled_texts > 0:
        table = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(dataloader):
            query_responses = torch.cat((data[INPUT_IDS_CHOSEN_KEY], data[INPUT_IDS_REJECTED_KEY]), dim=0)
            _, predicted_reward, _ = get_reward(model, query_responses, tokenizer.pad_token_id, 0)
            chosen_rewards = predicted_reward[: data[INPUT_IDS_CHOSEN_KEY].shape[0]]
            rejected_rewards = predicted_reward[data[INPUT_IDS_CHOSEN_KEY].shape[0] :]
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_chosen_rewards += chosen_rewards.mean().item()
            total_rejected_rewards += rejected_rewards.mean().item()
            total_reward_margin += (chosen_rewards - rejected_rewards).mean().item()
            total_batches += 1

            if table is not None and len(table["shared prompt text"]) < max_sampled_texts:
                chosen_texts = tokenizer.batch_decode(data[INPUT_IDS_CHOSEN_KEY])
                rejected_texts = tokenizer.batch_decode(data[INPUT_IDS_REJECTED_KEY])
                # remove padding
                chosen_texts = [item.replace(tokenizer.pad_token, "") for item in chosen_texts]
                rejected_texts = [item.replace(tokenizer.pad_token, "") for item in rejected_texts]
                rewards_rounded = [
                    [round(chosen.item(), 4), round(rejected.item(), 4)]
                    for chosen, rejected in zip(chosen_rewards, rejected_rewards)
                ]
                correct_prediction = [
                    bool((chosen > rejected)) for chosen, rejected in zip(chosen_rewards, rejected_rewards)
                ]
                shared_texts = [
                    find_shared_text(chosen_text, rejected_text)
                    for chosen_text, rejected_text in zip(chosen_texts, rejected_texts)
                ]
                chosen_response_texts = [
                    chosen_text[len(shared_text) :] for chosen_text, shared_text in zip(chosen_texts, shared_texts)
                ]
                rejected_response_texts = [
                    rejected_text[len(shared_text) :]
                    for rejected_text, shared_text in zip(rejected_texts, shared_texts)
                ]
                table["shared prompt text"].extend(shared_texts)
                table["chosen response text"].extend(chosen_response_texts)
                table["rejected response text"].extend(rejected_response_texts)
                table["chosen reward, rejected reward"].extend(rewards_rounded)
                table["correct prediction"].extend(correct_prediction)

    model.train()
    return {
        "eval/rm/accuracy": total_accuracy / total_batches,
        "eval/rm/loss": total_loss / total_batches,
        "eval/rm/chosen_rewards": total_chosen_rewards / total_batches,
        "eval/rm/rejected_rewards": total_rejected_rewards / total_batches,
        "eval/rm/reward_margin": total_reward_margin / total_batches,
    }, table

decompose_list = lambda long_list, lengths: [long_list[sum(lengths[:i]):sum(lengths[:i+1])] for i in range(len(lengths))]
def evaluate_prm(
    model: PreTrainedModel, dataloader: DataLoader, tokenizer: PreTrainedTokenizer, max_sampled_texts: int = 0
) -> Tuple[dict, dict]:
    model.eval()
    total_loss = 0
    total_accuracy = 0
    # total_chosen_rewards = 0
    # total_rejected_rewards = 0
    # total_reward_margin = 0
    total_batches = 0
    table = None
    if max_sampled_texts > 0:
        table = defaultdict(list)
    end_step_token_id = tokenizer.encode(" ки")[1]
    with torch.no_grad():
        for data in tqdm(dataloader):
            query_responses = data[INPUT_IDS_KEY]
            flatten_labels = data[LABELS_KEY][data[LABELS_KEY] != tokenizer.pad_token_id].flatten() # data['label'][data['label'] != -1].flatten()           
            input_label_lengths = (data[LABELS_KEY] != tokenizer.pad_token_id).sum(dim=1).tolist()
            _, predicted_reward, _, predcited_step_logits, _ = get_prm_reward(model, query_responses, tokenizer.pad_token_id, end_step_token_id, 0)
            assert predcited_step_logits.shape[0] == flatten_labels.shape[0], f"predicted logits and labels size do not match {predcited_step_logits.shape} vs {flatten_labels.shape}"
            _, predicted_labels = torch.max(predcited_step_logits, dim=1)
            accuracy = (predicted_labels == flatten_labels).float().mean()
            loss = F.cross_entropy(predcited_step_logits, flatten_labels)
            # predicted_probs = predcited_step_logits[: data[LABELS_KEY].shape[0]] #TODO: debug this


            # chosen_rewards = predicted_reward[: data[INPUT_IDS_CHOSEN_KEY].shape[0]]
            # rejected_rewards = predicted_reward[data[INPUT_IDS_CHOSEN_KEY].shape[0] :]
            # accuracy = (chosen_rewards > rejected_rewards).float().mean()
            # loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            # total_chosen_rewards += chosen_rewards.mean().item()
            # total_rejected_rewards += rejected_rewards.mean().item()
            # total_reward_margin += (chosen_rewards - rejected_rewards).mean().item()
            total_batches += 1

            if table is not None and len(table["input text"]) < max_sampled_texts:
                input_texts = tokenizer.batch_decode(data[INPUT_IDS_KEY])
                # remove padding
                input_texts = [item.replace(tokenizer.pad_token, "") for item in input_texts]
                # rewards_rounded = [
                #     [round(chosen.item(), 4), round(rejected.item(), 4)]
                #     for chosen, rejected in zip(chosen_rewards, rejected_rewards)
                # ]
                correct_prediction = [
                    bool((pred_label == gold_label)) for pred_label, gold_label in zip(predicted_labels, flatten_labels)
                ]
                correct_prediction_per_input = decompose_list(correct_prediction, input_label_lengths)
                # shared_texts = [
                #     find_shared_text(chosen_text, rejected_text)
                #     for chosen_text, rejected_text in zip(chosen_texts, rejected_texts)
                # ]
                # chosen_response_texts = [
                #     chosen_text[len(shared_text) :] for chosen_text, shared_text in zip(chosen_texts, shared_texts)
                # ]
                # rejected_response_texts = [
                #     rejected_text[len(shared_text) :]
                #     for rejected_text, shared_text in zip(rejected_texts, shared_texts)
                # ]
                table["input text"].extend(input_texts)
                # table["chosen response text"].extend(chosen_response_texts)
                # table["rejected response text"].extend(rejected_response_texts)
                # table["chosen reward, rejected reward"].extend(rewards_rounded)
                table["correct prediction"].extend(correct_prediction_per_input)

    model.train()
    return {
        "eval/rm/accuracy": total_accuracy / total_batches,
        "eval/rm/loss": total_loss / total_batches,
        # "eval/rm/chosen_rewards": total_chosen_rewards / total_batches,
        # "eval/rm/rejected_rewards": total_rejected_rewards / total_batches,
        # "eval/rm/reward_margin": total_reward_margin / total_batches,
    }, table


if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-14m", num_labels=1)
    dataset_config = DatasetConfig(
        dataset_name="trl-internal-testing/sentiment-trl-style", chat_template="simple_chat"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m", padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # NOTE: we do not resize the embedding
    tokenizer.chat_template = CHAT_TEMPLATES[dataset_config.chat_template]
    eval_dataset = load_dataset(dataset_config.dataset_name)["test"]
    dataset_processor = PreferenceDatasetProcessor(
        tokenizer=tokenizer,
        config=dataset_config,
    ) if 'math' not in dataset_config.dataset_name else PRMDatasetProcessor(tokenizer=tokenizer, config=dataset_config)
    eval_dataset = dataset_processor.tokenize(eval_dataset)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=8,
        collate_fn=SimplePreferenceCollator(tokenizer.pad_token_id) if 'math' not in dataset_config.dataset_name else SimplePRMCollator(pad_token_id=tokenizer.pad_token_id),
    )

    eval_func = evaluate  if 'math' not in dataset_config.dataset_name else evaluate_prm
    metrics, table = eval_func(model, dataloader, tokenizer, max_sampled_texts=5) 
    print(metrics)
    print_rich_table(pd.DataFrame(table))
    ...
