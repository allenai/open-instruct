from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer

from open_instruct.dataset_transformation import (
    CHOSEN_INPUT_IDS_KEY,
    REJECTED_INPUT_IDS_KEY,
    SimplePreferenceCollator,
    TokenizerConfig,
    get_cached_dataset_tulu,
)
from open_instruct.model_utils import get_reward, print_rich_table

api = HfApi()


def find_shared_text(chosen_text: str, rejected_text: str):
    """return shared (prompt) text between chosen and rejected text"""
    for i in range(min(len(chosen_text), len(rejected_text))):
        if chosen_text[i] != rejected_text[i]:
            break

    return chosen_text[:i]


def evaluate(
    model: PreTrainedModel, dataloader: DataLoader, tokenizer: PreTrainedTokenizer, max_sampled_texts: int = 0
) -> tuple[dict, dict]:
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
            query_responses = torch.cat((data[CHOSEN_INPUT_IDS_KEY], data[REJECTED_INPUT_IDS_KEY]), dim=0)
            _, predicted_reward, _ = get_reward(model, query_responses, tokenizer.pad_token_id, 0)
            chosen_rewards = predicted_reward[: data[CHOSEN_INPUT_IDS_KEY].shape[0]]
            rejected_rewards = predicted_reward[data[CHOSEN_INPUT_IDS_KEY].shape[0] :]
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_chosen_rewards += chosen_rewards.mean().item()
            total_rejected_rewards += rejected_rewards.mean().item()
            total_reward_margin += (chosen_rewards - rejected_rewards).mean().item()
            total_batches += 1

            if table is not None and len(table["shared prompt text"]) < max_sampled_texts:
                chosen_texts = tokenizer.batch_decode(data[CHOSEN_INPUT_IDS_KEY])
                rejected_texts = tokenizer.batch_decode(data[REJECTED_INPUT_IDS_KEY])
                # remove padding
                chosen_texts = [item.replace(tokenizer.pad_token, "") for item in chosen_texts]
                rejected_texts = [item.replace(tokenizer.pad_token, "") for item in rejected_texts]
                rewards_rounded = [
                    [round(chosen.item(), 4), round(rejected.item(), 4)]
                    for chosen, rejected in zip(chosen_rewards, rejected_rewards)
                ]
                correct_prediction = [
                    bool(chosen > rejected) for chosen, rejected in zip(chosen_rewards, rejected_rewards)
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


if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-14m", num_labels=1)
    tc = TokenizerConfig(tokenizer_name_or_path="EleutherAI/pythia-14m")
    tokenizer = tc.tokenizer
    eval_dataset = get_cached_dataset_tulu(
        ["trl-internal-testing/sentiment-trl-style", "1.0"],
        ["test"],
        tc,
        ["preference_tokenize_v1", "preference_filter_v1"],
        [{}, {"max_token_length": 1024, "max_prompt_token_length": 512}],
        target_columns=[CHOSEN_INPUT_IDS_KEY, REJECTED_INPUT_IDS_KEY],
        dataset_skip_cache=False,
    )
    dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=SimplePreferenceCollator(tokenizer.pad_token_id))
    metrics, table = evaluate(model, dataloader, tokenizer, max_sampled_texts=5)
    print(metrics)
    print_rich_table(pd.DataFrame(table))
