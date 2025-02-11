import ast
import random
from datasets import load_dataset, concatenate_datasets, Dataset
import copy
import httpx
import asyncio
import os
import json
from open_instruct.ground_truth_utils import verify_gsm8k_sample, verify_math_sample


semaphore1 = asyncio.Semaphore(500)
semaphore2 = asyncio.Semaphore(500)

async def call_hosted_model(messages, model, model_url, llm_config):
    headers = {"Content-Type": "application/json"}
    payload = copy.deepcopy(llm_config)
    payload["stream"] = False
    payload["messages"] = messages
    payload["model"] = model
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=model_url,
            headers=headers,
            json=payload,
            timeout=240,
        )
    try:
        model_output = response.json()["choices"][0]["message"]["content"]
    except:
        import pdb
        pdb.set_trace()

    return model_output


async def rate_limited_call_hosted_model(messages, model, model_url, llm_config, semaphore):
    async with semaphore:
        return await call_hosted_model(messages, model, model_url, llm_config)


async def main(model, dataset, dataset_name, semaphore):
    tasks = []
    
    for i, data in enumerate(dataset):

        # if "translated_messages" in data:
        #     messages = ast.literal_eval(data["translated_messages"])
        # else:
        messages = data["translated_messages"]
        task = asyncio.create_task(rate_limited_call_hosted_model(messages, model["model"], model["model_url"], llm_config, semaphore))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    verification_results = []
    positive_samples = []
    negative_samples = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error for sample {i}: {result}")
            continue

        if dataset_name == "gsm":
            verification_result = verify_gsm8k_sample(result, dataset[i]["ground_truth"])
        elif dataset_name == "math":
            verification_result = verify_math_sample(result, dataset[i]["ground_truth"])
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
        verification_results.append(verification_result)

        if verification_result:
            positive_samples.append(dataset[i])
        else:
            negative_samples.append(dataset[i])

    return verification_results, positive_samples, negative_samples
            


if __name__ == "__main__":

    llm_config = {
        "temperature": 0.1,
    }

    models = [
        # {
        #     "model": "/home/tanay_sarvam_ai/open-instruct/checkpoints/rlvr_llama3_8b_indic_gsm_checkpoints/step_200",
        #     "model_url": "http://10.67.27.1:8069/v1/chat/completions",
        # },
        # {
        #     "model": "meta-llama/Llama-3.1-8B-Instruct",
        #     "model_url": "http://10.67.27.15:8070/v1/chat/completions",
        # },
        {
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "model_url": "http://10.67.27.4:8071/v1/chat/completions",
        },
    ]

    NUM_SAMPLES = 10
    
    # for language in ["Hindi", "Punjabi", "Tamil", "Gujarati", "Telugu", "Kannada", "Malayalam", "Bengali", "Marathi", "Odia", "English"]:
        
    #     dataset_path = f"../datasets/local_data/translations_math_{language}.csv"

    #     if language == "English":
    #         dataset = load_dataset("allenai/RLVR-MATH").shuffle(seed=42)["train"].select(range(NUM_SAMPLES))
    #     elif os.path.exists(dataset_path):
    #         dataset = load_dataset("csv", data_files=dataset_path).shuffle(seed=42)["train"].select(range(NUM_SAMPLES)) 
    #     else:
    #         print(f"Dataset {dataset_path} does not exist")
    #         continue

    gsm_dataset = load_dataset("sarvam/RLVR-GSM-Indic").shuffle(seed=42)["train"]
    math_dataset = load_dataset("sarvam/RLVR-MATH-Indic").shuffle(seed=42)["train"]
    datasets = {"gsm": gsm_dataset, "math": math_dataset}
    
    combined_data_positive = []
    combined_data_negative = []
    
    for dataset_name, dataset in datasets.items():
        if dataset_name == "gsm":
            semaphore = semaphore1
        else:
            semaphore = semaphore2
        verification_results, positive_samples, negative_samples = asyncio.run(main(models[0], dataset, dataset_name, semaphore))
        if len(verification_results) > 0:
            print(f"Accuracy: {sum(verification_results)/len(verification_results)} for {len(verification_results)} samples")
        else:
            print(f"No accurate samples for {models[0]['model']}")

        combined_data_positive.extend(positive_samples)
        combined_data_negative.extend(negative_samples)

    random.shuffle(combined_data_positive)
    random.shuffle(combined_data_negative)
    positive_samples_dataset = Dataset.from_list(combined_data_positive[:int(0.25*len(combined_data_negative))])
    negative_samples_dataset = Dataset.from_list(combined_data_negative)
    # Save both positive and negative samples datasets

    positive_samples_dataset.save_to_disk("/data/open-instruct/datasets/pass_samples_gsm_math")
    negative_samples_dataset.save_to_disk("/data/open-instruct/datasets/fail_samples_gsm_math")

    if len(positive_samples_dataset) and len(negative_samples_dataset):
        combined_data = concatenate_datasets([positive_samples_dataset, negative_samples_dataset])
        combined_data.push_to_hub("sarvam/RLVR-Indic-MATH-GSM", token="")
    