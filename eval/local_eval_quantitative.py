from datasets import load_dataset, concatenate_datasets
import copy
import httpx
import asyncio
from open_instruct.ground_truth_utils import verify_gsm8k_sample

semaphore = asyncio.Semaphore(100)


async def call_hosted_model(messages, model, model_url, llm_config):
    async with semaphore:
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
                timeout=120,
            )
        try:
            model_output = response.json()["choices"][0]["message"]["content"]
        except:
            import pdb
            pdb.set_trace()
    
        return model_output


async def main(model):
    tasks = []
    
    for i, data in enumerate(dataset):

        messages = data["translated_messages"]
        if messages is None:
            messages = data["messages"]
        task = asyncio.create_task(call_hosted_model(messages, model["model"], model["model_url"], llm_config))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error for sample {i}: {result}")
            continue
        if i < NUM_SAMPLES:
            import pdb
            pdb.set_trace()
            accuracy_count["hindi"][model["model"].split("/")[-1]].append(verify_gsm8k_sample(result, dataset[i]["ground_truth"]))
        elif i < 2*NUM_SAMPLES:
            accuracy_count["punjabi"][model["model"].split("/")[-1]].append(verify_gsm8k_sample(result, dataset[i]["ground_truth"]))
        # elif i < 3*NUM_SAMPLES:
        #     accuracy_count["tamil"][model["model"].split("/")[-1]].append(verify_gsm8k_sample(result, dataset[i]["ground_truth"]))
        else:
            accuracy_count["english"][model["model"].split("/")[-1]].append(verify_gsm8k_sample(result, dataset[i]["ground_truth"]))


if __name__ == "__main__":

    llm_config = {
        "temperature": 0.1,
    }

    NUM_SAMPLES = 1

    hindi_dataset = load_dataset("sarvam/RLVR-GSM-Hindi").shuffle(seed=42)["train"].select(range(NUM_SAMPLES))
    punjabi_dataset = load_dataset("sarvam/RLVR-GSM-Punjabi").shuffle(seed=42)["train"].select(range(NUM_SAMPLES))
    # tamil_dataset = load_dataset("sarvam/RLVR-GSM-Tamil").shuffle(seed=42)["train"].select(range(NUM_SAMPLES))
    english_dataset = load_dataset("allenai/RLVR-GSM").shuffle(seed=42)["train"].select(range(NUM_SAMPLES))

    dataset = concatenate_datasets([hindi_dataset, punjabi_dataset, english_dataset])

    models = [
        {
            "model": "/home/tanay_sarvam_ai/Meta-Llama-3.1-8B-Instruct",
            "model_url": "http://10.67.27.10:8062/v1/chat/completions",
        },
        {
            "model": "/home/tanay_sarvam_ai/open-instruct/checkpoints/rlvr_llama3_8b_hi_pa_gsm",
            "model_url": "http://10.67.27.7:8063/v1/chat/completions",
        },
    ]

    accuracy_count = {"english": {"Meta-Llama-3.1-8B-Instruct": [], "rlvr_llama3_8b_hi_pa_gsm": []}, "hindi": {"Meta-Llama-3.1-8B-Instruct": [], "rlvr_llama3_8b_hi_pa_gsm": []}, "punjabi": {"Meta-Llama-3.1-8B-Instruct": [], "rlvr_llama3_8b_hi_pa_gsm": []}}

    for model in models:
        asyncio.run(main(model))
        
    # Calculate final scores by summing over lists
    final_scores = {}
    for lang in accuracy_count:
        final_scores[lang] = {}
        for model in accuracy_count[lang]:
            if len(accuracy_count[lang][model]) == 0:
                final_scores[lang][model] = 0
            else:
                final_scores[lang][model] = sum(accuracy_count[lang][model])/len(accuracy_count[lang][model])
    
    print("Raw accuracy counts:", accuracy_count)
    print("\nFinal scores:", final_scores)