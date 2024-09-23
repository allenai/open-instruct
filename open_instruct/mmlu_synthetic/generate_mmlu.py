import asyncio
import random
import json
from typing import List
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from huggingface_hub import HfApi


@dataclass
class LLMGenerationConfig:
    num_completions: int = 1
    model: str = "gpt-3.5-turbo-0125"
    max_parallel_requests: int = 50  # Adjust based on your API rate limits


@dataclass
class GenerationArgs:
    temperature: float = 0.8
    max_tokens: int = 500
    top_p: float = 0.95


class LLMProcessor:
    def __init__(self, config: LLMGenerationConfig):
        self.config = config
        self.async_client = AsyncOpenAI()

    def format_example(self, sample: dict) -> str:
        return f"""
Subject: {sample['subject']}
Question: {sample['question']}
A: {sample['choices'][0]}
B: {sample['choices'][1]}
C: {sample['choices'][2]}
D: {sample['choices'][3]}
Correct answer: {sample['choices'][sample['answer']]}
"""

    async def generate_synthetic_question(self, samples: List[dict], gen_args: GenerationArgs):
        examples = [self.format_example(sample) for sample in samples[:3]]
        examples_str = "\n".join(examples)

        prompt = f"""
Generate a new multiple-choice question similar to the following MMLU (Massive Multitask Language Understanding) questions:

{examples_str}

Create a new question that's different from the examples you've seen on one of the subjects shown above with four options and indicate the correct answer. Format your response as follows:
Subject: [Subject of your new question]
Question: [Your new question]
A: [Option A]
B: [Option B]
C: [Option C]
D: [Option D]
Correct answer: [Letter of correct option]
"""

        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system",
                     "content": "You are an AI assistant tasked with generating synthetic data for the MMLU dataset."},
                    {"role": "user", "content": prompt}
                ],
                temperature=gen_args.temperature,
                max_tokens=gen_args.max_tokens,
                top_p=gen_args.top_p,
                n=self.config.num_completions
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating question: {e}")
            return None

    async def process_batch(self, samples: List[dict], gen_args: GenerationArgs):
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)

        async def process_with_semaphore(sample_group):
            async with semaphore:
                return await self.generate_synthetic_question(sample_group, gen_args)

        # Group samples into sets of 3 for few-shot learning
        sample_groups = [samples[i:i + 3] for i in range(0, len(samples), 3)]
        # breakpoint()
        tasks = [process_with_semaphore(group) for group in sample_groups]
        return await tqdm.gather(*tasks)


def get_mmlu_sample(num_samples=300):  # Increased to 300 to have 100 groups of 3
    dataset = load_dataset("cais/mmlu", "all")
    samples = random.sample(list(dataset['test']), num_samples)
    return samples


def parse_generated_question(text):
    lines = text.strip().split('\n')
    subject = lines[0].replace('Subject: ', '').strip()
    question = lines[1].replace('Question: ', '').strip()
    choices = [line.split(': ', 1)[1].strip() for line in lines[2:6]]
    answer = ord(lines[6].replace('Correct answer: ', '').strip()) - ord('A')
    return {
        "question": question,
        "subject": subject,
        "choices": choices,
        "answer": answer
    }


def upload_to_huggingface(data, dataset_name):
    dataset = Dataset.from_dict({
        "question": [item["question"] for item in data],
        "subject": [item["subject"] for item in data],
        "choices": [item["choices"] for item in data],
        "answer": [item["answer"] for item in data]
    })

    dataset.push_to_hub(dataset_name)
    print(f"Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{dataset_name}")


async def main():
    config = LLMGenerationConfig()
    processor = LLMProcessor(config)
    samples = get_mmlu_sample(300)  # Generate 300 samples for 100 groups of 3
    gen_args = GenerationArgs()

    raw_synthetic_data = await processor.process_batch(samples, gen_args)
    synthetic_data = [
        parse_generated_question(data)
        for data in raw_synthetic_data
        if data is not None
    ]

    with open('synthetic_mmlu_data.json', 'w') as f:
        json.dump(synthetic_data, f, indent=2)

    print(f"Generated {len(synthetic_data)} synthetic MMLU questions saved to 'synthetic_mmlu_data.json'")

    # Upload to Hugging Face
    upload_to_huggingface(synthetic_data, "your-username/synthetic-mmlu-dataset")


if __name__ == "__main__":
    asyncio.run(main())