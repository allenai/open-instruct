import asyncio
import random
import json
import re
from typing import List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from huggingface_hub import HfApi


@dataclass
class LLMGenerationConfig:
    num_completions: int = 100
    model: str = "gpt-3.5-turbo-0125"
    max_parallel_requests: int = 50  # Adjust based on your API rate limits


@dataclass
class GenerationArgs:
    temperature: float = 0.7
    max_tokens: int = 300
    top_p: float = 0.95
    examples_per_subject: int = 5
    few_shot_examples: int = 3


class LLMProcessor:
    def __init__(self, config: LLMGenerationConfig):
        self.config = config
        self.async_client = AsyncOpenAI()

    def format_example(self, sample: dict) -> str:
        return f"""
Question: {sample['question']}
A: {sample['choices'][0]}
B: {sample['choices'][1]}
C: {sample['choices'][2]}
D: {sample['choices'][3]}
Correct answer: {sample['choices'][sample['answer']]}
"""

    async def generate_synthetic_question(self, subject: str, examples: List[dict], gen_args: GenerationArgs):
        examples_str = "\n".join([self.format_example(example) for example in examples[:gen_args.few_shot_examples]])

        prompt = f"""
Generate a new multiple-choice question similar to the following MMLU (Massive Multitask Language Understanding) questions on the subject of {subject}:

{examples_str}

Create a new question on {subject} with four options and indicate the correct answer. Format your response as follows:
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
            print(f"Error generating question for {subject}: {e}")
            return None

    async def process_subject(self, subject: str, samples: List[dict], gen_args: GenerationArgs):
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)

        async def process_with_semaphore():
            async with semaphore:
                return await self.generate_synthetic_question(subject, samples, gen_args)

        tasks = [process_with_semaphore() for _ in range(gen_args.examples_per_subject)]
        return await tqdm.gather(*tasks, desc=f"Generating {subject}")


def get_mmlu_samples_by_subject(num_samples_per_subject: int = 10) -> Dict[str, List[dict]]:
    dataset = load_dataset("cais/mmlu", "all")
    samples_by_subject = defaultdict(list)

    for sample in dataset['test']:
        if len(samples_by_subject[sample['subject']]) < num_samples_per_subject:
            samples_by_subject[sample['subject']].append(sample)

    return dict(samples_by_subject)


def parse_generated_question(text: str, subject: str) -> Optional[dict]:
    # Define regex patterns
    question_pattern = re.compile(r'Question:\s*(.+)')
    choice_pattern = re.compile(r'([A-D]):\s*(.+)')
    answer_pattern = re.compile(r'Correct answer:\s*([A-D])')

    # Extract question
    question_match = question_pattern.search(text)
    if not question_match:
        return None
    question = question_match.group(1).strip()

    # Extract choices
    choices = {}
    for match in choice_pattern.finditer(text):
        choices[match.group(1)] = match.group(2).strip()

    # Check if we have exactly 4 choices
    if len(choices) != 4 or set(choices.keys()) != set('ABCD'):
        return None

    # Extract answer
    answer_match = answer_pattern.search(text)
    if not answer_match or answer_match.group(1) not in 'ABCD':
        return None
    answer = ord(answer_match.group(1)) - ord('A')

    return {
        "question": question,
        "subject": subject,
        "choices": [choices[letter] for letter in 'ABCD'],
        "answer": answer
    }


def upload_to_huggingface(data: List[dict], dataset_name: str):
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
    gen_args = GenerationArgs()

    samples_by_subject = get_mmlu_samples_by_subject(gen_args.few_shot_examples + gen_args.examples_per_subject)

    all_synthetic_data = []

    for subject, samples in samples_by_subject.items():
        raw_synthetic_data = await processor.process_subject(subject, samples, gen_args)
        synthetic_data = [
            parsed_question
            for data in raw_synthetic_data
            if data is not None
            if (parsed_question := parse_generated_question(data, subject)) is not None
        ]
        all_synthetic_data.extend(synthetic_data)

    with open('synthetic_mmlu_data.json', 'w') as f:
        json.dump(all_synthetic_data, f, indent=2)

    print(
        f"Generated {len(all_synthetic_data)} valid synthetic MMLU questions across {len(samples_by_subject)} subjects, saved to 'synthetic_mmlu_data.json'")

    # Upload to Hugging Face
    upload_to_huggingface(all_synthetic_data, "your-username/synthetic-mmlu-dataset")


if __name__ == "__main__":
    asyncio.run(main())