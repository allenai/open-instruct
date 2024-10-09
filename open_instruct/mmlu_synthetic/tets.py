import asyncio
import json
import re
from typing import List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


@dataclass
class LLMGenerationConfig:
    num_completions_per_batch: int = 128
    model: str = "gpt-4"
    max_parallel_requests: int = 50
    max_examples_per_subject: int = 1000


@dataclass
class GenerationArgs:
    temperature: float = 0.9
    max_tokens: int = 500
    top_p: float = 0.95
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

    async def generate_synthetic_questions(self, subject: str, examples: List[dict], gen_args: GenerationArgs,
                                           batch_number: int):
        examples_str = "\n".join([self.format_example(example) for example in examples[: gen_args.few_shot_examples]])

        prompt = f"""
Generate {self.config.num_completions_per_batch} new, diverse multiple-choice questions similar to the following MMLU (Massive Multitask Language Understanding) questions on the subject of {subject}:

{examples_str}

Create new questions on {subject} with four options and indicate the correct answer. Ensure the questions are diverse and cover different aspects of the subject. Format your response as follows:

Question: [Your new question]
A: [Option A]
B: [Option B]
C: [Option C]
D: [Option D]
Correct answer: [Letter of correct option]

---

Repeat this format for all {self.config.num_completions_per_batch} questions. This is batch number {batch_number}, so make sure to generate different questions from previous batches. Don't say "Please let me know if you would like me to continue generating more questions in this format".
"""

        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system",
                     "content": "You are an AI assistant tasked with generating diverse synthetic data for the MMLU dataset."},
                    {"role": "user", "content": prompt},
                ],
                temperature=gen_args.temperature,
                max_tokens=gen_args.max_tokens,
                top_p=gen_args.top_p,
                n=1,  # We're now generating multiple questions in a single completion
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating questions for {subject} (batch {batch_number}): {e}")
            return None

    async def process_subject(self, subject: str, samples: List[dict], gen_args: GenerationArgs):
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)

        async def process_batch(batch_number):
            async with semaphore:
                return await self.generate_synthetic_questions(subject, samples, gen_args, batch_number)

        num_batches = (
                                  self.config.max_examples_per_subject + self.config.num_completions_per_batch - 1) // self.config.num_completions_per_batch
        tasks = [process_batch(i) for i in range(num_batches)]
        return await tqdm.gather(*tasks, desc=f"Generating {subject}")


def get_mmlu_samples_by_subject(num_samples_per_subject: int = 10) -> Dict[str, List[dict]]:
    dataset = load_dataset("cais/mmlu", "all")
    samples_by_subject = defaultdict(list)

    for sample in dataset["test"]:
        if len(samples_by_subject[sample["subject"]]) < num_samples_per_subject:
            samples_by_subject[sample["subject"]].append(sample)

    return dict(samples_by_subject)


def parse_generated_questions(text: str, subject: str) -> List[Optional[dict]]:
    questions = re.split(r'\n---\n', text)
    parsed_questions = []

    for question_text in questions:
        parsed_question = parse_generated_question(question_text, subject)
        if parsed_question:
            parsed_questions.append(parsed_question)

    return parsed_questions


def parse_generated_question(text: str, subject: str) -> Optional[dict]:
    # Define regex patterns
    question_pattern = re.compile(r"Question:\s*(.+)")
    choice_pattern = re.compile(r"([A-D]):\s*(.+)")
    answer_pattern = re.compile(r"Correct answer:\s*([A-D])")

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
    if len(choices) != 4 or set(choices.keys()) != set("ABCD"):
        return None

    # Extract answer
    answer_match = answer_pattern.search(text)
    if not answer_match or answer_match.group(1) not in "ABCD":
        return None
    answer = ord(answer_match.group(1)) - ord("A")

    return {
        "question": question,
        "subject": subject,
        "choices": [choices[letter] for letter in "ABCD"],
        "answer": answer,
    }


def upload_to_huggingface(data: List[dict], dataset_name: str):
    dataset = Dataset.from_dict(
        {
            "question": [item["question"] for item in data],
            "subject": [item["subject"] for item in data],
            "choices": [item["choices"] for item in data],
            "answer": [item["answer"] for item in data],
        }
    )

    dataset_name = "ai2-adapt-dev/synth-mmlu-mini-sample-new"
    dataset.push_to_hub(dataset_name)
    print(f"Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{dataset_name}")


async def main():
    config = LLMGenerationConfig()
    processor = LLMProcessor(config)
    gen_args = GenerationArgs()

    samples_by_subject = get_mmlu_samples_by_subject(gen_args.few_shot_examples + 5)  # We need fewer examples now

    all_synthetic_data = []
    for subject, samples in samples_by_subject.items():
        raw_synthetic_data = await processor.process_subject(subject, samples, gen_args)

        for batch in raw_synthetic_data:
            if batch is not None:
                synthetic_data = parse_generated_questions(batch, subject)
                all_synthetic_data.extend(synthetic_data)

        breakpoint()

        # Limit to max_examples_per_subject
        all_synthetic_data = all_synthetic_data[:config.max_examples_per_subject]

    with open(f"synthetic_mmlu_data_{config.model}_diverse.json", "w") as f:
        json.dump(all_synthetic_data, f, indent=2)

    print(
        f"Generated {len(all_synthetic_data)} valid synthetic MMLU questions across {len(samples_by_subject)} subjects, saved to 'synthetic_mmlu_data_{config.model}_diverse.json'")

    # Upload to Hugging Face
    upload_to_huggingface(all_synthetic_data, "ai2-adapt-dev/synthetic-mmlu-dataset-diverse")


if __name__ == "__main__":
    asyncio.run(main())