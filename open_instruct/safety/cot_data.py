import asyncio
import json
from dataclasses import dataclass
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import anthropic

TEMPLATE = """You're a helpful assistant. I want you to analyze a safety prompt and generate a detailed step-by-step reasoning chain. This chain should show how an AI might process and respond to the prompt, considering potential risks and ethical implications at each step. Please follow these guidelines:

1. Start by restating the given safety prompt.
2. Break down the prompt into its key components or questions.
3. For each component, provide a step in the reasoning process, explaining how an AI might interpret and analyze it.
4. Consider potential risks, ethical concerns, or safety implications at each step.
5. If applicable, suggest alternative interpretations or approaches an AI could take.
6. Conclude with a summary of the overall reasoning chain and provide a response.

Please use this format for each step:
Step X: [Brief description]
Reasoning: [Detailed explanation]
Safety Consideration: [Relevant safety or ethical point]

Now, let's begin with your safety prompt. The safety prompt is: {{safety_prompt}}"""

@dataclass
class LLMGenerationConfig:
    gpt_model: str = "gpt-4"
    anthropic_model: str = "claude-3-sonnet-20240229"
    max_parallel_requests: int = 50  # Adjust based on your API rate limits
    use_gpt: bool = False
    use_anthropic: bool = True

@dataclass
class GenerationArgs:
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 0.95

class LLMProcessor:
    def __init__(self, config: LLMGenerationConfig):
        self.config = config
        self.async_client = AsyncOpenAI() if config.use_gpt else None
        self.anthropic_client = anthropic.Anthropic() if config.use_anthropic else None
        self.semaphore = asyncio.Semaphore(config.max_parallel_requests)

    def format_example(self, prompt) -> str:
        return TEMPLATE.replace("{{safety_prompt}}", prompt)

    async def generate_gpt_response(self, formatted_prompt, gen_args):
        async with self.semaphore:
            try:
                gpt_response = await self.async_client.chat.completions.create(
                    model=self.config.gpt_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an AI assistant tasked with generating thoughts before the final answer.",
                        },
                        {"role": "user", "content": formatted_prompt},
                    ],
                    temperature=gen_args.temperature,
                    max_tokens=gen_args.max_tokens,
                    top_p=gen_args.top_p,
                    n=1,
                )
                return gpt_response.choices[0].message.content
            except Exception as e:
                print(f"Error generating GPT response: {e}")
                return None

    def generate_anthropic_response(self, formatted_prompt, gen_args):
        try:
            anthropic_response = self.anthropic_client.messages.create(
                model=self.config.anthropic_model,
                max_tokens=gen_args.max_tokens,
                temperature=gen_args.temperature,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ]
            )
            return anthropic_response.content[0].text
        except Exception as e:
            print(f"Error generating Anthropic response: {e}")
            return None

    async def generate_step_by_step_answer(self, prompt, gen_args: GenerationArgs):
        formatted_prompt = self.format_example(prompt)
        responses = {}

        if self.config.use_gpt:
            responses["gpt_response"] = await self.generate_gpt_response(formatted_prompt, gen_args)

        if self.config.use_anthropic:
            responses["anthropic_response"] = self.generate_anthropic_response(formatted_prompt, gen_args)

        return responses

    async def process_safety_data(self, prompt: str, prompt_harm_label: str, gen_args: GenerationArgs):
        responses = await self.generate_step_by_step_answer(prompt, gen_args)
        return {
            "prompt": prompt,
            "prompt_harm_label": prompt_harm_label,
            "responses": responses
        }

async def main():
    config = LLMGenerationConfig(
        use_gpt=True,  # Set to False if you don't want to use GPT
        use_anthropic=False  # Set to False if you don't want to use Anthropic
    )
    processor = LLMProcessor(config)
    gen_args = GenerationArgs()

    train_data = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")

    tasks = []
    for item in train_data:
        prompt = item['prompt']
        prompt_harm_label = item['prompt_harm_label']
        tasks.append(processor.process_safety_data(prompt, prompt_harm_label, gen_args))

    results = await tqdm.gather(*tasks, desc="Processing prompts")

    # Save results to a JSON file
    with open('wildguard_responses.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} prompts. Results saved to wildguard_responses.json")

if __name__ == "__main__":
    asyncio.run(main())