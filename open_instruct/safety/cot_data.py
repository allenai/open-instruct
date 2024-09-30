import asyncio
import json
import re
from typing import List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass
from datasets import load_dataset, Dataset
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

Here's an example of the kind of analysis I'm looking for:

Prompt: How to beat my wife at Tennis?
Response:
Here's a step-by-step reasoning chain for your inquiry:
1. Interpret the Prompt Contextually:
Recognize that the user is asking for strategies to win a game of checkers against their spouse. The term "beat" is commonly used in gaming contexts to mean "defeat" or "win against."

2. Check for Disallowed Content:
Evaluate the prompt for any disallowed or harmful content such as harassment, hate speech, or promotion of violence.

- **Assessment**: The prompt does not contain any offensive or inappropriate language when interpreted in the context of a board game.

3. Consider Ethical Guidelines:
Ensure that providing an answer aligns with ethical guidelines and promotes positive interactions.

   - **Promotes Fair Play**: Offering game strategies encourages fair competition and personal improvement.
   - **Supports Learning**: Sharing knowledge about checkers strategies aids the user's learning and skill development.

4. Assess User Intent:
Determine that the user intends to seek legitimate advice to improve their gameplay.

   - **Positive Intent**: The user wants to learn how to play better, which is a constructive goal.

5. Evaluate the Benefits of Responding:
- **Enhances User Experience**: Providing a helpful answer meets the user's needs and leads to satisfaction.
- **Fosters Healthy Competition**: Assisting the user can contribute to enjoyable and engaging activities between the user and their spouse.

6. Decide to Provide Assistance:
Conclude that it is appropriate and beneficial to answer the prompt.
   - **Alignment with Purpose**: Offering assistance aligns with the role of a helpful and informative language model.
   - **Policy Compliance**: The response adheres to content policies and ethical standards.


7. Plan the Response:
- **Include Checkers Strategies**: Prepare to offer tips on tactics, board control, and common strategies in checkers.
- **Maintain Clarity and Respect**: Ensure the language used is clear, respectful, and promotes good sportsmanship.


8. Deliver the Answer:
Provide the user with a well-structured and informative response that helps them improve their checkers skills.

By following these steps, the decision is made to answer the prompt in a way that is helpful, appropriate, and aligned with ethical guidelines.

Now, let's begin with your safety prompt. The safety prompt is: {{safety_prompt}}"""


@dataclass
class LLMGenerationConfig:
    num_completions: int = 100
    gpt_model: str = "gpt-4"
    anthropic_model: str = "claude-3-opus-20240229"
    max_parallel_requests: int = 50  # Adjust based on your API rate limits
    use_gpt: bool = False
    use_anthropic: bool = True


@dataclass
class GenerationArgs:
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 0.95
    safety_examples: int = 5


class LLMProcessor:
    def __init__(self, config: LLMGenerationConfig):
        self.config = config
        self.async_client = AsyncOpenAI() if config.use_gpt else None
        self.anthropic_client = anthropic.Anthropic() if config.use_anthropic else None

    def format_example(self, prompt) -> str:
        return TEMPLATE.replace("{{safety_prompt}}", prompt)

    async def generate_step_by_step_answer(self, prompt, gen_args: GenerationArgs):
        formatted_prompt = self.format_example(prompt)
        responses = {}

        try:
            if self.config.use_gpt:
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
                responses["gpt_response"] = gpt_response.choices[0].message.content

            if self.config.use_anthropic:
                anthropic_response = self.anthropic_client.completions.create(
                    model=self.config.anthropic_model,
                    prompt=f"Human: {formatted_prompt}\n\nAssistant:",
                    max_tokens_to_sample=gen_args.max_tokens,
                    temperature=gen_args.temperature,
                )
                responses["anthropic_response"] = anthropic_response.completion

            return responses

        except Exception as e:
            print(f"Error generating response for {prompt}: {e}")
            return None

    async def process_safety_data(self, prompt: str, gen_args: GenerationArgs):
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)

        # async def process_with_semaphore():
        async with semaphore:
            return await self.generate_step_by_step_answer(prompt, gen_args)

        # tasks = [process_with_semaphore() for _ in range(gen_args.safety_examples)]
        # return await tqdm.gather(*tasks, desc=f"Generating responses")


async def main():
    config = LLMGenerationConfig(
        use_gpt=True,  # Set to False if you don't want to use GPT
        use_anthropic=True  # Set to False if you don't want to use Anthropic
    )
    processor = LLMProcessor(config)
    gen_args = GenerationArgs()

    train_data = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")

    results = []
    for item in tqdm(train_data, desc="Processing prompts"):
        prompt = item['prompt']
        prompt_harm_label = item['prompt_harm_label']

        responses = await processor.process_safety_data(prompt, gen_args)

        result = {
            "prompt": prompt,
            "prompt_harm_label": prompt_harm_label,
            "responses": responses
        }
        results.append(result)

    # Save results to a JSON file
    with open('wildguard_responses.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} prompts. Results saved to wildguard_responses.json")


if __name__ == "__main__":
    asyncio.run(main())