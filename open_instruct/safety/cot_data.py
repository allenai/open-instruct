import asyncio
import json
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import anthropic

TEMPLATE = """You're a helpful assistant. I want you to analyze a safety prompt and generate a detailed step-by-step reasoning chain. This chain should show how an AI might process and respond to the prompt, considering potential risks and ethical implications at each step. Please follow these guidelines:

Follow exactly this format for each step:
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

Final Response: I apologize, but I cannot provide advice on that topic as stated. The phrase "beat my wife" could be misinterpreted in a harmful way. Instead, I'd be happy to offer tips on how to improve your tennis skills or suggest ways to engage in friendly competition with your spouse that don't use potentially concerning language. Perhaps we could rephrase it as "How can I improve my tennis game when playing against my spouse?" If you'd like advice on that, I'd be glad to help.

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
    max_tokens: int = 1048
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
        response = await self.generate_step_by_step_answer(prompt, gen_args)
        if response:
            return {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response.get('gpt_response') or response.get('anthropic_response')}
                ],
                "prompt_harm_label": prompt_harm_label,
                "model_used": "gpt" if self.config.use_gpt else "anthropic"
            }

def upload_to_huggingface(data: List[dict], dataset_name: str):
    dataset = Dataset.from_dict(
        {
            "messages": [item["messages"] for item in data],
            "prompt_harm_label": [item["prompt_harm_label"] for item in data],
            "model_used": [item["model_used"] for item in data],
        }
    )

    dataset.push_to_hub(dataset_name)
    print(f"Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{dataset_name}")


async def main():
    config = LLMGenerationConfig(
        use_gpt=True,  # Set to False if you don't want to use GPT
        use_anthropic=False  # Set to False if you don't want to use Anthropic
    )
    processor = LLMProcessor(config)
    gen_args = GenerationArgs()

    train_data = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")

    tasks = []
    for i, item in enumerate(train_data):
        if i == 2:
            break
        prompt = item['prompt']
        prompt_harm_label = item['prompt_harm_label']
        tasks.append(processor.process_safety_data(prompt, prompt_harm_label, gen_args))

    results = await tqdm.gather(*tasks, desc="Processing prompts")


    # Save results to a JSON file
    with open('wildguard_responses_test.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} prompts. Results saved to wildguard_responses_test.json")

    upload_to_huggingface(results, "ai2-adapt-dev/synthetic-cot-safety")

if __name__ == "__main__":
    asyncio.run(main())