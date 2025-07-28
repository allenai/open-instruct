from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
# model_name = "/weka/oe-adapt-default/jacobm/random-rewards/checkpoints/qwen2p5_7b_random_reward/olmo-32b-pref-mix/qwen2p5_7b_random_rewards__1__1751699850_checkpoints/step_1950"
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# prompt = "Can You Imagine 3 Fictional Cities That Are Based On Berlin During 1991?"
prompt = "Who is Natasha Jaques?"
inputs = tokenizer(prompt, return_tensors="pt")

responses = []
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=1024,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
responses.append(response[len(prompt):])  # Remove prompt from response

    
for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            temperature=t,  # Adjust for more/less randomness
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    responses.append(response[len(prompt):])  # Remove prompt from response

for i, response in enumerate(responses, 1):
    print(f"Response {i}: {response}\n")
