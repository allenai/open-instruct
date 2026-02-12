"""Minimal reproduction of rope_theta bug in olmo3_5_hybrid transformers fork."""

from transformers import AutoConfig, AutoModelForCausalLM

MODEL_PATH = "/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-long-context-drope/step23842-hf"
#MODEL_PATH = "/weka/oe-adapt-default/finbarrt/stego32/step17000-hf"

# Step 1: Load config and show the problem
print("=== Loading config ===")
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"rope_parameters: {config.rope_parameters}")
print(f"rope_theta in rope_parameters: {config.rope_parameters.get('rope_theta') if config.rope_parameters else 'N/A'}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)
