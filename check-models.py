import torch
from transformers import AutoModel

# Load both models
model1 = AutoModel.from_pretrained("/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7B-from-posttrained-math-pretrainednonFFN-frozen/step11921-hf")
model2 = AutoModel.from_pretrained("/weka/oe-training-default/sanjaya/flexolmo/checkpoints/intermediate-sft-math/step0-hf")

# Get state dictionaries
state_dict1 = model1.state_dict()
state_dict2 = model2.state_dict()

# Compare keys first
keys1 = set(state_dict1.keys())
keys2 = set(state_dict2.keys())

if keys1 != keys2:
    print("Models have different layer names!")
    print(f"Only in model1: {keys1 - keys2}")
    print(f"Only in model2: {keys2 - keys1}")
else:
    print("Both models have the same layer names ✓\n")

# Compare each layer
different_layers = []
for key in keys1.intersection(keys2):
    tensor1 = state_dict1[key]
    tensor2 = state_dict2[key]
    
    if tensor1.shape != tensor2.shape:
        different_layers.append(f"{key}: shape mismatch ({tensor1.shape} vs {tensor2.shape})")
    elif not torch.equal(tensor1, tensor2):
        # Calculate difference statistics
        diff = (tensor1 - tensor2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        different_layers.append(f"{key}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    else:
        different_layers.append(f"{key}: equal")

if different_layers:
    print(f"Found {len(different_layers)} different layers:\n")
    for layer in different_layers:
        print(layer)
else:
    print("All layers are identical! ✓")
