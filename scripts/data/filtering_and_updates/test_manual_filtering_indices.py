import numpy as np
from datasets import Dataset

# Simulate a small dataset with repetitive and non-repetitive examples
examples = [
    {"assistant": "Hello world!", "has_repetition": False},
    {"assistant": "Repeat this. Repeat this. Repeat this.", "has_repetition": True, "repetition_reason": "line_repeated_3x", "repetition_examples": ["Repeat this."]},
    {"assistant": "No repetition here.", "has_repetition": False},
    {"assistant": "Spam spam spam spam.", "has_repetition": True, "repetition_reason": "line_repeated_4x", "repetition_examples": ["spam spam spam spam."]},
]

dataset = Dataset.from_dict({k: [ex.get(k, None) for ex in examples] for k in examples[0].keys()})

# Simulate manual filtering: keep only the first repetitive example (index 1), remove the second (index 3)
manual_keep_map = {1: True, 3: False}

def set_manual_keep(example, idx):
    if example.get('has_repetition', False):
        return {'manual_keep': manual_keep_map.get(idx, False)}
    else:
        return {'manual_keep': True}

dataset = dataset.map(set_manual_keep, with_indices=True)

# Now filter by manual_keep
filtered_dataset = dataset.filter(lambda x: x.get('manual_keep', True))

# Check that only the correct examples remain
expected_assistants = ["Hello world!", "Repeat this. Repeat this. Repeat this.", "No repetition here."]
actual_assistants = filtered_dataset['assistant']

assert actual_assistants == expected_assistants, f"Expected {expected_assistants}, got {actual_assistants}"
print("Test passed: manual_keep column filtering works as expected.") 