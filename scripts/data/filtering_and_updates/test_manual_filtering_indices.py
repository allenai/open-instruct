import pyarrow as pa
from datasets import Dataset
import numpy as np

# Simulate a small dataset with repetitive and non-repetitive examples
examples = [
    {"assistant": "Hello world!", "has_repetition": False},
    {"assistant": "Repeat this. Repeat this. Repeat this.", "has_repetition": True, "repetition_reason": "line_repeated_3x", "repetition_examples": ["Repeat this."]},
    {"assistant": "No repetition here.", "has_repetition": False},
    {"assistant": "Spam spam spam spam.", "has_repetition": True, "repetition_reason": "line_repeated_4x", "repetition_examples": ["spam spam spam spam."]},
]

dataset = Dataset.from_dict({k: [ex.get(k, None) for ex in examples] for k in examples[0].keys()})

# Simulate repetitive_dataset as a filtered view
repetitive_dataset = dataset.filter(lambda x: x.get('has_repetition', False))

# Simulate user keeping only the first repetitive example (index 1 in original dataset)
# In real code, repetitive_dataset._indices is a pyarrow.ChunkedArray
# Let's simulate this:
repetitive_dataset._indices = pa.chunked_array([np.array([1])])  # Only keep index 1

# The code under test:
def get_indices_set(indices):
    # This is the logic that failed in the main script
    if hasattr(indices, 'to_pylist'):
        return set(indices.to_pylist())
    elif isinstance(indices, (list, np.ndarray)):
        return set(indices)
    else:
        return set(indices)

try:
    # This should not raise TypeError
    indices_set = get_indices_set(repetitive_dataset._indices)
    print(f"Indices set: {indices_set}")
    assert indices_set == {1}, f"Expected indices set to be {{1}}, got {indices_set}"
    print("Test passed: indices logic works with pyarrow.ChunkedArray.")
except Exception as e:
    print(f"Test failed: {e}")
    raise 