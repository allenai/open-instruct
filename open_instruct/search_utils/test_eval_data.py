from datasets import load_dataset
import json
from open_instruct.ground_truth_utils import f1_score


ds = load_dataset("rulins/nq_rlvr_no_prompt", split="test")

labels = [data["ground_truth"] for data in ds]

# Check if labels are JSON strings that need to be parsed
try:
    json.loads(labels[0])
    # If we get here, the labels are JSON strings
    labels = [json.loads(label) for label in labels]
except json.JSONDecodeError:
    # Labels are already plain strings, no processing needed
    labels = [[label] for label in labels]

predictions = [label[0] for label in labels]
f1_scores = [max([f1_score(predictions[i], label) for label in labels[i]], key=lambda x: x['f1']) for i in range(len(predictions))]
import pdb; pdb.set_trace()