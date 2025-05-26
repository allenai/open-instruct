"""
Analysis for RL-rag. Given an output with documents, is the answer present in them?
example:
python -m open_instruct.search_utils.calculate_doc_recall --output_file /path/to/output.jsonl
"""

import argparse
import json

from open_instruct.ground_truth_utils import f1_score

argparser = argparse.ArgumentParser(description="Calculate recall of documents.")
argparser.add_argument("--output_file", type=str, required=True, help="Path to generated output file")
args = argparser.parse_args()

# load the output file
with open(args.output_file, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

recalls = []
for sample in data:
    ground_truth = sample["ground_truth"]
    generation = sample["generation"]
    # extract text in snippets
    snippets = generation.split("<document>")[1:]
    snippets = [snippet.split("</document>")[0] for snippet in snippets]
    # check if ground truth is in any of the snippets
    metrics = [f1_score(s, ground_truth) for s in snippets]
    if not metrics:
        metrics.append({"recall": 0})
    # take max recall
    recall = max([m["recall"] for m in metrics])
    recalls.append(recall)

# calculate mean recall
mean_recall = sum(recalls) / len(recalls)
print(f"Mean recall: {mean_recall:.4f}")
