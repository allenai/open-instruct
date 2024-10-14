#!/usr/bin/env python3

import json
import os
import argparse
from typing import List, Dict

def load_dataset_statistics(output_dir: str) -> List[Dict]:
    statistics = []
    for root, dirs, files in os.walk(output_dir):
        for filename in files:
            if filename.endswith("_statistics.json"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Extract dataset name from the directory structure
                    dataset_name = os.path.basename(root) + "/" + filename.split("_statistics")[0]
                    
                    stat = {
                        'Dataset': dataset_name,
                        'Num Instances': data['num_instances'],
                        'Avg Total Length': round(data['total_lengths_summary']['mean'], 2),
                        'Max Total Length': round(data['total_lengths_summary']['max'], 2),
                        'Avg User Prompt': round(data['user_prompt_lengths_summary']['mean'], 2),
                        'Avg Assistant Response': round(data['assistant_response_lengths_summary']['mean'], 2),
                        'Avg Turns': round(data['turns_summary']['mean'], 2),
                        '% > 512': round(data['num_instances_with_total_length_gt_512'] / data['num_instances'] * 100, 2),
                        '% > 1024': round(data['num_instances_with_total_length_gt_1024'] / data['num_instances'] * 100, 2),
                        '% > 2048': round(data['num_instances_with_total_length_gt_2048'] / data['num_instances'] * 100, 2),
                        '% > 4096': round(data['num_instances_with_total_length_gt_4096'] / data['num_instances'] * 100, 2),
                    }
                    statistics.append(stat)
    return statistics

def print_markdown_table(statistics: List[Dict]):
    if not statistics:
        print("No statistics found.")
        return

    # Print markdown table header
    headers = [
        "Dataset", "Num Instances", "Avg Total Length", "Max Total Length", 
        "Avg User Prompt", "Avg Assistant Response", "Avg Turns",
        "% > 512", "% > 1024", "% > 2048", "% > 4096"
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---" for _ in headers]) + "|")

    # Print table rows
    for stat in sorted(statistics, key=lambda x: x['Dataset']):
        row = [
            stat['Dataset'],
            str(stat['Num Instances']),
            f"{stat['Avg Total Length']:.2f}",
            f"{stat['Max Total Length']:.2f}",
            f"{stat['Avg User Prompt']:.2f}",
            f"{stat['Avg Assistant Response']:.2f}",
            f"{stat['Avg Turns']:.2f}",
            f"{stat['% > 512']:.2f}%",
            f"{stat['% > 1024']:.2f}%",
            f"{stat['% > 2048']:.2f}%",
            f"{stat['% > 4096']:.2f}%"
        ]
        print("| " + " | ".join(row) + " |")

def main():
    parser = argparse.ArgumentParser(description="Generate a markdown table from dataset statistics JSON files.")
    parser.add_argument("output_dir", help="Directory containing the statistics JSON files")
    args = parser.parse_args()

    statistics = load_dataset_statistics(args.output_dir)
    print_markdown_table(statistics)

if __name__ == "__main__":
    main()