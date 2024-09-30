import json
from collections import Counter

def analyze_wildguard_responses(file_path):
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Initialize a Counter to keep track of harm labels
    harm_counter = Counter()

    # Iterate through the data and count harm labels
    for item in data:
        harm_label = item.get('prompt_harm_label')
        if harm_label is not None:
            harm_counter[harm_label] += 1

    # Print results
    print("Analysis of WildGuard Responses:")
    print(f"Total examples: {len(data)}")
    print("\nBreakdown of harm labels:")
    for label, count in harm_counter.items():
        print(f"{label}: {count}")

    # Categorize as harmful or unharmful
    harmful = harm_counter['harmful']
    unharmful = harm_counter['unharmful'] if 'unharmful' in harm_counter else 0

    print("\nSummary:")
    print(f"Harmful examples: {harmful}")
    print(f"Unharmful examples: {unharmful}")

    # Calculate percentages
    total = sum(harm_counter.values())
    if total > 0:
        print(f"\nPercentage harmful: {(harmful/total)*100:.2f}%")
        print(f"Percentage unharmful: {((total-harmful)/total)*100:.2f}%")

if __name__ == "__main__":
    file_path = 'wildguard_responses.json'
    analyze_wildguard_responses(file_path)