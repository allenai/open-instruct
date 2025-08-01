import os
import json

# Set the base directory where your JSON files are located
base_dir = "decontamination_results"  # Change this to your actual path
output_dir = os.path.join(base_dir, "filtered_results")
os.makedirs(output_dir, exist_ok=True)

# Process each .jsonl file
for filename in os.listdir(base_dir):
    if filename.endswith(".jsonl"):
        input_path = os.path.join(base_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".jsonl", "_filtered.jsonl"))

        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:
            for line in infile:
                try:
                    entry = json.loads(line)
                    if entry.get("score", 0) > 0.5:
                        outfile.write(json.dumps(entry) + "\n")
                except json.JSONDecodeError:
                    print(f"Skipping malformed line in {filename}")

print("✅ Done: Filtered files saved to:", output_dir)