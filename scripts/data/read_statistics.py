import argparse
import json
import os

"""
Usage:
python scripts/data/read_statistics.py --latex data/processed/

"""


def load_dataset_statistics(output_dir: str) -> list[dict]:
    statistics = []
    for root, dirs, files in os.walk(output_dir):
        for filename in files:
            if filename.endswith("_statistics.json"):
                file_path = os.path.join(root, filename)
                with open(file_path) as f:
                    data = json.load(f)
                    dataset_name = os.path.basename(root) + "/" + filename.split("_statistics")[0]

                    stat = {
                        "Dataset": dataset_name,
                        "Num Instances": data["num_instances"],
                        "Avg Total Length": data["total_lengths_summary"]["mean"],
                        "Max Total Length": data["total_lengths_summary"]["max"],
                        "Avg User Prompt": data["user_prompt_lengths_summary"]["mean"],
                        "Avg Assistant Response": data["assistant_response_lengths_summary"]["mean"],
                        "Avg Turns": data["turns_summary"]["mean"],
                        "% > 512": data["num_instances_with_total_length_gt_512"] / data["num_instances"] * 100,
                        "% > 1024": data["num_instances_with_total_length_gt_1024"] / data["num_instances"] * 100,
                        "% > 2048": data["num_instances_with_total_length_gt_2048"] / data["num_instances"] * 100,
                        "% > 4096": data["num_instances_with_total_length_gt_4096"] / data["num_instances"] * 100,
                    }
                    statistics.append(stat)
    return statistics


def print_markdown_table(statistics: list[dict], columns: list[str], roundings: dict[str, int]):
    if not statistics:
        print("No statistics found.")
        return

    # Print markdown table header
    headers = [col for col in columns]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---" for _ in headers]) + "|")

    # Print table rows
    for stat in sorted(statistics, key=lambda x: x["Dataset"]):
        row = []
        for col in columns:
            value = stat[col]
            if isinstance(value, float) and col in roundings:
                value = round(value, roundings[col])
            row.append(f"{value:.{roundings.get(col, 2)}f}" if isinstance(value, float) else str(value))
        print("| " + " | ".join(row) + " |")


def print_latex_table(statistics: list[dict], columns: list[str], roundings: dict[str, int]):
    if not statistics:
        print("No statistics found.")
        return

    columns_pretty = [col.replace("%", "\\%") for col in columns]
    # Print LaTeX table header
    print("\\begin{table}[]")
    print("\\centering")
    print("\\setlength\\tabcolsep{5pt}")
    print("\\adjustbox{max width=\\linewidth}{")
    print("\\begin{tabular}{@{}lcccccccc@{}}")
    print("\\toprule")
    print(" & ".join(columns_pretty) + "\\\\")
    print("\\midrule")

    # Print table rows
    for stat in sorted(statistics, key=lambda x: x["Dataset"]):
        row = []
        for col in columns:
            value = stat[col]
            if isinstance(value, float) and col in roundings:
                value = round(value, roundings[col])
            row.append(f"{value:.{roundings.get(col, 2)}f}" if isinstance(value, float) else str(value))
        print(" & ".join(row) + "\\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\caption{}")
    print("\\label{}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Generate a table from dataset statistics JSON files.")
    parser.add_argument("output_dir", help="Directory containing the statistics JSON files")
    parser.add_argument("--latex", action="store_true", help="Output in LaTeX format instead of markdown")
    args = parser.parse_args()

    statistics = load_dataset_statistics(args.output_dir)

    # Define the columns to include and rounding rules
    columns = [
        "Dataset",
        "Num Instances",
        "Avg Total Length",
        "Max Total Length",
        "Avg User Prompt",
        "Avg Assistant Response",
        "Avg Turns",
        "% > 512",
        "% > 1024",
        "% > 2048",  # Uncomment to include this column
        "% > 4096",  # Uncomment to include this column
    ]
    roundings = {
        "Avg Total Length": 2,
        "Max Total Length": 2,
        "Avg User Prompt": 2,
        "Avg Assistant Response": 2,
        "Avg Turns": 2,
        "% > 512": 2,
        "% > 1024": 2,
        "% > 2048": 2,
        "% > 4096": 2,
    }

    if args.latex:
        print_latex_table(statistics, columns, roundings)
    else:
        print_markdown_table(statistics, columns, roundings)


if __name__ == "__main__":
    main()
