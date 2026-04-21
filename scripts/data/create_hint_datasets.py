"""Create two hint-augmented variants of a math dataset for value model conditioning.

Each variant adds a `hint` column:
  - answer_hint:   "The correct answer to this question is {ground_truth}"
  - solution_hint: "A correct worked solution is: {solution}"  (requires a solution column)

Usage:
    python scripts/data/create_hint_datasets.py \
        --dataset hamishivi/DAPO-Math-17k-Processed_filtered \
        --solution_column solution \
        --hub_prefix hamishivi \
        --push_to_hub
"""

from __future__ import annotations

import argparse

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hamishivi/DAPO-Math-17k-Processed_filtered")
    parser.add_argument("--split", default="train")
    parser.add_argument("--ground_truth_column", default="ground_truth")
    parser.add_argument("--solution_column", default=None, help="Column containing worked solutions.")
    parser.add_argument("--hub_prefix", default="hamishivi")
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(ds)} examples from {args.dataset}")
    print(f"Columns: {ds.column_names}")

    gt_col = args.ground_truth_column

    # Variant 1: answer hint
    def add_answer_hint(row):
        gt = row[gt_col]
        if isinstance(gt, list):
            gt = gt[0] if gt else ""
        row["hint"] = f"The correct answer to this question is {gt}"
        return row

    ds_answer = ds.map(add_answer_hint)
    answer_name = f"{args.hub_prefix}/{args.dataset.split('/')[-1]}_hint_answer"
    print(f"\nAnswer hint dataset: {answer_name}")
    print(f"  Example hint: {ds_answer[0]['hint']}")
    if args.push_to_hub:
        ds_answer.push_to_hub(answer_name)
        print(f"  Pushed to {answer_name}")

    # Variant 2: solution hint (requires solution column)
    sol_col = args.solution_column
    if sol_col is not None:
        if sol_col not in ds.column_names:
            print(f"\nWarning: solution column '{sol_col}' not found. Available: {ds.column_names}")
        else:

            def add_solution_hint(row):
                sol = row[sol_col]
                if isinstance(sol, list):
                    sol = sol[0] if sol else ""
                row["hint"] = f"A correct worked solution is: {sol}"
                return row

            ds_solution = ds.map(add_solution_hint)
            solution_name = f"{args.hub_prefix}/{args.dataset.split('/')[-1]}_hint_solution"
            print(f"\nSolution hint dataset: {solution_name}")
            print(f"  Example hint: {ds_solution[0]['hint'][:200]}...")
            if args.push_to_hub:
                ds_solution.push_to_hub(solution_name)
                print(f"  Pushed to {solution_name}")
    else:
        print("\nNo --solution_column specified; skipping solution hint dataset.")
        print("Run with --solution_column <col> to create the solution hint variant.")


if __name__ == "__main__":
    main()
