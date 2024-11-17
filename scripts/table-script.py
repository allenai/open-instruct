import pandas as pd
import argparse
import sys

"""
Examples:

python -m plots.table-tulu3-90 --csv-path leaderboard/exported_results.csv --models ppo_ray_β_0.03__3__1730357435 Meta-Llama-3.1-8B-Instruct hf-ministral_8b_instruct_2410 hf-qwen2_5_7b_instruct valpy_dpo_70b_hslj_uflj_dalj_wciflj_iftaxlj_wcunusedlj hf-Llama-3.1-70B-Instruct hf-qwen2_5_72b_instruct

8B
python -m plots.table-tulu3-90 --csv-path leaderboard/exported_results.csv --models Meta-Llama-3.1-8B-Instruct hf-google_gemma-2-9b-it hf-NousResearch-Hermes-3-Llama-3.1-8B hf-qwen2_5_7b_instruct hf-ministral_8b_instruct_2410 L3.18B-math-mix-final-nc__meta-llama_Llama-3.1-8B__42__1729284525 dpo_tune___model__42__1729311739 ppo_ray_β_0.03__3__1730357435

70B
python -m plots.table-tulu3-90 --csv-path leaderboard/exported_results.csv --models hf-Meta-Llama-3.1-70B-Instruct hf-qwen2_5_72b_instruct hf-NousResearch-Hermes-3-Llama-3.1-70B hf-llama_3_1_nemotron_70B_instruct_hf L3.1-70B-v3.8-lr_2e-6-2_epochs 70B_ppo_ray_β_0.07_lr_1e-7__3__1730258118 L3.1-70B-v3.8-lr_2e-6-2_epochs-pif_dpo-5e-7 

Merging example:
python table-tulu3.py --csv-path ~/Downloads/exported_results_4.csv --models L3.1-8B-v3.8-nc-soup L3.1-8B-v3.9-nc-3__meta-llama_Llama-3.1-8B__456__1730332817 L3.1-8B-v3.9-nc-2__meta-llama_Llama-3.1-8B__123__1730333671 L3.1-8B-v3.9-nc__meta-llama_Llama-3.1-8B__42__1730330678
"""

model_label_conversion = {
    # llamas
    "Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
    "hf-Llama-3.1-70B-Instruct": "Llama 3.1 70B Instruct",
    "hf-Meta-Llama-3.1-70B-Instruct": "Llama 3.1 70B Instruct",
    #
    "hf-llama-3-tulu-2-8b": "Tulu 2 SFT",
    "hf-llama-3-tulu-2-dpo-8b": "Tulu 2 + DPO",
    "L3.1-8B-v3.8-nc-final__meta-llama_Llama-3.1-8B__42__1729991287": "Tulu 3 SFT",
    "L3.1-8B-v3.8-wip-persona_code_v3-2-pif_dpo___model__42__1729725103": "Tulu 3 + DPO",
    "ljrmvalue_lj_gsm_data_step_300": "Tulu 3 + RL",
    "hf-NousResearch-Hermes-3-Llama-3.1-8B": "Hermes 3 8B",
    "hf-NousResearch-Hermes-3-Llama-3.1-70B": "Hermes 3 70B",
    "hf-llama_3_tulu_2_dpo_70b": "Tulu 2 + DPO 70B",
    "L3.1-70B-v3.7-nc": "Tulu 3 70B SFT",
    "hf-google_gemma-2-9b-it": "Gemma 2 9B",
    "hf-ministral_8b_instruct_2410": "Ministral 8B",
    "hf-magpielm_8b_chat_v0_1": "Magpie 8B",
    "hf-gemma_2_9b_it_simpo": "Gemma 2 9B SimPO",
    "L3.1-8B-v3.8-nc-soup-pif_dpo-soup": "Tulu 3 + Merging + DPO",
    "L3.1-8B-v3.8-nc-soup": "Tulu 3 SFT Merge",
    "L3.1-8B-v3.9-nc-3__meta-llama_Llama-3.1-8B__456__1730332817": "Seed 1",
    "L3.1-8B-v3.9-nc-2__meta-llama_Llama-3.1-8B__123__1730333671": "Seed 2",
    "L3.1-8B-v3.9-nc__meta-llama_Llama-3.1-8B__42__1730330678": "Seed 3",
    # random SFT mixes
    "fae_llama3_sftmix_v3.4_personahub_if_v1__meta-llama_Meta-Llama-3-8B__42__1728059424": "Tulu v3.4 SFT",
    "sft_preview_mix_v3.5.10__meta-llama_Llama-3.1-8B__42__1729148912": "Tulu v3.6 SFT",
    "L3.18B-v3.7-c__meta-llama_Llama-3.1-8B__42__1729454073": "Tulu v3.7 SFT",
    "L3.1-8B-v3.8-nc-final__meta-llama_Llama-3.1-8B__42__1729991287": "Tulu v3.8 SFT",
    "L3.1-8B-v3.8-nc-soup": "Tulu v3.8 SFT + Merging",
    "hf-llama_3_tulu_2_70b": "Tulu 2 SFT 70B",
    "L3.1-70B-v3.8-lr_2e-6-2_epochs-pif_dpo-5e-7": "Tulu 3 DPO 70B",
    "L3.1-70B-v3.8-lr_2e-6-2_epochs": "Tulu 3 SFT 70B",
    # 7b rivals
    "hf-qwen2_5_7b_instruct": "Qwen 2.5 7B Instruct",
    "hf-ministral_8b_instruct_2410": "Ministral 8B Instruct",
    "hf-google_gemma-2-9b-it": "Gemma 2 9B",
    "hf-gemma_2_9b_it_simpo": "Gemma 2 9B SimPO",
    # 70b rivalsqw
    "hf-llama_3_1_nemotron_70b_instruct_hf": "Nemotron Llama 3.1 70B",
    "hf-llama_3_1_nemotron_70B_instruct_hf": "Nemotron Llama 3.1 70B",
    "hf-qwen2_5_72b_instruct": "Qwen 2.5 72B",
    # LMSYS version compare
    "L3.18B-math-mix-final-nc__meta-llama_Llama-3.1-8B__42__1729284525": "Tulu 3 SFT",
    "dpo_tune___model__42__1729311739": "Tulu 3 DPO",
    "ppo_ray_β_0.03__3__1730357435": "Tulu 3 8B",
    # 70b fine tunes
    "L3.1-70B-v3.8-lr_2e-6-2_epochs-pif_dpo-5e-7": "Tulu 70B DPO",
    "70B_ppo_ray_β_0.07_lr_1e-7__3__1730258118": "Tulu 70B RL",
    "valpy_dpo_70b_hslj_uflj_dalj_wciflj_iftaxlj_wcunusedlj": "Tulu 3 70B",
    "hf-NousResearch-Hermes-3-Llama-3.1-8B": "Hermes 3 8B",
    "hf-llama-3-tulu-2-8b": "Tulu 2 8B SFT",
    "L3.1-8B-v3.9-nc-fixed-2__meta-llama_Llama-3.1-8B__123__1730531285": "Tulu 3 8B SFT",
    "hf-NousResearch-Hermes-3-Llama-3.1-70B": "Hermes 3 70B",
    "hf-llama-3-tulu-2-70b": "Tulu 2 70B SFT",
    "L3.1-70B-v3.9-nc-2e-6-2_ep-fixed-3__meta-llama_Llama-3.1-70B__456__1731059165": "Tulu 3 70B SFT",
    "L3.1-8B-v3.9-nc-no-safety__meta-llama_Llama-3.1-8B__42__1731562927": "Tulu 3 8B SFT w/o Safety",
    "L3.1-8B-v3.9-nc-no-wc__meta-llama_Llama-3.1-8B__42__1731562946": "Tulu 3 8B SFT w/o WildChat",
    "L3.1-8B-v3.9-nc-no-synthetic__meta-llama_Llama-3.1-8B__42__1731613382": "Tulu 3 8B SFT w/o Synthetic Data (ours)",
    "L3.1-8B-v3.9-nc-no-math__meta-llama_Llama-3.1-8B__42__1731562937": "Tulu 3 8B SFT w/o Mathematics",
    "hf-RLHFlow-LLaMA3-SFT-v2": "RLHFlow SFT V2",
    "hf-MAmmoTH2-8B": "MAmmoTH2 8B",

    # downsampling
    "L3.1-8B-v3.9-nc-downsample-0.05__meta-llama_Llama-3.1-8B__42__1731214637": "Tulu 3 8B SFT (5\%)",
    "L3.1-8B-v3.9-nc-downsample-0.10__meta-llama_Llama-3.1-8B__42__1731214619": "Tulu 3 8B SFT (10\%)",
    "L3.1-8B-v3.9-nc-downsample-0.25__meta-llama_Llama-3.1-8B__42__1731214572": "Tulu 3 8B SFT (25\%)",
    "L3.1-8B-v3.9-nc-downsample-0.50__meta-llama_Llama-3.1-8B__42__1731214572": "Tulu 3 8B SFT (50\%)",
    "L3.1-8B-v3.9-nc-downsample-0.75__meta-llama_Llama-3.1-8B__42__1731214576": "Tulu 3 8B SFT (75\%)",
}

# Metric keys definition
metric_keys = {
    "MMLU": "mmlu:mc::tulu",
    "TruthfulQA": "truthfulqa",
    "PopQA": "popqa",
    "BigBenchHard": "bbh:cot::tulu",
    "HumanEval": "codex_humaneval",
    "HumanEval+": "codex_humanevalplus",
    "GSM8K": "gsm8k",
    "DROP": "drop",
    "MATH": "math::flex",
    "IFEval": "ifeval",
    "AlpacaEval 2": "alpaca_eval",
    "Safety": "overall_oe_safety_average",
}

eval_settings = {
    "MMLU": "5 shot",
    "TruthfulQA": "6 shot",
    "PopQA": "15 shot",
    "BigBenchHard": "3 shot, CoT",
    "HumanEval": "pass@10",
    "HumanEval+": "pass@10",
    "GSM8K": "8 shot, CoT",
    "DROP": "3 shot",
    "MATH": "4 shot CoT, Flex",
    "IFEval": "Strict",
    "AlpacaEval 2": "LC \% win",
    "Safety": "",
}

# Change this to change the table size
AVERAGE_KEYS = [
    "alpaca_eval",
    "bbh:cot::tulu",
    "codex_humaneval",
    "codex_humanevalplus",
    "drop",
    "gsm8k",
    "ifeval",
    "math::flex",
    "mmlu:mc::tulu",
    "popqa",
    "truthfulqa",
    "overall_oe_safety_average",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a table of model performance metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--csv-path", required=True, help="Path to the CSV file containing the results"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names to generate table for",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output in Markdown format instead of LaTeX",
    )
    parser.add_argument(
        "--extra_cols",
        type=int,
        default=0,
        help="Number of extra columns to add to the table",
    )

    return parser.parse_args()


def format_value(value, markdown=False):
    """Format a numeric value for table output."""
    if pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.1f}"
    except:
        return "N/A"


def create_performance_table_rows(csv_path, model_names, markdown=False, extra_cols=0):
    """
    Create performance table rows for the specified models.

    Parameters:
    csv_path (str): Path to the CSV file containing the results
    model_names (list): List of model names to generate table for
    markdown (bool): Whether to output in Markdown format
    extra_cols (int): Number of extra columns to add to the table
    """

    try:
        all_data = {}
        df = pd.read_csv(csv_path)
        rows = []

        for model_name in model_names:
            model_data = df[df["Model"] == model_name]
            if len(model_data) == 0:
                print(f"Warning: Model '{model_name}' not found in CSV file")
                continue

            # Get pretty model name from conversion dictionary
            pretty_name = model_label_conversion.get(model_name, model_name)

            # Replace "Tulu" with "\modelname" for LaTeX output only
            if not markdown:
                pretty_name = pretty_name.replace("Tulu ", "\\modelname~")

            all_data[pretty_name] = {}

            # Calculate average
            for key in AVERAGE_KEYS:
                model_data[key] = model_data[key].apply(
                    lambda x: float(x) if x != "nan" else None
                )
            average = model_data[AVERAGE_KEYS].mean(axis=1).iloc[0]
            all_data[pretty_name]["Avg."] = format_value(average, markdown)

            # add all the eval scores
            for metric_name, metric_key in metric_keys.items():
                value = model_data[metric_key].iloc[0]
                all_data[pretty_name][metric_name] = format_value(value, markdown)

        for metric_name in ["Avg."] + list(metric_keys.keys()):
            values = [metric_name]
            if metric_name == "Avg.":
                values.append("")
            else:
                values.append(f"\\small{{{eval_settings[metric_name]}}}")
            for pretty_name in all_data.keys():
                values.append(all_data[pretty_name][metric_name])

            values = ["-1" if i == "N/A" else i for i in values]
            numbers = [float(v) for v in values[2:]]
            max_index = numbers.index(max(numbers)) + 2
            values[max_index] = f"\\textbf{{{values[max_index]}}}"

            if markdown:
                # Markdown table row with pretty name
                r = f"|  | {' | '.join(values)} |"
                r += " |" * extra_cols
                rows.append(r)
            else:
                # LaTeX table row with pretty name
                r = f"{' & '.join(values)}"
                r += " &" * extra_cols
                r += " \\\\"
                rows.append(r)
            if metric_name == "Avg.":
                rows.append("\\midrule")

        return rows

    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {csv_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {csv_path} is empty")
        sys.exit(1)


def create_latex_table(model_names, extra_cols):
    """Return the LaTeX table header."""
    header = """\\begin{table}[]
\\centering
\\setlength\\tabcolsep{5pt}
\\adjustbox{max width=\\linewidth}{
"""
    column_spec = "ll"
    for model_name in model_names:
        if "Tulu" in model_label_conversion[model_name]:
            # P is defined via \newcolumntype{P}{>{\columncolor{ai2pink}}c}
            column_spec += "l"
        else:
            column_spec += "c"
    column_spec += "c" * extra_cols

    header += (
        """\\begin{NiceTabular}{@{}"""
        + column_spec
        + """@{}}
\\toprule
"""
    )
    header += """\\textbf{Benchmark} & \\textbf{Eval Setting}"""
    for model_name in model_names:
        pretty_name = model_label_conversion.get(model_name, model_name)
        if "Tulu" in pretty_name:
            pretty_name = pretty_name.replace("Tulu ", "\\modelname~")
            pretty_name = f"\\textbf{{{pretty_name}}}"
        header += " & \\rotatebox{90}{" + pretty_name + "}"
    for i in range(extra_cols):
        header += " & "
    header += """\\\\\\midrule"""
    return header


def create_latex_footer():
    """Return the LaTeX table footer."""
    return """\\bottomrule
\\end{NiceTabular}}
\\vspace{3pt}
\\caption{TODO}
\\label{tab:TODO}
\\end{table}"""


def main():
    """Main function to run the script."""
    args = parse_args()

    rows = create_performance_table_rows(
        csv_path=args.csv_path,
        model_names=args.models,
        markdown=args.markdown,
        extra_cols=args.extra_cols,
    )

    if not args.markdown:
        print(create_latex_table(model_names=args.models, extra_cols=args.extra_cols))

    for row in rows:
        print(row)

    if not args.markdown:
        print(create_latex_footer())


if __name__ == "__main__":
    main()
