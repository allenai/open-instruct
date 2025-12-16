import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from matplotlib.patches import Patch
from transformers import AutoTokenizer

import open_instruct.utils as open_instruct_utils

"""
Example commands
SFT plot in paper:
python scripts/data/get_token_distribution.py --dataset allenai/tulu-v.3.8-mix-preview-noncommercial --log_x --not_log_y --automatic_binning --hide_legend
python scripts/data/get_token_distribution.py --dataset allenai/tulu-v1-sft-mixture --log_x --not_log_y --automatic_binning --hide_legend --dont_split_histogram --set_max_y 60000
python scripts/data/get_token_distribution.py --dataset allenai/tulu-v2-sft-mixture --log_x --not_log_y --automatic_binning --hide_legend --dont_split_histogram --set_max_y 60000
python scripts/data/get_token_distribution.py --dataset teknium/OpenHermes-2.5 --column_name conversations --log_x --not_log_y --automatic_binning --hide_legend --dont_split_histogram --set_max_y 60000
"""


def plot_token_length_histogram(
    dataset_name,
    column_name="messages",
    tokenizer_name="baseten/Meta-Llama-3-tokenizer",
    num_proc=16,
    automatic_binning=False,
    log_x=False,
    not_log_y=False,
    hide_legend=False,
    plot_num_turns=False,
    dont_split_histogram=False,
    set_max_y=0,
    min_category_count=0,
):
    DATASET_NAME_MAPPING = {
        # EXTRA DATASETS
        # "NuminaMath-TIR": "NuminaMath-TIR",
        # "aya_dataset_converted": "Aya",
        # "flan_v2": "FLAN v2",
        # "open_math_2_gsm8k_converted": "OpenMathInstruct2",
        # "processed-wildjailbreak": "WildJailbreak",
        # "synthetic-finalresp-wildguarmixtrain": "WildGuard",
        # "table_gpt_converted": "TableGPT",
        # "wildchat_gpt4_converted": "WildChat GPT4",
        # "oasst1": "OASST1",
        # "open_orca": "OpenOrca",
        # "personahub_math_interm_algebra_50000": "Tulu 3 Persona MATH - Algebra",
        # "personahub_grade_math_v1_49980": "Tulu 3 Persona Grade School Math",
        # "sciriff_converted": "SciRIFF",
        # "Hardcoded": "Hardcoded",
        # "code_alpaca": "CodeAlpaca",
        # "cot": "FLAN CoT",
        # "sharegpt": "ShareGPT",
        # "dolly": "Dolly",
        # "gpt4_alpaca": "Alpaca (GPT4)",
        # "lima": "LIMA",
        # "science": "Science",
        # "wizardlm": "WizardLM",
        # "wizardlm_alpaca": "WizardLM (Alpaca)",
        # PERSONA DATASETS
        "tulu_v3.9_personahub_math_interm_algebra_20k": "Tulu 3 Persona MATH - Algebra",
        "personahub_math_v5_regen_149960": "Tulu 3 Persona MATH",
        "tulu-3-sft-personas-math-grade": "Tulu 3 Persona Grade School Math",
        "personahub_code_v2_34999": "Tulu 3 Persona Code",
        # GENERAL DATASETS
        "flan_v2_converted": "FLAN v2",
        "no_robots_converted": "No Robots",
        "oasst1_converted": "OASST1",
        "tulu_v3.9_wildchat_100k": "WildChat",
        # MATH DATASETS
        "numinamath_tir_math_decontaminated": "NuminaMath-TIR",
        "tulu_v3.9_open_math_2_gsm8k_50k": "OpenMathInstruct2",
        # SAFETY
        "coconot_converted": "CoCoNot",
        "tulu_v3.9_wildjailbreak_decontaminated_50k": "WildJailbreak",
        "tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k": "WildGuard",
        # OTHER
        "evol_codealpaca_heval_decontaminated": "Evol CodeAlpaca",
        "tulu_hard_coded_repeated_10": "Hardcoded",
        "tulu_v3.9_aya_100k": "Aya",
        "tulu_v3.9_sciriff_10k": "SciRIFF",
        "tulu_v3.9_table_gpt_5k": "TableGPT",
        "personahub_ifdata_manual_seed_v3_29980": "Tulu 3 Persona IF",
    }

    DATASET_NAME_MAPPING_PREF = {
        "helpsteer2-uf-pipeline-regen": "Tulu 3 HelpSteer2 Regen",
        "ultrafeedback_binarized_cleaned_train": "UltraFeedback",
        # Custom conversion of daring anteater synthetic data into preferences
        "tulu3.4-sft-replica-50k-gpt4-prefs-on-policy": "Tulu 3 UltraFeedback+",
        # Modifications of WildChat data to preferences with
        "personahub_if_pref_data_manualseed_v2_19890": "Tulu 3 Persona IF Preferences",
        # Custom IF Eval data with Llama 3.1 405B for chosen and Tulu 2 as rejected
        "Llama-3.1-if_taxonomy_tulu": "Tulu 3 IFEval",
    }

    # swap dataset mapping if preferences
    if column_name in ["chosen", "rejected"]:
        DATASET_NAME_MAPPING = DATASET_NAME_MAPPING_PREF

    print("Running analytics...")
    # Load the dataset
    dataset = load_dataset(dataset_name, num_proc=open_instruct_utils.max_num_processes())

    # Convert "from"/"value" format to "role"/"content" if needed
    def convert_to_messages(sample, column_name=column_name):
        new_messages = []
        for message in sample[column_name]:
            content = message["value"]
            role = message["from"]
            new_messages.append({"role": role, "content": content})
        sample[column_name] = new_messages
        return sample

    if "from" in dataset["train"][0][column_name][0].keys():
        dataset = dataset.map(convert_to_messages, num_proc=num_proc)

    # If allenai/tulu in name, turn on categories
    TRACK_CATEGORIES = "allenai/tulu" in dataset_name or "source" in dataset["train"].column_names
    if dont_split_histogram:
        TRACK_CATEGORIES = False

    if plot_num_turns:
        # Count turns (number of messages divided by 2)
        def process_sample(example):
            example["metric_value"] = len(example[column_name]) // 2
            return example

        xlabel = "Number of turns in conversation"
        metric_name = "turns"
    else:
        # Process tokens
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        class MapColumn:
            def __init__(self, column, tokenizer):
                self.column = column
                self.tokenizer = tokenizer

            def apply_template(self, example):
                tokenized = self.tokenizer.apply_chat_template(example[self.column])
                example["metric_value"] = len(tokenized)
                return example

        process_sample = MapColumn(column_name, tokenizer).apply_template
        xlabel = "Number of tokens in sample"
        metric_name = "tokens"

    # Process the dataset
    print(f"Processing {'turns' if plot_num_turns else 'tokens'} in conversations...")
    processed_dataset = dataset["train"].map(
        process_sample, num_proc=num_proc, desc=f"Processing {'turns' if plot_num_turns else 'tokens'}"
    )

    # Extract metric values and categories
    metric_values = processed_dataset["metric_value"]

    if TRACK_CATEGORIES:
        # if source in dataset, take those as categories
        if "source" in dataset["train"].column_names:
            categories = processed_dataset["source"]
        else:
            categories = processed_dataset["id"]
            categories = [category.rsplit("_", 1)[0] for category in categories]

            repeated_ids = ["sharegpt"]
            for repeated_id in repeated_ids:
                categories = [repeated_id if repeated_id in category else category for category in categories]

        if any("/" in category for category in categories):
            categories = [category.split("/")[1] if "/" in category else "Other" for category in categories]

        unique_categories = np.unique(categories)
        # if min_category_count is set, combine categories with less than min_category_count samples to "Other"
        if min_category_count > 0:
            category_counts = {
                category: sum([1 for c in categories if c == category]) for category in unique_categories
            }
            categories = [
                category if category_counts[category] >= min_category_count else "Other" for category in categories
            ]
            unique_categories = np.unique(categories)
            # add other to category counts
            DATASET_NAME_MAPPING["Other"] = "Other"
            category_counts["Other"] = sum([1 for c in categories if c == "Other"])
            # sort the unique categories by count
            unique_categories = sorted(unique_categories, key=lambda x: category_counts[x], reverse=True)
            # sort DATA_NAME_MAPPING by count
            DATASET_NAME_MAPPING = {category: DATASET_NAME_MAPPING[category] for category in unique_categories}

        else:
            # alphabetical the unique categories
            unique_categories = np.sort(unique_categories)

        # make colors for all of DATASET_NAME_MAPPING, then assign colors to unique_categories, only for unique values in dict
        # colors_full = plt.cm.tab20(np.linspace(0, 1, len(DATASET_NAME_MAPPING)))
        colors_full = plt.cm.tab20c(np.linspace(0, 1, len(DATASET_NAME_MAPPING)))
        category_colors = {category: colors_full[i] for i, category in enumerate(DATASET_NAME_MAPPING.keys())}

        # if other is a category, make it white
        if "Other" in unique_categories:
            category_colors["Other"] = (1.0, 1.0, 1.0, 1.0)

        category_metric_values = defaultdict(list)
        for value, category in zip(metric_values, categories):
            category_metric_values[category].append(value)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.rcParams.update({"font.size": 38, "font.family": "DeJavu Serif", "font.serif": ["Times New Roman"]})

    print("Plotting histogram...")
    # Prepare histogram bins
    if automatic_binning:
        max_value = max(metric_values)
        if plot_num_turns:
            max_value = min(max_value + 1, 10)  # Cap at 10 turns for readability
            bins = np.arange(0, max_value + 2)  # +2 to include the last turn
        else:
            max_value = min(max_value, 16384 + 1)
            if log_x:
                bins = np.logspace(np.log10(min(metric_values)), np.log10(max_value), 25)
            else:
                bins = np.linspace(0, max_value, 25)
    else:
        if plot_num_turns:
            bins = np.arange(0, max(metric_values) + 2)
        else:
            bins = [0, 2000, 4000, 6000, 8000, 10000, 12000]

    if TRACK_CATEGORIES:
        bin_counts = {
            category: np.histogram(category_metric_values[category], bins=bins)[0] for category in category_colors
        }
        bottom_counts = np.zeros(len(bins) - 1)

        for category, color in category_colors.items():
            counts = bin_counts[category]
            ax.bar(
                bins[:-1],
                counts,
                width=np.diff(bins),
                color=color,
                alpha=1.0,
                edgecolor="black",
                label=category,
                align="edge",
                bottom=bottom_counts,
            )
            bottom_counts += counts
    else:
        n, bins, patches = ax.hist(metric_values, bins=bins, color="grey", edgecolor="black")

    # Set axis properties
    if not automatic_binning and not plot_num_turns:
        ax.set_xlim(0, 12000)
        ax.set_xticks(bins)
        ax.set_xticks(np.array(bins[1:]) - 1000)
        ax.set_xticklabels([f"{int(center)}" for center in bins[1:]])
    else:
        if log_x and not plot_num_turns:
            ax.set_xscale("log")
            ticks = [16, 128, 512, 2048, 8192, 16384 * 2]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{tick}" for tick in ticks])
        else:
            if plot_num_turns:
                ax.set_xticks(bins[::2])  # Show every other tick for turns
                # set limit for turns to be 0 to 10 always
                ax.set_xlim(0, 10)
            else:
                max_power = int(np.ceil(np.log2(max_value)))
                major_ticks = [2**i for i in range(max_power)]
                block_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
                if max_value > 4096:
                    block_list += [512]
                major_ticks = [0] + [i for i in major_ticks if i not in block_list]
                ax.set_xticks(major_ticks)

    if log_x and not plot_num_turns:
        xlabel += " (log scale)"
    ax.set_xlabel(xlabel)

    # Set y-axis properties
    if not_log_y:
        # Linear scale
        # ax.set_ylabel('Count')

        # Create 4 major ticks
        if TRACK_CATEGORIES:
            max_count = max(bottom_counts)
        else:
            max_count = max(n)
        # Round up to the nearest 10000
        max_count = 10000 * (max_count // 10000 + 1)

        if set_max_y > 0:
            max_count = set_max_y
        major_ticks = np.linspace(0, max_count, 5)
        ax.set_yticks(major_ticks)
    else:
        # Log scale
        ax.set_yscale("log")
        max_count = max(metric_values)
        max_power = int(np.ceil(np.log10(max_count)))
        major_ticks = [10**i for i in range(max_power + 1)]
        minor_ticks = []
        for power in range(max_power + 1):
            for factor in range(2, 10):
                minor_ticks.append(factor * 10**power)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_yticklabels([f"$10^{int(np.log10(tick))}$" for tick in major_ticks])
        ax.tick_params(axis="y", which="minor", length=4)
        ax.tick_params(axis="y", which="major", length=8)
        ax.set_ylabel("Count (log scale)")

    # Add legend
    if TRACK_CATEGORIES and not hide_legend:
        legend_handles = [Patch(color=color, label=category) for category, color in category_colors.items()]
        ax.legend(handles=legend_handles, fontsize=6)
    else:
        plt.margins(x=0.01)
        plt.tight_layout(pad=0.1)

        # print the colors for the datasets for use in custome legend
        if TRACK_CATEGORIES:
            for category, color in category_colors.items():
                if category in unique_categories:
                    print(f"{category}: {color}")

            print("\n% Color blocks with labels")
            for category, color in category_colors.items():
                if category in unique_categories:
                    # Convert RGBA tuple to RGB for LaTeX (dropping alpha value)
                    r, g, b, _ = color
                    r_255 = round(r * 255)
                    g_255 = round(g * 255)
                    b_255 = round(b * 255)
                    # Print directly with \colorbox using rgb color model
                    print(f"\\cblock{{{r_255}}}{{{g_255}}}{{{b_255}}}~{DATASET_NAME_MAPPING[category]}, \\quad")

    # Print statistics
    print(f"Total samples: {len(metric_values)}")
    print(f"Mean {metric_name}: {np.mean(metric_values):.2f}")
    print(f"Median {metric_name}: {np.median(metric_values):.2f}")
    print(f"Max {metric_name}: {max(metric_values)}")
    print(f"Min {metric_name}: {min(metric_values)}")

    # Save plot
    dataset_name = dataset_name.split("/")[-1]
    metric_suffix = "turns" if plot_num_turns else "tokens"
    plt.savefig(f"output/{metric_suffix}_histogram_{dataset_name}_{column_name}.pdf")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(description="Plot token length or turns histogram from Hugging Face datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the Hugging Face dataset to load")
    parser.add_argument(
        "--column_name", type=str, default="messages", help="Column to extract text from (default: messages)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="baseten/Meta-Llama-3-tokenizer",
        help="Tokenizer to use (default: Meta-Llama-3)",
    )
    parser.add_argument(
        "--num_proc", type=int, default=16, help="Number of processes for parallel processing (default: 16)"
    )
    parser.add_argument("--automatic_binning", action="store_true", help="Use automatic binning for the histogram")
    parser.add_argument("--log_x", action="store_true", help="Use log scale for x-axis")
    parser.add_argument("--not_log_y", action="store_true", help="Use log scale for y-axis")
    parser.add_argument("--hide_legend", action="store_true", help="Hide the legend")
    parser.add_argument("--plot_num_turns", action="store_true", help="Plot number of turns instead of token length")
    parser.add_argument("--dont_split_histogram", action="store_true", help="Unicolor histogram")
    parser.add_argument("--set_max_y", type=int, default=0, help="Set max y value")
    parser.add_argument(
        "--min_category_count", type=int, default=0, help="Minimum number of samples for a category to be included"
    )
    args = parser.parse_args()

    fig, ax = plot_token_length_histogram(
        args.dataset,
        args.column_name,
        args.tokenizer,
        args.num_proc,
        args.automatic_binning,
        args.log_x,
        args.not_log_y,
        args.hide_legend,
        args.plot_num_turns,
        args.dont_split_histogram,
        args.set_max_y,
        args.min_category_count,
    )


if __name__ == "__main__":
    main()
