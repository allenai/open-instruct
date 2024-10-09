import argparse
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from matplotlib.patches import Patch
from collections import defaultdict

def plot_token_length_histogram(dataset_name, 
                                column_name='messages', 
                                tokenizer_name="baseten/Meta-Llama-3-tokenizer", 
                                num_proc=16, automatic_binning=False, 
                                log_x=False,
                                not_log_y=False,
                                hide_legend=False):
    
    print("Running analytics...")
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # If allenai/tulu in name, turn on categories
    TRACK_CATEGORIES = "allenai/tulu" in dataset_name
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    class MapColumn:
        def __init__(self, column, tokenizer):
            self.column = column
            self.tokenizer = tokenizer

        def apply_template(self, example):
            tokenized = self.tokenizer.apply_chat_template(example[self.column])
            example['token_length'] = len(tokenized)
            return example
    
    mapper = MapColumn(column_name, tokenizer)
    
    # Process the dataset in parallel using dataset.map()
    print("Tokenizing conversations...")
    processed_dataset = dataset['train'].map(
        mapper.apply_template,
        num_proc=num_proc,
        desc="Tokenizing conversations"
    )

    # Extract token lengths and categories
    token_lengths = processed_dataset['token_length']
    
    # Strip category column from last _ (take what is to left, e.g. AI-MO/NuminaMath-TIR_72428 -> AI-MO/NuminaMath-TIR)
    if TRACK_CATEGORIES:
        categories = processed_dataset['id']
        # split from last _, note some may have multiple _
        categories = [category.rsplit('_', 1)[0] for category in categories]

        # if "/" is not in category, replace it with "Other"
        categories = [category if "/" in category else "/Other" for category in categories]

        # Get unique categories
        unique_categories = np.unique(categories)

        # Create dict of categories mapping to colors
        category_colors = {category: plt.cm.tab20(i) for i, category in enumerate(unique_categories)}

        # Group token lengths by category
        category_token_lengths = defaultdict(list)
        for length, category in zip(token_lengths, categories):
            category_token_lengths[category].append(length)
    
    # Create figure and axis objects with specified size
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set font properties
    plt.rcParams.update({
        'font.size': 38,
        'font.family': 'DeJavu Serif',
        'font.serif': ['Times New Roman']
    })

    print("Plotting histogram...")
    # Prepare histogram bins
    if automatic_binning:
        max_length = min(max(token_lengths), 16384 + 1)
        if log_x:
            bins = np.logspace(np.log10(min(token_lengths)), np.log10(max_length), 25)
        else:
            bins = np.linspace(0, max_length, 25)
    else:
        bins = [0, 2000, 4000, 6000, 8000, 10000, 12000]
    
    if TRACK_CATEGORIES:
        # Calculate stacked bar heights for each category
        # bin_counts = {category: np.histogram([token_lengths[i] for i in range(len(token_lengths)) if categories[i] == category], bins=bins)[0] for category in category_colors}
        bin_counts = {category: np.histogram(category_token_lengths[category], bins=bins)[0] for category in category_colors}
        bottom_counts = np.zeros(len(bins) - 1)

        # Plot stacked bars
        for category, color in category_colors.items():
            counts = bin_counts[category]
            ax.bar(bins[:-1], counts, width=np.diff(bins), color=color, alpha=1.0, edgecolor='black', label=category, align='edge', bottom=bottom_counts)
            bottom_counts += counts

    else: 
        n, bins, patches = ax.hist(token_lengths, bins=bins, color='blue', edgecolor='black')
    
    # Set x limit and ticks for custom bins
    if not automatic_binning:
        ax.set_xlim(0, 12000)
        ax.set_xticks(bins)
        ax.set_xticks(np.array(bins[1:]) - 1000)
        ax.set_xticklabels([f'{int(center)}' for center in bins[1:]])
    else:
        if log_x:
            pass
        else:
            # set number of ticks to all the powers of 2 that fit in max(token_lengths)
            max_power = int(np.ceil(np.log2(max_length)))
            major_ticks = [2**i for i in range(max_power)]
            # Remove 1, 2, 4, 8, 16, 32, 64, from list, add 0
            block_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            if max_length > 4096:
                block_list += [512]
            major_ticks = [0] + [i for i in major_ticks if i not in block_list]
            ax.set_xticks(major_ticks)
    
    if log_x:
        ax.set_xscale('log')
        ax.set_xlabel('Number of tokens in sample (log scale)')
    else:
        ax.set_xlabel('Number of tokens in sample')
    
    # Set y-axis ticks
    max_count = max(token_lengths)
    max_power = int(np.ceil(np.log10(max_count)))
    major_ticks = [10**i for i in range(max_power + 1)]
    minor_ticks = []
    for power in range(max_power + 1):
        for factor in range(2, 10):
            minor_ticks.append(factor * 10**power)

    if not not_log_y:
        ax.set_yscale('log')
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_yticklabels([f'$10^{int(np.log10(tick))}$' for tick in major_ticks])
    ax.tick_params(axis='y', which='minor', length=4)
    ax.tick_params(axis='y', which='major', length=8)
    
    # Labels
    ax.set_ylabel('Count (log scale)')
    
    # Add legend
    if TRACK_CATEGORIES:
        if not hide_legend:
            legend_handles = [Patch(color=color, label=category.split("/")[1]) for category, color in category_colors.items()]
            ax.legend(handles=legend_handles, fontsize=6)
    else:
        plt.margins(x=0.01)
        plt.tight_layout(pad=0.1)
    
    # Print statistics
    print(f"Total samples: {len(token_lengths)}")
    print(f"Mean token length: {np.mean(token_lengths):.2f}")
    print(f"Median token length: {np.median(token_lengths):.2f}")
    print(f"Max token length: {max(token_lengths)}")
    print(f"Min token length: {min(token_lengths)}")
    
    # Adjust layout and save
    dataset_name = dataset_name.split('/')[-1]    
    plt.savefig(f"output/token_length_histogram_{dataset_name}_{column_name}.pdf")
    
    return fig, ax

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Plot token length histogram from Hugging Face datasets")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the Hugging Face dataset to load")
    parser.add_argument('--column_name', type=str, default='messages', help="Column to extract text from (default: messages)")
    parser.add_argument('--tokenizer', type=str, default='baseten/Meta-Llama-3-tokenizer', help="Tokenizer to use (default: Meta-Llama-3)")
    parser.add_argument('--num_proc', type=int, default=16, help="Number of processes to use for parallel processing (default: 16)")
    parser.add_argument('--automatic_binning', action='store_true', help="Use automatic binning for the histogram")
    parser.add_argument('--log_x', action='store_true', help="Use log scale for x-axis")
    parser.add_argument('--not_log_y', action='store_true', help="Use log scale for y-axis")
    parser.add_argument('--hide_legend', action='store_true', help="Hide the legend")

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function with provided arguments
    fig, ax = plot_token_length_histogram(args.dataset, args.column_name, args.tokenizer, args.num_proc, args.automatic_binning, args.log_x, args.not_log_y, args.hide_legend)

if __name__ == "__main__":
    main()