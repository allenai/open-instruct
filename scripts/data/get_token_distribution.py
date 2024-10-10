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
                                num_proc=16, 
                                automatic_binning=False, 
                                log_x=False,
                                not_log_y=False,
                                hide_legend=False,
                                plot_num_turns=False):
    
    print("Running analytics...")
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Convert "from"/"value" format to "role"/"content" if needed
    def convert_to_messages(sample, column_name=column_name):
        new_messages = []
        for message in sample[column_name]:
            content = message["value"]
            role = message["from"]
            new_messages.append({"role": role, "content": content})
        sample[column_name] = new_messages
        return sample

    if "from" in dataset['train'][0][column_name][0].keys():
        dataset = dataset.map(convert_to_messages, num_proc=num_proc)

    # If allenai/tulu in name, turn on categories
    TRACK_CATEGORIES = "allenai/tulu" in dataset_name
    
    if plot_num_turns:
        # Count turns (number of messages divided by 2)
        def process_sample(example):
            example['metric_value'] = len(example[column_name]) // 2
            return example
        
        xlabel = 'Number of turns in conversation'
        metric_name = 'turns'
    else:
        # Process tokens
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        class MapColumn:
            def __init__(self, column, tokenizer):
                self.column = column
                self.tokenizer = tokenizer

            def apply_template(self, example):
                tokenized = self.tokenizer.apply_chat_template(example[self.column])
                example['metric_value'] = len(tokenized)
                return example
        
        process_sample = MapColumn(column_name, tokenizer).apply_template
        xlabel = 'Number of tokens in sample'
        metric_name = 'tokens'
    
    # Process the dataset
    print(f"Processing {'turns' if plot_num_turns else 'tokens'} in conversations...")
    processed_dataset = dataset['train'].map(
        process_sample,
        num_proc=num_proc,
        desc=f"Processing {'turns' if plot_num_turns else 'tokens'}"
    )

    # Extract metric values and categories
    metric_values = processed_dataset['metric_value']
    
    if TRACK_CATEGORIES:
        categories = processed_dataset['id']
        categories = [category.rsplit('_', 1)[0] for category in categories]

        repeated_ids = ["sharegpt"]
        for repeated_id in repeated_ids:
            categories = [repeated_id if repeated_id in category else category for category in categories]
        
        if any("/" in category for category in categories):
            categories = [category.split("/")[1] if "/" in category else "Other" for category in categories]

        unique_categories = np.unique(categories)
        # alphabetical the unique categories
        unique_categories = np.sort(unique_categories)
        category_colors = {category: plt.cm.tab20(i) for i, category in enumerate(unique_categories)}

        category_metric_values = defaultdict(list)
        for value, category in zip(metric_values, categories):
            category_metric_values[category].append(value)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.rcParams.update({
        'font.size': 38,
        'font.family': 'DeJavu Serif',
        'font.serif': ['Times New Roman']
    })

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
        bin_counts = {category: np.histogram(category_metric_values[category], bins=bins)[0] 
                     for category in category_colors}
        bottom_counts = np.zeros(len(bins) - 1)

        for category, color in category_colors.items():
            counts = bin_counts[category]
            ax.bar(bins[:-1], counts, width=np.diff(bins), color=color, alpha=1.0, 
                  edgecolor='black', label=category, align='edge', bottom=bottom_counts)
            bottom_counts += counts
    else:
        n, bins, patches = ax.hist(metric_values, bins=bins, color='blue', edgecolor='black')
    
    # Set axis properties
    if not automatic_binning and not plot_num_turns:
        ax.set_xlim(0, 12000)
        ax.set_xticks(bins)
        ax.set_xticks(np.array(bins[1:]) - 1000)
        ax.set_xticklabels([f'{int(center)}' for center in bins[1:]])
    else:
        if log_x and not plot_num_turns:
            ax.set_xscale('log')
            ticks = [16, 128, 512, 2048, 8192, 16384*2]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f'{tick}' for tick in ticks])
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
        xlabel += ' (log scale)'
    ax.set_xlabel(xlabel)
    
    # Set y-axis properties
    if not_log_y:
        # Linear scale
        ax.set_ylabel('Count')
    else:
        # Log scale
        ax.set_yscale('log')
        max_count = max(metric_values)
        max_power = int(np.ceil(np.log10(max_count)))
        major_ticks = [10**i for i in range(max_power + 1)]
        minor_ticks = []
        for power in range(max_power + 1):
            for factor in range(2, 10):
                minor_ticks.append(factor * 10**power)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_yticklabels([f'$10^{int(np.log10(tick))}$' for tick in major_ticks])
        ax.tick_params(axis='y', which='minor', length=4)
        ax.tick_params(axis='y', which='major', length=8)
        ax.set_ylabel('Count (log scale)')
        
    # Add legend
    if TRACK_CATEGORIES and not hide_legend:
        legend_handles = [Patch(color=color, label=category) for category, color in category_colors.items()]
        ax.legend(handles=legend_handles, fontsize=6)
    else:
        plt.margins(x=0.01)
        plt.tight_layout(pad=0.1)
    
    # Print statistics
    print(f"Total samples: {len(metric_values)}")
    print(f"Mean {metric_name}: {np.mean(metric_values):.2f}")
    print(f"Median {metric_name}: {np.median(metric_values):.2f}")
    print(f"Max {metric_name}: {max(metric_values)}")
    print(f"Min {metric_name}: {min(metric_values)}")
    
    # Save plot
    dataset_name = dataset_name.split('/')[-1]
    metric_suffix = 'turns' if plot_num_turns else 'tokens'
    plt.savefig(f"output/{metric_suffix}_histogram_{dataset_name}_{column_name}.pdf")
    
    return fig, ax

def main():
    parser = argparse.ArgumentParser(description="Plot token length or turns histogram from Hugging Face datasets")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the Hugging Face dataset to load")
    parser.add_argument('--column_name', type=str, default='messages', help="Column to extract text from (default: messages)")
    parser.add_argument('--tokenizer', type=str, default='baseten/Meta-Llama-3-tokenizer', help="Tokenizer to use (default: Meta-Llama-3)")
    parser.add_argument('--num_proc', type=int, default=16, help="Number of processes for parallel processing (default: 16)")
    parser.add_argument('--automatic_binning', action='store_true', help="Use automatic binning for the histogram")
    parser.add_argument('--log_x', action='store_true', help="Use log scale for x-axis")
    parser.add_argument('--not_log_y', action='store_true', help="Use log scale for y-axis")
    parser.add_argument('--hide_legend', action='store_true', help="Hide the legend")
    parser.add_argument('--plot_num_turns', action='store_true', help="Plot number of turns instead of token length")

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
        args.plot_num_turns
    )

if __name__ == "__main__":
    main()