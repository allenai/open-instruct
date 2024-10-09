import argparse
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def plot_token_length_histogram(dataset_name, column_name='messages', tokenizer_name="baseten/Meta-Llama-3-tokenizer", num_proc=16, automatic_binning=False):
    # Load the dataset
    dataset = load_dataset(dataset_name)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # function that calls apply_chat_template on the specified column
    # def apply_chat_template_on_column(example, column, tokenizer):
    #     return tokenizer.apply_chat_template(example[column])

    class MapColumn:
        def __init__(self, column, tokenizer):
            self.column = column
            self.tokenizer = tokenizer

        def apply_template(self, example):
            tokenized = self.tokenizer.apply_chat_template(example[self.column])
            example['token_length'] = len(tokenized)
            del example[self.column]
            return example
    
    mapper = MapColumn(column_name, tokenizer)
    
    # Process the dataset in parallel using dataset.map()
    processed_dataset = dataset['train'].map(
        mapper.apply_template,
        num_proc=num_proc,
        desc="Tokenizing conversations"
    )

    # Extract token lengths
    token_lengths = processed_dataset['token_length']
    
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
    
    # Plot histogram
    if automatic_binning:
        n, bins, patches = ax.hist(token_lengths, bins=25, color='blue', 
                                 edgecolor='black')
    else:
        bins = [0, 2000, 4000, 6000, 8000, 10000, 12000]
        # bins = [0, 1000, 3000, 5000, 7000, 9000, 11000, 13000]
        n, bins, patches = ax.hist(token_lengths, bins=bins, color='blue',
                                 alpha=0.7, edgecolor='black')
        
        # set x limit and ticks to this
        ax.set_xlim(0, 12000)
        ax.set_xticks(bins)

        # Set x-ticks at the center of each bin
        ax.set_xticks(np.array(bins[1:])-1000)
        ax.set_xticklabels([f'{int(center)}' for center in bins[1:]])
    
    # Set scale and labels
    ax.set_yscale('log')
    
    # Set major and minor ticks for y-axis
    max_count = max(n)
    max_power = int(np.ceil(np.log10(max_count)))
    
    # Major ticks at powers of 10
    major_ticks = [10**i for i in range(max_power + 1)]
    
    # Minor ticks at 2-9 times each power of 10
    minor_ticks = []
    for power in range(max_power + 1):
        for factor in range(2, 10):
            minor_ticks.append(factor * 10**power)
    
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    
    # Format major tick labels as 10^N
    ax.set_yticklabels([f'$10^{int(np.log10(tick))}$' for tick in major_ticks])
    
    # Make minor ticks visible but shorter
    ax.tick_params(axis='y', which='minor', length=4)
    ax.tick_params(axis='y', which='major', length=8)
    
    # Labels
    ax.set_xlabel('Number of tokens in sample')
    ax.set_ylabel('Count (log scale)')

    # Print statistics
    print(f"Total samples: {len(token_lengths)}")
    print(f"Mean token length: {np.mean(token_lengths):.2f}")
    print(f"Median token length: {np.median(token_lengths):.2f}")
    print(f"Max token length: {max(token_lengths)}")
    print(f"Min token length: {min(token_lengths)}")
    
    # Adjust layout and save
    # Adjust margins to be very tight
    plt.margins(x=0.01)  # 1% margin on x-axis
    
    # Adjust layout with specific padding
    plt.tight_layout(pad=0.1)  # Reduce padding around the plot
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

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function with provided arguments
    fig, ax = plot_token_length_histogram(args.dataset, args.column_name, args.tokenizer, args.num_proc, args.automatic_binning)

if __name__ == "__main__":
    main()