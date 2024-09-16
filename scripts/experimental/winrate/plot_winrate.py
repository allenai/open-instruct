# Re-import necessary modules and define the previous variables

from dataclasses import dataclass
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Data:
    data: Dict[float, List[float]] # Dict of model sizes to [winrate, response len]
    color: str
    marker: str

# Data
results = {
    "SFT w/ H4/no_robots": Data({8e9: [0.31, 121.41]}, "#FFB898", "o"),
    "SFT + Online DPO w/ H4/no_robots": Data({8e9: [0.47, 153.838]}, "#eb4034", "v"),
    "SFT + PPO w/ H4/no_robots": Data({8e9: [0.446, 146.928]}, "#eb4034", "v"),
    "SFT + Offline DPO w/ H4/no_robots": Data({8e9: [0.424, 144.382]}, "#00FFFF", "2"),
    "SFT + Offline DPO w/ H4/no_robots two epochs": Data({8e9: [0.498, 140.248]}, "#9F2B68", "3"),
    "SFT + Offline DPO w/ H4/no_robots three epochs": Data({8e9: [0.536, 136.864]}, "#9F2B68", "3"),
    "llama-3.1-tulu-2-dpo-8": Data({8e9: [0.504, 172.248]}, "#68D39F", "X"),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": Data({8e9: [0.566, 151.506]}, "#8EC2FF", "s"),
}
first_key = list(results.keys())[0]

# Function to create and save the original plot
def create_original_plot(filename):
    fig, ax = plt.subplots(figsize=(11, 7))

    # Plot each method
    for method, item in results.items():
        sizes = list(item.data.keys())
        values = [value[0] for value in item.data.values()]
        ax.plot(sizes, values, label=method, color=item.color, marker=item.marker, markersize=12, linestyle='--' if method != 'Online DPO' else '-', linewidth=3)

    # Add percentage improvements relative to SFT for each parameter scale
    sft_values = np.array([value[0] for value in results[first_key].data.values()])
    for method, item in results.items():
        if method != first_key:
            sizes = list(item.data.keys())
            values = [value[0] for value in item.data.values()]
            relative_gains = (values - sft_values) / sft_values * 100
            for i, size in enumerate(sizes):
                ax.text(size, values[i] + 0.02, f"+{int(relative_gains[i])}%", fontsize=24, color=item.color, va='bottom')

    # Style the plot
    ax.set_xscale("log")
    ax.set_xticks([8e9])
    plt.axhline(y=0.5, color='black', linestyle='-.', label='reference response')
    ax.set_xticklabels(["8B"], fontsize=20)
    ax.set_xlabel("Model size (parameters)", fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylabel("Win rate vs. human completions", fontsize=20)
    ax.set_ylim(0, 0.7)

    # Ensure only the specified ticks are shown
    ax.get_xaxis().set_major_locator(plt.FixedLocator([8e9]))

    # Remove minor ticks
    ax.minorticks_off()

    # Adjust legend
    # ax.legend(title="", loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, fontsize=20)
    fig.legend(loc="upper center", ncol=1, bbox_transform=fig.transFigure, fontsize=10)

    # Grid and spines
    # ax.grid(True, linestyle=':', color='grey')

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

# Function to create and save the new plot (winrate vs model length)
def create_winrate_vs_length_plot(filename):
    fig, ax = plt.subplots(figsize=(11, 7))

    # Extract data for the plot
    for method, item in results.items():
        for size, values in item.data.items():
            winrate, length = values
            ax.scatter(length, winrate, label=method, color=item.color, marker=item.marker, s=150)

    # Style the plot
    ax.set_xlabel("Model response length", fontsize=20)
    plt.axhline(y=0.5, color='black', linestyle='-.', label='reference response')
    plt.axvline(x=179.726, color='black', linestyle='-', label='reference response len')
    ax.set_ylabel("Win rate vs. human completions", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Set reasonable axis limits
    ax.set_xlim(100, 200)
    ax.set_ylim(0.2, 0.7)

    # Adjust legend
    # ax.legend(title="", loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, fontsize=20)
    fig.legend(loc="upper center", ncol=1, bbox_transform=fig.transFigure, fontsize=10)

    # Grid and spines
    # ax.grid(True, linestyle=':', color='grey')

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

# Create the original plot (win rate vs model size)
create_original_plot("winrate_plot.png")

# Create the new plot (win rate vs model length)
create_winrate_vs_length_plot("winrate_vs_length_plot.png")