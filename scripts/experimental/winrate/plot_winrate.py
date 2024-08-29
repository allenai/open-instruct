# Re-import necessary modules and define the previous variables

import matplotlib.pyplot as plt
import numpy as np

# Data
results = {
    "SFT just on H4/no_robots": {8e9: 0.31},
    "llama-3.1-tulu-2-dpo-8": {8e9: 0.504},
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {8e9: 0.566},
}
first_key = list(results.keys())[0]

# Plotting variables
colors = {
    "SFT just on H4/no_robots": "#FFB898", # light orange
    "llama-3.1-tulu-2-dpo-8": "#68D39F", # light green
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "#8EC2FF", # light blue
}
markers = {
    "SFT just on H4/no_robots": "o",         # Circle
    "llama-3.1-tulu-2-dpo-8": "X",  # Stars
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "s", # Square
}

# Map to new legend labels
legend_labels = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-3.1-tulu-2-dpo-8": "llama-3.1-tulu-2-dpo-8",
    "SFT just on H4/no_robots": "SFT just on H4/no_robots"
}

# Double the size of the percentage annotations
fig, ax = plt.subplots(figsize=(11, 7))

# Plot each method
for method, data in results.items():
    sizes = list(data.keys())
    values = list(data.values())
    ax.plot(sizes, values, label=legend_labels[method], color=colors[method], marker=markers[method], markersize=12, linestyle='--' if method != 'Online DPO' else '-', linewidth=3)

# Add percentage improvements relative to SFT for each parameter scale, adjusted just above the line with doubled font size
sft_values = np.array(list(results[first_key].values()))
for method, data in results.items():
    if method != first_key:
        sizes = list(data.keys())
        values = np.array(list(data.values()))
        relative_gains = (values - sft_values) / sft_values * 100
        for i, size in enumerate(sizes):
            ax.text(size, values[i] + 0.02, f"+{int(relative_gains[i])}%", fontsize=24, color=colors[method], va='bottom')

# Style similar to the provided plot
ax.set_xscale("log")
ax.set_xticks([8e9])
ax.set_xticklabels(["8B"], fontsize=20)
ax.set_xlabel("Model size (parameters)", fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylabel("Win rate vs. human completions", fontsize=20)
ax.set_ylim(0, 0.7)

# Ensure only the 3 specified ticks are shown, removing any other automatic ticks
ax.get_xaxis().set_major_locator(plt.FixedLocator([8e9]))

# Remove minor ticks if any
ax.minorticks_off()

# Adjust legend position and labels
ax.legend(title="", loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, fontsize=20)

# Grid and spines similar to the original plot
ax.grid(True, linestyle=':', color='grey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("winrate_plot.png")