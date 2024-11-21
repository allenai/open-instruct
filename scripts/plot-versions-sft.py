import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create dictionary with all models
data = {
    # "L3.1-8B-v3.9-nc-fixed-2-pif_uf_hs_dpo___model__42__1730613882": {
    #     "Rank": 1, "Average": 62.42, "alpaca_eval": 28.6, "BBH": 68.9,
    #     "codex_humaneval": 85.1, "codex_humanevalplus": 81.4, "drop": 61.3,
    #     "GSM8K": 82.3, "IFEval": 78.4, "MATH": 41.2,
    #     "mmlu:cot::summarize": float('nan'), "MMLU": 63.1,
    #     "Safety": 76.5, "popqa": 29.1, "truthfulqa": 54.9
    # },
    # "fae_dpo_on_L3.1-8B-v3.9-nc-fixed-2_add_shp___model__42__1730847906": {
    #     "Rank": 2, "Average": 61.01, "alpaca_eval": 27.1, "BBH": 65.0,
    #     "codex_humaneval": 83.9, "codex_humanevalplus": 78.2, "drop": 58.0,
    #     "GSM8K": 82.7, "IFEval": 78.6, "MATH": 41.9,
    #     "mmlu:cot::summarize": float('nan'), "MMLU": 64.8,
    #     "Safety": 76.1, "popqa": 29.1, "truthfulqa": 48.5
    # },
    "Tülu v3.7": {
        "Rank": 3, "Average": 60.48, "alpaca_eval": 13.7, "BBH": 67.8,
        "codex_humaneval": 87.2, "codex_humanevalplus": 83.6, "drop": 60.6,
        "GSM8K": 75.1, "IFEval": 72.5, "MATH": 32.6,
        "mmlu:cot::summarize": 65.1, "MMLU": 63.8,
        "Safety": 94.7, "popqa": 29.4, "truthfulqa": 44.7
    },
    "Tülu v3.8": {
        "Rank": 4, "Average": 60.12, "alpaca_eval": 12.0, "BBH": 67.9,
        "codex_humaneval": 85.8, "codex_humanevalplus": 81.1, "drop": 60.4,
        "GSM8K": 77.2, "IFEval": 72.1, "MATH": 32.5,
        "mmlu:cot::summarize": 65.3, "MMLU": 63.2,
        "Safety": 93.5, "popqa": 29.3, "truthfulqa": 46.5
    },
    "Tülu v3.9": {
        "Rank": 5, "Average": 60.08, "alpaca_eval": 12.4, "BBH": 67.9,
        "codex_humaneval": 86.2, "codex_humanevalplus": 81.4, "drop": 61.3,
        "GSM8K": 76.2, "IFEval": 72.8, "MATH": 31.5,
        "mmlu:cot::summarize": float('nan'), "MMLU": 62.1,
        "Safety": 93.1, "popqa": 29.3, "truthfulqa": 46.8
    },
    "Tülu v3.4": {
        "Rank": 6, "Average": 56.79, "alpaca_eval": 11.4, "BBH": 65.3,
        "codex_humaneval": 86.2, "codex_humanevalplus": 78.3, "drop": 55.8,
        "GSM8K": 76.3, "IFEval": 52.9, "MATH": 25.5,
        "mmlu:cot::summarize": 62.0, "MMLU": 64.8,
        "Safety": 89.6, "popqa": 23.5, "truthfulqa": 51.9
    },
    "Tülu v3.1": {
        "Rank": 7, "Average": 55.46, "alpaca_eval": 10.5, "BBH": 64.6,
        "codex_humaneval": 83.8, "codex_humanevalplus": 80.8, "drop": 64.7,
        "GSM8K": 74.5, "IFEval": 52.5, "MATH": 19.5,
        "mmlu:cot::summarize": 63.7, "MMLU": 64.6,
        "Safety": 70.3, "popqa": 31.4, "truthfulqa": 48.3
    },
    "Tülu v3.0": {
        "Rank": 8, "Average": 55.18, "alpaca_eval": 11.3, "BBH": 63.3,
        "codex_humaneval": 85.4, "codex_humanevalplus": 81.2, "drop": 62.5,
        "GSM8K": 72.9, "IFEval": 48.8, "MATH": 24.2,
        "mmlu:cot::summarize": 62.8, "MMLU": 65.1,
        "Safety": 68.0, "popqa": 31.2, "truthfulqa": 48.2
    },
    # "Tülu v3.2": {
    #     "Rank": 9, "Average": 55.05, "alpaca_eval": 12.1, "BBH": 66.5,
    #     "codex_humaneval": 84.2, "codex_humanevalplus": 79.7, "drop": 63.1,
    #     "GSM8K": 73.1, "IFEval": 49.7, "MATH": 19.0,
    #     "mmlu:cot::summarize": 63.7, "MMLU": 64.1,
    #     "Safety": 68.9, "popqa": 31.6, "truthfulqa": 49.2
    # },
    # "hf-llama-3-tulu-2-dpo-8b": {
    #     "Rank": 10, "Average": 49.49, "alpaca_eval": 14.1, "BBH": 57.3,
    #     "codex_humaneval": 69.2, "codex_humanevalplus": 67.7, "drop": 58.3,
    #     "GSM8K": 63.6, "IFEval": 48.8, "MATH": 13.5,
    #     "mmlu:cot::summarize": float('nan'), "MMLU": 61.8,
    #     "Safety": 57.9, "popqa": 24.6, "truthfulqa": 59.8
    # },
    "Tülu v2.0": {
        "Rank": 11, "Average": 48.30, "alpaca_eval": 8.9, "BBH": 57.1,
        "codex_humaneval": 66.9, "codex_humanevalplus": 63.1, "drop": 61.7,
        "GSM8K": 60.4, "IFEval": 42.3, "MATH": 14.0,
        "mmlu:cot::summarize": float('nan'), "MMLU": 61.8,
        "Safety": 70.7, "popqa": 23.3, "truthfulqa": 49.4
    }
}

# Replace this dictionary with your preferred hex colors for each model
colors = {
    "Tülu v2.0": "#F7C8E2",
    "Tülu v3.0": "#E7EEEE",  # RGB(231, 238, 238)
    "Tülu v3.1": "#CEDCDD",  # RGB(206, 220, 221)
    "Tülu v3.4": "#9FB9BB",
    "Tülu v3.7": "#88A8AB",
    # "Tülu v3.2": "#000080",
    "Tülu v3.8": "#6E979A",
    # "Tülu v3.7": "#588689",
    # "Tülu v3.8": "#3F7478",
    "Tülu v3.9": "#F0529C",
    "fae_dpo_on_L3.1-8B-v3.9-nc-fixed-2_add_shp___model__42__1730847906": "#00FF00",
    "L3.1-8B-v3.9-nc-fixed-2-pif_uf_hs_dpo___model__42__1730613882": "#FF0000",
    "hf-llama-3-tulu-2-dpo-8b": "#808000",
}

    # "#B7CBCC",  # RGB(183, 203, 204)
    # "#9FB9BB",  # RGB(159, 185, 187)
    # "#88A8AB",  # RGB(136, 168, 171)
    # "#6E979A",  # RGB(110, 151, 154)
    # "#588689",  # RGB(88, 134, 137)
    # "#3F7478",  # RGB(
    # )
    # "#105257",  # RGB(16, 82, 87)
    # "#0A3235",  # RGB(10, 50, 53)
    # "#F0529C", # PINK

# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

# Get metrics (excluding Rank and Average)
# metrics = [col for col in df.columns if col not in ['Rank']]

metrics = [
    "Average",
    "BBH",
    "GSM8K",
    "IFEval",
    "MATH",
    "MMLU",
    "Safety",
]

# Set up the plot
fig, ax = plt.subplots(figsize=(15, 8))

# Create the grouped bar chart
# plt.figure(figsize=(20, 10))

# Set the width of each bar and positions of the bars
width = 0.08  # Reduced width to accommodate more bars
x = np.arange(len(metrics))

# Create bars for each model
for i, (model, model_data) in enumerate(sorted(df.iterrows())):
    plt.bar(x + i*width, 
            model_data[metrics], 
            width, 
            label=model.split('___')[0] if '___' in model else model,
            color=colors[model],
            edgecolor="black")

# Customize the plot
# plt.xlabel('Metrics', fontsize=12)
# plt.ylabel('Score', fontsize=12)
# plt.title('Model Performance Comparison Across Different Metrics', fontsize=14)
    
# Customize the plot
# ax.set_xlabel('Benchmarks', fontsize=14)
ax.set_ylabel('Performance', fontsize=18)
plt.tick_params(axis='y', labelsize=18)
# ax.set_title('Performance by Benchmark and SFT Percentage', fontsize=14)

# Set x-axis ticks and labels
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, ha="center", fontsize=18)

ax.spines[["right", "top"]].set_visible(False)

# Add legend

plt.xticks(x + width * len(df)/2, metrics, ha='center')
# plt.legend(bbox_to_anchor=(0.6, 0.75), loc='upper left')
# plt.grid(True, alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save and show the plot
plt.savefig('tulu_version_bars.pdf', bbox_inches='tight', dpi=300)
plt.show()