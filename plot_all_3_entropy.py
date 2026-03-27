import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Load all three dataframes
df_base = pd.read_csv("calc-likelihoods/entropy_results-base-100.csv")
df_sft = pd.read_csv("calc-likelihoods/entropy_results-SFT-100.csv")
df_dpo = pd.read_csv("calc-likelihoods/entropy_results-DPO-100.csv")

# Add source labels
df_base['source'] = 'Base'
df_sft['source'] = 'SFT'
df_dpo['source'] = 'DPO'

def extract_step(model_name):
    """Extract step number from model name."""
    if 'allenai/Olmo-3-1025-7B' in model_name or 'jacobmorrison/Olmo-3-7B-Instruct-SFT-do-sample' in model_name or 'jacobmorrison/Olmo-3-7B-Instruct-DPO-do-sample' in model_name:
        return 0
    
    match = re.search(r'step_(\d+)', model_name)
    if match:
        return int(match.group(1))
    
    return None

# Process each dataframe
for df in [df_base, df_sft, df_dpo]:
    df['step'] = df['model'].apply(extract_step)

# Combine all dataframes
df_all = pd.concat([df_base, df_sft, df_dpo], ignore_index=True)
df_all = df_all.dropna(subset=['step'])

# Define colors for each source
source_colors = {
    'Base': '#1f77b4',  # blue
    'SFT': '#ff7f0e',   # orange
    'DPO': '#2ca02c'    # green
}

# ============================================================================
# PLOT: All data points, one line per model (Base, SFT, DPO)
# ============================================================================
fig, ax = plt.subplots(figsize=(5, 4))  # Single column size

for source in ['Base', 'SFT', 'DPO']:
    subset = df_all[df_all['source'] == source]
    
    if len(subset) == 0:
        print(f"Warning: No data for {source}")
        continue
    
    # Group by step, averaging across all domains and distributions
    grouped = subset.groupby('step')['avg_entropy'].mean()
    grouped = grouped.sort_index()
    
    steps = grouped.index.values
    mean_vals = grouped.values
    
    color = source_colors[source]
    ax.plot(steps, mean_vals, color=color, linestyle='-', 
            linewidth=2.5, label=source, marker='o', markersize=6)

# Large fonts for single column figure
ax.set_xlabel('Training Step', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Entropy', fontsize=14, fontweight='bold')
ax.set_title('Entropy over Training', fontsize=16, fontweight='bold')

ax.legend(loc='best', framealpha=0.9, fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

# Save
filename = 'calc-likelihoods/plots-paper/comparison_entropy_all.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
print(f"✓ Plot saved: {filename}")
plt.close()

print("\n✓ Done!")