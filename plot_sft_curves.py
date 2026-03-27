import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# ============================================================================
# Load entropy and likelihood data
# ============================================================================
df_entropy = pd.read_csv("calc-likelihoods/entropy_results-SFT-100.csv")
df_likelihood = pd.read_csv("calc-likelihoods/log_likelihood_results-SFT-100.csv")

def extract_step(model_name):
    """Extract step number from model name."""
    if 'allenai/Olmo-3-1025-7B' in model_name:
        return 0
    match = re.search(r'step_(\d+)', model_name)
    if match:
        return int(match.group(1))
    return None

# Process dataframes
df_entropy['step'] = df_entropy['model'].apply(extract_step)
df_likelihood['step'] = df_likelihood['model'].apply(extract_step)
df_entropy = df_entropy.dropna(subset=['step'])
df_likelihood = df_likelihood.dropna(subset=['step'])

# Average across all domains and distributions
entropy_by_step = df_entropy.groupby('step')['avg_entropy'].mean().sort_index()
likelihood_by_step = df_likelihood.groupby('step')['avg_log_likelihood'].mean().sort_index()

# ============================================================================
# Parse eval data
# ============================================================================
eval_data = """model_name	MMLU	PopQA	SimpleQA	BBH	GPQA	ZebraLogic	AGI Eval	Minerva Math	GSM8K	Omega 500	AIME 2025	AIME 2024	Codex HE+	MBPP	LCB v3	Alpaca Eval v3	IFEval
olmo3-sft-rando-step_0	67.1	16.5	3.4	51	30	18	59.2	65.1	87.4	14.4	6.7	7.2	69.8	56.5	20	21.8	81.7
olmo3-sft-rando-step_50	65.76	16.23	3	49.92	36.16	15	57.34	63.2	86.5	10.2	5.31	6.88	69.88	55.98	16.49	24.95	79.11
olmo3-sft-rando-step_100	65.7	16.14	3.37	49.65	29.02	13.8	56.43	61.81	87.26	10.8	5.42	5.21	69.39	55.98	15.98	24	80.04
olmo3-sft-rando-step_150	65.23	16.14	3.03	50.55	33.48	13.3	56.31	60.11	86.73	10.2	4.17	5.42	68.78	56.16	15.52	25.46	79.3
olmo3-sft-rando-step_200	65.39	15.76	3.21	50.25	31.47	14.5	56.7	57.83	86.66	10	3.75	5	68.29	56.11	14.59	23.46	76.52
olmo3-sft-rando-step_250	65.9	16.15	3.13	52.57	32.59	14.7	57.2	54.8	84.46	10.6	4.58	4.79	68.29	55.08	13.17	24.49	79.85
olmo3-sft-rando-step_300	65.99	16.39	3.01	56.27	31.92	15.3	57.06	47.12	84.31	8.2	3.02	4.06	66.71	53.97	13.12	23.85	78.19
olmo3-sft-rando-step_350	66.21	16.24	3.18	59.42	33.26	15.9	56.98	44.36	80.29	7	3.33	3.75	66.22	53.17	16.85	20.39	77.82
olmo3-sft-rando-step_400	64.77	15.69	2.32	54.51	31.47	13.9	56.38	31.07	44.28	4.4	1.25	1.15	56.1	46.85	16.93	6.42	70.98
olmo3-sft-rando-step_450	44.31	7.66	0.51	17.97	25.89	3.3	40.3	8.06	9.1	1	0.21	0.21	24.27	22.04	6.32	0.69	34.2
olmo3-sft-rando-step_500	21.04	4.04	0.05	4.53	24.55	0	19.01	0.01	0.23	0	0	0	0.06	0.93	0	0	12.57
olmo3-sft-rando-step_550	null	null	null	null	24.78	null	null	null	null	0	null	null	null	null	null	null	null
olmo3-sft-rando-step_600	null	null	null	null	23.66	null	null	null	null	null	0	null	null	null	null	null	null
olmo3-sft-rando-step_650	null	null	null	null	24.55	0	null	null	0.15	0	0	0	null	null	null	null	null
olmo3-sft-rando-step_700	null	null	null	null	25.22	0	null	null	null	0	0	null	null	null	null	null	10.72
olmo3-sft-rando-step_750	null	null	null	null	24.78	0	null	null	0.53	0	0	0	0	null	null	null	10.54
olmo3-sft-rando-step_800	null	null	0.09	9.97	26.79	0	null	null	0.68	0	0	0	0	null	null	null	10.54
olmo3-sft-rando-step_850	null	null	0.09	null	22.1	null	19.12	null	null	0	0	null	0	null	null	null	null
olmo3-sft-rando-step_900	null	null	null	null	25.89	0	null	null	0.3	0	0	0	null	null	null	null	null
olmo3-sft-rando-step_950	null	null	null	null	24.33	null	null	null	null	null	null	null	null	null	null	null	null
olmo3-sft-rando-step_1000	null	null	0.14	null	27.68	null	null	null	null	null	null	null	null	null	null	null	null"""

# Parse into dataframe
from io import StringIO
df_evals = pd.read_csv(StringIO(eval_data), sep='\t', na_values=['null'])

# Extract step from model name
df_evals['step'] = df_evals['model_name'].apply(lambda x: int(re.search(r'step_(\d+)', x).group(1)))

# Select evals to plot: GSM8K (math), MMLU (general), IFEval (instruction following)
evals_to_plot = ['GSM8K', 'MMLU', 'IFEval']

# ============================================================================
# Create combined plot
# ============================================================================
fig, ax1 = plt.subplots(figsize=(7, 5))

# Colors
color_likelihood = '#1f77b4'  # blue
color_entropy = '#d62728'     # red
eval_colors = {
    'GSM8K': '#2ca02c',       # green
    'MMLU': '#9467bd',        # purple
    'IFEval': '#ff7f0e'       # orange
}

# Plot likelihood on left axis
ax1.plot(likelihood_by_step.index, likelihood_by_step.values, 
         color=color_likelihood, linestyle='-', linewidth=2.5, 
         label='Log Likelihood', marker='o', markersize=6)

# Plot entropy on left axis
ax1.plot(entropy_by_step.index, entropy_by_step.values, 
         color=color_entropy, linestyle='--', linewidth=2.5, 
         label='Entropy', marker='s', markersize=6)

ax1.set_xlabel('Training Step', fontsize=14, fontweight='bold')
ax1.set_ylabel('Log Likelihood / Entropy', fontsize=14, fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.tick_params(axis='y', labelcolor='black')

# Create second y-axis for evals
ax2 = ax1.twinx()

for eval_name in evals_to_plot:
    eval_series = df_evals[['step', eval_name]].dropna()
    if len(eval_series) > 0:
        ax2.plot(eval_series['step'], eval_series[eval_name],
                 color=eval_colors[eval_name], linestyle='-', linewidth=2.5,
                 label=eval_name, marker='^', markersize=6)

ax2.set_ylabel('Eval Score (%)', fontsize=14, fontweight='bold')
ax2.tick_params(axis='y', labelsize=12)
ax2.set_ylim(0, 100)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11, framealpha=0.9)

ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_title('SFT Training: Likelihood, Entropy, and Evals', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('calc-likelihoods/plots-paper/sft_combined_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('calc-likelihoods/plots-paper/sft_combined_plot.pdf', bbox_inches='tight')
print("✓ Combined plot saved: calc-likelihoods/plots-paper/sft_combined_plot.png")
plt.close()

print("\n✓ Done!")