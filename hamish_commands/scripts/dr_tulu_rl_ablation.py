import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D # Import Line2D for creating custom legend handles

# --- 1. Data Loading ---
file_path_updated = 'RL-RAG2_updated.csv' # Assumes the CSV is in the same directory as the script
df = pd.read_csv(file_path_updated)

# --- 2. Data Cleaning and Preprocessing ---
relevant_columns = [
    'healthbench',
    'Asta-overall',
    'Asta-Rubric',
    'Asta-C-overall',
    'Asta-C-Rubric',
    'Helpful-overall',
    'Helpful-Rubric',
    'DRB-overall',
    'DRB-# valid citation',
    'Faithful-overall',
    'Faithful-Rubric',
    'Factual-overall',
    'Factual-Rubric',
    'Asta-Answer P',
    'Asta-Citation P',
    'Asta-Citation R',
    'DRB-Comp.',
    'DRB-Depth',
    'DRB-Inst.',
    'DRB-Read.',
    'DRB-citaiton Acc.'
]

# Create a copy to avoid modifying the original DataFrame directly
df_cleaned = df.copy()

existing_relevant_columns = []
for col in relevant_columns:
    if col in df_cleaned.columns:
        # Remove '%' and convert to numeric, handling errors by coercing to NaN
        df_cleaned[col] = df_cleaned[col].astype(str).str.replace('%', '', regex=False)
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        existing_relevant_columns.append(col)
    # else:
        # print(f"Warning: Column '{col}' not found in DataFrame.") # Suppress warnings in final script

# --- 3. Unique Results Extraction ---
qwen3_data = df_cleaned[df_cleaned['Model Name'] == 'Qwen3']
qwen3_base_rl_data = df_cleaned[df_cleaned['Model Name'] == 'Qwen3 Base + RL']
qwen3_8b_data = df_cleaned[df_cleaned['Model Name'] == 'Qwen3 8B']

qwen3_unique_results = {}
qwen3_base_rl_unique_results = {}
qwen3_8b_unique_results = {}

for col in existing_relevant_columns:
    # Get unique non-NaN values for Qwen3
    unique_qwen3_values = qwen3_data[col].dropna().unique()
    if len(unique_qwen3_values) == 1:
        qwen3_unique_results[col] = unique_qwen3_values[0]
    elif len(unique_qwen3_values) > 1:
        qwen3_unique_results[col] = unique_qwen3_values.tolist()
    else:
        qwen3_unique_results[col] = None

    # Get unique non-NaN values for Qwen3 Base + RL
    unique_qwen3_base_rl_values = qwen3_base_rl_data[col].dropna().unique()
    if len(unique_qwen3_base_rl_values) == 1:
        qwen3_base_rl_unique_results[col] = unique_qwen3_base_rl_values[0]
    elif len(unique_qwen3_base_rl_values) > 1:
        qwen3_base_rl_unique_results[col] = unique_qwen3_base_rl_values.tolist()
    else:
        qwen3_base_rl_unique_results[col] = None

    # Get unique non-NaN values for Qwen3 8B
    unique_qwen3_8b_values = qwen3_8b_data[col].dropna().unique()
    if len(unique_qwen3_8b_values) == 1:
        qwen3_8b_unique_results[col] = unique_qwen3_8b_values[0]
    elif len(unique_qwen3_8b_values) > 1:
        qwen3_8b_unique_results[col] = unique_qwen3_8b_values.tolist()
    else:
        qwen3_8b_unique_results[col] = None

# --- 4. Combined Plot Generation ---

# Set custom figure size for the combined plot (1 row, 4 columns)
plt.rcParams['figure.figsize'] = (20, 5) # Adjusted for 1x4 layout

# Define custom colors for dynamic models in the line plot
custom_palette = {
    'On policy SFT': '#B11BE8',  # 'onpolicy sft' = Hex. B11BE8
    'base SFT': '#F0529C',       # 'base SFT' = Hex. F0529C
    'Undertrained': '#0FCBBC',   # 'Undertrained' = Hex. 0FCBBC
}

# Define the color for static lines (Qwen3, Qwen3 Base, and Qwen3 8B)
static_line_color = '#105257'

# Metrics to include in the combined plot
selected_metrics = ['healthbench', 'Asta-overall', 'DRB-overall', 'Asta-Citation P']

fig, axes = plt.subplots(1, 4, sharey=False, figsize=(20, 5)) # 1 row, 4 columns of subplots
axes = axes.flatten() # Flatten the 1x4 array of axes for easy iteration

unique_legend_data = {} # Use a dictionary to store unique label -> handle mapping

# --- Manually create custom legend handles for all models ---

# Dynamic models (lineplot styles)
for model_name, color in custom_palette.items():
    display_label = model_name
    if model_name == 'base SFT':
        display_label = 'SFT'
    if display_label not in unique_legend_data:
        unique_legend_data[display_label] = Line2D([0], [0], color=color, marker='o', markersize=8, label=display_label)

# Static models (axhline styles)
# Qwen3
if any(v is not None and not np.isnan(v) for v in qwen3_unique_results.values()):
    display_label = 'Qwen 3 (no RL)'
    if display_label not in unique_legend_data:
        unique_legend_data[display_label] = Line2D([0], [0], color=static_line_color, linestyle='-', label=display_label)

# Qwen3 Base + RL
if any(v is not None and not np.isnan(v) for v in qwen3_base_rl_unique_results.values()):
    display_label = 'Qwen3 Base'
    if display_label not in unique_legend_data:
        unique_legend_data[display_label] = Line2D([0], [0], color=static_line_color, linestyle='--', label=display_label)

# Qwen3 8B
if any(v is not None and not np.isnan(v) for v in qwen3_8b_unique_results.values()):
    display_label = 'Qwen 3 8B (no RL)'
    if display_label not in unique_legend_data:
        unique_legend_data[display_label] = Line2D([0], [0], color=static_line_color, linestyle=':', label=display_label)


for i, metric in enumerate(selected_metrics):
    ax = axes[i]

    # Filter df_cleaned to exclude static models (Qwen3, Qwen3 Base + RL, Qwen3 8B) for line plot.
    df_filtered_for_line = df_cleaned[~df_cleaned['Model Name'].isin(['Qwen3', 'Qwen3 Base + RL', 'Qwen3 8B'])]

    # Plot the performance of the filtered models over 'Time Step'
    sns.lineplot(data=df_filtered_for_line, x='Time Step', y=metric, hue='Model Name',
                 marker='o', markersize=8, ax=ax, palette=custom_palette, legend=False)

    # Add horizontal line for Qwen3 if a single numeric value is available
    if metric in qwen3_unique_results and isinstance(qwen3_unique_results[metric], (int, float)) and not np.isnan(qwen3_unique_results[metric]):
        ax.axhline(y=qwen3_unique_results[metric], color=static_line_color, linestyle='-', label='Qwen 3 (no RL)')

    # Add horizontal line for Qwen3 Base + RL if a single numeric value is available
    if metric in qwen3_base_rl_unique_results and isinstance(qwen3_base_rl_unique_results[metric], (int, float)) and not np.isnan(qwen3_base_rl_unique_results[metric]):
        ax.axhline(y=qwen3_base_rl_unique_results[metric], color=static_line_color, linestyle='--', label='Qwen3 Base')

    # Add horizontal line for Qwen3 8B if a single numeric value is available
    if metric in qwen3_8b_unique_results and isinstance(qwen3_8b_unique_results[metric], (int, float)) and not np.isnan(qwen3_8b_unique_results[metric]):
        ax.axhline(y=qwen3_8b_unique_results[metric], color=static_line_color, linestyle=':', label='Qwen 3 8B (no RL)')

    # Set plot labels with increased font size
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(metric, fontsize=14) # Set subplot title


plt.tight_layout(rect=[0, 0.15, 1, 0.95]) # Adjust layout to make space for the common legend and remove overall title

# Extract final handles and labels from the unique_legend_data dictionary
final_labels = list(unique_legend_data.keys())
final_handles = [unique_legend_data[label] for label in final_labels]

# Create a common legend at the bottom of the figure
fig.legend(final_handles, final_labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0), fontsize=10, title='Model Name', title_fontsize=12)

# plt.show() # Uncomment to display the plot when running locally

# Save the figure as a PDF
pdf_file_name = 'combined_performance_plots_1x4.pdf'
fig.savefig(pdf_file_name, bbox_inches='tight')
print(f"Combined plots saved to {pdf_file_name}")