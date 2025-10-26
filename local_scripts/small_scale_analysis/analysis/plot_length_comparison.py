import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

# visualize the data_info
data_info = {
    "static_rubric": "/checkpoint/comem/rulin/open-instruct/output_toy_rag_survey_analysis/toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761279009/",
    "adaptive_rubric": "/checkpoint/comem/rulin/open-instruct/output_toy_rag_survey_analysis/toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761283810",
}

def calculate_average_length(data_dir, step_range):
    """Calculate the average response length (in words) for a given step range."""
    all_responses_filenames = os.listdir(data_dir)
    total_words = 0
    total_responses = 0
    
    for response_filename in all_responses_filenames:
        pattern = r"eval_step_(\d+).json"
        match = re.search(pattern, response_filename)
        if match:
            step = int(match.group(1))
            if step < step_range[0] or step > step_range[1]:
                continue
            response_path = os.path.join(data_dir, response_filename)
            with open(response_path, "r") as f:
                responses = json.load(f)
                responses = [item["response"] for item in responses["samples"]]
                for response in responses:
                    # Count words by splitting on whitespace
                    word_count = len(response.split())
                    total_words += word_count
                    total_responses += 1
    
    if total_responses == 0:
        return 0.0
    return total_words / total_responses

# Get all available steps from both datasets
all_steps = set()
for key, data_dir in data_info.items():
    all_responses_filenames = os.listdir(data_dir)
    for response_filename in all_responses_filenames:
        pattern = r"eval_step_(\d+).json"
        match = re.search(pattern, response_filename)
        if match:
            step = int(match.group(1))
            all_steps.add(step)

all_steps = sorted(list(all_steps))
print(f"Available steps: {len(all_steps)} steps")
min_step = min(all_steps)
max_step = min(300, max(all_steps))
print(f"Min step: {min_step}, Max step: {max_step}")

# Calculate lengths for different step ranges with intervals
step_interval = 10  # Can be changed to 10, 50, etc.
# Start from the minimum step and go up by intervals
step_thresholds = [min_step]  # Include the minimum step
current_step = min_step + step_interval
while current_step <= max_step:
    step_thresholds.append(current_step)
    current_step += step_interval
# Ensure max_step is included if not already
if step_thresholds[-1] != max_step:
    step_thresholds.append(max_step)
print(f"Step thresholds: {step_thresholds}")

lengths_by_type = {key: [] for key in data_info.keys()}
lengths_by_type_binned = {key: [] for key in data_info.keys()}

# Calculate cumulative average length for each threshold
print("\nCumulative average lengths:")
for max_step_threshold in step_thresholds:
    for key, data_dir in data_info.items():
        avg_length = calculate_average_length(data_dir, (0, max_step_threshold))
        lengths_by_type[key].append(avg_length)
        print(f"{key} - Steps 0-{max_step_threshold}: {avg_length:.2f} words")

# Calculate average length for each bin
print("\nPer-bin average lengths:")
for i, step_threshold in enumerate(step_thresholds):
    # Determine the bin range
    if i == 0:
        bin_start = min_step
    else:
        bin_start = step_thresholds[i-1] + 1
    bin_end = step_threshold
    
    for key, data_dir in data_info.items():
        avg_length = calculate_average_length(data_dir, (bin_start, bin_end))
        lengths_by_type_binned[key].append(avg_length)
        if i % 5 == 0:  # Print every 5th bin
            print(f"{key} - Bin [{bin_start}, {bin_end}]: {avg_length:.2f} words")

# ============================================================================
# First figure: Cumulative average length
# ============================================================================

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot average lengths
colors = ['blue', 'green']
for idx, (key, lengths) in enumerate(lengths_by_type.items()):
    ax1.plot(step_thresholds, lengths, marker='o', label=f'{key}', 
             linewidth=2, markersize=6, color=colors[idx])

ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('Average Response Length (words)', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=10)

plt.title('Average Response Length: Static vs Adaptive Rubrics (Cumulative)', fontsize=14)
plt.tight_layout()

# Save the plot
output_plot_path = '/checkpoint/comem/rulin/open-instruct/local_scripts/small_scale_analysis/analysis/length_comparison.png'
plt.savefig(output_plot_path, dpi=300)
print(f"\nPlot saved to: {output_plot_path}")

# ============================================================================
# Second figure: Average length within each bin (with smoothing)
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 6))

# Smoothing parameters
window_size = 25  # Size of smoothing window for smooth curves

# Plot binned average lengths
colors = ['blue', 'green']
for idx, (key, lengths) in enumerate(lengths_by_type_binned.items()):
    lengths_array = np.array(lengths)
    
    # Plot original data with transparency (background) - no smoothing
    ax2.plot(step_thresholds, lengths, alpha=0.15, 
             linewidth=1, color=colors[idx])
    
    # Apply smoothing
    if len(lengths_array) >= window_size:
        smoothed_lengths = uniform_filter1d(lengths_array.astype(float), size=window_size, mode='nearest')
    else:
        smoothed_lengths = lengths_array
    
    # Plot smoothed data (foreground) - smooth curve
    ax2.plot(step_thresholds, smoothed_lengths, label=f'{key}', 
             linewidth=3, color=colors[idx])

ax2.set_xlabel('Training Steps', fontsize=12)
ax2.set_ylabel('Average Response Length (words per bin)', fontsize=12, color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=10)

plt.title('Average Response Length per Bin: Static vs Adaptive Rubrics (Smoothed)', fontsize=14)
plt.tight_layout()

# Save the second plot
output_plot_path_binned = '/checkpoint/comem/rulin/open-instruct/local_scripts/small_scale_analysis/analysis/length_binned.png'
plt.savefig(output_plot_path_binned, dpi=300)
print(f"Binned plot saved to: {output_plot_path_binned}")

