import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

# Define keywords to check
keywords = ["dataset", "benchmark", "evaluation", "leaderboard", "metric"]

# visualize the data_info
data_info = {
    "static_rubric": "/checkpoint/comem/rulin/open-instruct/output_toy_rag_survey_analysis/toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761279009/",
    "adaptive_rubric": "/checkpoint/comem/rulin/open-instruct/output_toy_rag_survey_analysis/toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761283810",
}

# Path to adaptive rubrics JSONL file
adaptive_rubrics_path = "/checkpoint/comem/rulin/open-instruct/output_toy_rag_survey_analysis/toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761283810/adaptive_rubrics_toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761283810.jsonl"

def calculate_keyword_ratio(data_dir, step_range):
    """Calculate the ratio of responses containing keywords for a given step range."""
    all_responses_filenames = os.listdir(data_dir)
    num_keyword_responses = 0
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
                    if any(keyword in response.lower() for keyword in keywords):
                        num_keyword_responses += 1
                    total_responses += 1
    
    if total_responses == 0:
        return 0.0
    return num_keyword_responses / total_responses

def load_adaptive_rubrics_data(rubrics_path):
    """Load all adaptive rubrics data."""
    with open(rubrics_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def count_cumulative_keyword_positive_rubrics(data, max_step):
    """Count cumulative unique positive rubrics containing keywords from step 0 to max_step."""
    unique_keyword_rubrics = set()
    
    for item in data:
        step = item["training_step"]
        if step > max_step:
            continue
        
        # Collect unique positive rubrics containing keywords
        for rubric in item["adaptive_rubric_scores"][0]["positive_rubrics"]:
            if any(keyword in rubric["description"].lower() for keyword in keywords):
                # Use description as unique identifier
                unique_keyword_rubrics.add(rubric["description"])
    
    return len(unique_keyword_rubrics)

def count_keyword_positive_rubrics_in_bin(data, min_step, max_step):
    """Count positive rubrics containing keywords within a step range [min_step, max_step]."""
    keyword_rubric_count = 0
    
    for item in data:
        step = item["training_step"]
        if step < min_step or step > max_step:
            continue
        
        # Count positive rubrics containing keywords in this step
        for rubric in item["adaptive_rubric_scores"][0]["positive_rubrics"]:
            if any(keyword in rubric["description"].lower() for keyword in keywords):
                keyword_rubric_count += 1
    
    return keyword_rubric_count

# Load adaptive rubrics data
print("Loading adaptive rubrics data...")
adaptive_rubrics_data = load_adaptive_rubrics_data(adaptive_rubrics_path)
print(f"Loaded {len(adaptive_rubrics_data)} steps from adaptive rubrics")

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

# Also add steps from adaptive rubrics
for item in adaptive_rubrics_data:
    all_steps.add(item["training_step"])

all_steps = sorted(list(all_steps))
print(f"Available steps: {all_steps}")
min_step = min(all_steps)
max_step = min(500, max(all_steps))
print(f"Min step: {min_step}, Max step: {max_step}")

# Calculate ratios for different step ranges with intervals
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

ratios_by_type = {key: [] for key in data_info.keys()}
ratios_by_type_binned = {key: [] for key in data_info.keys()}

for max_step_threshold in step_thresholds:
    for key, data_dir in data_info.items():
        ratio = calculate_keyword_ratio(data_dir, (0, max_step_threshold))
        ratios_by_type[key].append(ratio)
        print(f"{key} - Steps 0-{max_step_threshold}: {ratio:.4f}")

# Calculate ratios for each bin
for i, step_threshold in enumerate(step_thresholds):
    # Determine the bin range
    if i == 0:
        bin_start = min_step
    else:
        bin_start = step_thresholds[i-1] + 1
    bin_end = step_threshold
    
    for key, data_dir in data_info.items():
        ratio = calculate_keyword_ratio(data_dir, (bin_start, bin_end))
        ratios_by_type_binned[key].append(ratio)
        print(f"{key} - Bin [{bin_start}, {bin_end}]: {ratio:.4f}")

# Get cumulative keyword count at each step threshold
keyword_counts_at_thresholds = []
for step_threshold in step_thresholds:
    # Count cumulative unique keyword positive rubrics from step 0 to step_threshold
    count = count_cumulative_keyword_positive_rubrics(adaptive_rubrics_data, step_threshold)
    keyword_counts_at_thresholds.append(count)
    print(f"Cumulative keyword positive rubrics count (0 to {step_threshold}): {count}")

# Get keyword count in each bin
keyword_counts_binned = []
for i, step_threshold in enumerate(step_thresholds):
    # Determine the bin range
    if i == 0:
        bin_start = min_step
    else:
        bin_start = step_thresholds[i-1] + 1
    bin_end = step_threshold
    
    # Count keyword positive rubrics within this bin
    count = count_keyword_positive_rubrics_in_bin(adaptive_rubrics_data, bin_start, bin_end)
    keyword_counts_binned.append(count)
    print(f"Keyword positive rubrics count in bin [{bin_start}, {bin_end}]: {count}")

# Plot the results with dual y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot ratios on the left y-axis
colors = ['blue', 'green']
for idx, (key, ratios) in enumerate(ratios_by_type.items()):
    ax1.plot(step_thresholds, ratios, marker='o', label=key, 
             linewidth=2, markersize=6, color=colors[idx])

ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('Ratio of Keyword Responses', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)

# Create second y-axis for keyword count
ax2 = ax1.twinx()
ax2.plot(step_thresholds, keyword_counts_at_thresholds, marker='s', 
         label='Cumulative Positive Rubrics with Keywords', linewidth=2, markersize=6, 
         color='red', linestyle='--')
ax2.set_ylabel('Cumulative # of Positive Rubrics with Keywords', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

plt.title(f'Keyword Response Ratio and Cumulative Positive Rubrics Count vs Training Steps\nKeywords: {", ".join(keywords)}', fontsize=14)
plt.tight_layout()

# Save the plot
output_plot_path = '/checkpoint/comem/rulin/open-instruct/local_scripts/small_scale_analysis/analysis/keyword_ratio_comparison.png'
plt.savefig(output_plot_path, dpi=300)
print(f"\nPlot saved to: {output_plot_path}")

# ============================================================================
# Second figure: Keyword ratio within each bin (with smoothing)
# ============================================================================

fig2, ax3 = plt.subplots(figsize=(12, 6))

# Smoothing parameters
window_size = 25  # Size of smoothing window for smooth curves

# Plot binned ratios on the left y-axis
colors = ['blue', 'green']
for idx, (key, ratios) in enumerate(ratios_by_type_binned.items()):
    ratios_array = np.array(ratios)
    
    # Plot original data with transparency (background) - no smoothing
    ax3.plot(step_thresholds, ratios, alpha=0.15, 
             linewidth=1, color=colors[idx])
    
    # Apply smoothing
    if len(ratios_array) >= window_size:
        smoothed_ratios = uniform_filter1d(ratios_array, size=window_size, mode='nearest')
    else:
        smoothed_ratios = ratios_array
    
    # Plot smoothed data (foreground) - smooth curve
    ax3.plot(step_thresholds, smoothed_ratios, label=key, 
             linewidth=3, color=colors[idx])

ax3.set_xlabel('Training Steps', fontsize=12)
ax3.set_ylabel('Proportion of Keyword Responses (per bin)', fontsize=12, color='black')
ax3.tick_params(axis='y', labelcolor='black')
ax3.grid(True, alpha=0.3)

# Create second y-axis for binned keyword count
ax4 = ax3.twinx()

# Convert to numpy array
counts_array = np.array(keyword_counts_binned)

# Plot original count data with transparency (background) - no smoothing
ax4.plot(step_thresholds, keyword_counts_binned, alpha=0.15,
         linewidth=1, color='red', linestyle='--')

# Apply smoothing to count data
if len(counts_array) >= window_size:
    smoothed_counts = uniform_filter1d(counts_array.astype(float), size=window_size, mode='nearest')
else:
    smoothed_counts = counts_array

# Plot smoothed count data (foreground) - smooth curve
ax4.plot(step_thresholds, smoothed_counts,
         label='Positive Rubrics with Keywords (per bin)', linewidth=3, 
         color='red', linestyle='--')
ax4.set_ylabel('# of Positive Rubrics with Keywords (per bin)', fontsize=12, color='red')
ax4.tick_params(axis='y', labelcolor='red')

# Combine legends
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc='best', fontsize=10)

plt.title(f'Keyword Response Ratio and Positive Rubrics Count per Bin vs Training Steps (Smoothed)\nKeywords: {", ".join(keywords)}', fontsize=14)
plt.tight_layout()

# Save the second plot
output_plot_path_binned = '/checkpoint/comem/rulin/open-instruct/local_scripts/small_scale_analysis/analysis/keyword_ratio_binned.png'
plt.savefig(output_plot_path_binned, dpi=300)
print(f"Binned plot saved to: {output_plot_path_binned}")


