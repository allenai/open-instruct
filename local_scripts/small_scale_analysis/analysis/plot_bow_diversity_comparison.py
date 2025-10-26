import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

# No keywords needed for BOW diversity - we'll count unique words

# visualize the data_info
data_info = {
    "static_rubric": "/checkpoint/comem/rulin/open-instruct/output_toy_rag_survey_analysis/toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761279009/",
    "adaptive_rubric": "/checkpoint/comem/rulin/open-instruct/output_toy_rag_survey_analysis/toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761283810",
}

# Path to adaptive rubrics JSONL file
adaptive_rubrics_path = "/checkpoint/comem/rulin/open-instruct/output_toy_rag_survey_analysis/toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761283810/adaptive_rubrics_toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761283810.jsonl"

def tokenize_text(text):
    """Simple tokenization: lowercase and split by non-alphanumeric characters."""
    text = text.lower()
    # Split by non-alphanumeric characters and filter empty strings
    words = re.findall(r'\b\w+\b', text)
    return words

def get_rubric_vocabulary(adaptive_rubrics_data, max_step):
    """Get the set of unique words from positive rubrics up to max_step."""
    rubric_words = set()
    for item in adaptive_rubrics_data:
        step = item["training_step"]
        if step > max_step:
            continue
        for rubric in item["adaptive_rubric_scores"][0]["positive_rubrics"]:
            words = tokenize_text(rubric["description"])
            rubric_words.update(words)
    return rubric_words

def get_word_frequencies_from_responses(data_dir, step_range):
    """Get word frequency counts from responses."""
    from collections import Counter
    word_counts = Counter()
    all_responses_filenames = os.listdir(data_dir)
    
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
                    words = tokenize_text(response)
                    word_counts.update(words)
    
    return word_counts

def get_common_words_across_all_sources(data_info, adaptive_rubrics_data, max_step, top_percent=70):
    """Find words that account for the top X% of cumulative frequency across all sources."""
    from collections import Counter
    
    # Get words from static rubric responses
    static_words = get_word_frequencies_from_responses(
        data_info["static_rubric"], (0, max_step))
    
    # Get words from adaptive rubric responses
    adaptive_words = get_word_frequencies_from_responses(
        data_info["adaptive_rubric"], (0, max_step))
    
    # Get words from rubrics
    rubric_word_counts = Counter()
    for item in adaptive_rubrics_data:
        step = item["training_step"]
        if step > max_step:
            continue
        for rubric in item["adaptive_rubric_scores"][0]["positive_rubrics"]:
            words = tokenize_text(rubric["description"])
            rubric_word_counts.update(words)
    
    # Find words that appear in all three sources
    static_set = set(static_words.keys())
    adaptive_set = set(adaptive_words.keys())
    rubric_set = set(rubric_word_counts.keys())
    
    common_words_all = static_set & adaptive_set & rubric_set
    
    # Get frequency sum for each common word across all sources
    total_freq = Counter()
    for word in common_words_all:
        total_freq[word] = static_words[word] + adaptive_words[word] + rubric_word_counts[word]
    
    # Calculate cumulative frequency cutoff
    sorted_words = total_freq.most_common()
    total_count = sum(total_freq.values())
    cumulative_freq = 0
    cutoff_threshold = total_count * (top_percent / 100.0)
    
    words_to_exclude = []
    for word, count in sorted_words:
        cumulative_freq += count
        words_to_exclude.append(word)
        if cumulative_freq >= cutoff_threshold:
            break
    
    print(f"\nTop {top_percent}% of frequency accounts for {len(words_to_exclude)} words")
    print(f"Total frequency: {total_count}, Cutoff: {cumulative_freq} ({cumulative_freq/total_count*100:.1f}%)")
    print(f"\nFirst 50 words being excluded:")
    print(", ".join(words_to_exclude[:50]))  # Print first 50 for inspection
    
    return set(words_to_exclude)

def calculate_bow_diversity(data_dir, step_range, rubric_vocabulary):
    """Calculate the number of unique words in responses that appear in rubric vocabulary."""
    all_responses_filenames = os.listdir(data_dir)
    unique_words = set()
    
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
                    words = set(tokenize_text(response))
                    # Only count words that appear in rubric vocabulary
                    unique_words.update(words & rubric_vocabulary)
    
    return len(unique_words)

def load_adaptive_rubrics_data(rubrics_path):
    """Load all adaptive rubrics data."""
    with open(rubrics_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def calculate_cumulative_rubric_bow_diversity(data, max_step):
    """Calculate cumulative BOW diversity of positive rubrics from step 0 to max_step."""
    unique_words = set()
    
    for item in data:
        step = item["training_step"]
        if step > max_step:
            continue
        
        # Collect words from all positive rubrics
        for rubric in item["adaptive_rubric_scores"][0]["positive_rubrics"]:
            words = tokenize_text(rubric["description"])
            unique_words.update(words)
    
    return len(unique_words)

def calculate_rubric_bow_diversity_in_bin(data, min_step, max_step):
    """Calculate BOW diversity of positive rubrics within a step range [min_step, max_step]."""
    unique_words = set()
    
    for item in data:
        step = item["training_step"]
        if step < min_step or step > max_step:
            continue
        
        # Collect words from positive rubrics in this step range
        for rubric in item["adaptive_rubric_scores"][0]["positive_rubrics"]:
            words = tokenize_text(rubric["description"])
            unique_words.update(words)
    
    return len(unique_words)

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
max_step = min(300, max(all_steps))
print(f"Min step: {min_step}, Max step: {max_step}")

# Calculate BOW diversity for different step ranges with intervals
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

# First, identify the most common words shared across all sources
print("\n" + "="*80)
print("Identifying words accounting for top 70% of frequency across all sources...")
print("="*80)
common_words_to_exclude = get_common_words_across_all_sources(
    data_info, adaptive_rubrics_data, max_step, top_percent=70)
print(f"\nExcluding {len(common_words_to_exclude)} most frequent words")
print("="*80 + "\n")

diversity_by_type = {key: [] for key in data_info.keys()}
diversity_by_type_binned = {key: [] for key in data_info.keys()}

# Calculate cumulative BOW diversity for each threshold
for max_step_threshold in step_thresholds:
    # Get rubric vocabulary up to this threshold
    rubric_vocab = get_rubric_vocabulary(adaptive_rubrics_data, max_step_threshold)
    # Filter out common words
    rubric_vocab_filtered = rubric_vocab - common_words_to_exclude
    print(f"Step {max_step_threshold}: Rubric vocab size: {len(rubric_vocab)} -> {len(rubric_vocab_filtered)} (after filtering)")
    
    for key, data_dir in data_info.items():
        diversity = calculate_bow_diversity(data_dir, (0, max_step_threshold), rubric_vocab_filtered)
        diversity_by_type[key].append(diversity)
        print(f"  {key}: {diversity} unique distinctive words from rubrics")

# Calculate BOW diversity for each bin
print("\nCalculating per-bin diversity...")
for i, step_threshold in enumerate(step_thresholds):
    # Determine the bin range
    if i == 0:
        bin_start = min_step
    else:
        bin_start = step_thresholds[i-1] + 1
    bin_end = step_threshold
    
    # Get rubric vocabulary for this bin and filter out common words
    rubric_vocab_bin = get_rubric_vocabulary(adaptive_rubrics_data, bin_end)
    rubric_vocab_bin_filtered = rubric_vocab_bin - common_words_to_exclude
    
    for key, data_dir in data_info.items():
        diversity = calculate_bow_diversity(data_dir, (bin_start, bin_end), rubric_vocab_bin_filtered)
        diversity_by_type_binned[key].append(diversity)
        if i % 5 == 0:  # Print every 5th bin to reduce clutter
            print(f"{key} - Bin [{bin_start}, {bin_end}]: {diversity} unique distinctive words")

# Get cumulative BOW diversity of positive rubrics at each step threshold
rubric_diversity_at_thresholds = []
for step_threshold in step_thresholds:
    # Calculate cumulative BOW diversity of positive rubrics from step 0 to step_threshold
    diversity = calculate_cumulative_rubric_bow_diversity(adaptive_rubrics_data, step_threshold)
    rubric_diversity_at_thresholds.append(diversity)
    print(f"Cumulative positive rubrics BOW diversity (0 to {step_threshold}): {diversity} unique words")

# Get BOW diversity of rubrics in each bin
rubric_diversity_binned = []
for i, step_threshold in enumerate(step_thresholds):
    # Determine the bin range
    if i == 0:
        bin_start = min_step
    else:
        bin_start = step_thresholds[i-1] + 1
    bin_end = step_threshold
    
    # Calculate BOW diversity of positive rubrics within this bin
    diversity = calculate_rubric_bow_diversity_in_bin(adaptive_rubrics_data, bin_start, bin_end)
    rubric_diversity_binned.append(diversity)
    print(f"Positive rubrics BOW diversity in bin [{bin_start}, {bin_end}]: {diversity} unique words")

# Plot the results - only response BOW diversity comparison
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot BOW diversity comparison
colors = ['blue', 'green']
for idx, (key, diversities) in enumerate(diversity_by_type.items()):
    ax1.plot(step_thresholds, diversities, marker='o', label=f'{key}', 
             linewidth=2, markersize=6, color=colors[idx])

ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('# Distinctive Rubric Words Used (Cumulative)', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=10)

plt.title('Vocabulary Alignment: Distinctive Rubric Words in Responses\n(Excluding Top 70% Most Frequent Common Words)', fontsize=13)
plt.tight_layout()

# Save the plot
output_plot_path = '/checkpoint/comem/rulin/open-instruct/local_scripts/small_scale_analysis/analysis/bow_diversity_comparison.png'
plt.savefig(output_plot_path, dpi=300)
print(f"\nPlot saved to: {output_plot_path}")

# ============================================================================
# Second figure: BOW diversity within each bin (with smoothing)
# ============================================================================

fig2, ax3 = plt.subplots(figsize=(12, 6))

# Smoothing parameters
window_size = 25  # Size of smoothing window for smooth curves

# Plot binned BOW diversity - only responses comparison
colors = ['blue', 'green']
for idx, (key, diversities) in enumerate(diversity_by_type_binned.items()):
    diversities_array = np.array(diversities)
    
    # Plot original data with transparency (background) - no smoothing
    ax3.plot(step_thresholds, diversities, alpha=0.15, 
             linewidth=1, color=colors[idx])
    
    # Apply smoothing
    if len(diversities_array) >= window_size:
        smoothed_diversities = uniform_filter1d(diversities_array.astype(float), size=window_size, mode='nearest')
    else:
        smoothed_diversities = diversities_array
    
    # Plot smoothed data (foreground) - smooth curve
    ax3.plot(step_thresholds, smoothed_diversities, label=f'{key}', 
             linewidth=3, color=colors[idx])

ax3.set_xlabel('Training Steps', fontsize=12)
ax3.set_ylabel('# Distinctive Rubric Words Used (per bin)', fontsize=12, color='black')
ax3.tick_params(axis='y', labelcolor='black')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best', fontsize=10)

plt.title('Vocabulary Alignment: Distinctive Rubric Words per Bin (Smoothed)\n(Excluding Top 70% Most Frequent Common Words)', fontsize=13)
plt.tight_layout()

# Save the second plot
output_plot_path_binned = '/checkpoint/comem/rulin/open-instruct/local_scripts/small_scale_analysis/analysis/bow_diversity_binned.png'
plt.savefig(output_plot_path_binned, dpi=300)
print(f"Binned plot saved to: {output_plot_path_binned}")


