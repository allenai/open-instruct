import csv
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

LENGTH_TSV_PATH = "/weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/7B/median_lengths.tsv"
SCORE_TSV_PATH = "/weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/7B/primary_scores.tsv"
Ai2_colors = {
       "Dark teal": "#0a3235",
       "Cream": "#faf2e9",
       "Light teal": "#105257",
       "Pink": "#f0529c",
       "Purple": "#b11be8",
       "Deep sea": "#0fcb8c"
}

def parse_score_and_length_tsv(length_tsv_path, score_tsv_path):
    # Parse length TSV - average across all eval columns
    # First pass: identify columns with complete data (no missing values)
    length_data = {}
    with open(length_tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
        
        if not rows:
            return {}
        
        # Get all column names except model
        all_cols = [k for k in rows[0].keys() if k not in ['model', 'model_name']]
        
        # Identify columns with any missing values
        valid_cols = []
        excluded_cols = {}  # Map column -> list of models with missing values
        
        for col in all_cols:
            missing_models = []
            for row in rows:
                val = row.get(col, '').strip()
                model_name = row.get('model', row.get('model_name', '')).strip()
                if not val:
                    missing_models.append(model_name)
                else:
                    try:
                        float(val)
                    except ValueError:
                        missing_models.append(model_name)
            
            if missing_models:
                excluded_cols[col] = missing_models
            else:
                valid_cols.append(col)
        
        if excluded_cols:
            print(f"\n[LENGTH TSV] Excluding {len(excluded_cols)} column(s) with missing values:")
            for col in sorted(excluded_cols.keys()):
                models_str = ', '.join(excluded_cols[col])
                print(f"  - {col}")
                print(f"    Missing in: {models_str}")
        
        print(f"[LENGTH TSV] Using {len(valid_cols)} complete column(s) for averaging")
        
        # Second pass: compute averages using only valid columns
        for row in rows:
            model_name = row.get('model', row.get('model_name', '')).strip()
            if not model_name:
                continue
            
            values = []
            for col in valid_cols:
                val = row[col].strip()
                values.append(float(val))
            
            if values:
                length_data[model_name] = sum(values) / len(values)
    
    # Parse score TSV - average across all eval columns
    score_data = {}
    with open(score_tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
        
        if not rows:
            return {}
        
        # Get all column names except model
        all_cols = [k for k in rows[0].keys() if k not in ['model', 'model_name']]
        
        # Identify columns with any missing values
        valid_cols = []
        excluded_cols = {}  # Map column -> list of models with missing values
        
        for col in all_cols:
            missing_models = []
            for row in rows:
                val = row.get(col, '').strip()
                model_name = row.get('model', row.get('model_name', '')).strip()
                if not val:
                    missing_models.append(model_name)
                else:
                    try:
                        float(val)
                    except ValueError:
                        missing_models.append(model_name)
            
            if missing_models:
                excluded_cols[col] = missing_models
            else:
                valid_cols.append(col)
        
        if excluded_cols:
            print(f"\n[SCORE TSV] Excluding {len(excluded_cols)} column(s) with missing values:")
            for col in sorted(excluded_cols.keys()):
                models_str = ', '.join(excluded_cols[col])
                print(f"  - {col}")
                print(f"    Missing in: {models_str}")
        
        print(f"[SCORE TSV] Using {len(valid_cols)} complete column(s) for averaging\n")
        
        # Second pass: compute averages using only valid columns
        for row in rows:
            model_name = row.get('model', row.get('model_name', '')).strip()
            if not model_name:
                continue
            
            values = []
            for col in valid_cols:
                val = row[col].strip()
                values.append(float(val))
            
            if values:
                score_data[model_name] = sum(values) / len(values)
    
    # Combine into the desired dictionary format
    result = {}
    all_models = set(length_data.keys()) | set(score_data.keys())
    
    for model in all_models:
        if model in length_data and model in score_data:
            result[model] = {
                "eval_average": score_data[model],
                "reasoning_length": length_data[model]
            }
    
    return result

"""{
    "hf-Qwen3-8B-3": {"eval_average": 68.16, "reasoning_length": 2223},
    "hf-Qwen3-8B-3 no thinking": {"eval_average": 52.03, "reasoning_length": 315.3333333},
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-7B-deepseek-configs": {"eval_average": 49.99, "reasoning_length": 1581},
    "hf-OpenThinker3-7B-3": {"eval_average": 55.9, "reasoning_length": 2450},
    "hf-DeepSeek-R1-Distill-Qwen-7B-3": {"eval_average": 50.29, "reasoning_length": 1581},
    "Qwen 3 8B with reasoning": {"eval_average": 68.03, "reasoning_length": 2223},
    "Openthinker 3 7B": {"eval_average": 55.17, "reasoning_length": 3982.5},
    "DeepSeek R1 Distill Qwen 7B": {"eval_average": 50.1, "reasoning_length": 1581},
    "Nemotron Nano 9B V2": {"eval_average": 64.52, "reasoning_length": 1048.250},
    "olmo3-instruct-SFT": {"eval_average": 41.03, "reasoning_length": 239.500},
    "olmo3-instruct-DPO": {"eval_average": 47.5, "reasoning_length": 257.500},
    "olmo-3-thinking": {"eval_average": 67.22, "reasoning_length": 2681.333},
    "hf-NVIDIA-Nemotron-Nano-9B-v2-2": {"eval_average": 64.38, "reasoning_length": 1048.250}
}"""

data = parse_score_and_length_tsv(LENGTH_TSV_PATH, SCORE_TSV_PATH)

# Check if we have any data
if not data:
    print("\n⚠️  ERROR: No valid data to plot!")
    print("All columns have missing values in at least one row.")
    print("Consider using a dataset with more complete data or relaxing the filtering criteria.\n")
    exit(1)

# Extract data for plotting
model_names = list(data.keys())
reasoning_lengths = [data[model]["reasoning_length"] for model in model_names]
eval_averages = [data[model]["eval_average"] for model in model_names]

# Assign colors based on model name
colors = []
edge_colors = []
for model in model_names:
    if "olmo" in model.lower():
        colors.append(Ai2_colors["Pink"])
        edge_colors.append(Ai2_colors["Purple"])
    else:
        colors.append(Ai2_colors["Deep sea"])
        edge_colors.append(Ai2_colors["Dark teal"])

# Identify Pareto frontier
# A point is on the Pareto frontier if no other point has both lower reasoning_length AND higher eval_average
def is_pareto_efficient(point_idx, lengths, averages):
    """Check if a point is on the Pareto frontier"""
    for i in range(len(lengths)):
        if i != point_idx:
            # If another point has lower length AND higher average, current point is dominated
            if lengths[i] <= lengths[point_idx] and averages[i] >= averages[point_idx]:
                # If strictly better in at least one dimension
                if lengths[i] < lengths[point_idx] or averages[i] > averages[point_idx]:
                    return False
    return True

pareto_indices = []
for i in range(len(model_names)):
    if is_pareto_efficient(i, reasoning_lengths, eval_averages):
        pareto_indices.append(i)

# Sort Pareto points by reasoning length for drawing the line
pareto_points = [(reasoning_lengths[i], eval_averages[i], i) for i in pareto_indices]
pareto_points.sort(key=lambda x: x[0])  # Sort by reasoning length


def labels_overlap(text_objects, renderer, padding_px=2):
    """Return True when any label bounding boxes overlap after a small padding."""
    if not text_objects:
        return False

    padded_bboxes = []
    for text in text_objects:
        bbox = text.get_window_extent(renderer=renderer)
        width = max(bbox.width, 1)
        height = max(bbox.height, 1)
        scale_x = 1 + (padding_px * 2) / width
        scale_y = 1 + (padding_px * 2) / height
        padded_bboxes.append(bbox.expanded(scale_x, scale_y))

    for i in range(len(padded_bboxes)):
        for j in range(i + 1, len(padded_bboxes)):
            if padded_bboxes[i].overlaps(padded_bboxes[j]):
                return True
    return False


def create_plot(use_log_scale=True):
    """Create a single plot with either log or linear scale"""
    # Set style
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(Ai2_colors["Cream"])
    ax.set_facecolor(Ai2_colors["Cream"])
    
    # Create scatter plot
    plt.scatter(reasoning_lengths, eval_averages, s=150, alpha=0.7, 
                c=colors, edgecolors=edge_colors, linewidth=2)
    
    # Draw Pareto frontier line
    if len(pareto_points) > 1:
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        plt.plot(pareto_x, pareto_y, linestyle='--', linewidth=2.5, 
                 color=Ai2_colors["Purple"], alpha=0.8, label='Pareto Frontier', zorder=5)
        
        # Highlight Pareto frontier points with a subtle glow
        for px, py in zip(pareto_x, pareto_y):
            plt.scatter(px, py, s=300, alpha=0.15, color=Ai2_colors["Purple"], zorder=4)
    
    # Add labels for each point with adjustText to avoid overlap
    texts = []
    y_offset = (max(eval_averages) - min(eval_averages)) * 0.01
    if y_offset < 0.25:
        y_offset = 0.25
    
    for i, model in enumerate(model_names):
        x = reasoning_lengths[i]
        y = eval_averages[i] + y_offset
        text = ax.text(
            x,
            y,
            model,
            fontsize=8,
            alpha=0.9,
            ha='center',
            va='bottom',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='none',
                alpha=0.7,
            ),
        )
        texts.append(text)
    
    fig.canvas.draw()  # Needed so text bounding boxes are available
    renderer = fig.canvas.get_renderer()
    
    # Adjust text positions only if labels collide
    if labels_overlap(texts, renderer):
        adjust_text(
            texts,
            expand_points=(1.05, 1.2),
            expand_text=(1.05, 1.2),
            force_points=(0.05, 0.1),
            force_text=(0.05, 0.1),
            lim=50,  # Reduced from 200 for faster performance
            only_move={'points': 'xy', 'text': 'y'},
            avoid_self=False,
            arrowprops=dict(
                arrowstyle='->',
                color='gray',
                lw=0.5,
                alpha=0.6,
                shrinkA=5,
                shrinkB=5,
            ),
        )
    
    # Set scale on x-axis
    if use_log_scale:
        plt.xscale('log')
        scale_label = 'log scale'
        filename = 'length_pareto_plot_log.png'
    else:
        scale_label = 'linear scale'
        filename = 'length_pareto_plot_linear.png'
    
    # Labels and title
    plt.xlabel(f'Reasoning Length ({scale_label})', fontsize=12, fontweight='bold', color=Ai2_colors["Dark teal"])
    plt.ylabel('Eval Average', fontsize=12, fontweight='bold', color=Ai2_colors["Dark teal"])
    plt.title('Model Performance: Eval Average vs Reasoning Length', fontsize=14, fontweight='bold', 
              color=Ai2_colors["Dark teal"], pad=20)
    
    # Grid styling
    plt.grid(True, alpha=0.3, color=Ai2_colors["Light teal"])
    
    # Add legend
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save and show
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=Ai2_colors["Cream"])
    print(f"Saved plot to {filename}")
    plt.show()


# Create both plots
print("\nCreating plot with logarithmic x-axis...")
create_plot(use_log_scale=True)

print("\nCreating plot with linear x-axis...")
create_plot(use_log_scale=False)
