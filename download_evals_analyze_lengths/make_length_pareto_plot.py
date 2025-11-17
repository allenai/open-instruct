import argparse
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from pathlib import Path

# Import the category definitions from download_and_analyze
from download_and_analyze import DEFAULT_ALIASES

DEFAULT_LENGTH_TSV_PATH = "/weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/7B/median_lengths.tsv"
DEFAULT_SCORE_TSV_PATH = "/weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/7B/primary_scores.tsv"
DEFAULT_OUTPUT_DIR = "/weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/plots"

Ai2_colors = {
       "Dark teal": "#0a3235",
       "Cream": "#faf2e9",
       "Light teal": "#105257",
       "Pink": "#f0529c",
       "Purple": "#b11be8",
       "Deep sea": "#0fcb8c"
}

def parse_score_and_length_tsv_by_category(length_tsv_path, score_tsv_path, category_name, category_aliases):
    """Parse TSV files and average only the columns belonging to the specified category."""
    # Parse length TSV - average across category eval columns
    length_data = {}
    with open(length_tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
        
        if not rows:
            return {}
        
        # Get all column names except model
        all_cols = [k for k in rows[0].keys() if k not in ['model', 'model_name']]
        
        # Filter to only category aliases that exist in the TSV
        category_cols = [col for col in all_cols if col in category_aliases]
        
        if not category_cols:
            print(f"[{category_name} - LENGTH] No matching columns found in TSV")
            return {}
        
        # Identify columns with any missing values (within this category)
        valid_cols = []
        excluded_cols = {}
        
        for col in category_cols:
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
            print(f"[{category_name} - LENGTH] Excluding {len(excluded_cols)} column(s) with missing values:")
            for col in sorted(excluded_cols.keys()):
                models_str = ', '.join(excluded_cols[col])
                print(f"  - {col}")
                print(f"    Missing in: {models_str}")
        
        print(f"[{category_name} - LENGTH] Using {len(valid_cols)} column(s) for averaging")
        
        # Compute averages using only valid columns in this category
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
    
    # Parse score TSV - average across category eval columns
    score_data = {}
    with open(score_tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
        
        if not rows:
            return {}
        
        all_cols = [k for k in rows[0].keys() if k not in ['model', 'model_name']]
        category_cols = [col for col in all_cols if col in category_aliases]
        
        if not category_cols:
            print(f"[{category_name} - SCORE] No matching columns found in TSV")
            return {}
        
        valid_cols = []
        excluded_cols = {}
        
        for col in category_cols:
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
            print(f"[{category_name} - SCORE] Excluding {len(excluded_cols)} column(s) with missing values:")
            for col in sorted(excluded_cols.keys()):
                models_str = ', '.join(excluded_cols[col])
                print(f"  - {col}")
                print(f"    Missing in: {models_str}")
        
        print(f"[{category_name} - SCORE] Using {len(valid_cols)} column(s) for averaging\n")
        
        # Compute averages using only valid columns in this category
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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Pareto plots for eval scores vs reasoning lengths by category."
    )
    parser.add_argument(
        "--length-tsv",
        default=DEFAULT_LENGTH_TSV_PATH,
        help=f"Path to the median lengths TSV file (default: {DEFAULT_LENGTH_TSV_PATH})",
    )
    parser.add_argument(
        "--score-tsv",
        default=DEFAULT_SCORE_TSV_PATH,
        help=f"Path to the primary scores TSV file (default: {DEFAULT_SCORE_TSV_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where plots will be saved (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--separate-label-legend",
        action="store_true",
        help="Use numbered labels on points with a separate legend mapping numbers to model names (useful for crowded plots)",
    )
    return parser.parse_args()


def create_plot(data, category_name, use_log_scale=True, output_dir=DEFAULT_OUTPUT_DIR, use_numbered_labels=False):
    """Create a single plot with either log or linear scale for a specific category"""
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
    def is_pareto_efficient(point_idx, lengths, averages):
        """Check if a point is on the Pareto frontier"""
        for i in range(len(lengths)):
            if i != point_idx:
                if lengths[i] <= lengths[point_idx] and averages[i] >= averages[point_idx]:
                    if lengths[i] < lengths[point_idx] or averages[i] > averages[point_idx]:
                        return False
        return True
    
    pareto_indices = []
    for i in range(len(model_names)):
        if is_pareto_efficient(i, reasoning_lengths, eval_averages):
            pareto_indices.append(i)
    
    # Sort Pareto points by reasoning length for drawing the line
    pareto_points = [(reasoning_lengths[i], eval_averages[i], i) for i in pareto_indices]
    pareto_points.sort(key=lambda x: x[0])
    
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
    
    # Add labels for each point
    if use_numbered_labels:
        # Use numbered labels on the plot with a separate legend
        for i, model in enumerate(model_names):
            x = reasoning_lengths[i]
            y = eval_averages[i]
            # Add small numbered label on each point
            ax.text(
                x,
                y,
                str(i + 1),
                fontsize=7,
                fontweight='bold',
                ha='center',
                va='center',
                color='white',
                zorder=10
            )
        
        # Create legend mapping numbers to model names
        # Split into multiple columns if there are many models
        legend_text = []
        for i, model in enumerate(model_names):
            legend_text.append(f"{i + 1}. {model}")
        
        # Add legend box outside the plot area
        legend_str = '\n'.join(legend_text)
        plt.subplots_adjust(right=0.7)  # Make room for legend
        ax.text(
            1.05, 0.5,
            legend_str,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment='center',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor=Ai2_colors["Light teal"],
                alpha=0.9,
            ),
        )
    else:
        # Use traditional text labels with adjustText
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
        filename = f'{category_name.replace(" ", "_").replace("&", "and")}_pareto_log.png'
    else:
        scale_label = 'linear scale'
        filename = f'{category_name.replace(" ", "_").replace("&", "and")}_pareto_linear.png'
    
    # Labels and title
    plt.xlabel(f'Reasoning Length ({scale_label})', fontsize=12, fontweight='bold', color=Ai2_colors["Dark teal"])
    plt.ylabel('Eval Average', fontsize=12, fontweight='bold', color=Ai2_colors["Dark teal"])
    plt.title(f'{category_name}: Eval Average vs Reasoning Length', fontsize=14, fontweight='bold', 
              color=Ai2_colors["Dark teal"], pad=20)
    
    # Grid styling
    plt.grid(True, alpha=0.3, color=Ai2_colors["Light teal"])
    
    # Add legend
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Create output directory and save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    full_path = output_path / filename
    plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor=Ai2_colors["Cream"])
    print(f"Saved plot to {full_path}")
    plt.close()


# Main execution: loop over categories and create plots
if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("GENERATING CATEGORY-LEVEL PARETO PLOTS")
    print("="*70)
    print(f"Length TSV: {args.length_tsv}")
    print(f"Score TSV: {args.score_tsv}")
    print(f"Output Directory: {args.output_dir}")
    print("="*70)
    
    for category_name, category_aliases in DEFAULT_ALIASES.items():
        print(f"\n{'='*70}")
        print(f"Processing Category: {category_name}")
        print(f"{'='*70}")
        
        # Parse data for this category
        data = parse_score_and_length_tsv_by_category(
            args.length_tsv, 
            args.score_tsv, 
            category_name, 
            category_aliases
        )
        
        # Check if we have any data
        if not data:
            print(f"⚠️  WARNING: No valid data for {category_name}. Skipping plots.\n")
            continue
        
        print(f"\nFound data for {len(data)} models in {category_name}")
        
        # Create both log and linear plots for this category
        print(f"Creating logarithmic plot for {category_name}...")
        create_plot(data, category_name, use_log_scale=True, output_dir=args.output_dir, use_numbered_labels=args.separate_label_legend)
        
        print(f"Creating linear plot for {category_name}...")
        create_plot(data, category_name, use_log_scale=False, output_dir=args.output_dir, use_numbered_labels=args.separate_label_legend)
    
    # Generate overall plot across all categories
    print(f"\n{'='*70}")
    print("Processing Overall (All Categories)")
    print(f"{'='*70}")
    
    # Collect all aliases from all categories
    all_aliases = []
    for category_aliases in DEFAULT_ALIASES.values():
        all_aliases.extend(category_aliases)
    
    # Parse data for all aliases
    overall_data = parse_score_and_length_tsv_by_category(
        args.length_tsv, 
        args.score_tsv, 
        "Overall", 
        all_aliases
    )
    
    if overall_data:
        print(f"\nFound data for {len(overall_data)} models in Overall")
        
        print(f"Creating logarithmic plot for Overall...")
        create_plot(overall_data, "Overall", use_log_scale=True, output_dir=args.output_dir, use_numbered_labels=args.separate_label_legend)
        
        print(f"Creating linear plot for Overall...")
        create_plot(overall_data, "Overall", use_log_scale=False, output_dir=args.output_dir, use_numbered_labels=args.separate_label_legend)
    else:
        print(f"⚠️  WARNING: No valid data for Overall. Skipping plots.\n")
    
    print("\n" + "="*70)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
