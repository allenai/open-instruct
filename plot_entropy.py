import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

df = pd.read_csv("calc-likelihoods/entropy_results-100.csv")  # Update filename as needed

def extract_step(model_name):
    """Extract step number from model name."""
    if 'allenai/Olmo-3-1025-7B' in model_name:
        return 0
    
    # Extract step number from checkpoint path
    match = re.search(r'step_(\d+)', model_name)
    if match:
        return int(match.group(1))
    
    return None


# Add step column
df['step'] = df['model'].apply(extract_step)

# Remove rows where step couldn't be extracted
df = df.dropna(subset=['step'])

# Define colors and line styles for each category
domains = ['math', 'code', 'ifeval']
colors = {'math': '#1f77b4', 'code': '#ff7f0e', 'ifeval': '#2ca02c'}
line_styles = {True: '-', False: '--'}  # solid for in-distribution, dashed for out-of-distribution

# ============================================================================
# 1. COMBINED PLOT (all curves together)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

for domain in domains:
    for in_dist in [True, False]:
        mask = (df['domain'] == domain) & (df['in_distribution'] == in_dist)
        subset = df[mask]
        
        if len(subset) == 0:
            continue
        
        grouped = subset.groupby('step')['avg_entropy'].agg(['mean', 'min', 'max'])
        grouped = grouped.sort_index()
        
        steps = grouped.index.values
        mean_vals = grouped['mean'].values
        min_vals = grouped['min'].values
        max_vals = grouped['max'].values
        
        dist_label = 'in-dist' if in_dist else 'out-of-dist'
        label = f'{domain} ({dist_label})'
        
        color = colors[domain]
        linestyle = line_styles[in_dist]
        ax.plot(steps, mean_vals, color=color, linestyle=linestyle, 
                linewidth=2, label=label, marker='o', markersize=4)
        
        ax.fill_between(steps, min_vals, max_vals, 
                        color=color, alpha=0.15)

ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Entropy', fontsize=12, fontweight='bold')
ax.set_title('Average Entropy over Training Steps', fontsize=14, fontweight='bold')
ax.legend(loc='best', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('calc-likelihoods/plots/entropy_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('calc-likelihoods/plots/entropy_plot.pdf', bbox_inches='tight')
print("✓ Combined plot saved")
plt.close()

# ============================================================================
# 2. INDIVIDUAL PLOTS (one for each domain × distribution combination)
# ============================================================================
for domain in domains:
    for in_dist in [True, False]:
        mask = (df['domain'] == domain) & (df['in_distribution'] == in_dist)
        subset = df[mask]
        
        if len(subset) == 0:
            continue
        
        grouped = subset.groupby('step')['avg_entropy'].agg(['mean', 'min', 'max'])
        grouped = grouped.sort_index()
        
        steps = grouped.index.values
        mean_vals = grouped['mean'].values
        min_vals = grouped['min'].values
        max_vals = grouped['max'].values
        
        # Create individual plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        dist_label = 'in-dist' if in_dist else 'out-of-dist'
        color = colors[domain]
        linestyle = line_styles[in_dist]
        
        ax.plot(steps, mean_vals, color=color, linestyle=linestyle, 
                linewidth=2.5, marker='o', markersize=5)
        ax.fill_between(steps, min_vals, max_vals, color=color, alpha=0.2)
        
        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Entropy', fontsize=12, fontweight='bold')
        ax.set_title(f'{domain.upper()} - {dist_label.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save with descriptive filename
        filename = f'calc-likelihoods/plots/entropy_individual_{domain}_{dist_label.replace("-", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"✓ Individual plot saved: {filename}")
        plt.close()

# ============================================================================
# 3. DOMAIN-SPECIFIC PLOTS (one plot per domain, showing both distributions)
# ============================================================================
# Define distinct colors for in-dist vs out-of-dist within each domain
dist_colors = {
    'math': {True: '#1f77b4', False: '#aec7e8'},  # darker blue for in-dist, lighter for out-of-dist
    'code': {True: '#ff7f0e', False: '#ffbb78'},  # darker orange, lighter orange
    'ifeval': {True: '#2ca02c', False: '#98df8a'}  # darker green, lighter green
}

for domain in domains:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for in_dist in [True, False]:
        mask = (df['domain'] == domain) & (df['in_distribution'] == in_dist)
        subset = df[mask]
        
        if len(subset) == 0:
            continue
        
        grouped = subset.groupby('step')['avg_entropy'].agg(['mean', 'min', 'max'])
        grouped = grouped.sort_index()
        
        steps = grouped.index.values
        mean_vals = grouped['mean'].values
        min_vals = grouped['min'].values
        max_vals = grouped['max'].values
        
        dist_label = 'in-dist' if in_dist else 'out-of-dist'
        color = dist_colors[domain][in_dist]
        linestyle = line_styles[in_dist]
        
        # Plot mean line with matching color
        ax.plot(steps, mean_vals, color=color, linestyle=linestyle, 
                linewidth=2.5, label=dist_label, marker='o', markersize=5)
        
        # Plot shaded region with matching color and higher alpha for better visibility
        ax.fill_between(steps, min_vals, max_vals, color=color, alpha=0.3)
    
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Entropy', fontsize=12, fontweight='bold')
    ax.set_title(f'{domain.upper()} - Average Entropy over Training', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    filename = f'calc-likelihoods/plots/entropy_domain_{domain}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Domain plot saved: {filename}")
    plt.close()

print("\n✓ All entropy plots generated successfully!")