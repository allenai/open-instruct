import matplotlib.pyplot as plt
import numpy as np

# Data
methods = [
    'SFT + Offline DPO (length-normalized) (126k episodes)',
    'SFT + Online DPO (126k episodes)',
    'SFT + PPO (126k episodes)',
]

rm_rates = {
    'vwxyzjn/reward_modeling__allenai_open_instruct_dev': [45.20, 56.20, 59.60],
    'allenai/llama-3-tulu-2-8b-uf-mean-rm': [61.20, 64.00, 63.40],
    "reward_modeling__1__1726175049": [63.00, 63.00, 58.40],
}

x = np.arange(len(methods))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
axes = []
for i, (rm, rates) in enumerate(rm_rates.items()):
    rects = ax.bar(x + i * width, rates, width, label=rm)
    axes.append(rects)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Agreement Rate (%)', fontsize=12)
ax.set_title('Agreement Rates w/ GPT 4 judgement for Different Reward Models and Training Methods', fontsize=14)
ax.set_xticks(x + width)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.legend()

ax.set_ylim(0, 100)  # Set y-axis limit from 0 to 100%

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

for rects in axes:
    autolabel(rects)

fig.tight_layout()

plt.savefig('agreement_rates_comparison.png')
plt.close()

print("Plot saved as 'agreement_rates_comparison.png'")