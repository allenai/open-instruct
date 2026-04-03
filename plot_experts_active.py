import matplotlib.pyplot as plt

experts_active = [1, 2, 3, 4, 5]
overall = [31.1, 38.8, 36.8, 48.2, 48.8]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(experts_active, overall, marker='o', linewidth=2.5, markersize=9, color='#1f77b4')

ax.set_xlabel('Experts Active', fontsize=16)
ax.set_ylabel('Overall Average', fontsize=16)
ax.set_title('FlexOLMo 5x7B: Overall Performance vs. Active Experts', fontsize=17)
ax.set_xticks(experts_active)
ax.set_ylim(25, 55)
ax.tick_params(axis='both', labelsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experts_vs_overall.png', dpi=150)
plt.show()