import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- 1. Load the pivot table -----------------------------------
pivot = pd.read_csv(
    "likelihoods_per_token.csv",
    index_col=[0, 1],          # 0 → ex_id, 1 → c_id
)

# (optional) give the index levels names again if you like
pivot.index.set_names(["ex_id", "c_id"], inplace=True)

# ---- 2. Make sure the step columns are in numeric order --------
step_numbers = [int(c.split("_")[1]) for c in pivot.columns]
pivot = pivot.loc[:, pivot.columns[np.argsort(step_numbers)]]

# ---- 3. Plot (same code as before) -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for (ex_id, c_id), series in pivot.iterrows():
    ax.plot(
        step_numbers,
        series.values,
        marker="o",
        linewidth=1.4,
        label=f"prompt {ex_id} – comp {c_id}",
    )

ax.set_xlabel("Training step")
ax.set_ylabel("Average log-likelihood per token (nats)")
ax.set_title("Length-normalized likelihood through training")
ax.grid(True, linewidth=0.3)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
plt.tight_layout()
plt.show()

