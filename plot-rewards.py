import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# ── Settings ──────────────────────────────────────────────────────────
PROJECT = "ai2-llm/open_instruct_internal"
METRIC = "objective/verifiable_reward"
MAX_STEPS = 250

runs_info = {
    "grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615": "Frozen",
    "grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026": "Unfrozen",
}

# ── Publication styling ───────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Half-page width ~ 3.3 in for most venues (NeurIPS/ICML/ACL column width)
fig, ax = plt.subplots(figsize=(3.3, 2.2))

colors = {"Frozen": "#1f77b4", "Unfrozen": "#d62728"}

api = wandb.Api()


def ema_with_bands(values, alpha=0.05, window=15):
    """Return EMA line and a rolling-std band around it."""
    ema = pd.Series(values).ewm(alpha=alpha, adjust=True).mean().values
    # Rolling std of the residuals for the shaded band
    residuals = values - ema
    rolling_std = (
        pd.Series(residuals)
        .rolling(window=window, center=True, min_periods=1)
        .std()
        .values
    )
    return ema, rolling_std


for run_name, label in runs_info.items():
    runs = api.runs(PROJECT, filters={"display_name": run_name})
    if not runs:
        print(f"Warning: run '{run_name}' not found, skipping.")
        continue
    run = runs[0]

    hist = run.history(keys=[METRIC], samples=50000)
    hist = hist.dropna(subset=[METRIC]).reset_index(drop=True)

    steps = np.arange(len(hist))
    values = hist[METRIC].values

    mask = steps < MAX_STEPS
    steps, values = steps[mask], values[mask]

    ema, std = ema_with_bands(values, alpha=0.05, window=20)
    c = colors[label]

    ax.fill_between(
        steps, ema - std, ema + std, color=c, alpha=0.18, edgecolor="none"
    )
    ax.plot(steps, ema, label=label, color=c, linewidth=1.5)

ax.set_xlabel("Step")
ax.set_ylabel("Verifiable Reward")
ax.set_xlim(0, MAX_STEPS)
ax.legend(frameon=True, fancybox=False, edgecolor="0.8", loc="upper left")
ax.grid(True, alpha=0.15, linewidth=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.3)
fig.savefig("reward_curves.png")
fig.savefig("reward_curves.pdf")
print("Saved reward_curves.png and reward_curves.pdf")
