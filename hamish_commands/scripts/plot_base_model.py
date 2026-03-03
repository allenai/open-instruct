import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib as mpl

# =========================
# Data (with 1000 + 1900 for Our SFT)
# =========================

models = {
    "On Policy SFT": {
        "steps":  np.array([0, 100, 200, 300, 400, 500, 600]),
        "health": np.array([37.51, 37.21, 36.79, 38.10, 37.00, 37.48, 36.30]),
        "sqav2":  np.array([73.40, 78.40, 79.90, 80.80, 82.20, 82.10, 80.60]),
        "drb":    np.array([39.40, 38.91, 38.87, 38.45, 38.77, 38.99, 38.69]),
    },
    "Our SFT": {
        "steps":  np.array([0, 100, 200, 300, 400, 500, 600, 1000, 1900]),
        "health": np.array([37.50, 36.55, 37.8, 37.17, 38.71, 39.40, 38.90, 42.6, 50.2]),
        "sqav2":  np.array([72.40, 78.30, 79.9, 82.00, 84.60, 82.50, 84.90, 86.2, 86.80]),
        "drb":    np.array([38.70, 38.97, 37.60, 39.04, 39.35, 38.32, 38.70, 41.38, 43.42]),
    },
    "Undertrained SFT": {
        "steps":  np.array([0, 100, 200, 300, 400, 500, 600]),
        "health": np.array([29.50, 29.83, 29.70, 29.62, 31.20, 32.10, 34.10]),
        "sqav2":  np.array([75.70, 80.20, 80.30, 81.90, 81.30, 82.10, 82.20]),
        "drb":    np.array([33.51, 34.25, 34.44, 33.40, 35.06, 34.73, 35.25]),
    },
    "No SFT": {
        "steps":  np.array([0, 100, 200, 300, 400, 500, 600]),
        "health": np.array([5.91, 9.80, 7.80, 9.70, 12.40, 13.50, 15.90]),
        "sqav2":  np.array([57.2, 61.70, 69.60, 68.80, 70.50, 74.63, 76.80]),
        "drb":    np.array([17.68, 16.30, 18.43, 20.82, 22.35, 22.44, 22.96]),
    },
}

palette = {
    "On Policy SFT":    "#B11BE8",
    "Our SFT":          "#F0529C",
    "Undertrained SFT": "#0FCBBC",
    "No SFT":           "#105257",
}

metrics = [
    ("health", "Healthbench"),
    ("sqav2",  "SQAv2"),
    ("drb",    "DRB"),
]

# =========================
# X-transform so that:
#   600 -> near middle,
#   1000 & 1900 are further right and NOT evenly spaced
#   (e.g. 600 ~ 1.0, 1000 ~ 2.2, 1900 ~ 2.9)
# =========================
def transform_steps(s):
    """
    Tighter fake spacing:
    - 0..600   -> [0..1]
    - 1000     -> ~1.4
    - 1900     -> ~1.8
    """
    s = np.asarray(s)
    x = np.zeros_like(s, dtype=float)

    # early region: linear 0..600 -> 0..1
    mask_early = s <= 600
    x[mask_early] = s[mask_early] / 600.0 * 1.0

    # anchors
    x_1000 = 1.4
    x_1900 = 1.8

    mask_1000 = (s == 1000)
    mask_1900 = (s == 1900)
    x[mask_1000] = x_1000
    x[mask_1900] = x_1900

    # any in-between late points if they ever exist
    mask_late = (s > 600) & (~mask_1000) & (~mask_1900)
    if np.any(mask_late):
        x[mask_late] = x_1000 + (s[mask_late] - 1000) / (1900 - 1000) * (x_1900 - x_1000)

    return x


# Precompute for ticks
ticks_values = [0, 200, 400, 600, 1000, 1900]
ticks_locs = transform_steps(np.array(ticks_values))

# =========================
# Styling
# =========================

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#b0b0b0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.6
plt.rcParams["grid.alpha"] = 0.7

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
})

# =========================
# Figure
# =========================
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.24, top=0.90, wspace=0.33)

# full shading: everything AFTER 600
shade_start = transform_steps(np.array([600]))[0]

# set nicer compact x-limits
xmax = 1.9

for ax, (metric_key, title) in zip(axes, metrics):

    # shade region after step 600
    ax.axvspan(shade_start, xmax, color="#F4F4F4", zorder=0)

    # plot each model
    for name, dct in models.items():
        s = dct["steps"]
        v = dct[metric_key]
        x = transform_steps(s)

        # Solid <= 600
        mask_solid = s <= 600
        if np.any(mask_solid):
            ax.plot(
                x[mask_solid],
                v[mask_solid],
                marker="o",
                color=palette[name],
                label=name,
                linestyle="-",
                zorder=10
            )

        # Dashed >= 600
        mask_dashed = s >= 600
        if np.sum(mask_dashed) > 1:
            ax.plot(
                x[mask_dashed],
                v[mask_dashed],
                marker="o",
                color=palette[name],
                linestyle="--",
                zorder=10
            )

    # ticks at transformed positions
    ax.set_xticks(ticks_locs)
    ax.set_xticklabels([str(t) for t in ticks_values])

    ax.set_xlim(0.0, xmax)
    ax.set_title(title)
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("RL training steps")

    # global fixed ylim (shared across all subplots)
    # ax.set_ylim(0, 90)

# Shared legend under plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, 0.02),
)

# plt.show()
plt.savefig("base_model_comparison.pdf", bbox_inches="tight")