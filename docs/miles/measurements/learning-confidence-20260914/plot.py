import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
d = json.loads((root / "recovered-evidence.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
colors = {"Core": "#007c91", "Megatron": "#cb6817"}
ax = axes[0, 0]
c = d["runs"]["light_core200"]["native_curve"]
m = d["light_megatron_reference"]["native_eval"]["correct_by_step"]
ax.plot([x["completed_updates"] for x in c], [100 * x["correct"] / 128 for x in c], label="Core", color=colors["Core"])
ax.plot([int(x) for x in m], [100 * x / 128 for x in m.values()], label="Megatron", color=colors["Megatron"])
ax.set(
    title="Light SFT: held-out GSM8K (128 questions)",
    xlabel="Completed optimizer updates",
    ylabel="Correct (%)",
    ylim=(0, 100),
)
ax.legend()
ax = axes[0, 1]
for name, label in [("core500", "Core"), ("megatron500", "Megatron")]:
    c = d["runs"][name]["curve"]
    ax.plot([x["completed_updates"] for x in c], [100 * x["score"] for x in c], label=label, color=colors[label])
ax.set(
    title="Heavy SFT: held-out GSM8K (128 questions)",
    xlabel="Completed optimizer updates",
    ylabel="Correct (%)",
    ylim=(65, 95),
)
ax.legend()
ax = axes[1, 0]
for i, (label, before, after) in enumerate([("Core", 229, 268), ("Megatron", 230, 274)]):
    ax.plot(
        [0, 200],
        [100 * before / 1319, 100 * after / 1319],
        "-o",
        label=f"{label}: {before} → {after}",
        color=colors[label],
    )
    ax.annotate(
        f"+{100 * (after - before) / 1319:.2f} pp",
        (200, 100 * after / 1319),
        xytext=(-62, 8 if i else -16),
        textcoords="offset points",
        color=colors[label],
    )
ax.set(
    title="Light SFT: separate raw-prompt test (1,319)",
    xlabel="Completed optimizer updates",
    ylabel="Correct (%)",
    xticks=[0, 200],
    xlim=(-10, 245),
    ylim=(15, 23),
)
ax.legend(loc="upper left")
ax = axes[1, 1]
for name, label in [("core500", "Core"), ("megatron500", "Megatron")]:
    c = d["runs"][name]["curve"]
    ax.plot([x["completed_updates"] for x in c], [100 * x["capped"] for x in c], label=label, color=colors[label])
ax.set(
    title="Heavy SFT: held-out answers hitting 4K cap",
    xlabel="Completed optimizer updates",
    ylabel="Capped (%)",
    ylim=(0, 50),
)
ax.legend()
for ax in axes.flat:
    ax.grid(alpha=0.2)
fig.suptitle("Existing learning evidence — different evaluation protocols remain separate", fontsize=15)
fig.savefig(root / "learning-curves.png", dpi=150)
fig.savefig(root / "learning-curves.svg")
print(root / "learning-curves.png")
