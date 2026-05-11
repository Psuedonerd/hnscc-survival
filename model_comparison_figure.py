import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

df = pd.read_csv(
    "CPTAC/results/full_model_comparison_20260214_155709/full_model_comparison.csv"
)

model_order = ["CoxPH", "CoxNet", "Ridge Cox", "GBS", "RSF"]
cond_order = ["Clinical", "Omics", "Combined"]
metric_order = ["Train_C", "Test_C"]
row_titles = ["Train C-index (TCGA)", "Test C-index (CPTAC)"]
col_titles = ["Clinical", "Omics", "Combined"]

model_colors = {
    "CoxPH": "#EF553B",
    "CoxNet": "#636EFA",
    "Ridge Cox": "#00CC96",
    "GBS": "#FFA15A",
    "RSF": "#AB63FA",
}

fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey="row")

for row, metric in enumerate(metric_order):
    for col, cond in enumerate(cond_order):
        ax = axes[row, col]

        subset = df[df["Condition"] == cond].set_index("Model").loc[model_order]
        vals = subset[metric].values
        x = np.arange(len(model_order))

        colors = [model_colors[m] for m in model_order]
        ax.scatter(x, vals, s=90, c=colors, zorder=3, edgecolors="white", linewidths=0.8)

        for i, v in enumerate(vals):
            ax.annotate(
                f"{v:.3f}", (x[i], v),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=7.5, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(model_order, fontsize=9, rotation=45, ha="right")
        ax.set_xlim(-0.6, len(model_order) - 0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        if row == 0:
            ax.set_title(col_titles[col], fontsize=12, fontweight="bold", pad=10)
        if col == 0:
            ax.set_ylabel(row_titles[row], fontsize=11)

axes[0, 0].set_ylim(0.50, 0.95)
axes[1, 0].set_ylim(0.44, 0.72)

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=model_colors[m],
           markersize=9, markeredgecolor="white", markeredgewidth=0.5, label=m)
    for m in model_order
]
fig.legend(
    handles=legend_handles, loc="center right",
    fontsize=9, frameon=True, title="Model", title_fontsize=10,
    bbox_to_anchor=(1.01, 0.5),
)

fig.suptitle(
    "Train and Test Set C-indices Across Survival Models",
    fontsize=14, fontweight="bold", y=1.01,
)

plt.tight_layout()
fig.subplots_adjust(right=0.88)
plt.savefig("CPTAC/results/model_comparison_figure.png", dpi=800, bbox_inches="tight")
plt.savefig("CPTAC/results/model_comparison_figure.pdf", bbox_inches="tight")
print("Saved to CPTAC/results/model_comparison_figure.{png,pdf}")
