#!/usr/bin/env python3
"""
Dot plot of ECM and protease pathway enrichment from the 74-gene signature.
Loads saved Enrichr results, filters to ECM/protease terms, plots.
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

RESULTS_DIR = Path(
    "/restricted/projectnb/montilab-p/personal/pnarnur/projects/hnscc_genomics_misc/"
    "CPTAC/results/enrichment_v2_20260103_012150"
)

PATHWAY_DBS = {
    "GO_BP_results.csv": "GO BP",
    "GO_MF_results.csv": "GO MF",
    "GO_CC_results.csv": "GO CC",
    "Reactome_results.csv": "Reactome",
    "Hallmark_results.csv": "Hallmark",
    "WikiPathways_results.csv": "WikiPathways",
    "KEGG_results.csv": "KEGG",
}

ECM_PROTEASE_PATTERN = (
    r"(?i)extracellular matrix|collagen|matrix metallo|focal adhes|integrin|"
    r"serine.type.*(peptidase|endopeptidase)|zymogen|plasminogen|"
    r"protease|proteolysis|peptidase|endopeptidase|"
    r"procollagen|laminin|fibronectin|ECM|"
    r"epithelial mesenchymal transition"
)


def clean_label(term):
    term = re.sub(r"\s*R-HSA-\d+", "", term)
    term = re.sub(r"\s*WP\d+", "", term)
    term = re.sub(r"\s*\(GO:\d+\)", "", term)
    return term.strip()


def main():
    frames = []
    for fname, label in PATHWAY_DBS.items():
        df = pd.read_csv(RESULTS_DIR / fname)
        df["Database"] = label
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    mask = (
        combined["Term"].str.contains(ECM_PROTEASE_PATTERN, regex=True, na=False)
        & (combined["Adjusted P-value"] <= 0.25)
    )
    filtered = combined[mask].sort_values(["Adjusted P-value", "P-value"])
    exclude = (
        r"(?i)Osteogenesis Imperfecta"
        r"|^Serine-Type Peptidase Activity"
        r"|^Endopeptidase Activity"
        r"|^Regulation Of Plasminogen Activation"
        r"|^Negative Regulation Of Plasminogen Activation"
        r"|^Collagen Formation"
    )
    filtered = filtered[~filtered["Term"].str.contains(exclude, regex=True, na=False)]
    filtered = filtered.head(15).copy()

    nums = filtered["Overlap"].str.split("/", expand=True).astype(int)
    filtered["Count"] = nums[0]
    filtered["Ratio"] = nums[0] / nums[1]
    filtered["Label"] = filtered.apply(
        lambda r: f"{clean_label(r['Term'])} ({r['Database']})", axis=1
    )
    filtered["FDR"] = filtered["Adjusted P-value"]
    filtered = filtered.sort_values("FDR", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 6))

    sc = ax.scatter(
        filtered["Ratio"],
        range(len(filtered)),
        s=filtered["Count"] * 80,
        c=filtered["FDR"],
        cmap="viridis_r",
        vmin=0,
        vmax=0.25,
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )

    ax.set_yticks(range(len(filtered)))
    ax.set_yticklabels(filtered["Label"], fontsize=10)
    ax.set_xlabel("Gene Ratio", fontsize=11)
    ax.set_title(
        "Enriched Pathways in 74-Gene Prognostic Signature",
        fontsize=12, fontweight="bold",
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.4, pad=0.02, aspect=15)
    cbar.set_label("FDR", fontsize=10)

    sizes = sorted(filtered["Count"].unique())
    size_picks = [sizes[0], sizes[len(sizes) // 2], sizes[-1]]
    legend_dots = []
    for s in size_picks:
        legend_dots.append(
            ax.scatter([], [], s=s * 80, c="white", edgecolors="black",
                       linewidths=0.4, label=str(s))
        )
    ax.legend(
        handles=legend_dots, title="Gene\nCount", loc="lower right",
        fontsize=9, title_fontsize=9, framealpha=0.9,
        labelspacing=1.5, handletextpad=0.5,
    )

    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = RESULTS_DIR / "enrichment_dotplot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
