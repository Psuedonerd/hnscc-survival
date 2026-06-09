#!/usr/bin/env python3
"""Create an ECM/protease-focused dot plot from Enrichr result CSV files."""

import argparse
import os
import re
import tempfile
from pathlib import Path

np = None
pd = None
plt = None


def load_plotting_dependencies():
    global np, pd, plt
    if np is None:
        import numpy as _np
        np = _np
    if pd is None:
        import pandas as _pd
        pd = _pd
    if plt is None:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / "hnscc-survival-matplotlib"),
        )
        try:
            import matplotlib as _matplotlib
        except ImportError as exc:
            raise ImportError("Plotting requires matplotlib. Install it with `pip install matplotlib`.") from exc
        _matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        plt = _plt


DEFAULT_DATABASE_LABELS = {
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
    r"serine.type.*(?:peptidase|endopeptidase)|zymogen|plasminogen|protease|"
    r"proteolysis|peptidase|endopeptidase|procollagen|laminin|fibronectin|ECM|"
    r"epithelial mesenchymal transition"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ECM/protease pathway terms from pathway enrichment results"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing *_results.csv files")
    parser.add_argument("--output", required=True, help="Output image path, usually .png or .pdf")
    parser.add_argument(
        "--adjusted-p-cutoff",
        type=float,
        default=0.25,
        help="Maximum adjusted p-value to include (default: 0.25)",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Maximum terms to plot")
    return parser.parse_args()


def clean_label(term):
    term = re.sub(r"\s*R-HSA-\d+", "", str(term))
    term = re.sub(r"\s*WP\d+", "", term)
    term = re.sub(r"\s*\(GO:\d+\)", "", term)
    return term.strip()


def read_enrichment_tables(input_dir):
    load_plotting_dependencies()
    input_dir = Path(input_dir)
    frames = []
    for path in sorted(input_dir.glob("*_results.csv")):
        df = pd.read_csv(path)
        if "Term" not in df.columns or "Adjusted P-value" not in df.columns:
            continue
        df["Database"] = DEFAULT_DATABASE_LABELS.get(path.name, path.stem.replace("_results", ""))
        frames.append(df)
    if not frames:
        raise ValueError(f"No Enrichr-style *_results.csv files found in {input_dir}")
    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_args()
    load_plotting_dependencies()
    combined = read_enrichment_tables(args.input_dir)
    mask = (
        combined["Term"].str.contains(ECM_PROTEASE_PATTERN, regex=True, na=False)
        & (pd.to_numeric(combined["Adjusted P-value"], errors="coerce") <= args.adjusted_p_cutoff)
    )
    filtered = combined[mask].copy()
    if filtered.empty:
        raise ValueError("No ECM/protease terms passed the filter")

    filtered["Clean_Term"] = filtered["Term"].map(clean_label)
    filtered["Adjusted P-value"] = pd.to_numeric(filtered["Adjusted P-value"], errors="coerce")
    if "Overlap" in filtered.columns:
        overlap_values = filtered["Overlap"]
    else:
        overlap_values = pd.Series("0/1", index=filtered.index)
    filtered["Overlap_Size"] = overlap_values.astype(str).str.split("/").str[0]
    filtered["Overlap_Size"] = pd.to_numeric(filtered["Overlap_Size"], errors="coerce").fillna(1)
    if "Combined Score" not in filtered.columns:
        filtered["Combined Score"] = 1.0
    filtered = filtered.sort_values(["Adjusted P-value", "Combined Score"], ascending=[True, False]).head(args.top_n)
    filtered = filtered.iloc[::-1]

    fig_height = max(4, 0.35 * len(filtered) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    y = np.arange(len(filtered))
    x = -np.log10(filtered["Adjusted P-value"].clip(lower=1e-300))
    sizes = 35 + filtered["Overlap_Size"] * 18
    scatter = ax.scatter(x, y, s=sizes, c=x, cmap="viridis", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(filtered["Clean_Term"], fontsize=8)
    ax.set_xlabel("-log10 adjusted p-value")
    ax.set_title("ECM and protease pathway enrichment")
    ax.grid(axis="x", alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="-log10 adjusted p-value")
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    print(f"Wrote dot plot to {output}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
