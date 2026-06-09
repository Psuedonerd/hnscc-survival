#!/usr/bin/env python3
"""Plot train/test C-index values from model-comparison CSV outputs."""

import argparse
import os
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


def parse_args():
    parser = argparse.ArgumentParser(description="Create a model-comparison C-index figure")
    parser.add_argument(
        "--results-csv",
        required=True,
        help="CSV produced by train_survival_models.py or a full model-comparison table",
    )
    parser.add_argument("--output", required=True, help="Output figure path, usually .png or .pdf")
    parser.add_argument("--top-n", type=int, default=12, help="Rows to include when plotting all_results.csv")
    return parser.parse_args()


def prepare_results(df, top_n):
    if df.empty:
        raise ValueError("No rows found in model results CSV")
    required = {"Train_C", "Test_C"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    prepared_input = df.copy()
    prepared_input["Train_C"] = pd.to_numeric(prepared_input["Train_C"], errors="coerce")
    prepared_input["Test_C"] = pd.to_numeric(prepared_input["Test_C"], errors="coerce")
    prepared_input = prepared_input.dropna(subset=["Train_C", "Test_C"], how="all")
    if prepared_input.empty:
        raise ValueError("No finite Train_C or Test_C values found in model results CSV")
    if {"Condition", "Model"}.issubset(df.columns):
        prepared = prepared_input.copy()
        prepared["Label"] = prepared["Condition"].astype(str) + " / " + prepared["Model"].astype(str)
    elif {"Experiment", "Config"}.issubset(df.columns):
        prepared = prepared_input.sort_values("Test_C", ascending=False, na_position="last").head(top_n).copy()
        prepared["Label"] = prepared["Experiment"].astype(str) + "\n" + prepared["Config"].astype(str)
    else:
        prepared = prepared_input.head(top_n).copy()
        prepared["Label"] = [f"Model {i + 1}" for i in range(len(prepared))]
    return prepared.reset_index(drop=True)


def main():
    args = parse_args()
    load_plotting_dependencies()
    df = pd.read_csv(args.results_csv)
    plot_df = prepare_results(df, args.top_n)
    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(plot_df)), 5.5))
    ax.bar(x - width / 2, plot_df["Train_C"], width, label="Train C-index", color="#4C78A8")
    ax.bar(x + width / 2, plot_df["Test_C"], width, label="Test C-index", color="#F58518")
    ax.set_ylabel("Concordance index")
    ax.set_title("Survival model performance")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Label"], rotation=45, ha="right", fontsize=8)
    finite_scores = plot_df[["Train_C", "Test_C"]].to_numpy(dtype=float, copy=True)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    y_max = max(0.75, finite_scores.max() + 0.05) if len(finite_scores) else 0.75
    ax.set_ylim(0.45, y_max)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    print(f"Wrote model-comparison figure to {output}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
