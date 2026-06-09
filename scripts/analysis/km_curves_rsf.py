#!/usr/bin/env python3
"""Create Kaplan-Meier curves for RSF risk groups on the CPTAC validation cohort."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

np = None
pd = None
plt = None
KaplanMeierFitter = None
add_at_risk_counts = None
logrank_test = None


def load_analysis_dependencies():
    global np, pd, plt, KaplanMeierFitter, add_at_risk_counts, logrank_test
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
            raise ImportError("KM plotting requires matplotlib. Install it with `pip install matplotlib`.") from exc
        _matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        plt = _plt
    if KaplanMeierFitter is None:
        try:
            from lifelines import KaplanMeierFitter as _KaplanMeierFitter
            from lifelines.plotting import add_at_risk_counts as _add_at_risk_counts
            from lifelines.statistics import logrank_test as _logrank_test
        except ImportError as exc:
            raise ImportError("KM plotting requires lifelines. Install it with `pip install lifelines`.") from exc
        KaplanMeierFitter = _KaplanMeierFitter
        add_at_risk_counts = _add_at_risk_counts
        logrank_test = _logrank_test

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "scripts" / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

# Local project module: reuses the preprocessing and RSF helpers from training.
import train_survival_models as survival


CLINICAL_COLS = [
    "Age", "Gender", "Stage", "T_Stage", "N_Stage",
    "Grade", "Alcohol_History", "Pack_Years",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a block-constrained RSF and plot CPTAC validation KM risk groups"
    )
    parser.add_argument("--train-data", required=True, help="Harmonized TCGA discovery CSV")
    parser.add_argument("--test-data", required=True, help="Harmonized CPTAC validation CSV")
    parser.add_argument("--output", required=True, help="Output figure path, usually .png or .pdf")
    parser.add_argument("--n-rna", type=int, default=70, help="Number of RNA features")
    parser.add_argument("--n-scna", type=int, default=4, help="Number of SCNA/CNV features")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs for RSF tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def prepare_data(args):
    load_analysis_dependencies()
    survival.load_modeling_dependencies()
    survival.np.random.seed(args.seed)
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)
    required_cols = CLINICAL_COLS + ["OS_days", "OS_event"]
    survival.ensure_columns(train_df, required_cols, "Training data")
    survival.ensure_columns(test_df, required_cols, "Test data")
    train_df, test_df = survival.harmonize_clinical_data(train_df, test_df)
    for df in (train_df, test_df):
        df["OS_days"] = pd.to_numeric(df["OS_days"], errors="coerce")
        df["OS_event"] = survival.coerce_event_column(df["OS_event"])
    train_df = train_df.dropna(subset=["OS_days"]).query("OS_days > 0").copy()
    test_df = test_df.dropna(subset=["OS_days"]).query("OS_days > 0").copy()
    return train_df, test_df


def make_survival_array(df):
    return np.array(
        list(zip(df["OS_event"], df["OS_days"])),
        dtype=[("event", bool), ("time", float)],
    )


def split_risk_groups(risk_scores):
    """Split risk scores into non-empty lower/higher groups."""
    risk_scores = np.asarray(risk_scores, dtype=float)
    if len(risk_scores) < 2:
        raise ValueError("At least two validation samples are required for KM risk groups")
    order = np.argsort(risk_scores, kind="mergesort")
    high_mask = np.zeros(len(risk_scores), dtype=bool)
    high_mask[order[len(risk_scores) // 2:]] = True
    low_mask = ~high_mask
    return low_mask, high_mask


def main():
    args = parse_args()
    load_analysis_dependencies()
    train_df, test_df = prepare_data(args)
    y_train = make_survival_array(train_df)
    y_test = make_survival_array(test_df)

    rna_cols = survival.common_omic_columns(train_df, test_df, "RNA_")
    cnv_cols = survival.common_omic_columns(train_df, test_df, "CNV_")
    X_train_clin, X_test_clin, _ = survival.process_clinical_features(train_df, test_df, CLINICAL_COLS)
    X_train_rna, X_test_rna = survival.impute_and_scale_block(train_df, test_df, rna_cols, "RNA")
    X_train_cnv, X_test_cnv = survival.impute_and_scale_block(train_df, test_df, cnv_cols, "CNV/SCNA")

    rna_scores = survival.univariate_cindex_scores(X_train_rna, y_train)
    cnv_scores = survival.univariate_cindex_scores(X_train_cnv, y_train)
    top_rna_idx = survival.top_feature_indices(rna_scores, args.n_rna)
    top_cnv_idx = survival.top_feature_indices(cnv_scores, args.n_scna)
    X_train = np.hstack([X_train_clin, X_train_rna[:, top_rna_idx], X_train_cnv[:, top_cnv_idx]])
    X_test = np.hstack([X_test_clin, X_test_rna[:, top_rna_idx], X_test_cnv[:, top_cnv_idx]])

    print(f"Training RSF with {X_train.shape[1]} features...")
    model, _ = survival.tune_rsf_aggressive(X_train, y_train, n_jobs=args.n_jobs, seed=args.seed)
    if model is None:
        raise RuntimeError("RSF tuning did not return a successful model")
    train_c = model.score(X_train, y_train)
    test_c = model.score(X_test, y_test)
    risk_test = model.predict(X_test)
    low_mask, high_mask = split_risk_groups(risk_test)

    test_months = test_df["OS_days"].to_numpy(dtype=float, copy=True) / 30.44
    test_events = test_df["OS_event"].to_numpy(dtype=bool, copy=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()
    kmf_low.fit(test_months[low_mask], event_observed=test_events[low_mask], label=f"Lower risk (n={low_mask.sum()})")
    kmf_high.fit(test_months[high_mask], event_observed=test_events[high_mask], label=f"Higher risk (n={high_mask.sum()})")
    kmf_low.plot_survival_function(ax=ax, ci_show=False, color="#2196F3", linewidth=2)
    kmf_high.plot_survival_function(ax=ax, ci_show=False, color="#F44336", linewidth=2)
    lr = logrank_test(
        test_months[high_mask],
        test_months[low_mask],
        event_observed_A=test_events[high_mask],
        event_observed_B=test_events[low_mask],
    )
    p_str = f"p = {lr.p_value:.1e}" if lr.p_value < 1e-3 else f"p = {lr.p_value:.3f}"
    ax.text(
        0.95,
        0.05,
        f"Log-rank {p_str}\nTrain C-index = {train_c:.3f}\nTest C-index = {test_c:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax.set_title("RSF risk stratification (CPTAC validation)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left")
    add_at_risk_counts(kmf_low, kmf_high, ax=ax, fontsize=8)
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    print(f"Wrote KM figure to {output}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
