#!/usr/bin/env python3
"""
Reproduce the block-constrained RSF configuration highlighted in the manuscript.

The script retrains the RNA70/SCNA4 random survival forest using the same
preprocessing utilities as the main modeling workflow, prints the selected RSF
hyperparameters, and reports train/test concordance for quick verification.

Example:
    python scripts/training/check_v7_params.py \
        --train-data CPTAC/data/TCGA_Discovery_Harmonized_Full_Data.csv \
        --test-data CPTAC/data/CPTAC_Validation_Harmonized_Full_Data.csv
"""

import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import train_survival_models as survival


CLINICAL_COLS = [
    "Age", "Gender", "Stage", "T_Stage", "N_Stage",
    "Grade", "Alcohol_History", "Pack_Years",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain the manuscript block-constrained RNA70/SCNA4 RSF model"
    )
    parser.add_argument("--train-data", required=True, help="Harmonized TCGA discovery CSV")
    parser.add_argument("--test-data", required=True, help="Harmonized CPTAC validation CSV")
    parser.add_argument("--n-rna", type=int, default=70, help="Number of RNA features")
    parser.add_argument("--n-scna", type=int, default=4, help="Number of SCNA/CNV features")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs for RSF tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    survival.load_modeling_dependencies()
    survival.np.random.seed(args.seed)

    train_df = survival.pd.read_csv(args.train_data)
    test_df = survival.pd.read_csv(args.test_data)

    required_cols = CLINICAL_COLS + ["OS_days", "OS_event"]
    survival.ensure_columns(train_df, required_cols, "Training data")
    survival.ensure_columns(test_df, required_cols, "Test data")

    train_df, test_df = survival.harmonize_clinical_data(train_df, test_df)
    train_df["OS_days"] = survival.pd.to_numeric(train_df["OS_days"], errors="coerce")
    test_df["OS_days"] = survival.pd.to_numeric(test_df["OS_days"], errors="coerce")
    train_df["OS_event"] = survival.coerce_event_column(train_df["OS_event"])
    test_df["OS_event"] = survival.coerce_event_column(test_df["OS_event"])
    train_df = train_df.dropna(subset=["OS_days"]).query("OS_days > 0").copy()
    test_df = test_df.dropna(subset=["OS_days"]).query("OS_days > 0").copy()

    y_train = survival.np.array(
        list(zip(train_df["OS_event"], train_df["OS_days"])),
        dtype=[("event", bool), ("time", float)],
    )
    y_test = survival.np.array(
        list(zip(test_df["OS_event"], test_df["OS_days"])),
        dtype=[("event", bool), ("time", float)],
    )

    rna_cols = survival.common_omic_columns(train_df, test_df, "RNA_")
    cnv_cols = survival.common_omic_columns(train_df, test_df, "CNV_")

    X_train_clin, X_test_clin, clin_names = survival.process_clinical_features(
        train_df, test_df, CLINICAL_COLS
    )
    X_train_rna, X_test_rna = survival.impute_and_scale_block(train_df, test_df, rna_cols, "RNA")
    X_train_cnv, X_test_cnv = survival.impute_and_scale_block(train_df, test_df, cnv_cols, "CNV/SCNA")

    print("Calculating univariate C-index scores...")
    rna_scores = survival.univariate_cindex_scores(X_train_rna, y_train)
    cnv_scores = survival.univariate_cindex_scores(X_train_cnv, y_train)
    top_rna_idx = survival.top_feature_indices(rna_scores, args.n_rna)
    top_cnv_idx = survival.top_feature_indices(cnv_scores, args.n_scna)

    X_train_sel = survival.np.hstack([
        X_train_clin,
        X_train_rna[:, top_rna_idx],
        X_train_cnv[:, top_cnv_idx],
    ])
    X_test_sel = survival.np.hstack([
        X_test_clin,
        X_test_rna[:, top_rna_idx],
        X_test_cnv[:, top_cnv_idx],
    ])
    selected_features = (
        clin_names
        + [rna_cols[i] for i in top_rna_idx]
        + [cnv_cols[i] for i in top_cnv_idx]
    )

    print(
        f"Tuning RSF for Block_Constrained RNA{len(top_rna_idx)}_"
        f"SCNA{len(top_cnv_idx)} ({len(selected_features)} total features)..."
    )
    model, best_params = survival.tune_rsf_aggressive(
        X_train_sel, y_train, n_jobs=args.n_jobs, seed=args.seed
    )
    if model is None:
        raise RuntimeError("RSF tuning did not return a successful model")

    print("\nBest RSF parameters")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("\nConcordance")
    print(f"  OOB:   {model.oob_score_:.4f}")
    print(f"  Train: {model.score(X_train_sel, y_train):.4f}")
    print(f"  Test:  {model.score(X_test_sel, y_test):.4f}")


if __name__ == "__main__":
    main()
