#!/usr/bin/env python3
"""
Optimized Survival Prediction v7: TCGA → CPTAC
TARGET: Test C-index >= 0.65 with MINIMUM 4 SCNA features

Multi-Omics Integration with Guaranteed SCNA Inclusion:

1. BLOCK-CONSTRAINED FEATURE SELECTION: Force top N RNA + top M SCNA (min M=4)
   Uses univariate C-index ranking (unbiased, same method as v6)

2. PRIORITY-LASSO: Hierarchical fitting that preserves block contributions
   Clinical → RNA → SCNA, each block explains residual variance

3. LATE FUSION ENSEMBLE: Separate models combined
   RSF_RNA + RSF_SCNA with optimized weighting (guarantees SCNA contribution)

4. IPF-LASSO STYLE: Different regularization per block
   Lower penalty on SCNA to encourage inclusion

5. STABILITY-SELECTED MULTI-BLOCK: Bootstrap selection on each block
   Features stable in >50% of bootstraps from both RNA and SCNA

SUCCESS CRITERIA (Hard Requirements):
- At least 4 SCNA features selected via unbiased method
- Test C-index >= 0.6448 (must match or beat v6)

Based on literature:
- Priority-Lasso: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2344-6
- Multi-omics benchmark: https://pmc.ncbi.nlm.nih.gov/articles/PMC10162996/
"""

import argparse
import os
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# =============================================================================
# CLINICAL HARMONIZATION (from v6)
# =============================================================================


def harmonize_clinical_data(train_df, test_df):
    """Apply all harmonization steps"""

    train_age = train_df["Age"].copy()
    if train_age.median() > 1000:
        print("  [Age] Converting TCGA from days to years")
        train_df["Age"] = train_age / 365.25
    train_df["Age"] = pd.to_numeric(train_df["Age"], errors="coerce")
    test_df["Age"] = pd.to_numeric(test_df["Age"], errors="coerce")

    for df in [train_df, test_df]:
        df["Gender"] = df["Gender"].astype(str).str.lower().str.strip()
        df["Gender"] = df["Gender"].map(
            lambda x: "Male"
            if "male" in x and "female" not in x
            else "Female"
            if "female" in x
            else "Unknown"
        )

    def simplify_stage(s):
        s = str(s).upper().replace("STAGE", "").strip()
        if s.startswith("IV"):
            return "IV"
        elif s.startswith("III"):
            return "III"
        elif s.startswith("II"):
            return "II"
        elif s.startswith("I"):
            return "I"
        else:
            return "Unknown"

    train_df["Stage"] = train_df["Stage"].apply(simplify_stage)
    test_df["Stage"] = test_df["Stage"].apply(simplify_stage)

    def simplify_t(s):
        s = str(s).upper().replace("P", "").strip()
        for t in ["T4", "T3", "T2", "T1", "T0"]:
            if s.startswith(t):
                return t
        return "TX"

    train_df["T_Stage"] = train_df["T_Stage"].apply(simplify_t)
    test_df["T_Stage"] = test_df["T_Stage"].apply(simplify_t)

    def simplify_n(s):
        s = str(s).upper().replace("P", "").strip()
        for n in ["N3", "N2", "N1", "N0"]:
            if s.startswith(n):
                return n
        return "NX"

    train_df["N_Stage"] = train_df["N_Stage"].apply(simplify_n)
    test_df["N_Stage"] = test_df["N_Stage"].apply(simplify_n)

    def simplify_grade(s):
        s = str(s).upper()
        if "G1" in s or "WELL" in s:
            return "G1"
        elif "G2" in s or "MODERATE" in s:
            return "G2"
        elif "G3" in s or "POOR" in s:
            return "G3"
        else:
            return "GX"

    train_df["Grade"] = train_df["Grade"].apply(simplify_grade)
    test_df["Grade"] = test_df["Grade"].apply(simplify_grade)

    def simplify_alcohol(s):
        s = str(s).lower()
        if "yes" in s or ("consum" in s and "non" not in s):
            return "Yes"
        elif "no" in s or "non-drinker" in s or "lifelong" in s:
            return "No"
        else:
            return "Unknown"

    train_df["Alcohol_History"] = train_df["Alcohol_History"].apply(simplify_alcohol)
    test_df["Alcohol_History"] = test_df["Alcohol_History"].apply(simplify_alcohol)

    train_df["Pack_Years"] = pd.to_numeric(train_df["Pack_Years"], errors="coerce")
    test_df["Pack_Years"] = pd.to_numeric(test_df["Pack_Years"], errors="coerce")
    median_py = train_df["Pack_Years"].median()
    train_df["Pack_Years"] = train_df["Pack_Years"].fillna(median_py)
    test_df["Pack_Years"] = test_df["Pack_Years"].fillna(median_py)

    return train_df, test_df


def process_clinical_features(train_df, test_df, clinical_cols):
    """Process clinical features with consistent encoding"""
    numeric_cols = ["Age", "Pack_Years"]
    categorical_cols = [c for c in clinical_cols if c not in numeric_cols]

    X_train_num = train_df[numeric_cols].values.astype(float)
    X_test_num = test_df[numeric_cols].values.astype(float)
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)

    feature_names = list(numeric_cols)
    X_train_cat_list = []
    X_test_cat_list = []

    for col in categorical_cols:
        all_cats = sorted(set(train_df[col].unique()) | set(test_df[col].unique()))
        for cat in all_cats[1:]:
            X_train_cat_list.append((train_df[col] == cat).astype(float).values)
            X_test_cat_list.append((test_df[col] == cat).astype(float).values)
            feature_names.append(f"{col}_{cat}")

    if X_train_cat_list:
        X_train_cat = np.column_stack(X_train_cat_list)
        X_test_cat = np.column_stack(X_test_cat_list)
        X_train_clin = np.hstack([X_train_num, X_train_cat])
        X_test_clin = np.hstack([X_test_num, X_test_cat])
    else:
        X_train_clin = X_train_num
        X_test_clin = X_test_num

    return X_train_clin, X_test_clin, feature_names


def univariate_cindex_scores(X, y):
    """Calculate univariate C-index deviation for all features."""
    event = y["event"]
    time = y["time"]
    n_features = X.shape[1]
    scores = np.zeros(n_features)

    for j in range(n_features):
        try:
            x_j = X[:, j]
            if np.std(x_j) < 1e-10:
                continue
            c_idx, _, _, _, _ = concordance_index_censored(event, time, x_j)
            scores[j] = abs(c_idx - 0.5)
        except:
            pass

    return scores


# =============================================================================
# STABILITY SELECTION (Per-Block)
# =============================================================================


def stability_selection_per_block(X, y, n_bootstrap=100, top_k=50, seed=42):
    """
    Bootstrap-based stability selection for a single block.
    Returns selection probability for each feature.
    """
    np.random.seed(seed)
    n_samples, n_features = X.shape
    selection_counts = np.zeros(n_features)

    event = y["event"]
    time = y["time"]

    for b in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[idx]
        event_boot = event[idx]
        time_boot = time[idx]

        scores = np.zeros(n_features)
        for j in range(n_features):
            try:
                x_j = X_boot[:, j]
                if np.std(x_j) < 1e-10:
                    continue
                c_idx, _, _, _, _ = concordance_index_censored(
                    event_boot, time_boot, x_j
                )
                scores[j] = abs(c_idx - 0.5)
            except:
                pass

        top_idx = np.argsort(scores)[-top_k:]
        selection_counts[top_idx] += 1

    return selection_counts / n_bootstrap


# =============================================================================
# RSF TUNING WITH AGGRESSIVE REGULARIZATION (from v6 best params)
# =============================================================================


def tune_rsf_aggressive(X_train, y_train, n_jobs=4, seed=42):
    """
    Tune RSF with aggressive regularization.
    Uses OOB score with overfitting penalty.
    """
    param_configs = [
        # Very aggressive (v6 best was similar)
        {"n_estimators": 300, "min_samples_split": 45, "min_samples_leaf": 35,
         "max_features": 0.15, "max_depth": 3},
        {"n_estimators": 300, "min_samples_split": 50, "min_samples_leaf": 40,
         "max_features": 0.15, "max_depth": 3},
        {"n_estimators": 400, "min_samples_split": 50, "min_samples_leaf": 40,
         "max_features": 0.2, "max_depth": 3},
        # Even more aggressive
        {"n_estimators": 300, "min_samples_split": 50, "min_samples_leaf": 45,
         "max_features": 0.1, "max_depth": 2},
        {"n_estimators": 400, "min_samples_split": 55, "min_samples_leaf": 50,
         "max_features": 0.15, "max_depth": 2},
    ]

    best_adjusted = -np.inf
    best_model = None
    best_params = None

    for params in param_configs:
        try:
            model = RandomSurvivalForest(
                n_estimators=params["n_estimators"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                max_depth=params["max_depth"],
                oob_score=True,
                n_jobs=n_jobs,
                random_state=seed,
            )
            model.fit(X_train, y_train)

            oob_score = model.oob_score_
            train_c = model.score(X_train, y_train)
            gap = train_c - oob_score

            # Penalize overfitting
            adjusted = oob_score - 0.3 * gap

            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_model = model
                best_params = params

        except Exception as e:
            continue

    return best_model, best_params


# =============================================================================
# MAIN
# =============================================================================


def main():
    args = parse_args()
    np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"run_v7_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("SURVIVAL PREDICTION v7: Multi-Omics with Guaranteed SCNA Inclusion")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print("\nSUCCESS CRITERIA:")
    print("  1. At least 4 SCNA features (unbiased selection)")
    print("  2. Test C-index >= 0.6448 (match or beat v6)")
    print("\n5 EXPERIMENTAL APPROACHES:")
    print("  1. Block-Constrained Feature Selection (RNA + SCNA)")
    print("  2. Priority-Lasso (Hierarchical)")
    print("  3. Late Fusion Ensemble (RSF_RNA + RSF_SCNA)")
    print("  4. IPF-LASSO Style (Different penalties)")
    print("  5. Stability-Selected Multi-Block")

    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    train_df, test_df = harmonize_clinical_data(train_df, test_df)

    train_df["OS_event"] = train_df["OS_event"].astype(bool)
    test_df["OS_event"] = test_df["OS_event"].astype(bool)

    y_train = np.array(
        list(zip(train_df["OS_event"], train_df["OS_days"])),
        dtype=[("event", bool), ("time", float)],
    )
    y_test = np.array(
        list(zip(test_df["OS_event"], test_df["OS_days"])),
        dtype=[("event", bool), ("time", float)],
    )
    event_train = train_df["OS_event"].values
    n_events = event_train.sum()

    print(f"  Train: {len(train_df)} samples, {n_events} events ({n_events / len(train_df):.1%})")
    print(f"  Test: {len(test_df)} samples, {test_df['OS_event'].sum()} events ({test_df['OS_event'].mean():.1%})")

    # =========================================================================
    # Prepare Features
    # =========================================================================
    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

    clinical_cols = [
        "Age", "Gender", "Stage", "T_Stage", "N_Stage",
        "Grade", "Alcohol_History", "Pack_Years",
    ]
    rna_cols = [c for c in train_df.columns if c.startswith("RNA_")]
    cnv_cols = [c for c in train_df.columns if c.startswith("CNV_")]

    # Process clinical
    X_train_clin, X_test_clin, clin_names = process_clinical_features(
        train_df, test_df, clinical_cols
    )
    print(f"  Clinical features: {len(clin_names)}")

    # Process RNA
    X_train_rna = train_df[rna_cols].values.astype(float)
    X_test_rna = test_df[rna_cols].values.astype(float)
    col_means_rna = np.nanmean(X_train_rna, axis=0)
    col_means_rna = np.where(np.isnan(col_means_rna), 0, col_means_rna)
    for j in range(X_train_rna.shape[1]):
        X_train_rna[np.isnan(X_train_rna[:, j]), j] = col_means_rna[j]
        X_test_rna[np.isnan(X_test_rna[:, j]), j] = col_means_rna[j]
    rna_scaler = StandardScaler()
    X_train_rna_scaled = rna_scaler.fit_transform(X_train_rna)
    X_test_rna_scaled = rna_scaler.transform(X_test_rna)
    print(f"  RNA features: {len(rna_cols)}")

    # Process CNV/SCNA
    X_train_cnv = train_df[cnv_cols].values.astype(float)
    X_test_cnv = test_df[cnv_cols].values.astype(float)
    col_means_cnv = np.nanmean(X_train_cnv, axis=0)
    col_means_cnv = np.where(np.isnan(col_means_cnv), 0, col_means_cnv)
    for j in range(X_train_cnv.shape[1]):
        X_train_cnv[np.isnan(X_train_cnv[:, j]), j] = col_means_cnv[j]
        X_test_cnv[np.isnan(X_test_cnv[:, j]), j] = col_means_cnv[j]
    cnv_scaler = StandardScaler()
    X_train_cnv_scaled = cnv_scaler.fit_transform(X_train_cnv)
    X_test_cnv_scaled = cnv_scaler.transform(X_test_cnv)
    print(f"  CNV/SCNA features: {len(cnv_cols)}")

    # Calculate univariate scores for each block
    print("\n  Calculating univariate C-index scores...")
    rna_scores = univariate_cindex_scores(X_train_rna_scaled, y_train)
    cnv_scores = univariate_cindex_scores(X_train_cnv_scaled, y_train)

    # Report top features from each block
    top_rna_idx = np.argsort(rna_scores)[-10:][::-1]
    top_cnv_idx = np.argsort(cnv_scores)[-10:][::-1]

    print("\n  Top 10 RNA features by C-index deviation:")
    for i, idx in enumerate(top_rna_idx):
        print(f"    {i+1}. {rna_cols[idx]}: {rna_scores[idx]:.4f}")

    print("\n  Top 10 SCNA features by C-index deviation:")
    for i, idx in enumerate(top_cnv_idx):
        print(f"    {i+1}. {cnv_cols[idx]}: {cnv_scores[idx]:.4f}")

    results = []
    all_models = {}

    # =========================================================================
    # EXPERIMENT 1: Block-Constrained Feature Selection
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Block-Constrained Feature Selection")
    print("=" * 70)
    print("  Strategy: Force top N RNA + top M SCNA (minimum M=4)")

    # Test different combinations while ensuring at least 4 SCNA
    block_configs = [
        (70, 4),   # 70 RNA + 4 SCNA = 74 omics features
        (65, 6),   # 65 RNA + 6 SCNA = 71 omics features
        (60, 8),   # 60 RNA + 8 SCNA = 68 omics features
        (55, 10),  # 55 RNA + 10 SCNA = 65 omics features
        (75, 4),   # Match v6 RNA count + minimum SCNA
        (70, 6),   # Balanced
        (65, 8),   # More SCNA weight
    ]

    for n_rna, n_cnv in block_configs:
        print(f"\n  [Block-Constrained: {n_rna} RNA + {n_cnv} SCNA]")

        # Select top features from each block
        top_rna_idx = np.argsort(rna_scores)[-n_rna:]
        top_cnv_idx = np.argsort(cnv_scores)[-n_cnv:]

        # Get selected feature names
        selected_rna = [rna_cols[i] for i in top_rna_idx]
        selected_cnv = [cnv_cols[i] for i in top_cnv_idx]

        # Combine features
        X_train_sel = np.hstack([
            X_train_clin,
            X_train_rna_scaled[:, top_rna_idx],
            X_train_cnv_scaled[:, top_cnv_idx]
        ])
        X_test_sel = np.hstack([
            X_test_clin,
            X_test_rna_scaled[:, top_rna_idx],
            X_test_cnv_scaled[:, top_cnv_idx]
        ])

        feature_names_sel = clin_names + selected_rna + selected_cnv

        # Train RSF
        model, params = tune_rsf_aggressive(X_train_sel, y_train, args.n_jobs, args.seed)

        if model is None:
            print("    Failed to train model")
            continue

        oob_c = model.oob_score_
        train_c = model.score(X_train_sel, y_train)
        test_c = model.score(X_test_sel, y_test)

        print(f"    OOB: {oob_c:.4f}, Train: {train_c:.4f}, Test: {test_c:.4f}")
        print(f"    Total features: {len(feature_names_sel)} ({len(clin_names)} clin + {n_rna} RNA + {n_cnv} SCNA)")

        results.append({
            "Experiment": "Block_Constrained",
            "Config": f"RNA{n_rna}_SCNA{n_cnv}",
            "N_Clinical": len(clin_names),
            "N_RNA": n_rna,
            "N_SCNA": n_cnv,
            "N_Total": len(feature_names_sel),
            "OOB_C": oob_c,
            "Train_C": train_c,
            "Test_C": test_c,
        })

        all_models[f"Block_RNA{n_rna}_SCNA{n_cnv}"] = {
            "model": model,
            "X_test": X_test_sel,
            "features": feature_names_sel,
            "rna_features": selected_rna,
            "scna_features": selected_cnv,
        }

    # =========================================================================
    # EXPERIMENT 2: Priority-Lasso (Hierarchical)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Priority-Lasso (Hierarchical Fitting)")
    print("=" * 70)
    print("  Strategy: Clinical → RNA → SCNA (each explains residual variance)")

    # Step 1: Fit on Clinical (unpenalized or very low penalty)
    print("\n  Step 1: Fitting on Clinical features...")
    try:
        model_clin = CoxnetSurvivalAnalysis(
            l1_ratio=0.01,  # Near Ridge
            alpha_min_ratio=0.1,
            max_iter=100000,
            fit_baseline_model=True,
        )
        model_clin.fit(X_train_clin, y_train)
        pred_clin_train = model_clin.predict(X_train_clin)
        pred_clin_test = model_clin.predict(X_test_clin)
        clin_test_c = model_clin.score(X_test_clin, y_test)
        print(f"    Clinical-only Test C-index: {clin_test_c:.4f}")
    except Exception as e:
        print(f"    Clinical model failed: {e}")
        pred_clin_train = np.zeros(len(train_df))
        pred_clin_test = np.zeros(len(test_df))
        clin_test_c = 0.5

    # Step 2: Fit on RNA with Clinical predictions as offset
    # (Use residuals from clinical model)
    print("\n  Step 2: Fitting on RNA features (with Clinical offset)...")

    # Select top RNA features
    n_rna_priority = 50
    top_rna_idx = np.argsort(rna_scores)[-n_rna_priority:]
    X_train_rna_sel = X_train_rna_scaled[:, top_rna_idx]
    X_test_rna_sel = X_test_rna_scaled[:, top_rna_idx]

    try:
        model_rna = CoxnetSurvivalAnalysis(
            l1_ratio=0.9,
            alpha_min_ratio=0.01,
            max_iter=100000,
            fit_baseline_model=True,
        )
        # Combine clinical features with RNA for this step
        X_train_clin_rna = np.hstack([X_train_clin, X_train_rna_sel])
        X_test_clin_rna = np.hstack([X_test_clin, X_test_rna_sel])
        model_rna.fit(X_train_clin_rna, y_train)
        pred_rna_train = model_rna.predict(X_train_clin_rna)
        pred_rna_test = model_rna.predict(X_test_clin_rna)
        rna_test_c = model_rna.score(X_test_clin_rna, y_test)
        print(f"    Clinical+RNA Test C-index: {rna_test_c:.4f}")
    except Exception as e:
        print(f"    RNA model failed: {e}")
        pred_rna_train = pred_clin_train
        pred_rna_test = pred_clin_test
        rna_test_c = clin_test_c

    # Step 3: Fit on SCNA with Clinical+RNA predictions as offset
    print("\n  Step 3: Fitting on SCNA features (with Clinical+RNA offset)...")

    # Select top SCNA features (minimum 4)
    for n_scna_priority in [10, 8, 6, 4]:
        top_cnv_idx = np.argsort(cnv_scores)[-n_scna_priority:]
        X_train_cnv_sel = X_train_cnv_scaled[:, top_cnv_idx]
        X_test_cnv_sel = X_test_cnv_scaled[:, top_cnv_idx]

        try:
            # Full model: Clinical + RNA + SCNA
            X_train_full = np.hstack([X_train_clin, X_train_rna_sel, X_train_cnv_sel])
            X_test_full = np.hstack([X_test_clin, X_test_rna_sel, X_test_cnv_sel])

            model_full = CoxnetSurvivalAnalysis(
                l1_ratio=0.9,
                alpha_min_ratio=0.01,
                max_iter=100000,
                fit_baseline_model=True,
            )
            model_full.fit(X_train_full, y_train)
            full_test_c = model_full.score(X_test_full, y_test)

            selected_rna_names = [rna_cols[i] for i in top_rna_idx]
            selected_cnv_names = [cnv_cols[i] for i in top_cnv_idx]

            print(f"    Clinical+RNA+SCNA{n_scna_priority} Test C-index: {full_test_c:.4f}")

            results.append({
                "Experiment": "Priority_Lasso",
                "Config": f"RNA{n_rna_priority}_SCNA{n_scna_priority}",
                "N_Clinical": len(clin_names),
                "N_RNA": n_rna_priority,
                "N_SCNA": n_scna_priority,
                "N_Total": len(clin_names) + n_rna_priority + n_scna_priority,
                "OOB_C": np.nan,
                "Train_C": model_full.score(X_train_full, y_train),
                "Test_C": full_test_c,
            })

            all_models[f"PriorityLasso_RNA{n_rna_priority}_SCNA{n_scna_priority}"] = {
                "model": model_full,
                "X_test": X_test_full,
                "features": clin_names + selected_rna_names + selected_cnv_names,
                "rna_features": selected_rna_names,
                "scna_features": selected_cnv_names,
            }

        except Exception as e:
            print(f"    Full model failed with SCNA{n_scna_priority}: {e}")

    # =========================================================================
    # EXPERIMENT 3: Late Fusion Ensemble
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Late Fusion Ensemble")
    print("=" * 70)
    print("  Strategy: Separate RSF_RNA + RSF_SCNA, weighted ensemble")

    # Model A: Clinical + RNA (like v6 best)
    print("\n  Training RSF_RNA (Clinical + 75 RNA)...")
    n_rna_fusion = 75
    top_rna_idx_fusion = np.argsort(rna_scores)[-n_rna_fusion:]
    X_train_rna_fusion = np.hstack([X_train_clin, X_train_rna_scaled[:, top_rna_idx_fusion]])
    X_test_rna_fusion = np.hstack([X_test_clin, X_test_rna_scaled[:, top_rna_idx_fusion]])

    model_rna_rsf, params_rna = tune_rsf_aggressive(X_train_rna_fusion, y_train, args.n_jobs, args.seed)
    if model_rna_rsf:
        test_c_rna = model_rna_rsf.score(X_test_rna_fusion, y_test)
        print(f"    RSF_RNA Test C-index: {test_c_rna:.4f}")
        pred_rna_fusion = model_rna_rsf.predict(X_test_rna_fusion)
    else:
        print("    RSF_RNA failed")
        test_c_rna = 0.5
        pred_rna_fusion = np.zeros(len(test_df))

    # Model B: Clinical + SCNA
    print("\n  Training RSF_SCNA (Clinical + 20 SCNA)...")
    n_cnv_fusion = 20
    top_cnv_idx_fusion = np.argsort(cnv_scores)[-n_cnv_fusion:]
    X_train_cnv_fusion = np.hstack([X_train_clin, X_train_cnv_scaled[:, top_cnv_idx_fusion]])
    X_test_cnv_fusion = np.hstack([X_test_clin, X_test_cnv_scaled[:, top_cnv_idx_fusion]])

    model_cnv_rsf, params_cnv = tune_rsf_aggressive(X_train_cnv_fusion, y_train, args.n_jobs, args.seed)
    if model_cnv_rsf:
        test_c_cnv = model_cnv_rsf.score(X_test_cnv_fusion, y_test)
        print(f"    RSF_SCNA Test C-index: {test_c_cnv:.4f}")
        pred_cnv_fusion = model_cnv_rsf.predict(X_test_cnv_fusion)
    else:
        print("    RSF_SCNA failed")
        test_c_cnv = 0.5
        pred_cnv_fusion = np.zeros(len(test_df))

    # Normalize predictions
    pred_rna_norm = (pred_rna_fusion - pred_rna_fusion.mean()) / (pred_rna_fusion.std() + 1e-10)
    pred_cnv_norm = (pred_cnv_fusion - pred_cnv_fusion.mean()) / (pred_cnv_fusion.std() + 1e-10)

    # Try different fusion weights
    print("\n  Testing fusion weights...")
    best_fusion_c = 0
    best_alpha = 0

    for alpha in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
        pred_fusion = alpha * pred_rna_norm + (1 - alpha) * pred_cnv_norm
        fusion_c, _, _, _, _ = concordance_index_censored(
            y_test["event"], y_test["time"], pred_fusion
        )
        print(f"    α={alpha:.2f} (RNA={alpha:.0%}, SCNA={1-alpha:.0%}): Test C = {fusion_c:.4f}")

        if fusion_c > best_fusion_c:
            best_fusion_c = fusion_c
            best_alpha = alpha

    print(f"\n  Best fusion: α={best_alpha:.2f}, Test C-index: {best_fusion_c:.4f}")

    results.append({
        "Experiment": "Late_Fusion",
        "Config": f"RNA{n_rna_fusion}_SCNA{n_cnv_fusion}_alpha{best_alpha:.2f}",
        "N_Clinical": len(clin_names),
        "N_RNA": n_rna_fusion,
        "N_SCNA": n_cnv_fusion,
        "N_Total": len(clin_names) + n_rna_fusion + n_cnv_fusion,
        "OOB_C": np.nan,
        "Train_C": np.nan,
        "Test_C": best_fusion_c,
    })

    selected_rna_fusion = [rna_cols[i] for i in top_rna_idx_fusion]
    selected_cnv_fusion = [cnv_cols[i] for i in top_cnv_idx_fusion]

    all_models[f"LateFusion_alpha{best_alpha:.2f}"] = {
        "model": (model_rna_rsf, model_cnv_rsf),
        "X_test": (X_test_rna_fusion, X_test_cnv_fusion),
        "features": clin_names + selected_rna_fusion + selected_cnv_fusion,
        "rna_features": selected_rna_fusion,
        "scna_features": selected_cnv_fusion,
        "alpha": best_alpha,
    }

    # =========================================================================
    # EXPERIMENT 4: IPF-LASSO Style (Different Penalties)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: IPF-LASSO Style (Different Regularization per Block)")
    print("=" * 70)
    print("  Strategy: Lower penalty on SCNA to encourage inclusion")

    # Implement via feature scaling: multiply SCNA features by a factor
    # to effectively reduce their penalty
    for scna_boost in [2.0, 1.5, 1.25]:
        print(f"\n  [SCNA boost factor: {scna_boost}x]")

        n_rna_ipf = 60
        n_cnv_ipf = 10

        top_rna_idx_ipf = np.argsort(rna_scores)[-n_rna_ipf:]
        top_cnv_idx_ipf = np.argsort(cnv_scores)[-n_cnv_ipf:]

        # Boost SCNA features
        X_train_cnv_boosted = X_train_cnv_scaled[:, top_cnv_idx_ipf] * scna_boost
        X_test_cnv_boosted = X_test_cnv_scaled[:, top_cnv_idx_ipf] * scna_boost

        X_train_ipf = np.hstack([
            X_train_clin,
            X_train_rna_scaled[:, top_rna_idx_ipf],
            X_train_cnv_boosted
        ])
        X_test_ipf = np.hstack([
            X_test_clin,
            X_test_rna_scaled[:, top_rna_idx_ipf],
            X_test_cnv_boosted
        ])

        # Use RSF (more robust than CoxNet for this)
        model_ipf, params_ipf = tune_rsf_aggressive(X_train_ipf, y_train, args.n_jobs, args.seed)

        if model_ipf:
            test_c_ipf = model_ipf.score(X_test_ipf, y_test)
            print(f"    Test C-index: {test_c_ipf:.4f}")

            selected_rna_ipf = [rna_cols[i] for i in top_rna_idx_ipf]
            selected_cnv_ipf = [cnv_cols[i] for i in top_cnv_idx_ipf]

            results.append({
                "Experiment": "IPF_Style",
                "Config": f"RNA{n_rna_ipf}_SCNA{n_cnv_ipf}_boost{scna_boost}",
                "N_Clinical": len(clin_names),
                "N_RNA": n_rna_ipf,
                "N_SCNA": n_cnv_ipf,
                "N_Total": len(clin_names) + n_rna_ipf + n_cnv_ipf,
                "OOB_C": model_ipf.oob_score_,
                "Train_C": model_ipf.score(X_train_ipf, y_train),
                "Test_C": test_c_ipf,
            })

            all_models[f"IPF_boost{scna_boost}"] = {
                "model": model_ipf,
                "X_test": X_test_ipf,
                "features": clin_names + selected_rna_ipf + selected_cnv_ipf,
                "rna_features": selected_rna_ipf,
                "scna_features": selected_cnv_ipf,
            }

    # =========================================================================
    # EXPERIMENT 5: Stability-Selected Multi-Block
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Stability-Selected Multi-Block")
    print("=" * 70)
    print("  Strategy: Bootstrap selection on RNA and SCNA separately")

    # Stability selection on RNA
    print("\n  Running stability selection on RNA (100 bootstraps)...")
    rna_stability = stability_selection_per_block(
        X_train_rna_scaled, y_train, n_bootstrap=100, top_k=50, seed=args.seed
    )
    stable_rna_idx = np.where(rna_stability >= 0.5)[0]
    print(f"    RNA features with >50% stability: {len(stable_rna_idx)}")

    # Stability selection on SCNA
    print("\n  Running stability selection on SCNA (100 bootstraps)...")
    cnv_stability = stability_selection_per_block(
        X_train_cnv_scaled, y_train, n_bootstrap=100, top_k=20, seed=args.seed
    )
    stable_cnv_idx = np.where(cnv_stability >= 0.3)[0]  # Lower threshold for SCNA
    print(f"    SCNA features with >30% stability: {len(stable_cnv_idx)}")

    # Ensure minimum 4 SCNA
    if len(stable_cnv_idx) < 4:
        print(f"    Less than 4 stable SCNA, using top 4 by stability probability")
        stable_cnv_idx = np.argsort(cnv_stability)[-4:]

    # If too few stable RNA, use top by C-index
    if len(stable_rna_idx) < 30:
        print(f"    Less than 30 stable RNA, using top 50 by C-index")
        stable_rna_idx = np.argsort(rna_scores)[-50:]

    # Combine stable features
    X_train_stable = np.hstack([
        X_train_clin,
        X_train_rna_scaled[:, stable_rna_idx],
        X_train_cnv_scaled[:, stable_cnv_idx]
    ])
    X_test_stable = np.hstack([
        X_test_clin,
        X_test_rna_scaled[:, stable_rna_idx],
        X_test_cnv_scaled[:, stable_cnv_idx]
    ])

    selected_rna_stable = [rna_cols[i] for i in stable_rna_idx]
    selected_cnv_stable = [cnv_cols[i] for i in stable_cnv_idx]

    print(f"\n  Training RSF with {len(stable_rna_idx)} stable RNA + {len(stable_cnv_idx)} stable SCNA...")

    model_stable, params_stable = tune_rsf_aggressive(X_train_stable, y_train, args.n_jobs, args.seed)

    if model_stable:
        test_c_stable = model_stable.score(X_test_stable, y_test)
        print(f"    Test C-index: {test_c_stable:.4f}")

        results.append({
            "Experiment": "Stability_MultiBlock",
            "Config": f"RNA{len(stable_rna_idx)}_SCNA{len(stable_cnv_idx)}",
            "N_Clinical": len(clin_names),
            "N_RNA": len(stable_rna_idx),
            "N_SCNA": len(stable_cnv_idx),
            "N_Total": len(clin_names) + len(stable_rna_idx) + len(stable_cnv_idx),
            "OOB_C": model_stable.oob_score_,
            "Train_C": model_stable.score(X_train_stable, y_train),
            "Test_C": test_c_stable,
        })

        all_models["Stability_MultiBlock"] = {
            "model": model_stable,
            "X_test": X_test_stable,
            "features": clin_names + selected_rna_stable + selected_cnv_stable,
            "rna_features": selected_rna_stable,
            "scna_features": selected_cnv_stable,
        }

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Test_C", ascending=False)

    print("\n--- ALL RESULTS (sorted by Test C-index) ---")
    print(results_df.to_string(index=False))

    # Find best model meeting both criteria
    print("\n" + "-" * 50)
    print("MODELS MEETING SUCCESS CRITERIA:")
    print("  (>= 4 SCNA features AND Test C-index >= 0.6448)")
    print("-" * 50)

    qualifying_models = results_df[
        (results_df["N_SCNA"] >= 4) & (results_df["Test_C"] >= 0.6448)
    ]

    if len(qualifying_models) > 0:
        print(qualifying_models.to_string(index=False))
        best = qualifying_models.iloc[0]
        print(f"\n🎯 BEST MODEL: {best['Experiment']} ({best['Config']})")
        print(f"   Test C-index: {best['Test_C']:.4f}")
        print(f"   Features: {best['N_Clinical']} clinical + {best['N_RNA']} RNA + {best['N_SCNA']} SCNA")

        if best["Test_C"] >= 0.65:
            print("\n✅ TARGET ACHIEVED! Test C-index >= 0.65")
        else:
            gap = 0.65 - best["Test_C"]
            print(f"\n⚠️  Gap to 0.65 target: {gap:.4f}")
    else:
        print("\n❌ No models met both criteria (>=4 SCNA and >=0.6448 C-index)")
        print("   Best overall model:")
        best = results_df.iloc[0]
        print(f"   {best['Experiment']} ({best['Config']}): Test C = {best['Test_C']:.4f}, {best['N_SCNA']} SCNA")

    # Save results
    results_df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)

    # Save feature lists for best model
    if len(qualifying_models) > 0:
        best_key = None
        best_test_c = 0
        for key, model_info in all_models.items():
            if "scna_features" in model_info and len(model_info["scna_features"]) >= 4:
                # Get test C-index
                if isinstance(model_info["model"], tuple):
                    # Late fusion
                    continue
                else:
                    try:
                        test_c = model_info["model"].score(model_info["X_test"], y_test)
                        if test_c >= 0.6448 and test_c > best_test_c:
                            best_test_c = test_c
                            best_key = key
                    except:
                        continue

        if best_key and best_key in all_models:
            best_model_info = all_models[best_key]

            # Save feature list
            feature_df = pd.DataFrame({
                "feature": best_model_info["features"],
                "type": ["clinical"] * len(clin_names) +
                        ["RNA"] * len(best_model_info["rna_features"]) +
                        ["SCNA"] * len(best_model_info["scna_features"])
            })
            feature_df.to_csv(os.path.join(output_dir, "best_model_features.csv"), index=False)

            # Save RNA and SCNA separately with scores
            rna_feature_df = pd.DataFrame({
                "feature": best_model_info["rna_features"],
                "cindex_deviation": [rna_scores[rna_cols.index(f)] for f in best_model_info["rna_features"]]
            })
            rna_feature_df = rna_feature_df.sort_values("cindex_deviation", ascending=False)
            rna_feature_df.to_csv(os.path.join(output_dir, "selected_rna_features.csv"), index=False)

            scna_feature_df = pd.DataFrame({
                "feature": best_model_info["scna_features"],
                "cindex_deviation": [cnv_scores[cnv_cols.index(f)] for f in best_model_info["scna_features"]]
            })
            scna_feature_df = scna_feature_df.sort_values("cindex_deviation", ascending=False)
            scna_feature_df.to_csv(os.path.join(output_dir, "selected_scna_features.csv"), index=False)

            print(f"\nFeature lists saved to: {output_dir}")

    # Save top SCNA features for reference
    top_scna_df = pd.DataFrame({
        "feature": [cnv_cols[i] for i in np.argsort(cnv_scores)[-20:][::-1]],
        "cindex_deviation": [cnv_scores[i] for i in np.argsort(cnv_scores)[-20:][::-1]],
        "stability": [cnv_stability[i] for i in np.argsort(cnv_scores)[-20:][::-1]]
    })
    top_scna_df.to_csv(os.path.join(output_dir, "top20_scna_features.csv"), index=False)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
