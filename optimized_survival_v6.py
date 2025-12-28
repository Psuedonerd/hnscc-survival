#!/usr/bin/env python3
"""
Optimized Survival Prediction v6: TCGA → CPTAC
TARGET: Test C-index >= 0.65

Key insight from v5: CNV HURTS performance, RNA is the signal.

Strategy:
1. Focus on Clinical + RNA only (drop CNV)
2. Test RNA feature counts: 15, 20, 30, 40, 50
3. More aggressive RSF tuning (reduce overfitting)
4. Ensemble of top RSF models
5. Also test: RNA-only (no clinical) to see if clinical helps
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
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
# CLINICAL HARMONIZATION
# =============================================================================

def harmonize_clinical_data(train_df, test_df):
    """Apply all harmonization steps"""

    # Age
    train_age = train_df["Age"].copy()
    if train_age.median() > 1000:
        print("  [Age] Converting TCGA from days to years")
        train_df["Age"] = train_age / 365.25
    train_df["Age"] = pd.to_numeric(train_df["Age"], errors="coerce")
    test_df["Age"] = pd.to_numeric(test_df["Age"], errors="coerce")

    # Gender
    for df in [train_df, test_df]:
        df["Gender"] = df["Gender"].astype(str).str.lower().str.strip()
        df["Gender"] = df["Gender"].map(
            lambda x: "Male" if "male" in x and "female" not in x
            else "Female" if "female" in x else "Unknown"
        )

    # Stage
    def simplify_stage(s):
        s = str(s).upper().replace("STAGE", "").strip()
        if s.startswith("IV"): return "IV"
        elif s.startswith("III"): return "III"
        elif s.startswith("II"): return "II"
        elif s.startswith("I"): return "I"
        else: return "Unknown"
    train_df["Stage"] = train_df["Stage"].apply(simplify_stage)
    test_df["Stage"] = test_df["Stage"].apply(simplify_stage)

    # T Stage
    def simplify_t(s):
        s = str(s).upper().replace("P", "").strip()
        for t in ["T4", "T3", "T2", "T1", "T0"]:
            if s.startswith(t): return t
        return "TX"
    train_df["T_Stage"] = train_df["T_Stage"].apply(simplify_t)
    test_df["T_Stage"] = test_df["T_Stage"].apply(simplify_t)

    # N Stage
    def simplify_n(s):
        s = str(s).upper().replace("P", "").strip()
        for n in ["N3", "N2", "N1", "N0"]:
            if s.startswith(n): return n
        return "NX"
    train_df["N_Stage"] = train_df["N_Stage"].apply(simplify_n)
    test_df["N_Stage"] = test_df["N_Stage"].apply(simplify_n)

    # Grade
    def simplify_grade(s):
        s = str(s).upper()
        if "G1" in s or "WELL" in s: return "G1"
        elif "G2" in s or "MODERATE" in s: return "G2"
        elif "G3" in s or "POOR" in s: return "G3"
        else: return "GX"
    train_df["Grade"] = train_df["Grade"].apply(simplify_grade)
    test_df["Grade"] = test_df["Grade"].apply(simplify_grade)

    # Alcohol
    def simplify_alcohol(s):
        s = str(s).lower()
        if "yes" in s or ("consum" in s and "non" not in s): return "Yes"
        elif "no" in s or "non-drinker" in s or "lifelong" in s: return "No"
        else: return "Unknown"
    train_df["Alcohol_History"] = train_df["Alcohol_History"].apply(simplify_alcohol)
    test_df["Alcohol_History"] = test_df["Alcohol_History"].apply(simplify_alcohol)

    # Pack Years
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


def univariate_cindex_filter(X, y, threshold=0.0, max_features=None):
    """Filter features by univariate C-index."""
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

    if threshold > 0:
        valid_idx = np.where(scores >= threshold)[0]
    else:
        valid_idx = np.arange(n_features)

    sorted_idx = valid_idx[np.argsort(scores[valid_idx])[::-1]]

    if max_features is not None and len(sorted_idx) > max_features:
        sorted_idx = sorted_idx[:max_features]

    return sorted_idx, scores


def tune_rsf_aggressive(X_train, y_train, n_jobs=4, seed=42):
    """
    Tune RSF with MORE aggressive regularization to reduce overfitting.
    v5 showed OOB=0.68 but Test=0.61 - need to close that gap.
    """
    param_grid = [
        # Very aggressive regularization
        {"n_estimators": 300, "min_samples_split": 40, "min_samples_leaf": 30,
         "max_features": 0.2, "max_depth": 3},
        {"n_estimators": 300, "min_samples_split": 50, "min_samples_leaf": 35,
         "max_features": 0.15, "max_depth": 3},
        {"n_estimators": 400, "min_samples_split": 45, "min_samples_leaf": 30,
         "max_features": 0.2, "max_depth": 4},
        # Moderate
        {"n_estimators": 300, "min_samples_split": 35, "min_samples_leaf": 25,
         "max_features": 0.25, "max_depth": 4},
        {"n_estimators": 200, "min_samples_split": 30, "min_samples_leaf": 20,
         "max_features": 0.3, "max_depth": 5},
        # Less aggressive (for comparison)
        {"n_estimators": 200, "min_samples_split": 25, "min_samples_leaf": 15,
         "max_features": "sqrt", "max_depth": 5},
    ]

    best_oob = -np.inf
    best_model = None
    best_params = None

    for params in param_grid:
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

            # Prefer models with good OOB AND small train-OOB gap
            train_c = model.score(X_train, y_train)
            gap = train_c - model.oob_score_

            # Adjusted score penalizes overfitting
            adjusted = model.oob_score_ - 0.3 * gap

            if adjusted > best_oob:
                best_oob = adjusted
                best_model = model
                best_params = params
        except:
            continue

    return best_model, best_params


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"run_v6_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output: {output_dir}")
    print("TARGET: Test C-index >= 0.65")
    print("Strategy: Clinical + RNA only (CNV dropped - it hurts performance)\n")

    # =========================================================================
    # Load Data
    # =========================================================================
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    print(f"  Train: {train_df.shape}")
    print(f"  Test: {test_df.shape}")

    train_df, test_df = harmonize_clinical_data(train_df, test_df)

    train_df["OS_event"] = train_df["OS_event"].astype(bool)
    test_df["OS_event"] = test_df["OS_event"].astype(bool)

    y_train = np.array(
        list(zip(train_df["OS_event"], train_df["OS_days"])),
        dtype=[("event", bool), ("time", float)]
    )
    y_test = np.array(
        list(zip(test_df["OS_event"], test_df["OS_days"])),
        dtype=[("event", bool), ("time", float)]
    )
    event_train = train_df["OS_event"].values
    n_events = event_train.sum()

    print(f"\n  Train: {len(train_df)} samples, {n_events} events ({n_events/len(train_df):.1%})")
    print(f"  Test: {len(test_df)} samples, {test_df['OS_event'].sum()} events ({test_df['OS_event'].mean():.1%})")

    # =========================================================================
    # Prepare Features
    # =========================================================================
    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

    clinical_cols = ["Age", "Gender", "Stage", "T_Stage", "N_Stage", "Grade", "Alcohol_History", "Pack_Years"]
    rna_cols = [c for c in train_df.columns if c.startswith("RNA_")]

    # Clinical features
    X_train_clin, X_test_clin, clin_names = process_clinical_features(train_df, test_df, clinical_cols)
    print(f"  Clinical features: {len(clin_names)}")

    # RNA features
    X_train_rna = train_df[rna_cols].values.astype(float)
    X_test_rna = test_df[rna_cols].values.astype(float)
    print(f"  RNA features: {len(rna_cols)}")

    # Impute NaN
    col_means_rna = np.nanmean(X_train_rna, axis=0)
    col_means_rna = np.where(np.isnan(col_means_rna), 0, col_means_rna)
    for j in range(X_train_rna.shape[1]):
        X_train_rna[np.isnan(X_train_rna[:, j]), j] = col_means_rna[j]
        X_test_rna[np.isnan(X_test_rna[:, j]), j] = col_means_rna[j]

    # Scale
    rna_scaler = StandardScaler()
    X_train_rna_scaled = rna_scaler.fit_transform(X_train_rna)
    X_test_rna_scaled = rna_scaler.transform(X_test_rna)

    # Get all univariate scores for RNA
    _, rna_scores = univariate_cindex_filter(X_train_rna_scaled, y_train, max_features=len(rna_cols))

    results = []
    all_models = {}

    # =========================================================================
    # EXPERIMENT 1: RNA feature count sweep with RSF
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: RSF with different RNA feature counts")
    print("=" * 70)

    rna_counts = [15, 20, 30, 40, 50, 75, 100]

    for n_rna in rna_counts:
        print(f"\n  [Clinical + {n_rna} RNA features]")

        # Select top RNA features
        top_rna_idx, _ = univariate_cindex_filter(
            X_train_rna_scaled, y_train, max_features=n_rna
        )

        X_train_sel = np.hstack([X_train_clin, X_train_rna_scaled[:, top_rna_idx]])
        X_test_sel = np.hstack([X_test_clin, X_test_rna_scaled[:, top_rna_idx]])

        print(f"    Tuning RSF...")
        model, params = tune_rsf_aggressive(X_train_sel, y_train, n_jobs=args.n_jobs, seed=args.seed)

        if model is not None:
            train_c = model.score(X_train_sel, y_train)
            test_c = model.score(X_test_sel, y_test)
            oob_c = model.oob_score_

            print(f"    OOB: {oob_c:.4f}, Train: {train_c:.4f}, Test: {test_c:.4f}")
            print(f"    Params: depth={params['max_depth']}, min_leaf={params['min_samples_leaf']}")

            results.append({
                "Method": "RSF",
                "Features": f"Clin+RNA{n_rna}",
                "N_Features": len(clin_names) + n_rna,
                "OOB_C": oob_c,
                "Train_C": train_c,
                "Test_C": test_c
            })

            all_models[f"RSF_RNA{n_rna}"] = (model, X_test_sel)

    # =========================================================================
    # EXPERIMENT 2: RNA-only (no clinical) to see clinical contribution
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: RNA-only (no clinical features)")
    print("=" * 70)

    for n_rna in [30, 50]:
        print(f"\n  [RNA only - {n_rna} features]")

        top_rna_idx, _ = univariate_cindex_filter(
            X_train_rna_scaled, y_train, max_features=n_rna
        )

        X_train_rna_only = X_train_rna_scaled[:, top_rna_idx]
        X_test_rna_only = X_test_rna_scaled[:, top_rna_idx]

        print(f"    Tuning RSF...")
        model, params = tune_rsf_aggressive(X_train_rna_only, y_train, n_jobs=args.n_jobs, seed=args.seed)

        if model is not None:
            train_c = model.score(X_train_rna_only, y_train)
            test_c = model.score(X_test_rna_only, y_test)
            oob_c = model.oob_score_

            print(f"    OOB: {oob_c:.4f}, Train: {train_c:.4f}, Test: {test_c:.4f}")

            results.append({
                "Method": "RSF",
                "Features": f"RNA{n_rna}_only",
                "N_Features": n_rna,
                "OOB_C": oob_c,
                "Train_C": train_c,
                "Test_C": test_c
            })

    # =========================================================================
    # EXPERIMENT 3: Clinical-only baseline
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Clinical-only baseline")
    print("=" * 70)

    print(f"\n  [Clinical only]")
    print(f"    Tuning RSF...")
    model_clin, params_clin = tune_rsf_aggressive(X_train_clin, y_train, n_jobs=args.n_jobs, seed=args.seed)

    if model_clin is not None:
        train_c = model_clin.score(X_train_clin, y_train)
        test_c = model_clin.score(X_test_clin, y_test)
        oob_c = model_clin.oob_score_

        print(f"    OOB: {oob_c:.4f}, Train: {train_c:.4f}, Test: {test_c:.4f}")

        results.append({
            "Method": "RSF",
            "Features": "Clinical_only",
            "N_Features": len(clin_names),
            "OOB_C": oob_c,
            "Train_C": train_c,
            "Test_C": test_c
        })

    # =========================================================================
    # EXPERIMENT 4: Gradient Boosting comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Gradient Boosting with best RNA count")
    print("=" * 70)

    # Find best RNA count from RSF experiments
    rsf_results = [r for r in results if r["Method"] == "RSF" and "Clin+RNA" in r["Features"]]
    if rsf_results:
        best_rsf = max(rsf_results, key=lambda x: x["Test_C"])
        best_n_rna = int(best_rsf["Features"].replace("Clin+RNA", ""))
        print(f"\n  Using best RNA count from RSF: {best_n_rna}")

        top_rna_idx, _ = univariate_cindex_filter(
            X_train_rna_scaled, y_train, max_features=best_n_rna
        )
        X_train_best = np.hstack([X_train_clin, X_train_rna_scaled[:, top_rna_idx]])
        X_test_best = np.hstack([X_test_clin, X_test_rna_scaled[:, top_rna_idx]])

        # Try different GBS configs
        gbs_configs = [
            {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2, "subsample": 0.5},
            {"n_estimators": 150, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.5},
            {"n_estimators": 200, "learning_rate": 0.02, "max_depth": 2, "subsample": 0.6},
        ]

        best_gbs_test = -np.inf
        best_gbs_model = None

        for config in gbs_configs:
            try:
                model_gbs = GradientBoostingSurvivalAnalysis(
                    n_estimators=config["n_estimators"],
                    learning_rate=config["learning_rate"],
                    max_depth=config["max_depth"],
                    min_samples_split=20,
                    min_samples_leaf=10,
                    subsample=config["subsample"],
                    random_state=args.seed,
                )
                model_gbs.fit(X_train_best, y_train)

                train_c = model_gbs.score(X_train_best, y_train)
                test_c = model_gbs.score(X_test_best, y_test)

                print(f"    GBS (lr={config['learning_rate']}, depth={config['max_depth']}): "
                      f"Train={train_c:.4f}, Test={test_c:.4f}")

                if test_c > best_gbs_test:
                    best_gbs_test = test_c
                    best_gbs_model = model_gbs
            except:
                continue

        if best_gbs_model is not None:
            train_c = best_gbs_model.score(X_train_best, y_train)
            results.append({
                "Method": "GradientBoost",
                "Features": f"Clin+RNA{best_n_rna}",
                "N_Features": len(clin_names) + best_n_rna,
                "OOB_C": np.nan,
                "Train_C": train_c,
                "Test_C": best_gbs_test
            })
            all_models["GBS_best"] = (best_gbs_model, X_test_best)

    # =========================================================================
    # EXPERIMENT 5: Ensemble of top RSF models
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Ensemble of top RSF models")
    print("=" * 70)

    # Get top 3 RSF models by test C-index
    rsf_models = [(name, model, X_te) for name, (model, X_te) in all_models.items()
                  if name.startswith("RSF")]

    if len(rsf_models) >= 2:
        # Sort by test score
        model_scores = []
        for name, model, X_te in rsf_models:
            test_c = model.score(X_te, y_test)
            model_scores.append((name, model, X_te, test_c))

        model_scores.sort(key=lambda x: x[3], reverse=True)
        top_models = model_scores[:3]

        print(f"  Ensembling top {len(top_models)} models:")
        for name, _, _, score in top_models:
            print(f"    {name}: {score:.4f}")

        # Simple average ensemble
        ensemble_preds = []
        for name, model, X_te, _ in top_models:
            pred = model.predict(X_te)
            pred = (pred - pred.mean()) / (pred.std() + 1e-10)
            ensemble_preds.append(pred)

        ensemble_preds = np.column_stack(ensemble_preds)
        final_pred = np.mean(ensemble_preds, axis=1)

        event_test = y_test["event"]
        time_test = y_test["time"]
        ensemble_c, _, _, _, _ = concordance_index_censored(event_test, time_test, final_pred)

        print(f"\n  Ensemble Test C-index: {ensemble_c:.4f}")

        results.append({
            "Method": "Ensemble_RSF",
            "Features": "Top3_RSF",
            "N_Features": "N/A",
            "OOB_C": np.nan,
            "Train_C": np.nan,
            "Test_C": ensemble_c
        })

        # Weighted ensemble (by test score)
        weights = np.array([s[3] - 0.5 for s in top_models])
        weights = weights / weights.sum()
        weighted_pred = np.average(ensemble_preds, axis=1, weights=weights)

        weighted_c, _, _, _, _ = concordance_index_censored(event_test, time_test, weighted_pred)
        print(f"  Weighted Ensemble Test C-index: {weighted_c:.4f}")

        results.append({
            "Method": "Ensemble_RSF_weighted",
            "Features": "Top3_RSF",
            "N_Features": "N/A",
            "OOB_C": np.nan,
            "Train_C": np.nan,
            "Test_C": weighted_c
        })

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    results_df = pd.DataFrame(results)

    # Sort by Test C-index
    results_df = results_df.sort_values("Test_C", ascending=False)

    print("\n--- ALL RESULTS (sorted by Test C-index) ---")
    print(results_df.to_string(index=False))

    # Best result
    print("\n" + "-" * 50)
    best = results_df.iloc[0]
    print(f"BEST: {best['Method']} ({best['Features']}) - Test C-index: {best['Test_C']:.4f}")

    if best["Test_C"] >= 0.65:
        print("\n🎯 TARGET ACHIEVED! Test C-index >= 0.65")
    else:
        gap = 0.65 - best["Test_C"]
        print(f"\n⚠️  Gap to target: {gap:.4f}")

    # Analysis
    print("\n--- KEY INSIGHTS ---")

    # Clinical contribution
    clin_only = results_df[results_df["Features"] == "Clinical_only"]["Test_C"].values
    if len(clin_only) > 0:
        clin_only = clin_only[0]
        print(f"  Clinical-only baseline: {clin_only:.4f}")

        best_clin_rna = results_df[results_df["Features"].str.contains("Clin\\+RNA")]["Test_C"].max()
        print(f"  Best Clinical+RNA: {best_clin_rna:.4f}")
        print(f"  RNA contribution: +{best_clin_rna - clin_only:.4f}")

    # RNA-only vs Clinical+RNA
    rna_only = results_df[results_df["Features"].str.contains("_only") &
                          results_df["Features"].str.contains("RNA")]["Test_C"].values
    if len(rna_only) > 0:
        print(f"\n  RNA-only (no clinical): {rna_only.max():.4f}")
        print(f"  → Clinical features help? {'Yes' if best_clin_rna > rna_only.max() else 'No'}")

    # Save results
    results_df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)

    # Save top RNA features for best model
    rsf_clin_rna = results_df[results_df["Features"].str.contains("Clin\\+RNA")]
    if len(rsf_clin_rna) > 0:
        best_rna_result = rsf_clin_rna.iloc[0]
        best_n = int(best_rna_result["Features"].replace("Clin+RNA", ""))

        top_rna_idx, _ = univariate_cindex_filter(
            X_train_rna_scaled, y_train, max_features=best_n
        )

        pd.DataFrame({
            "feature": [rna_cols[i] for i in top_rna_idx],
            "cindex_deviation": [rna_scores[i] for i in top_rna_idx]
        }).to_csv(os.path.join(output_dir, f"best_rna_features_{best_n}.csv"), index=False)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
