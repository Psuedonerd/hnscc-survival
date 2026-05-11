#!/usr/bin/env python3
"""
Minimal reproduction of the winning v7 model (Block_Constrained RNA70_SCNA4)
to extract the best hyperparameters from tune_rsf_aggressive.
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from optimized_survival_v7 import (
    harmonize_clinical_data,
    process_clinical_features,
    univariate_cindex_scores,
    tune_rsf_aggressive,
)

warnings.filterwarnings("ignore")

PROJECT_DIR = "/restricted/projectnb/montilab-p/personal/pnarnur/projects/hnscc_genomics_misc"
SEED = 42
np.random.seed(SEED)

train_df = pd.read_csv(f"{PROJECT_DIR}/CPTAC/data/TCGA_Discovery_Harmonized_Full_Data.csv")
test_df = pd.read_csv(f"{PROJECT_DIR}/CPTAC/data/CPTAC_Validation_Harmonized_Full_Data.csv")
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

clinical_cols = [
    "Age", "Gender", "Stage", "T_Stage", "N_Stage",
    "Grade", "Alcohol_History", "Pack_Years",
]
rna_cols = [c for c in train_df.columns if c.startswith("RNA_")]
cnv_cols = [c for c in train_df.columns if c.startswith("CNV_")]

X_train_clin, X_test_clin, clin_names = process_clinical_features(
    train_df, test_df, clinical_cols
)

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

print("Calculating univariate C-index scores...")
rna_scores = univariate_cindex_scores(X_train_rna_scaled, y_train)
cnv_scores = univariate_cindex_scores(X_train_cnv_scaled, y_train)

n_rna, n_cnv = 70, 4
top_rna_idx = np.argsort(rna_scores)[-n_rna:]
top_cnv_idx = np.argsort(cnv_scores)[-n_cnv:]

X_train_sel = np.hstack([
    X_train_clin,
    X_train_rna_scaled[:, top_rna_idx],
    X_train_cnv_scaled[:, top_cnv_idx],
])
X_test_sel = np.hstack([
    X_test_clin,
    X_test_rna_scaled[:, top_rna_idx],
    X_test_cnv_scaled[:, top_cnv_idx],
])

print(f"Tuning RSF for Block_Constrained RNA{n_rna}_SCNA{n_cnv}...")
model, best_params = tune_rsf_aggressive(X_train_sel, y_train, n_jobs=4, seed=SEED)

print(f"\n{'='*50}")
print(f"BEST PARAMS: {best_params}")
print(f"  max_depth:        {best_params['max_depth']}")
print(f"  n_estimators:     {best_params['n_estimators']}")
print(f"  min_samples_split:{best_params['min_samples_split']}")
print(f"  min_samples_leaf: {best_params['min_samples_leaf']}")
print(f"  max_features:     {best_params['max_features']}")
print(f"{'='*50}")

oob_c = model.oob_score_
train_c = model.score(X_train_sel, y_train)
test_c = model.score(X_test_sel, y_test)
print(f"\nVerification (should match logged values):")
print(f"  OOB:   {oob_c:.4f}  (expected: 0.6638)")
print(f"  Train: {train_c:.4f}  (expected: 0.7987)")
print(f"  Test:  {test_c:.4f}  (expected: 0.6639)")
