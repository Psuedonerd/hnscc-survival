#!/usr/bin/env python3
"""
Figure 2: Kaplan-Meier survival curves for RSF risk stratification
on the CPTAC validation cohort (Combined features: Clinical + Omics).

Patients stratified into high/low risk using a cutoff defined from the TRAIN set
(median predicted risk on train), then applied to the TEST set.

Reuses preprocessing from full_model_comparison.py (identical v7 pipeline).
"""

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")

PROJECT = "/restricted/projectnb/montilab-p/personal/pnarnur/projects/hnscc_genomics_misc"
sys.path.insert(0, os.path.join(PROJECT, "CPTAC", "scripts"))
from full_model_comparison import (
    harmonize_clinical_data,
    process_clinical_features,
    univariate_cindex_scores,
    train_rsf,
)

SEED = 42
N_JOBS = int(os.environ.get("N_JOBS", 4))
np.random.seed(SEED)

# =====================================================================
# Load and preprocess data (identical to full_model_comparison.py)
# =====================================================================
print("Loading data...")
train_df = pd.read_csv(f"{PROJECT}/CPTAC/data/TCGA_Discovery_Harmonized_Full_Data.csv")
test_df  = pd.read_csv(f"{PROJECT}/CPTAC/data/CPTAC_Validation_Harmonized_Full_Data.csv")
train_df, test_df = harmonize_clinical_data(train_df, test_df)

train_df["OS_event"] = train_df["OS_event"].astype(bool)
test_df["OS_event"]  = test_df["OS_event"].astype(bool)

y_train = np.array(
    list(zip(train_df["OS_event"], train_df["OS_days"])),
    dtype=[("event", bool), ("time", float)],
)
y_test = np.array(
    list(zip(test_df["OS_event"], test_df["OS_days"])),
    dtype=[("event", bool), ("time", float)],
)

print(f"  Train: n={len(train_df)} ({train_df['OS_event'].sum()} events)")
print(f"  Test:  n={len(test_df)} ({test_df['OS_event'].sum()} events)")

# =====================================================================
# Feature preparation (identical to full_model_comparison.py)
# =====================================================================
clinical_cols = [
    "Age", "Gender", "Stage", "T_Stage", "N_Stage",
    "Grade", "Alcohol_History", "Pack_Years",
]
rna_cols = [c for c in train_df.columns if c.startswith("RNA_")]
cnv_cols = [c for c in train_df.columns if c.startswith("CNV_")]

X_train_clin, X_test_clin, clin_names = process_clinical_features(
    train_df, test_df, clinical_cols
)

# --- RNA ---
X_train_rna = train_df[rna_cols].to_numpy(dtype=float)
X_test_rna  = test_df[rna_cols].to_numpy(dtype=float)

col_means_rna = np.nanmean(X_train_rna, axis=0)
col_means_rna = np.where(np.isnan(col_means_rna), 0.0, col_means_rna)

X_train_rna = np.where(np.isnan(X_train_rna), col_means_rna, X_train_rna)
X_test_rna  = np.where(np.isnan(X_test_rna),  col_means_rna, X_test_rna)

rna_scaler = StandardScaler()
X_train_rna_scaled = rna_scaler.fit_transform(X_train_rna)
X_test_rna_scaled  = rna_scaler.transform(X_test_rna)

# --- CNV ---
X_train_cnv = train_df[cnv_cols].to_numpy(dtype=float)
X_test_cnv  = test_df[cnv_cols].to_numpy(dtype=float)

col_means_cnv = np.nanmean(X_train_cnv, axis=0)
col_means_cnv = np.where(np.isnan(col_means_cnv), 0.0, col_means_cnv)

X_train_cnv = np.where(np.isnan(X_train_cnv), col_means_cnv, X_train_cnv)
X_test_cnv  = np.where(np.isnan(X_test_cnv),  col_means_cnv, X_test_cnv)

cnv_scaler = StandardScaler()
X_train_cnv_scaled = cnv_scaler.fit_transform(X_train_cnv)
X_test_cnv_scaled  = cnv_scaler.transform(X_test_cnv)

# --- Feature selection (univariate c-index) ---
rna_scores = univariate_cindex_scores(X_train_rna_scaled, y_train)
cnv_scores = univariate_cindex_scores(X_train_cnv_scaled, y_train)

top_rna_idx = np.argsort(rna_scores)[-70:]
top_cnv_idx = np.argsort(cnv_scores)[-4:]

X_train_omics = np.hstack([
    X_train_rna_scaled[:, top_rna_idx],
    X_train_cnv_scaled[:, top_cnv_idx],
])
X_test_omics = np.hstack([
    X_test_rna_scaled[:, top_rna_idx],
    X_test_cnv_scaled[:, top_cnv_idx],
])

X_train_combined = np.hstack([X_train_clin, X_train_omics])
X_test_combined  = np.hstack([X_test_clin,  X_test_omics])

print(f"  Clinical: {len(clin_names)} features")
print(f"  Omics: 74 features (70 RNA + 4 SCNA)")
print(f"  Combined: {len(clin_names) + 74} features")

# =====================================================================
# Train RSF (Combined) and generate KM curve
# =====================================================================
test_times  = test_df["OS_days"].to_numpy()
test_events = test_df["OS_event"].to_numpy()
test_months = test_times / 30.44
max_followup = float(np.nanmax(test_months))

print(f"\nTraining RSF (Combined, {X_train_combined.shape[1]} features)...")
model, _ = train_rsf(X_train_combined, y_train, n_jobs=N_JOBS, seed=SEED)
train_c = model.score(X_train_combined, y_train)
test_c  = model.score(X_test_combined, y_test)
print(f"  Train C={train_c:.4f}, Test C={test_c:.4f}")

risk_train = model.predict(X_train_combined)
cutoff = np.median(risk_train)

risk_test = model.predict(X_test_combined)
high_mask = risk_test >= cutoff
low_mask  = ~high_mask

n_high = int(high_mask.sum())
n_low  = int(low_mask.sum())
print(f"  Higher risk: n={n_high}, Lower risk: n={n_low}")
print(f"  Train risk: median={np.median(risk_train):.3f}, "
      f"IQR={np.quantile(risk_train, [0.25, 0.75])}")
print(f"  Test  risk: median={np.median(risk_test):.3f}, "
      f"IQR={np.quantile(risk_test, [0.25, 0.75])}")
print(f"  Events high={test_events[high_mask].sum()}, "
      f"Events low={test_events[low_mask].sum()}")

fig, ax = plt.subplots(figsize=(7, 5.5))

kmf_low = KaplanMeierFitter()
kmf_low.fit(
    test_months[low_mask],
    event_observed=test_events[low_mask],
    label=f"Lower risk (n={n_low})",
)
kmf_low.plot_survival_function(ax=ax, ci_show=False, color="#2196F3", linewidth=2,
                               show_censors=True, censor_styles={"ms": 6, "marker": "|"})

kmf_high = KaplanMeierFitter()
kmf_high.fit(
    test_months[high_mask],
    event_observed=test_events[high_mask],
    label=f"Higher risk (n={n_high})",
)
kmf_high.plot_survival_function(ax=ax, ci_show=False, color="#F44336", linewidth=2,
                                show_censors=True, censor_styles={"ms": 6, "marker": "|"})

lr = logrank_test(
    test_months[high_mask], test_months[low_mask],
    event_observed_A=test_events[high_mask],
    event_observed_B=test_events[low_mask],
)

if lr.p_value < 1e-3:
    p_str = f"p = {lr.p_value:.1e}"
elif lr.p_value < 1e-2:
    p_str = f"p = {lr.p_value:.3f}"
else:
    p_str = f"p = {lr.p_value:.2f}"

ax.text(
    0.95, 0.05,
    f"Log-rank {p_str}\nTest C-index = {test_c:.3f}",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="#cccccc", alpha=0.9),
)

ax.set_title(
    "Figure 2: RSF risk stratification (CPTAC validation)",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("Time (months)", fontsize=11)
ax.set_ylabel("Survival Probability", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, max_followup)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", fontsize=10, frameon=True, fancybox=True)
ax.tick_params(labelsize=10)

add_at_risk_counts(kmf_low, kmf_high, ax=ax, fontsize=9)

plt.tight_layout()

out_base = Path(PROJECT) / "CPTAC" / "results" / "km_curves_rsf"
out_base.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(f"{out_base}.png", dpi=800, bbox_inches="tight")
fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
print(f"\nSaved to {out_base}.{{png,pdf}}")
