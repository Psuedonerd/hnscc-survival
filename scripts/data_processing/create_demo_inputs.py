#!/usr/bin/env python3
"""Create small anonymized/noisy CSVs for a cloneable HNSCC survival demo."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "patient_id",
    "Age",
    "Gender",
    "Stage",
    "T_Stage",
    "N_Stage",
    "Grade",
    "Alcohol_History",
    "Pack_Years",
    "OS_days",
    "OS_event",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build minimal anonymized demo inputs from full harmonized CSVs"
    )
    parser.add_argument("--train-data", required=True, help="Full TCGA discovery CSV")
    parser.add_argument("--test-data", required=True, help="Full CPTAC validation CSV")
    parser.add_argument("--output-dir", default="demo_data", help="Directory for demo CSVs")
    parser.add_argument("--train-rows", type=int, default=160, help="Number of demo training rows")
    parser.add_argument("--test-rows", type=int, default=80, help="Number of demo validation rows")
    parser.add_argument("--rna-features", type=int, default=80, help="Number of RNA columns to keep")
    parser.add_argument("--cnv-features", type=int, default=20, help="Number of CNV columns to keep")
    parser.add_argument("--noise", type=float, default=0.05, help="Numeric noise fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--keep-gene-names",
        action="store_true",
        help="Keep original RNA_/CNV_ gene feature names instead of renaming to AAA IDs",
    )
    return parser.parse_args()


def ensure_columns(df, columns, label):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def event_as_int(series):
    values = series.astype(str).str.strip().str.lower()
    return values.isin(["1", "true", "t", "yes", "y", "dead", "deceased", "event"]).astype(int)


def stratified_sample(df, n_rows, seed):
    if n_rows >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    event = event_as_int(df["OS_event"])
    sampled_parts = []
    remaining = n_rows
    groups = list(df.groupby(event))

    for i, (_, group) in enumerate(groups):
        if i == len(groups) - 1:
            take = remaining
        else:
            take = int(round(n_rows * len(group) / len(df)))
            take = max(1, min(take, len(group)))
        sampled_parts.append(group.sample(n=take, random_state=seed + i))
        remaining -= take

    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > n_rows:
        sampled = sampled.sample(n=n_rows, random_state=seed)
    return sampled.sample(frac=1, random_state=seed + 99).reset_index(drop=True)


def select_variable_features(train_df, test_df, prefix, n_features):
    train_cols = [c for c in train_df.columns if c.startswith(prefix)]
    test_cols = set(c for c in test_df.columns if c.startswith(prefix))
    shared_cols = [c for c in train_cols if c in test_cols]
    if len(shared_cols) < n_features:
        raise ValueError(
            f"Requested {n_features} {prefix} features, but only found {len(shared_cols)} shared columns"
        )

    numeric = train_df[shared_cols].apply(pd.to_numeric, errors="coerce")
    variances = numeric.var(axis=0, skipna=True).fillna(0).sort_values(ascending=False)
    return variances.head(n_features).index.tolist()


def add_numeric_noise(df, columns, rng, noise_fraction):
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        scale = values.std(skipna=True)
        if pd.isna(scale) or scale == 0:
            scale = 1.0
        df[col] = values + rng.normal(0, noise_fraction * scale, size=len(df))


def anonymize_and_noise(df, feature_cols, start_id, rng, noise_fraction):
    df = df.copy()
    df["patient_id"] = [f"AAA{i}" for i in range(start_id, start_id + len(df))]

    add_numeric_noise(df, ["Age", "Pack_Years", "OS_days"], rng, noise_fraction)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").clip(lower=30, upper=90).round(1)
    df["Pack_Years"] = pd.to_numeric(df["Pack_Years"], errors="coerce").clip(lower=0).round(1)
    df["OS_days"] = pd.to_numeric(df["OS_days"], errors="coerce").clip(lower=30).round().astype(int)
    df["OS_event"] = event_as_int(df["OS_event"])

    add_numeric_noise(df, feature_cols, rng, noise_fraction)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    return df


def rename_feature_columns(train_demo, test_demo, rna_cols, cnv_cols, keep_gene_names):
    if keep_gene_names:
        return train_demo, test_demo

    rename_map = {}
    rename_map.update({col: f"RNA_AAA{i + 1}" for i, col in enumerate(rna_cols)})
    rename_map.update({col: f"CNV_AAA{i + 1}" for i, col in enumerate(cnv_cols)})
    return train_demo.rename(columns=rename_map), test_demo.rename(columns=rename_map)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)
    ensure_columns(train_df, REQUIRED_COLUMNS, "Training data")
    ensure_columns(test_df, REQUIRED_COLUMNS, "Validation data")

    rna_cols = select_variable_features(train_df, test_df, "RNA_", args.rna_features)
    cnv_cols = select_variable_features(train_df, test_df, "CNV_", args.cnv_features)
    selected_cols = REQUIRED_COLUMNS + rna_cols + cnv_cols

    train_sample = stratified_sample(train_df[selected_cols], args.train_rows, args.seed)
    test_sample = stratified_sample(test_df[selected_cols], args.test_rows, args.seed + 1000)

    train_demo = anonymize_and_noise(train_sample, rna_cols + cnv_cols, 1, rng, args.noise)
    test_demo = anonymize_and_noise(test_sample, rna_cols + cnv_cols, len(train_demo) + 1, rng, args.noise)
    train_demo, test_demo = rename_feature_columns(
        train_demo, test_demo, rna_cols, cnv_cols, args.keep_gene_names
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / "TCGA_Discovery_Demo.csv"
    test_out = output_dir / "CPTAC_Validation_Demo.csv"
    train_demo.to_csv(train_out, index=False)
    test_demo.to_csv(test_out, index=False)

    print(f"Wrote {train_out} with shape {train_demo.shape}")
    print(f"Wrote {test_out} with shape {test_demo.shape}")
    print("These files are anonymized/noisy demo inputs and are not for scientific interpretation.")


if __name__ == "__main__":
    main()
