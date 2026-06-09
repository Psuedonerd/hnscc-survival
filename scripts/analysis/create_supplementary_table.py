#!/usr/bin/env python3

import argparse
from pathlib import Path

pd = None


def load_pandas():
    global pd
    if pd is None:
        import pandas as _pd
        pd = _pd



CLINICAL_DESCRIPTIONS = {
    "Age": "Patient age at diagnosis (years)",
    "Pack_Years": "Cumulative tobacco exposure (pack-years)",
    "Gender_Male": "Male sex (vs female reference)",
    "Stage_II": "Tumor stage II (vs stage I reference)",
    "Stage_III": "Tumor stage III (vs stage I reference)",
    "Stage_IV": "Tumor stage IV (vs stage I reference)",
    "Stage_Unknown": "Tumor stage unknown (vs stage I reference)",
    "T_Stage_T1": "Primary tumor T1 (vs T0 reference)",
    "T_Stage_T2": "Primary tumor T2 (vs T0 reference)",
    "T_Stage_T3": "Primary tumor T3 (vs T0 reference)",
    "T_Stage_T4": "Primary tumor T4 (vs T0 reference)",
    "T_Stage_TX": "Primary tumor TX (vs T0 reference)",
    "N_Stage_N1": "Lymph-node stage N1 (vs N0 reference)",
    "N_Stage_N2": "Lymph-node stage N2 (vs N0 reference)",
    "N_Stage_N3": "Lymph-node stage N3 (vs N0 reference)",
    "N_Stage_NX": "Lymph-node stage NX (vs N0 reference)",
    "Grade_G2": "Moderately differentiated tumor grade (vs G1 reference)",
    "Grade_G3": "Poorly differentiated tumor grade (vs G1 reference)",
    "Grade_GX": "Unknown tumor grade (vs G1 reference)",
    "Alcohol_History_Unknown": "Alcohol history unknown (vs no reference)",
    "Alcohol_History_Yes": "History of alcohol consumption (vs no reference)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a supplementary feature table from best_model_features.csv or metadata.joblib"
    )
    parser.add_argument(
        "--features",
        help="CSV with `feature` and optional `type` columns, such as best_model_features.csv",
    )
    parser.add_argument(
        "--metadata",
        help="Optional saved-model metadata.joblib with feature_names/n_clinical/n_rna/n_scna",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for the formatted supplementary feature table",
    )
    return parser.parse_args()


def feature_type(feature, explicit_type=None):
    if explicit_type and str(explicit_type).lower() not in {"nan", "none", ""}:
        value = str(explicit_type).strip().lower()
        if value == "rna":
            return "Transcriptomic"
        if value in {"scna", "cnv", "copy number", "copy_number"}:
            return "Copy Number"
        if value == "clinical":
            return "Clinical"
        return str(explicit_type).strip()
    if str(feature).startswith("RNA_"):
        return "Transcriptomic"
    if str(feature).startswith(("CNV_", "SCNA_")):
        return "Copy Number"
    return "Clinical"


def describe_feature(feature):
    feature = str(feature)
    if feature in CLINICAL_DESCRIPTIONS:
        return CLINICAL_DESCRIPTIONS[feature]
    if feature.startswith("RNA_"):
        return f"{feature.removeprefix('RNA_')} gene expression"
    if feature.startswith("CNV_"):
        return f"{feature.removeprefix('CNV_')} copy-number alteration"
    if feature.startswith("SCNA_"):
        return f"{feature.removeprefix('SCNA_')} copy-number alteration"
    return "Clinical covariate"


def load_from_features_csv(path):
    load_pandas()
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError("--features CSV must contain a `feature` column")
    type_values = df["type"].tolist() if "type" in df.columns else [None] * len(df)
    return pd.DataFrame({"feature": df["feature"].tolist(), "explicit_type": type_values})


def load_from_metadata(path):
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Reading --metadata requires joblib; install joblib or use --features") from exc
    metadata = joblib.load(path)
    features = metadata["feature_names"]
    n_clinical = int(metadata.get("n_clinical", 0))
    n_rna = int(metadata.get("n_rna", 0))
    n_scna = int(metadata.get("n_scna", 0))
    types = ["clinical"] * n_clinical + ["RNA"] * n_rna + ["SCNA"] * n_scna
    if len(types) != len(features):
        types = [None] * len(features)
    load_pandas()
    return pd.DataFrame({"feature": features, "explicit_type": types})


def main():
    args = parse_args()
    if not args.features and not args.metadata:
        raise ValueError("Provide either --features or --metadata")
    raw = load_from_features_csv(args.features) if args.features else load_from_metadata(args.metadata)
    load_pandas()
    table = pd.DataFrame({
        "Feature_Number": range(1, len(raw) + 1),
        "Feature_Name": raw["feature"],
        "Feature_Type": [feature_type(f, t) for f, t in zip(raw["feature"], raw["explicit_type"])],
        "Description": [describe_feature(f) for f in raw["feature"]],
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Wrote {len(table)} features to {output}")


if __name__ == "__main__":
    main()
