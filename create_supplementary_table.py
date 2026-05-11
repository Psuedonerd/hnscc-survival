#!/usr/bin/env python3
"""
Create Supplementary Feature Table for RSF Model

Simple script to list all 95 features in the RSF model with descriptions.
Output is printed to console for easy copy-paste to Google Docs.
"""

import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("/restricted/projectnb/montilab-p/personal/pnarnur/projects/hnscc_genomics_misc/CPTAC/results/saved_model_20260126_165704")

def create_descriptions(feature_names):
  """Create human-readable descriptions for all features."""
  descriptions = []

  clinical_desc = {
    'Age': 'Patient age at diagnosis (years)',
    'Pack_Years': 'Cumulative tobacco exposure (pack-years)',
    'Gender_Male': 'Male sex (vs Female reference)',
    'Stage_II': 'Tumor stage II (vs Stage I reference)',
    'Stage_III': 'Tumor stage III (vs Stage I reference)',
    'Stage_IV': 'Tumor stage IV (vs Stage I reference)',
    'Stage_Unknown': 'Tumor stage unknown (vs Stage I reference)',
    'T_Stage_T1': 'Primary tumor T1 (vs T0 reference)',
    'T_Stage_T2': 'Primary tumor T2 (vs T0 reference)',
    'T_Stage_T3': 'Primary tumor T3 (vs T0 reference)',
    'T_Stage_T4': 'Primary tumor T4 (vs T0 reference)',
    'T_Stage_TX': 'Primary tumor TX (vs T0 reference)',
    'N_Stage_N1': 'Lymph node stage N1 (vs N0 reference)',
    'N_Stage_N2': 'Lymph node stage N2 (vs N0 reference)',
    'N_Stage_N3': 'Lymph node stage N3 (vs N0 reference)',
    'N_Stage_NX': 'Lymph node stage NX (vs N0 reference)',
    'Grade_G2': 'Moderately differentiated (vs G1 reference)',
    'Grade_G3': 'Poorly differentiated (vs G1 reference)',
    'Grade_GX': 'Grade unknown (vs G1 reference)',
    'Alcohol_History_Unknown': 'Alcohol history unknown (vs No reference)',
    'Alcohol_History_Yes': 'History of alcohol consumption (vs No reference)'
  }

  for fname in feature_names:
    if fname in clinical_desc:
      descriptions.append(clinical_desc[fname])
    elif fname.startswith("RNA_"):
      gene = fname.replace("RNA_", "")
      descriptions.append(f"{gene} gene expression (log2 normalized)")
    elif fname.startswith("CNV_"):
      gene = fname.replace("CNV_", "")
      descriptions.append(f"{gene} copy number alteration")
    else:
      descriptions.append("Unknown feature")

  return descriptions


def main():
  print("Loading model metadata...")
  metadata = joblib.load(MODEL_DIR / "metadata.joblib")

  feature_names = metadata["feature_names"]
  n_clinical = metadata["n_clinical"]
  n_rna = metadata["n_rna"]
  n_scna = metadata["n_scna"]

  print(f"\nExtracted {len(feature_names)} features:")
  print(f"  - Clinical: {n_clinical}")
  print(f"  - RNA: {n_rna}")
  print(f"  - SCNA: {n_scna}")

  feature_types = (
    ["Clinical"] * n_clinical +
    ["Transcriptomic"] * n_rna +
    ["Copy Number"] * n_scna
  )

  descriptions = create_descriptions(feature_names)

  table = pd.DataFrame({
    'Feature_Name': feature_names,
    'Feature_Type': feature_types,
    'Description': descriptions
  })

  table.insert(0, 'Feature_Number', range(1, len(table) + 1))

  print("\n" + "=" * 120)
  print("SUPPLEMENTARY TABLE: RSF MODEL FEATURES (95 features)")
  print("=" * 120)
  print()

  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_colwidth', 60)
  pd.set_option('display.width', 120)

  print(table.to_string(index=False))

  print("\n" + "=" * 120)
  print(f"Summary: {len(table)} total features")
  print(f"  - Clinical: {sum(table['Feature_Type'] == 'Clinical')}")
  print(f"  - Transcriptomic: {sum(table['Feature_Type'] == 'Transcriptomic')}")
  print(f"  - Copy Number: {sum(table['Feature_Type'] == 'Copy Number')}")
  print("=" * 120)

  csv_path = "supplementary_table_features.csv"
  table.to_csv(csv_path, index=False)
  print(f"\n✓ Also saved to {csv_path} for backup")


if __name__ == "__main__":
  main()
