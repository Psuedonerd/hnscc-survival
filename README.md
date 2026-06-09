# HNSCC survival demo

This repository demonstrates a survival-modeling workflow for HNSCC using small demo CSV files. The demo files preserve the structure expected by the pipeline while using anonymized patient/sample IDs and perturbed numeric values. Gene feature names are retained in this demo so the enrichment scripts can run with interpretable gene symbols.

The full project workflow uses harmonized discovery and validation cohorts. This repository is configured as a lightweight demo so it can be cloned and run without distributing the full local data files.

## Requirements

Use Python 3.10 or 3.11 if possible. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The required Python packages are:

- `numpy`
- `pandas`
- `scikit-learn`
- `scikit-survival`
- `matplotlib`
- `gseapy`
- `lifelines`
- `joblib`

`scikit-survival` is the package most likely to be platform-sensitive. If pip cannot install it, use a conda environment with `scikit-survival` from `conda-forge`, then install the remaining pip packages from `requirements.txt`.

## Demo inputs

The demo uses:

```text
demo_data_gene_names/TCGA_Discovery_Demo.csv
demo_data_gene_names/CPTAC_Validation_Demo.csv
```

Each file contains:

```text
patient_id
Age, Gender, Stage, T_Stage, N_Stage, Grade, Alcohol_History, Pack_Years
OS_days, OS_event
RNA_* feature columns
CNV_* feature columns
```

The patient/sample IDs are anonymized. The RNA and CNV feature names are retained as gene-level feature labels so pathway enrichment can be demonstrated.

## Main workflow

Run the fast block-constrained survival-model sweep:

```bash
python scripts/training/train_survival_models.py \
  --train-data demo_data_gene_names/TCGA_Discovery_Demo.csv \
  --test-data demo_data_gene_names/CPTAC_Validation_Demo.csv \
  --output results/demo_model_runs \
  --mode block \
  --n-jobs 1 \
  --seed 42
```

The command prints a timestamped result folder such as:

```text
results/demo_model_runs/run_v7_YYYYMMDD_HHMMSS
```

Use that printed result path in the downstream commands.

The run folder contains:

```text
all_results.csv
best_model_features.csv
selected_rna_features.csv
selected_scna_features.csv
top20_scna_features.csv
```

## Model verification

Rerun one selected block-constrained RSF configuration:

```bash
python scripts/training/check_v7_params.py \
  --train-data demo_data_gene_names/TCGA_Discovery_Demo.csv \
  --test-data demo_data_gene_names/CPTAC_Validation_Demo.csv \
  --n-rna 60 \
  --n-scna 8 \
  --n-jobs 1 \
  --seed 42
```

For the included demo data, `RNA60_SCNA8` was the strongest block-constrained configuration during testing.

## Model comparison figure

Use the result path printed by the training run:

```bash
python scripts/analysis/model_comparison_figure.py \
  --results-csv results/demo_model_runs/run_v7_YYYYMMDD_HHMMSS/all_results.csv \
  --output results/figures/model_comparison_demo.png
```

## Supplementary feature table

```bash
python scripts/analysis/create_supplementary_table.py \
  --features results/demo_model_runs/run_v7_YYYYMMDD_HHMMSS/best_model_features.csv \
  --output results/tables/supplementary_model_features_demo.csv
```

## Kaplan-Meier risk groups

```bash
python scripts/analysis/km_curves_rsf.py \
  --train-data demo_data_gene_names/TCGA_Discovery_Demo.csv \
  --test-data demo_data_gene_names/CPTAC_Validation_Demo.csv \
  --output results/figures/km_curves_rsf_demo.png \
  --n-rna 60 \
  --n-scna 8 \
  --n-jobs 1 \
  --seed 42
```

This retrains the selected block-constrained RSF model and plots lower-risk versus higher-risk groups in the validation cohort.

## Pathway enrichment

Because this demo keeps gene feature names, pathway enrichment can be run from the selected model features:

```bash
python scripts/analysis/run_pathway_enrichment.py \
  --gene-file results/demo_model_runs/run_v7_YYYYMMDD_HHMMSS/best_model_features.csv \
  --output results/enrichment_demo \
  --organism human
```

This step uses `gseapy` to query Enrichr, so it requires internet access. It writes a timestamped enrichment folder under:

```text
results/enrichment_demo/
```

Use the enrichment result path printed by the enrichment script:

```text
results/enrichment_demo/enrichment_YYYYMMDD_HHMMSS
```

## Enrichment dot plot

Use a completed enrichment folder as input:

```bash
python scripts/analysis/enrichment_dotplot.py \
  --input-dir results/enrichment_demo/enrichment_YYYYMMDD_HHMMSS \
  --output results/figures/enrichment_dotplot_demo.png
```

## Rebuilding demo inputs

The included demo inputs are already prepared. If you have local harmonized source CSVs and need to rebuild the demo files, run:

```bash
python scripts/data_processing/create_demo_inputs.py \
  --train-data /path/to/TCGA_Discovery_Harmonized_Full_Data.csv \
  --test-data /path/to/CPTAC_Validation_Harmonized_Full_Data.csv \
  --output-dir demo_data_gene_names \
  --keep-gene-names \
  --seed 42
```

Use `--keep-gene-names` when you want pathway enrichment to remain interpretable. Without that flag, omics feature names are anonymized as `RNA_AAA*` and `CNV_AAA*`, which is safer but not useful for biological enrichment.

## What this demo shows

The demo shows that the repository can:

- load harmonized discovery and validation CSV files
- harmonize clinical variables
- select top RNA and CNV/SCNA features by univariate survival association
- train block-constrained random survival forest models
- evaluate validation C-index
- export model result tables and selected features
- generate model-comparison, Kaplan-Meier, enrichment, and supplementary-table outputs
