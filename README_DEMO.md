# HNSCC survival demo

This demo runs the survival-modeling pipeline on small anonymized CSV files. The demo files keep the same column structure as the full harmonized inputs, but patient/sample IDs and omics feature names are anonymized and numeric values have been perturbed. They are meant for reproducibility and presentation, not biological interpretation.

## Requirements

Use Python 3.10 or 3.11 if possible. From the repository root, create and activate a virtual environment:

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

`scikit-survival` is the package most likely to need extra attention. If pip cannot install it on a laptop, use conda instead:

```bash
conda env create -f environment.yml
conda activate hnscc-survival
```

## Demo inputs

The demo uses:

```text
demo_data/TCGA_Discovery_Demo.csv
demo_data/CPTAC_Validation_Demo.csv
```

Each file contains the required clinical/survival columns plus anonymized RNA and CNV/SCNA feature columns:

```text
patient_id
Age, Gender, Stage, T_Stage, N_Stage, Grade, Alcohol_History, Pack_Years
OS_days, OS_event
RNA_AAA*
CNV_AAA*
```

## Main demo workflow

Run the block-constrained survival-model sweep:

```bash
python scripts/training/train_survival_models.py \
  --train-data demo_data/TCGA_Discovery_Demo.csv \
  --test-data demo_data/CPTAC_Validation_Demo.csv \
  --output results/demo_model_runs \
  --mode block \
  --n-jobs 1 \
  --seed 42
```

After the command finishes, it prints a timestamped run folder such as:

```text
results/demo_model_runs/run_v7_YYYYMMDD_HHMMSS
```

Use that folder path in the commands below.

The run folder should contain:

```text
all_results.csv
best_model_features.csv
selected_rna_features.csv
selected_scna_features.csv
top20_scna_features.csv
```

## Verification script

Use this to rerun one selected block-constrained RSF model:

```bash
python scripts/training/check_v7_params.py \
  --train-data demo_data/TCGA_Discovery_Demo.csv \
  --test-data demo_data/CPTAC_Validation_Demo.csv \
  --n-rna 60 \
  --n-scna 8 \
  --n-jobs 1 \
  --seed 42
```

For the anonymized demo files, `RNA60_SCNA8` was the strongest block-constrained configuration during testing.

## Figures and tables

Create a model-comparison plot:

```bash
mkdir -p results/figures

python scripts/analysis/model_comparison_figure.py \
  --results-csv results/demo_model_runs/run_v7_YYYYMMDD_HHMMSS/all_results.csv \
  --output results/figures/model_comparison_demo.png
```

Create a formatted feature table:

```bash
mkdir -p results/tables

python scripts/analysis/create_supplementary_table.py \
  --features results/demo_model_runs/run_v7_YYYYMMDD_HHMMSS/best_model_features.csv \
  --output results/tables/supplementary_model_features_demo.csv
```

Create Kaplan-Meier risk-group curves:

```bash
python scripts/analysis/km_curves_rsf.py \
  --train-data demo_data/TCGA_Discovery_Demo.csv \
  --test-data demo_data/CPTAC_Validation_Demo.csv \
  --output results/figures/km_curves_rsf_demo.png \
  --n-rna 60 \
  --n-scna 8 \
  --n-jobs 1 \
  --seed 42
```

## Pathway enrichment

The anonymized demo feature names are not real gene symbols, so enrichment is only a pipeline check in this mode:

```bash
mkdir -p results/enrichment_demo

python scripts/analysis/run_pathway_enrichment.py \
  --gene-file results/demo_model_runs/run_v7_YYYYMMDD_HHMMSS/best_model_features.csv \
  --output results/enrichment_demo \
  --organism human
```

This writes an `input_genes.csv` and an enrichment summary report. Real enrichment tables require real gene symbols and network access to Enrichr through `gseapy`.

If enrichment creates `*_results.csv` files, make the dot plot:

```bash
python scripts/analysis/enrichment_dotplot.py \
  --input-dir results/enrichment_demo/enrichment_YYYYMMDD_HHMMSS \
  --output results/figures/enrichment_dotplot_demo.png
```

It is normal for the fully anonymized demo to skip this plot if no real enrichment terms are returned.

## Rebuilding demo files

Rebuild the fully anonymized demo inputs:

```bash
python scripts/data_processing/create_demo_inputs.py \
  --train-data real_data/TCGA_Discovery_Harmonized_Full_Data.csv \
  --test-data real_data/CPTAC_Validation_Harmonized_Full_Data.csv \
  --output-dir demo_data \
  --seed 42
```

If you need pathway-enrichment output for a presentation, preserve gene symbols while still anonymizing patient IDs:

```bash
python scripts/data_processing/create_demo_inputs.py \
  --train-data real_data/TCGA_Discovery_Harmonized_Full_Data.csv \
  --test-data real_data/CPTAC_Validation_Harmonized_Full_Data.csv \
  --output-dir demo_data_gene_names \
  --keep-gene-names \
  --seed 42
```

Then replace `demo_data/...` with `demo_data_gene_names/...` in the commands above.

## What the demo shows

The demo shows that the repository can load harmonized discovery and validation tables, harmonize clinical variables, select top RNA and CNV/SCNA features, train block-constrained random survival forest models, evaluate validation C-index, and generate downstream result tables and figures.
