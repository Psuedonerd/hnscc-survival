# HNSCC Survival

Code for a head and neck squamous cell carcinoma (HNSCC) survival-analysis project focused on HPV-negative disease. The workflow prepares matched TCGA discovery and CPTAC validation cohorts, trains clinicogenomic survival models from clinical variables, RNA expression, and copy-number alteration features, and generates downstream biological-interpretation and figure outputs.

The current manuscript artifact is available at [`docs/HNSCC_Survival_Paper.pdf`](docs/HNSCC_Survival_Paper.pdf).

## Repository layout

```text
hnscc-survival/
├── docs/
│   └── HNSCC_Survival_Paper.pdf
├── scripts/
│   ├── analysis/
│   │   ├── create_supplementary_table.py
│   │   ├── enrichment_dotplot.py
│   │   ├── km_curves_rsf.py
│   │   ├── model_comparison_figure.py
│   │   └── run_pathway_enrichment.py
│   ├── data_processing/
│   │   └── prepare_tcga_cptac_data.Rmd
│   └── training/
│       ├── check_v7_params.py
│       └── train_survival_models.py
├── requirements.txt
└── README.md
```

## Data and generated outputs

Raw, intermediate, and generated result files are intentionally not committed. By default, the scripts assume local project folders such as:

```text
CPTAC/data/
results/model_runs/
results/enrichment/
results/figures/
results/tables/
```

Expected harmonized modeling inputs after data preparation are:

```text
CPTAC/data/TCGA_Discovery_Harmonized_Full_Data.csv
CPTAC/data/CPTAC_Validation_Harmonized_Full_Data.csv
```

These wide CSVs should contain standardized survival columns (`OS_days`, `OS_event`), clinical columns (`Age`, `Gender`, `Stage`, `T_Stage`, `N_Stage`, `Grade`, `Alcohol_History`, `Pack_Years`), RNA features prefixed with `RNA_`, and copy-number/SCNA features prefixed with `CNV_`.

## Environment setup

### Python

Use Python 3.10+ if possible. Create and activate a virtual environment, then install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`scikit-survival` may require platform-specific compiled dependencies. If `pip install scikit-survival` is not supported on your system, install it with conda/mamba instead:

```bash
mamba create -n hnscc-survival python=3.11 numpy pandas scikit-learn scikit-survival matplotlib lifelines gseapy joblib
mamba activate hnscc-survival
```

### R

The data processing file is an R Markdown workflow. Install the R packages used in `scripts/data_processing/prepare_tcga_cptac_data.Rmd`, including `tidyverse`, `here`, `SummarizedExperiment`, `cBioPortalData`, `curatedTCGAData`, `TCGAbiolinks`, `biomaRt`, `janitor`, and related Bioconductor dependencies.

## Workflow

### 1. Prepare TCGA and CPTAC survival tables

Either Knit or render the R Markdown notebook from the repository root:

```r
rmarkdown::render("scripts/data_processing/prepare_tcga_cptac_data.Rmd")
```

The notebook prepares HPV-negative HNSCC discovery and validation cohorts, aligns clinical and survival variables, restricts omics features to shared genes between TCGA and CPTAC, and writes the two harmonized CSVs under `CPTAC/data/`.

### 2. Train and compare survival models

Run the main Python modeling script after the harmonized CSVs exist:

```bash
python scripts/modeling/train_survival_models.py \
  --train-data CPTAC/data/TCGA_Discovery_Harmonized_Full_Data.csv \
  --test-data CPTAC/data/CPTAC_Validation_Harmonized_Full_Data.csv \
  --output results/model_runs \
  --n-jobs 4 \
  --seed 42
```

The script writes a timestamped output directory under `results/model_runs/`. Important outputs include:

| Output | Description |
| --- | --- |
| `all_results.csv` | Model-comparison table sorted by CPTAC validation C-index. |
| `best_model_features.csv` | Clinical, RNA, and SCNA features from the selected model. |
| `selected_rna_features.csv` | Selected RNA features and univariate C-index deviation scores. |
| `selected_scna_features.csv` | Selected copy-number features and univariate C-index deviation scores. |
| `top20_scna_features.csv` | Highest-ranking SCNA features for interpretation. |

To rerun the manuscript-highlighted block-constrained RSF configuration directly:

```bash
python scripts/modeling/check_v7_params.py \
  --train-data CPTAC/data/TCGA_Discovery_Harmonized_Full_Data.csv \
  --test-data CPTAC/data/CPTAC_Validation_Harmonized_Full_Data.csv \
  --n-rna 70 \
  --n-scna 4 \
  --n-jobs 4
```

### 3. Run pathway enrichment

Use the selected model features from a model run:

```bash
python scripts/analysis/run_pathway_enrichment.py \
  --gene-file results/model_runs/<run_timestamp>/best_model_features.csv \
  --output results/enrichment
```

The enrichment script extracts gene symbols from `RNA_`, `CNV_`, and `SCNA_` feature names, runs Enrichr libraries through `gseapy`, and writes timestamped enrichment tables, plots, and a summary report.

### 4. Generate figures and tables

Create an ECM/protease-focused dot plot from enrichment result tables:

```bash
python scripts/analysis/enrichment_dotplot.py \
  --input-dir results/enrichment/<run_timestamp> \
  --output results/figures/enrichment_dotplot.png
```

Create a performance comparison plot from the modeling output:

```bash
python scripts/analysis/model_comparison_figure.py \
  --results-csv results/model_runs/<run_timestamp>/all_results.csv \
  --output results/figures/model_comparison.png
```

Create Kaplan-Meier curves for RSF validation-cohort risk groups:

```bash
python scripts/analysis/km_curves_rsf.py \
  --train-data CPTAC/data/TCGA_Discovery_Harmonized_Full_Data.csv \
  --test-data CPTAC/data/CPTAC_Validation_Harmonized_Full_Data.csv \
  --output results/figures/km_curves_rsf.png \
  --n-rna 70 \
  --n-scna 4 \
  --n-jobs 4
```

Create a supplementary feature table from model-selected features:

```bash
python scripts/analysis/create_supplementary_table.py \
  --features results/model_runs/<run_timestamp>/best_model_features.csv \
  --output results/tables/supplementary_model_features.csv
```

## Reproducibility notes

- The intended training/validation design is TCGA discovery followed by CPTAC validation.
- Overall survival is represented by `OS_days` and `OS_event`.
- The current feature blocks are clinical variables, RNA expression features, and SCNA/CNV features.
