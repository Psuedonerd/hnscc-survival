# HNSCC Survival

This repository contains code and documentation for a head and neck squamous cell carcinoma (HNSCC) survival-analysis project focused on HPV-negative disease and clinicogenomic prediction.

The project uses data from:

- The Cancer Genome Atlas (TCGA)
- Clinical Proteomic Tumor Analysis Consortium (CPTAC)

The current paper/manuscript artifact is available at [`docs/HNSCC_Survival_Paper.pdf`](docs/HNSCC_Survival_Paper.pdf).

## Repository layout

```text
hnscc-survival/
├── docs/
│   └── HNSCC_Survival_Paper.pdf
├── scripts/
│   ├── analysis/
│   │   └── run_pathway_enrichment.py
│   ├── data_processing/
│   │   └── prepare_tcga_cptac_data.Rmd
│   └── modeling/
│       └── train_survival_models.py
├── .gitignore
└── README.md
```

## File inventory and include/exclude recommendation

| File | Include? | Role | Why it matters |
| --- | --- | --- | --- |
| `scripts/data_processing/prepare_tcga_cptac_data.Rmd` | **Yes** | Main data-preparation notebook. | This is the script you should be able to discuss in methods: it downloads/loads CPTAC data, prepares TCGA and CPTAC RNA/GISTIC objects, filters to HPV-negative cases with usable survival data, harmonizes shared features, and exports modeling CSVs. |
| `scripts/modeling/train_survival_models.py` | **Yes** | Main model-training and validation script. | This is the core analysis script: it trains TCGA-discovery models, validates on CPTAC, compares multiple RNA/SCNA inclusion strategies, and saves model-comparison and selected-feature tables. |
| `scripts/analysis/run_pathway_enrichment.py` | **Yes** | Downstream biological interpretation script. | This turns selected RNA/CNV model features into gene symbols and runs Enrichr pathway/gene-set analyses. |
| `docs/HNSCC_Survival_Paper.pdf` | **Yes, if allowed by your mentor/project policy** | Manuscript/paper artifact. | Useful for orienting readers and connecting code to the written project narrative. Remove it only if there are sharing/copyright/privacy concerns. |
| Local raw data, `CPTAC/data/`, `data/`, `results/`, `outputs/` | **No** | Downloaded data and generated outputs. | These can be large, sensitive, or machine-specific. Keep them local or share via an approved data store, not git. |
| Python/R cache files, editor files | **No** | Local machine artifacts. | These are ignored by `.gitignore` and should not be committed. |

## End-to-end workflow

### 1. Prepare harmonized TCGA and CPTAC tables

Run the R Markdown notebook:

```r
rmarkdown::render("scripts/data_processing/prepare_tcga_cptac_data.Rmd")
```

Expected final modeling inputs:

```text
CPTAC/data/TCGA_Discovery_Harmonized_Full_Data.csv
CPTAC/data/CPTAC_Validation_Harmonized_Full_Data.csv
```

What the notebook does:

- loads CPTAC LinkedOmics/freeze files and TCGA processed objects;
- filters to HPV-negative HNSCC samples with non-missing overall survival time/event fields;
- creates RNA-seq and GISTIC/SCNA `SummarizedExperiment` objects;
- restricts features to protein-coding autosomal genes;
- intersects genes present in both TCGA and CPTAC so discovery and validation features match; and
- exports wide CSV tables with standardized clinical columns plus `RNA_*` and `CNV_*` feature columns.

Important methods points to know:

- **Discovery cohort:** TCGA.
- **Validation cohort:** CPTAC.
- **Endpoint:** overall survival, represented as `OS_days` and `OS_event`.
- **Disease subset:** HPV-negative HNSCC.
- **Feature blocks:** clinical variables, RNA expression features, and CNV/SCNA GISTIC features.
- **Current limitation:** BMI is intentionally omitted until it can be harmonized across both cohorts.

### 2. Train and compare survival models

After the CSV files exist locally, run:

```bash
python scripts/modeling/train_survival_models.py \
  --train-data CPTAC/data/TCGA_Discovery_Harmonized_Full_Data.csv \
  --test-data CPTAC/data/CPTAC_Validation_Harmonized_Full_Data.csv \
  --output results/model_runs \
  --n-jobs 4 \
  --seed 42
```

The script creates a timestamped directory under `results/model_runs/`. It now validates required clinical/survival columns, converts common `OS_event` encodings safely, aligns RNA/CNV feature columns shared by both cohorts, imputes/scales omics blocks using training data only, and always writes a best-model feature table for downstream enrichment. Key outputs include:

| Output | Meaning |
| --- | --- |
| `all_results.csv` | Model-comparison table sorted by CPTAC test C-index. |
| `best_model_features.csv` | Clinical/RNA/SCNA feature list for the best qualifying model when success criteria are met. |
| `selected_rna_features.csv` | RNA features selected for the best model with univariate C-index deviation scores. |
| `selected_scna_features.csv` | SCNA/CNV features selected for the best model with univariate C-index deviation scores. |
| `top20_scna_features.csv` | Highest-ranking SCNA features for reference and interpretation. |

Modeling strategies currently implemented:

1. **Block-constrained feature selection:** forces RNA and SCNA feature blocks to be represented.
2. **Priority-lasso-style hierarchical fitting:** evaluates clinical → RNA → SCNA blocks.
3. **Late-fusion ensemble:** trains separate RSF-style components and combines their risk predictions.
4. **IPF-lasso-style penalty strategy:** encourages SCNA inclusion by changing penalty treatment across blocks.
5. **Stability-selected multi-block model:** uses bootstrap feature-selection frequency to identify stable RNA and SCNA candidates.

The hard success criteria in the script are:

- at least 4 selected SCNA features; and
- CPTAC test C-index at least `0.6448`, with `0.65` as the aspirational target.

### 3. Run pathway enrichment on selected model genes

Once model outputs exist, run enrichment from the selected-feature table:

```bash
python scripts/analysis/run_pathway_enrichment.py \
  --gene-file results/model_runs/<run_directory>/best_model_features.csv \
  --output results/enrichment
```

The script extracts gene symbols from `RNA_*`, `CNV_*`, and `SCNA_*` features, records whether each gene came from RNA, SCNA, or both, and runs Enrichr analyses across GO, KEGG, Reactome, Hallmark, WikiPathways, transcription-factor, cell-marker, and disease-oriented libraries. If you do not yet have `best_model_features.csv`, it can also use `--train-data` to create an exploratory high-variance fallback gene list.

Key enrichment outputs include:

| Output | Meaning |
| --- | --- |
| `input_genes.csv` | Gene list and source modality (`RNA`, `SCNA`, or both). |
| `*_results.csv` | Full Enrichr results for each library. |
| `*_barplot.png` | Top-term barplots when plotting dependencies are available. |
| `enrichment_summary_report.txt` | Human-readable pathway-enrichment summary. |
| `all_results_combined.csv` | Combined enrichment results across libraries. |
| `significant_terms_combined.csv` or `suggestive_terms_combined.csv` | Significant or relaxed-threshold combined terms, depending on results. |

## Dependencies

The project uses both R and Python.

### R packages used by data preparation

- `tidyverse`
- `data.table`
- `janitor`
- `here`
- `SummarizedExperiment`
- `TCGAbiolinks`
- `Biobase`
- `S4Vectors`
- `matrixStats`
- `org.Hs.eg.db`
- `AnnotationDbi`
- `testthat` through explicit `testthat::expect_equal()` calls

### Python packages used by modeling/enrichment

- `numpy`
- `pandas`
- `scikit-learn`
- `scikit-survival`
- `gseapy` for Enrichr enrichment
- `matplotlib` for enrichment plots

A future improvement would be to add a pinned `environment.yml` or `requirements.txt`, but this commit intentionally avoids creating new files beyond the existing repository files.

## What to say if asked to explain the project

A concise explanation:

> This project builds a survival-prediction workflow for HPV-negative HNSCC. TCGA is used as the discovery/training cohort and CPTAC is used as an external validation cohort. The data-processing notebook harmonizes clinical survival fields and matched RNA/CNV gene features across cohorts. The modeling script compares several multi-omic survival-model strategies that force or encourage SCNA inclusion, then evaluates them by C-index on CPTAC. The enrichment script interprets the model-selected genes using pathway and gene-set enrichment.

## Refactoring performed

The current refactor did not split scripts into additional files. It only renamed and clarified existing files:

| Previous name | Current name | Reason |
| --- | --- | --- |
| `00_CPTAC_and_TCGA_data_processing.Rmd` | `scripts/data_processing/prepare_tcga_cptac_data.Rmd` | Describes the actual action: prepare harmonized TCGA/CPTAC modeling data. |
| `optimized_survival_v7.py` | `scripts/modeling/train_survival_models.py` | Describes the script as the main model-training entry point rather than a historical version. |
| `pathway_enrichment_v7.py` | `scripts/analysis/run_pathway_enrichment.py` | Describes the script as the pathway-enrichment entry point rather than a historical version. |
| `HNSCC_Survival_Paper.pdf` | `docs/HNSCC_Survival_Paper.pdf` | Keeps manuscript material separate from runnable code. |

The earlier separate `REFACTOR_PLAN.md` was removed because this README is now the single project guide.

## Notes about mentor-repository files

The files currently present here are sufficient for the visible workflow above. If additional mentor scripts are later added, include them only if they serve one of these reproducibility purposes:

- regenerate a manuscript figure;
- regenerate a supplementary table;
- reproduce final model parameters;
- create Kaplan-Meier/risk-group plots used in the paper or presentation; or
- perform a documented sensitivity/model-comparison analysis.

Do not include one-off scratch scripts unless their logic is folded into the main data, modeling, or analysis scripts.

## Data and output policy

Do not commit raw downloaded data, generated model outputs, generated figures, or local cache files. The `.gitignore` excludes common local folders such as `CPTAC/data/`, `data/`, `outputs/`, and `results/`.
