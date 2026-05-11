# HNSCC Survival

This repository contains code and documentation for a head and neck squamous cell carcinoma (HNSCC) survival-analysis project focused on HPV-negative disease and clinicogenomic prediction.

The project uses data from:

- The Cancer Genome Atlas (TCGA)
- Clinical Proteomic Tumor Analysis Consortium (CPTAC)

The current paper/manuscript artifact is available at [`docs/HNSCC_Survival_Paper.pdf`](docs/HNSCC_Survival_Paper.pdf).

## Repository layout

```text
scripts/
  data_processing/   Data download, cleaning, harmonization, and analysis-ready dataset creation
  modeling/          Survival model training, feature selection, tuning, and validation
  analysis/          Downstream enrichment, figures, KM curves, and supplementary outputs
docs/                Paper, slide decks, and project documentation
```

## Current key files

- [`scripts/data_processing/00_CPTAC_and_TCGA_data_processing.Rmd`](scripts/data_processing/00_CPTAC_and_TCGA_data_processing.Rmd): CPTAC/TCGA data processing notebook.
- [`scripts/modeling/optimized_survival_v7.py`](scripts/modeling/optimized_survival_v7.py): main optimized survival-model training/testing script.
- [`scripts/analysis/pathway_enrichment_v7.py`](scripts/analysis/pathway_enrichment_v7.py): pathway-enrichment analysis for selected prognostic RNA/CNV features.
- [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md): checklist for pulling in mentor-repository files and completing the refactor.

## Refactor status

This repository is being refactored from project scripts into a cleaner, reproducible layout. The immediate goals are:

1. Keep data processing, modeling, and analysis scripts separated.
2. Pull selected scripts from the private mentor repository `hnscc_survival_2026/CPTAC/scripts`.
3. Consolidate important model-saving and v7-comparison logic into the main modeling workflow.
4. Document inputs, outputs, and example commands for each script.

See [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) for the detailed checklist.

## Private mentor repository note

The mentor repository does **not** need to be made public. Clone or download it locally, then copy only the needed scripts into this repository. Suggested destinations are listed in [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md).

## Important data note

Large raw data, generated outputs, and intermediate result folders should not be committed to git. The `.gitignore` excludes common local data and output locations such as `data/`, `CPTAC/data/`, `outputs/`, and `results/`.
