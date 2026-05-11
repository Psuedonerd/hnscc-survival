# HNSCC Survival Refactor Plan

This document is a practical checklist for turning the project into a clean, explainable repository. It is intentionally written as a low-stress roadmap: complete one box at a time.

## Current repository structure

```text
hnscc-survival/
├── docs/
│   └── HNSCC_Survival_Paper.pdf
├── scripts/
│   ├── analysis/
│   │   └── pathway_enrichment_v7.py
│   ├── data_processing/
│   │   └── 00_CPTAC_and_TCGA_data_processing.Rmd
│   └── modeling/
│       └── optimized_survival_v7.py
├── README.md
└── REFACTOR_PLAN.md
```

## What each current file does

| File | Purpose | What to be able to explain |
| --- | --- | --- |
| `scripts/data_processing/00_CPTAC_and_TCGA_data_processing.Rmd` | Downloads and harmonizes CPTAC/TCGA-style data inputs and prepares analysis-ready objects/tables. | Data sources, HPV-negative filtering, survival endpoint construction, tumor sample filtering, protein-coding/autosomal gene filtering, and how clinical variables are harmonized. |
| `scripts/modeling/optimized_survival_v7.py` | Main survival-model training and testing script for TCGA-to-CPTAC validation. | Train/test datasets, clinical harmonization, RNA/SCNA feature blocks, feature selection, Cox/RSF/ensemble logic, C-index evaluation, and saved outputs. |
| `scripts/analysis/pathway_enrichment_v7.py` | Downstream enrichment analysis for selected RNA/CNV model features. | How gene symbols are extracted from model features, what gene-set libraries are tested, and what tables/plots are produced. |
| `docs/HNSCC_Survival_Paper.pdf` | Current paper/manuscript artifact. | Use as a written reference for the biological question, methods, and results narrative. |

## Files to pull from mentor repo later

Mentor repo path from notes: `hnscc_survival_2026/CPTAC/scripts`.

When you have local access to that private repo, copy these into the matching folders below:

| Mentor file | Suggested destination | Status / note |
| --- | --- | --- |
| `check_v7_params.py` | `scripts/modeling/check_v7_params.py` | Use to document/reproduce the winning model parameters. |
| `create_supplementary_table.py` | `scripts/analysis/create_supplementary_table.py` | Presentation/manuscript supplementary table generation. |
| `enrichment_dotplot.py` | `scripts/analysis/enrichment_dotplot.py` | Plotting helper for enrichment results. |
| `km_curves_rsf.py` | `scripts/analysis/km_curves_rsf.py` | Kaplan-Meier plots using RSF/model risk groups. |
| `model_comparison_figure.py` | `scripts/analysis/model_comparison_figure.py` | Main model comparison figure. |
| `full_model_comparison.py` | `scripts/analysis/full_model_comparison.py` | Needs review: likely compares all trained model variants and exports comparison metrics/figures. |
| `save_v7_model.py` | Do **not** keep as a separate long-term script unless needed. Move important save/load logic into `scripts/modeling/optimized_survival_v7.py`. | Mentor note said to extract the important parts. |
| `v7_comparison.py` | Do **not** keep as a separate long-term script unless needed. Move important comparison logic into `scripts/modeling/optimized_survival_v7.py` or `scripts/analysis/full_model_comparison.py`. | Mentor note said to transport important pieces. |
| slide deck PDF | `docs/` | Add the deck as a PDF and link it from the README. |

## How to get files from the private mentor repo

You do **not** need to make that private repo public. Any of these approaches works:

### Option A: clone both repos locally, then copy files

```bash
# In a parent folder, not inside this repo
git clone git@github.com:<mentor-or-org>/hnscc_survival_2026.git

# Then from this repo root
cp ../hnscc_survival_2026/CPTAC/scripts/check_v7_params.py scripts/modeling/
cp ../hnscc_survival_2026/CPTAC/scripts/create_supplementary_table.py scripts/analysis/
cp ../hnscc_survival_2026/CPTAC/scripts/enrichment_dotplot.py scripts/analysis/
cp ../hnscc_survival_2026/CPTAC/scripts/km_curves_rsf.py scripts/analysis/
cp ../hnscc_survival_2026/CPTAC/scripts/model_comparison_figure.py scripts/analysis/
cp ../hnscc_survival_2026/CPTAC/scripts/full_model_comparison.py scripts/analysis/
```

### Option B: download a ZIP from GitHub

If GitHub access works in your browser but SSH is annoying, download the mentor repo as a ZIP, unzip it outside this repository, and copy the scripts from `CPTAC/scripts` into the folders above.

### Option C: add mentor repo as a temporary remote

Only do this if you are comfortable with git remotes:

```bash
git remote add mentor git@github.com:<mentor-or-org>/hnscc_survival_2026.git
git fetch mentor
```

After fetching, you can inspect files from the mentor branch without merging the whole project.

## Refactor milestones

### Milestone 1: Make the repository understandable

- [x] Put data-processing, modeling, analysis, and docs into separate folders.
- [x] Add a checklist of files still needed from the mentor repo.
- [ ] Update script headers/docstrings so every script says: inputs, outputs, and example command.
- [ ] Add a single `requirements.txt` or `environment.yml` once dependencies are confirmed.

### Milestone 2: Make the data-processing story explainable

- [ ] In the R Markdown, add a short paragraph before each major section explaining why that processing step exists.
- [ ] Clearly identify which objects/files are final outputs of data processing.
- [ ] Separate unfinished exploratory chunks from required pipeline chunks.

### Milestone 3: Consolidate model training

- [ ] Keep `scripts/modeling/optimized_survival_v7.py` as the main training/testing entry point.
- [ ] Move important model-saving logic from `save_v7_model.py` into `optimized_survival_v7.py`.
- [ ] Move important v7-comparison logic into `optimized_survival_v7.py` or `scripts/analysis/full_model_comparison.py`.
- [ ] Add comments/docstrings for stability selection and RSF tuning so you can explain them in plain language.

### Milestone 4: Consolidate analysis and figures

- [ ] Keep pathway/enrichment logic in `scripts/analysis/`.
- [ ] Keep KM curves, enrichment dotplots, supplementary tables, and model-comparison figures in `scripts/analysis/`.
- [ ] Add output folders consistently, for example `results/tables/` and `results/figures/`.

## Immediate next thing to do

Do **not** try to understand everything at once. The next useful action is simply:

1. Get local access to the private mentor repo.
2. Copy only the listed files from `CPTAC/scripts` into this organized repo.
3. Run `git status` and verify the files appear in the intended folders.
4. Then review one script at a time.

You are not cooked. This is a normal refactor problem: first organize, then document, then consolidate.
