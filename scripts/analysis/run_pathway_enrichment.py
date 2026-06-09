#!/usr/bin/env python3
"""
Run pathway enrichment for HNSCC survival-model genes.

Purpose:
    Take model-selected RNA/CNV features, extract gene symbols, run Enrichr
    overrepresentation analysis across multiple gene-set libraries, and write
    enrichment tables, plots, and a text summary report.

Inputs:
    --gene-file: CSV with a `feature` column containing RNA_* and/or CNV_*
        feature names, usually produced by the modeling script as
        `best_model_features.csv`.
    --train-data: Optional harmonized training CSV. If `--gene-file` is not
        supplied, the script selects the most variable RNA/CNV genes from this
        table as a fallback exploratory gene list.
    --output: Directory where timestamped enrichment outputs are written.

Example:
    python scripts/analysis/run_pathway_enrichment.py \
        --gene-file results/model_runs/<run>/best_model_features.csv \
        --output results/enrichment
"""

import argparse
import importlib
import importlib.util
import os
import tempfile
from datetime import datetime
from pathlib import Path

np = None
pd = None
plt = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pathway enrichment analysis for prognostic HNSCC model genes"
    )
    parser.add_argument(
        "--gene-file",
        help=(
            "CSV containing selected model features. Preferred columns are "
            "`feature` plus optional `type`; a `gene_symbol` column is also accepted."
        ),
    )
    parser.add_argument(
        "--train-data",
        help=(
            "Fallback harmonized training CSV. Used only when --gene-file is not "
            "provided; top genes are chosen by feature variance."
        ),
    )
    parser.add_argument(
        "--n-genes",
        type=int,
        default=75,
        help="Number of fallback genes to select from --train-data (default: 75)",
    )
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--organism", default="human", help="Enrichr organism (default: human)")
    return parser.parse_args()


def load_core_dependencies():
    """Import required analysis dependencies after argparse so --help always works."""
    global np, pd
    np = importlib.import_module("numpy")
    pd = importlib.import_module("pandas")


def load_matplotlib():
    """Load matplotlib only when plotting is requested and available."""
    global plt
    if importlib.util.find_spec("matplotlib") is None:
        return False
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "hnscc-survival-matplotlib"),
    )
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    return True


def get_gseapy():
    """Return the gseapy module if it is installed; otherwise return None."""
    if importlib.util.find_spec("gseapy") is None:
        return None
    return importlib.import_module("gseapy")


def clean_gene_symbol(gene):
    """Normalize a raw extracted gene name and drop obvious empty values."""
    if gene is None or pd.isna(gene):
        return None
    gene = str(gene).strip()
    if not gene or gene.lower() in {"nan", "none", "null"}:
        return None
    return gene


def add_gene_source(gene_sources, gene, source):
    """Add a gene/source pair while preserving first-seen order."""
    gene = clean_gene_symbol(gene)
    if gene is None:
        return
    if gene not in gene_sources:
        gene_sources[gene] = []
    if source not in gene_sources[gene]:
        gene_sources[gene].append(source)


def extract_gene_symbols(feature_names, feature_types=None):
    """
    Extract gene symbols from model feature names.

    Recognized omics prefixes:
        RNA_<GENE>   -> RNA source
        CNV_<GENE>   -> SCNA source
        SCNA_<GENE>  -> SCNA source

    Clinical features and unrecognized values are skipped.
    """
    gene_sources = {}
    feature_types = feature_types if feature_types is not None else [None] * len(feature_names)

    for raw_feature, raw_type in zip(feature_names, feature_types):
        if raw_feature is None or pd.isna(raw_feature):
            continue
        feature = str(raw_feature).strip()
        feature_type = "" if raw_type is None or pd.isna(raw_type) else str(raw_type).strip().lower()

        if feature.startswith("RNA_"):
            add_gene_source(gene_sources, feature.removeprefix("RNA_"), "RNA")
        elif feature.startswith("CNV_"):
            add_gene_source(gene_sources, feature.removeprefix("CNV_"), "SCNA")
        elif feature.startswith("SCNA_"):
            add_gene_source(gene_sources, feature.removeprefix("SCNA_"), "SCNA")
        elif feature_type in {"rna", "expression"}:
            add_gene_source(gene_sources, feature, "RNA")
        elif feature_type in {"scna", "cnv", "copy_number", "copy-number"}:
            add_gene_source(gene_sources, feature, "SCNA")

    return list(gene_sources.keys()), gene_sources


def load_gene_list_from_file(gene_file):
    """Load gene symbols and modality sources from a model/enrichment CSV."""
    gene_df = pd.read_csv(gene_file)
    print(f"  Columns in file: {list(gene_df.columns)}")
    print(f"  Shape: {gene_df.shape}")

    if "feature" in gene_df.columns:
        feature_types = gene_df["type"].tolist() if "type" in gene_df.columns else None
        return extract_gene_symbols(gene_df["feature"].tolist(), feature_types)

    if "gene_symbol" in gene_df.columns:
        gene_sources = {}
        source_col = None
        for candidate in ["source", "type", "modality"]:
            if candidate in gene_df.columns:
                source_col = candidate
                break
        for _, row in gene_df.iterrows():
            gene = clean_gene_symbol(row["gene_symbol"])
            if gene is None:
                continue
            raw_source = row[source_col] if source_col else "unknown"
            sources = str(raw_source).replace(";", "+").split("+")
            for source in sources:
                source = source.strip().upper()
                if source in {"CNV", "SCNA", "COPY_NUMBER", "COPY-NUMBER"}:
                    add_gene_source(gene_sources, gene, "SCNA")
                elif source in {"RNA", "EXPRESSION"}:
                    add_gene_source(gene_sources, gene, "RNA")
                else:
                    add_gene_source(gene_sources, gene, "unknown")
        return list(gene_sources.keys()), gene_sources

    # Last-resort behavior: treat the first column as feature names.
    return extract_gene_symbols(gene_df.iloc[:, 0].tolist())


def load_gene_list_from_train_data(train_data, n_genes):
    """
    Fallback gene-list creation from a harmonized training table.

    This is not a replacement for using selected model features. It lets the
    script produce exploratory enrichment output when only the harmonized CSV is
    available by selecting the highest-variance RNA/CNV columns.
    """
    train_df = pd.read_csv(train_data)
    feature_cols = [
        col for col in train_df.columns
        if col.startswith("RNA_") or col.startswith("CNV_") or col.startswith("SCNA_")
    ]
    if not feature_cols:
        raise ValueError("No RNA_/CNV_/SCNA_ feature columns found in --train-data")

    numeric_features = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    variances = numeric_features.var(axis=0, skipna=True).fillna(0).sort_values(ascending=False)
    selected_features = variances.head(max(1, n_genes)).index.tolist()
    print(
        f"  Selected {len(selected_features)} high-variance omics features "
        f"from {len(feature_cols)} candidates in --train-data"
    )
    return extract_gene_symbols(selected_features)


def format_pvalue(value):
    """Safely format p-values that may arrive as strings or missing values."""
    try:
        return f"{float(value):.2e}"
    except (TypeError, ValueError):
        return str(value)


def normalize_enrichr_results(results):
    """Ensure Enrichr results have the columns and numeric values downstream code needs."""
    if results is None or len(results) == 0:
        return None
    results = results.copy()
    if "Adjusted P-value" not in results.columns:
        if "P-value" in results.columns:
            results["Adjusted P-value"] = results["P-value"]
        else:
            raise ValueError("Enrichr results do not contain `Adjusted P-value` or `P-value`")
    results["Adjusted P-value"] = pd.to_numeric(results["Adjusted P-value"], errors="coerce")
    if "P-value" in results.columns:
        results["P-value"] = pd.to_numeric(results["P-value"], errors="coerce")
    results = results.dropna(subset=["Adjusted P-value"])
    return results.sort_values("Adjusted P-value")


def run_enrichr_analysis(gene_list, gene_sets, output_dir, organism="human", prefix="enrichr"):
    """Run Enrichr overrepresentation analysis and save the full results table."""
    gp = get_gseapy()
    if gp is None:
        print("gseapy is not installed; skipping Enrichr calls but keeping gene-list/report outputs")
        return None

    print(f"\nRunning Enrichr analysis with {len(gene_list)} genes...")
    print(f"Gene sets: {gene_sets}")
    print(
        f"Input genes: {gene_list[:10]}..." if len(gene_list) > 10 else f"Input genes: {gene_list}"
    )

    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism=organism,
            outdir=None,
            cutoff=1.0,
            verbose=False,
        )
        results = normalize_enrichr_results(enr.results)
    except Exception as exc:
        print(f"  Error running Enrichr for {prefix}: {exc}")
        return None

    if results is None or len(results) == 0:
        print("  No results returned from Enrichr")
        return None

    output_file = os.path.join(output_dir, f"{prefix}_results.csv")
    results.to_csv(output_file, index=False)
    print(f"  Saved {len(results)} total terms to {output_file}")

    sig_strict = results[results["Adjusted P-value"] < 0.05]
    sig_relaxed = results[results["Adjusted P-value"] < 0.25]
    print("  Results summary:")
    print(f"    - adj. p < 0.05 (strict): {len(sig_strict)}")
    print(f"    - adj. p < 0.25 (relaxed): {len(sig_relaxed)}")
    if "P-value" in results.columns:
        sig_nominal = results[results["P-value"] < 0.05]
        print(f"    - nominal p < 0.05: {len(sig_nominal)}")

    print("\n  Top 5 terms (by adjusted p-value):")
    for i, (_, row) in enumerate(results.head(5).iterrows(), start=1):
        term = str(row.get("Term", "<no term>"))
        term = term[:50] + "..." if len(term) > 50 else term
        pval = row.get("P-value", "N/A")
        adj_pval = row["Adjusted P-value"]
        overlap = row.get("Overlap", "N/A")
        print(f"    {i}. {term}")
        print(f"       p={format_pvalue(pval)}, adj.p={format_pvalue(adj_pval)}, overlap={overlap}")

    return results


def create_barplot(results_df, title, output_file, top_n=15, pval_threshold=0.25):
    """Create a horizontal bar plot of top enriched terms."""
    if results_df is None or len(results_df) == 0:
        return
    if not load_matplotlib():
        print("  matplotlib is not installed; skipping plot")
        return

    pval_col = "Adjusted P-value"
    term_col = "Term"
    if pval_col not in results_df.columns or term_col not in results_df.columns:
        print("  Required plotting columns are missing; skipping plot")
        return

    sig_results = results_df[results_df[pval_col] < pval_threshold].copy()
    if len(sig_results) == 0:
        sig_results = results_df.head(top_n).copy()
        title = f"{title} (Top {len(sig_results)}, none below adj. p < {pval_threshold})"

    sig_results = sig_results.sort_values(pval_col).head(top_n)
    sig_results["Term_short"] = sig_results[term_col].apply(
        lambda x: str(x)[:40] + "..." if len(str(x)) > 40 else str(x)
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(sig_results) * 0.4)))
    y_pos = range(len(sig_results))
    pvals = sig_results[pval_col].astype(float).clip(lower=1e-300)
    neg_log_p = -np.log10(pvals.values)
    colors = ["steelblue" if p < 0.05 else "lightsteelblue" for p in pvals.values]

    ax.barh(y_pos, neg_log_p, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sig_results["Term_short"].values)
    ax.invert_yaxis()
    ax.set_xlabel("-log10(adjusted p-value)")
    ax.set_title(title)
    ax.axvline(x=-np.log10(0.05), color="red", linestyle="--", alpha=0.5, label="adj.p=0.05")
    ax.axvline(x=-np.log10(0.25), color="orange", linestyle=":", alpha=0.5, label="adj.p=0.25")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {output_file}")


def create_summary_report(all_results, gene_list, output_dir, gene_sources=None):
    """Create a human-readable summary report of all enrichment analyses."""
    report_lines = [
        "=" * 70,
        "PATHWAY ENRICHMENT ANALYSIS REPORT",
        "HNSCC Prognostic Multi-Omics Gene Signature",
        "=" * 70,
        "",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Number of input genes: {len(gene_list)}",
    ]

    if gene_sources:
        rna_genes = [g for g, srcs in gene_sources.items() if "RNA" in srcs]
        scna_genes = [g for g, srcs in gene_sources.items() if "SCNA" in srcs]
        both_genes = [g for g, srcs in gene_sources.items() if "RNA" in srcs and "SCNA" in srcs]
        report_lines.extend([
            f"  - From RNA expression: {len(rna_genes)}",
            f"  - From SCNA/copy number: {len(scna_genes)}",
            f"  - Present in both modalities: {len(both_genes)}",
        ])

    report_lines.extend(["", "Input Genes:", "-" * 40])
    for i in range(0, len(gene_list), 5):
        chunk = gene_list[i:i + 5]
        annotated = []
        for gene in chunk:
            src = gene_sources.get(gene, []) if gene_sources else []
            if "RNA" in src and "SCNA" in src:
                annotated.append(f"{gene}[RNA+SCNA]")
            elif "SCNA" in src:
                annotated.append(f"{gene}[SCNA]")
            elif "RNA" in src:
                annotated.append(f"{gene}[RNA]")
            else:
                annotated.append(gene)
        report_lines.append("  " + ", ".join(annotated))

    if not any(df is not None and len(df) > 0 for df in all_results.values()):
        report_lines.extend([
            "",
            "No enrichment result tables were generated. This usually means gseapy is not installed,",
            "Enrichr was unreachable, or no tested gene-set library returned results.",
        ])

    for analysis_name, results_df in all_results.items():
        report_lines.extend(["", "=" * 70, f"Analysis: {analysis_name}", "=" * 70])
        if results_df is None or len(results_df) == 0:
            report_lines.append("  No results returned.")
            continue

        sig_strict = results_df[results_df["Adjusted P-value"] < 0.05]
        sig_relaxed = results_df[results_df["Adjusted P-value"] < 0.25]
        report_lines.extend([
            f"  Total terms tested: {len(results_df)}",
            f"  Significant (adj. p < 0.05): {len(sig_strict)}",
            f"  Suggestive (adj. p < 0.25): {len(sig_relaxed)}",
            "",
            "  Top 10 Enriched Terms:",
            "  " + "-" * 60,
        ])

        for idx, row in results_df.sort_values("Adjusted P-value").head(10).iterrows():
            term = str(row.get("Term", idx))
            term = term[:47] + "..." if len(term) > 50 else term
            pval = row["Adjusted P-value"]
            sig_marker = "***" if pval < 0.05 else ("*" if pval < 0.25 else "")
            overlap = row.get("Overlap", "N/A")
            report_lines.append(f"    {term} {sig_marker}")
            report_lines.append(f"      adj.p: {format_pvalue(pval)}, Overlap: {overlap}")

    report_text = "\n".join(report_lines)
    report_file = os.path.join(output_dir, "enrichment_summary_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nSummary report saved to: {report_file}")
    return report_text


def write_combined_results(all_results, output_dir):
    """Write combined all/significant/suggestive enrichment tables."""
    combined_all = []
    combined_sig = []
    combined_suggestive = []

    for name, df in all_results.items():
        if df is None or len(df) == 0:
            continue
        df_copy = df.copy()
        df_copy["Analysis"] = name
        combined_all.append(df_copy)
        combined_sig.append(df_copy[df_copy["Adjusted P-value"] < 0.05])
        combined_suggestive.append(df_copy[df_copy["Adjusted P-value"] < 0.25])

    if not combined_all:
        print("No enrichment result tables to combine")
        return

    combined_all_df = pd.concat(combined_all, ignore_index=True).sort_values("Adjusted P-value")
    combined_all_df.to_csv(os.path.join(output_dir, "all_results_combined.csv"), index=False)
    print(f"Saved {len(combined_all_df)} total terms to all_results_combined.csv")

    combined_sig_df = (
        pd.concat(combined_sig, ignore_index=True).sort_values("Adjusted P-value")
        if any(len(df) > 0 for df in combined_sig)
        else pd.DataFrame()
    )
    if len(combined_sig_df) > 0:
        combined_sig_df.to_csv(os.path.join(output_dir, "significant_terms_combined.csv"), index=False)
        print(f"Saved {len(combined_sig_df)} significant terms (adj.p < 0.05)")
    else:
        print("No terms with adj.p < 0.05 found")

    combined_suggestive_df = (
        pd.concat(combined_suggestive, ignore_index=True).sort_values("Adjusted P-value")
        if any(len(df) > 0 for df in combined_suggestive)
        else pd.DataFrame()
    )
    if len(combined_suggestive_df) > 0:
        combined_suggestive_df.to_csv(os.path.join(output_dir, "suggestive_terms_combined.csv"), index=False)
        print(f"Saved {len(combined_suggestive_df)} suggestive terms (adj.p < 0.25)")


def main():
    args = parse_args()
    load_core_dependencies()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"enrichment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PATHWAY ENRICHMENT ANALYSIS")
    print("=" * 70)
    print(f"Output directory: {output_dir}")

    if args.gene_file:
        if not os.path.exists(args.gene_file):
            raise FileNotFoundError(f"--gene-file does not exist: {args.gene_file}")
        print(f"\nLoading genes from: {args.gene_file}")
        gene_symbols, gene_sources = load_gene_list_from_file(args.gene_file)
    elif args.train_data:
        if not os.path.exists(args.train_data):
            raise FileNotFoundError(f"--train-data does not exist: {args.train_data}")
        print(f"\nNo --gene-file supplied; selecting fallback genes from: {args.train_data}")
        gene_symbols, gene_sources = load_gene_list_from_train_data(args.train_data, args.n_genes)
    else:
        raise ValueError("Provide either --gene-file or --train-data")

    gene_symbols = [g for g in gene_symbols if clean_gene_symbol(g) is not None]
    if not gene_symbols:
        raise ValueError("No RNA/CNV/SCNA genes could be extracted from the provided input")

    rna_genes = [g for g, srcs in gene_sources.items() if "RNA" in srcs]
    scna_genes = [g for g, srcs in gene_sources.items() if "SCNA" in srcs]
    both_genes = [g for g, srcs in gene_sources.items() if "RNA" in srcs and "SCNA" in srcs]

    print(f"\n{'=' * 50}")
    print("GENE LIST SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total unique genes: {len(gene_symbols)}")
    print(f"  From RNA expression: {len(rna_genes)}")
    print(f"  From SCNA/copy number: {len(scna_genes)}")
    print(f"  In both modalities: {len(both_genes)}")

    gene_source_df = pd.DataFrame([
        {"gene_symbol": gene, "source": "+".join(gene_sources.get(gene, ["unknown"]))}
        for gene in gene_symbols
    ])
    gene_source_df.to_csv(os.path.join(output_dir, "input_genes.csv"), index=False)
    print("\nSaved gene list with sources to input_genes.csv")

    analyses = [
        ("GO Biological Process", ["GO_Biological_Process_2023"], "GO_BP"),
        ("GO Molecular Function", ["GO_Molecular_Function_2023"], "GO_MF"),
        ("GO Cellular Component", ["GO_Cellular_Component_2023"], "GO_CC"),
        ("KEGG Pathways", ["KEGG_2021_Human"], "KEGG"),
        ("Reactome Pathways", ["Reactome_2022"], "Reactome"),
        ("MSigDB Hallmark", ["MSigDB_Hallmark_2020"], "Hallmark"),
        ("WikiPathways", ["WikiPathway_2023_Human"], "WikiPathways"),
        ("ENCODE TF ChIP-seq", ["ENCODE_TF_ChIP-seq_2015"], "ENCODE_TF"),
        ("ChEA TF Targets", ["ChEA_2022"], "ChEA"),
        ("Human Gene Atlas", ["Human_Gene_Atlas"], "HumanGeneAtlas"),
        ("CellMarker", ["CellMarker_2024"], "CellMarker"),
        ("PanglaoDB", ["PanglaoDB_Augmented_2021"], "PanglaoDB"),
        ("Cancer Cell Line Encyclopedia", ["Cancer_Cell_Line_Encyclopedia"], "CCLE"),
        ("OMIM Disease", ["OMIM_Disease"], "OMIM"),
        ("DisGeNET", ["DisGeNET"], "DisGeNET"),
    ]

    all_results = {}
    for analysis_name, gene_sets, prefix in analyses:
        print("\n" + "=" * 70)
        print(analysis_name)
        print("=" * 70)
        results = run_enrichr_analysis(
            gene_symbols,
            gene_sets,
            output_dir,
            organism=args.organism,
            prefix=prefix,
        )
        all_results[analysis_name] = results
        if results is not None:
            create_barplot(
                results,
                analysis_name,
                os.path.join(output_dir, f"{prefix}_barplot.png"),
                pval_threshold=0.25,
            )

    print("\n" + "=" * 70)
    print("CREATING SUMMARY REPORT")
    print("=" * 70)
    report = create_summary_report(all_results, gene_symbols, output_dir, gene_sources)
    print(report)

    print("\n" + "=" * 70)
    print("COMBINED SIGNIFICANT RESULTS")
    print("=" * 70)
    write_combined_results(all_results, output_dir)

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
