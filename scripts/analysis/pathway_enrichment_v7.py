#!/usr/bin/env python3
"""
Pathway Enrichment Analysis for HNSCC Prognostic Genes - v7

Input: CSV file with features from survival model (RNA_* and CNV_* format)
Output: Enrichment results tables and visualization plots
"""

import argparse
import os
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Try to import gseapy
try:
    import gseapy as gp
    from gseapy import enrichr

    GSEAPY_AVAILABLE = True
    print(f"gseapy version: {gp.__version__}")
except ImportError:
    GSEAPY_AVAILABLE = False
    print("Warning: gseapy not installed. Install with: pip install gseapy")

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pathway enrichment analysis for prognostic genes"
    )
    parser.add_argument(
        "--gene-file",
        required=False,
        help="CSV file with gene features (column 'feature' with RNA_GENENAME format)",
    )
    parser.add_argument(
        "--train-data", required=False, help="Training data CSV to re-compute top genes"
    )
    parser.add_argument(
        "--n-genes",
        type=int,
        default=75,
        help="Number of top genes to analyze (default: 75)",
    )
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--organism", default="human", help="Organism (default: human)")
    return parser.parse_args()


def extract_gene_symbols(feature_names):
    """
    Extract gene symbols from RNA_GENENAME or CNV_GENENAME format.

    Returns:
        tuple: (gene_list, gene_sources_dict)
        - gene_list: deduplicated list of gene symbols
        - gene_sources_dict: {gene: [sources]} e.g., {"TP53": ["RNA", "SCNA"]}
    """
    gene_sources = {}

    for f in feature_names:
        if f.startswith("RNA_"):
            gene = f.replace("RNA_", "")
            source = "RNA"
        elif f.startswith("CNV_"):
            gene = f.replace("CNV_", "")
            source = "SCNA"
        else:
            # Skip clinical features or other non-gene features
            continue

        if gene not in gene_sources:
            gene_sources[gene] = []
        if source not in gene_sources[gene]:
            gene_sources[gene].append(source)

    # Return deduplicated gene list and source mapping
    genes = list(gene_sources.keys())
    return genes, gene_sources


def run_enrichr_analysis(gene_list, gene_sets, output_dir, prefix="enrichr"):
    """
    Run Enrichr overrepresentation analysis.

    Key change: cutoff=1.0 to get ALL results, then filter ourselves.
    """
    if not GSEAPY_AVAILABLE:
        print("gseapy not available, skipping Enrichr analysis")
        return None

    print(f"\nRunning Enrichr analysis with {len(gene_list)} genes...")
    print(f"Gene sets: {gene_sets}")
    print(
        f"Input genes: {gene_list[:10]}..."
        if len(gene_list) > 10
        else f"Input genes: {gene_list}"
    )

    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism="human",
            outdir=None,
            cutoff=1.0,  # IMPORTANT: Get ALL results, not just significant
            verbose=True,
        )

        results = enr.results

        if results is None or len(results) == 0:
            print("  No results returned from Enrichr")
            return None

        # Sort by p-value
        results = results.sort_values("Adjusted P-value")

        # Save ALL results
        output_file = os.path.join(output_dir, f"{prefix}_results.csv")
        results.to_csv(output_file, index=False)
        print(f"  Saved {len(results)} total terms to {output_file}")

        # Report at multiple thresholds
        sig_strict = results[results["Adjusted P-value"] < 0.05]
        sig_relaxed = results[results["Adjusted P-value"] < 0.25]

        # Check if P-value column exists
        if "P-value" in results.columns:
            sig_nominal = results[results["P-value"] < 0.05]
            print(f"  Results summary:")
            print(f"    - adj. p < 0.05 (strict): {len(sig_strict)}")
            print(f"    - adj. p < 0.25 (relaxed): {len(sig_relaxed)}")
            print(f"    - nominal p < 0.05: {len(sig_nominal)}")
        else:
            print(f"  Results summary:")
            print(f"    - adj. p < 0.05 (strict): {len(sig_strict)}")
            print(f"    - adj. p < 0.25 (relaxed): {len(sig_relaxed)}")

        # Show top 5 terms regardless of significance
        print(f"\n  Top 5 terms (by adjusted p-value):")
        for i, (_, row) in enumerate(results.head(5).iterrows()):
            term = (
                row["Term"][:50] + "..." if len(str(row["Term"])) > 50 else row["Term"]
            )
            pval = row.get("P-value", "N/A")
            adj_pval = row["Adjusted P-value"]
            overlap = row.get("Overlap", "N/A")
            print(f"    {i + 1}. {term}")
            if pval != "N/A":
                print(f"       p={pval:.2e}, adj.p={adj_pval:.2e}, overlap={overlap}")
            else:
                print(f"       adj.p={adj_pval:.2e}, overlap={overlap}")

        return results

    except Exception as e:
        print(f"  Error running Enrichr: {e}")
        import traceback

        traceback.print_exc()
        return None


def create_barplot(results_df, title, output_file, top_n=15, pval_threshold=0.25):
    """Create bar plot of top enriched terms."""
    if not MATPLOTLIB_AVAILABLE:
        return

    if results_df is None or len(results_df) == 0:
        return

    # Use relaxed threshold for plotting
    pval_col = "Adjusted P-value"
    term_col = "Term"

    # Get results under relaxed threshold
    sig_results = results_df[results_df[pval_col] < pval_threshold].copy()

    if len(sig_results) == 0:
        # If no significant results, plot top N anyway
        sig_results = results_df.head(top_n).copy()
        title = f"{title} (Top {len(sig_results)}, none significant)"

    # Sort and get top N
    sig_results = sig_results.sort_values(pval_col).head(top_n)

    # Truncate long term names
    sig_results["Term_short"] = sig_results[term_col].apply(
        lambda x: str(x)[:40] + "..." if len(str(x)) > 40 else str(x)
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(sig_results) * 0.4)))

    y_pos = range(len(sig_results))
    neg_log_p = -np.log10(sig_results[pval_col].values + 1e-300)

    # Color by significance
    colors = [
        "steelblue" if p < 0.05 else "lightsteelblue"
        for p in sig_results[pval_col].values
    ]

    bars = ax.barh(y_pos, neg_log_p, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sig_results["Term_short"].values)
    ax.invert_yaxis()
    ax.set_xlabel("-log10(adjusted p-value)")
    ax.set_title(title)

    # Add significance threshold lines
    ax.axvline(
        x=-np.log10(0.05), color="red", linestyle="--", alpha=0.5, label="p=0.05"
    )
    ax.axvline(
        x=-np.log10(0.25), color="orange", linestyle=":", alpha=0.5, label="p=0.25"
    )
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved plot to {output_file}")


def create_summary_report(all_results, gene_list, output_dir, gene_sources=None):
    """Create a summary report of all enrichment analyses."""

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("PATHWAY ENRICHMENT ANALYSIS REPORT (v7)")
    report_lines.append("HNSCC Prognostic Multi-Omics Gene Signature")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    report_lines.append(f"Number of input genes: {len(gene_list)}")

    # Add gene source breakdown if available
    if gene_sources:
        rna_genes = [g for g, srcs in gene_sources.items() if "RNA" in srcs]
        scna_genes = [g for g, srcs in gene_sources.items() if "SCNA" in srcs]
        report_lines.append(f"  - From RNA expression: {len(rna_genes)}")
        report_lines.append(f"  - From SCNA (copy number): {len(scna_genes)}")

    report_lines.append("")
    report_lines.append("Input Genes:")
    report_lines.append("-" * 40)

    # List genes with source annotation
    for i in range(0, len(gene_list), 5):
        chunk = gene_list[i : i + 5]
        if gene_sources:
            annotated = []
            for g in chunk:
                src = gene_sources.get(g, [])
                if "RNA" in src and "SCNA" in src:
                    annotated.append(f"{g}[R+S]")
                elif "SCNA" in src:
                    annotated.append(f"{g}[S]")
                else:
                    annotated.append(g)
            report_lines.append("  " + ", ".join(annotated))
        else:
            report_lines.append("  " + ", ".join(chunk))

    report_lines.append("")

    # Summarize each analysis
    for analysis_name, results_df in all_results.items():
        report_lines.append("")
        report_lines.append("=" * 70)
        report_lines.append(f"Analysis: {analysis_name}")
        report_lines.append("=" * 70)

        if results_df is None or len(results_df) == 0:
            report_lines.append("  No results returned.")
            continue

        pval_col = "Adjusted P-value"
        term_col = "Term"

        # Get significant at different thresholds
        sig_strict = results_df[results_df[pval_col] < 0.05]
        sig_relaxed = results_df[results_df[pval_col] < 0.25]

        report_lines.append(f"  Total terms tested: {len(results_df)}")
        report_lines.append(f"  Significant (adj. p < 0.05): {len(sig_strict)}")
        report_lines.append(f"  Significant (adj. p < 0.25): {len(sig_relaxed)}")
        report_lines.append("")

        # Show top 10 terms
        report_lines.append("  Top 10 Enriched Terms:")
        report_lines.append("  " + "-" * 60)

        top_results = results_df.sort_values(pval_col).head(10)

        for idx, row in top_results.iterrows():
            term = row[term_col] if term_col in row else str(idx)
            if len(str(term)) > 50:
                term = str(term)[:47] + "..."
            pval = row[pval_col]

            sig_marker = "***" if pval < 0.05 else ("*" if pval < 0.25 else "")

            if "Overlap" in row:
                overlap = row["Overlap"]
                report_lines.append(f"    {term} {sig_marker}")
                report_lines.append(f"      adj.p: {pval:.2e}, Overlap: {overlap}")
            else:
                report_lines.append(f"    {term}: adj.p={pval:.2e} {sig_marker}")

    # Write report
    report_text = "\n".join(report_lines)
    report_file = os.path.join(output_dir, "enrichment_summary_report.txt")
    with open(report_file, "w") as f:
        f.write(report_text)

    print(f"\nSummary report saved to: {report_file}")

    return report_text


def main():
    args = parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"enrichment_v7_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PATHWAY ENRICHMENT ANALYSIS (v7 - Multi-Omics)")
    print("=" * 70)
    print(f"Output directory: {output_dir}")

    # Get gene list
    if args.gene_file and os.path.exists(args.gene_file):
        print(f"\nLoading genes from: {args.gene_file}")
        gene_df = pd.read_csv(args.gene_file)

        print(f"  Columns in file: {list(gene_df.columns)}")
        print(f"  Shape: {gene_df.shape}")

        if "feature" in gene_df.columns:
            gene_symbols, gene_sources = extract_gene_symbols(gene_df["feature"].tolist())
        elif "gene_symbol" in gene_df.columns:
            gene_symbols = gene_df["gene_symbol"].tolist()
            gene_sources = {g: ["unknown"] for g in gene_symbols}
        else:
            gene_symbols, gene_sources = extract_gene_symbols(gene_df.iloc[:, 0].tolist())

    else:
        print("ERROR: Must provide --gene-file")
        return

    # Remove any empty or None values
    gene_symbols = [g for g in gene_symbols if g and str(g) != "nan"]

    # Count by source
    rna_genes = [g for g, srcs in gene_sources.items() if "RNA" in srcs]
    scna_genes = [g for g, srcs in gene_sources.items() if "SCNA" in srcs]
    both_genes = [g for g, srcs in gene_sources.items() if "RNA" in srcs and "SCNA" in srcs]

    print(f"\n{'=' * 50}")
    print(f"GENE LIST SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total unique genes: {len(gene_symbols)}")
    print(f"  From RNA expression: {len(rna_genes)}")
    print(f"  From SCNA (copy number): {len(scna_genes)}")
    print(f"  In both modalities: {len(both_genes)}")

    print(f"\nRNA genes ({len(rna_genes)}):")
    for i in range(0, min(20, len(rna_genes)), 5):
        print(f"  {', '.join(rna_genes[i : i + 5])}")
    if len(rna_genes) > 20:
        print(f"  ... and {len(rna_genes) - 20} more")

    print(f"\nSCNA genes ({len(scna_genes)}):")
    for g in scna_genes:
        print(f"  {g}")

    # Save gene list with source information
    gene_source_df = pd.DataFrame([
        {"gene_symbol": g, "source": "+".join(gene_sources.get(g, ["unknown"]))}
        for g in gene_symbols
    ])
    gene_source_df.to_csv(os.path.join(output_dir, "input_genes.csv"), index=False)
    print(f"\nSaved gene list with sources to input_genes.csv")

    # Store all results
    all_results = {}

    # Define all gene set libraries to test
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
        ("CellMarker 2024", ["CellMarker_2024"], "CellMarker"),
        ("PanglaoDB", ["PanglaoDB_Augmented_2021"], "PanglaoDB"),
        ("Cancer Cell Line Encyclopedia", ["Cancer_Cell_Line_Encyclopedia"], "CCLE"),
        ("OMIM Disease", ["OMIM_Disease"], "OMIM"),
        ("DisGeNET", ["DisGeNET"], "DisGeNET"),
    ]

    for analysis_name, gene_sets, prefix in analyses:
        print("\n" + "=" * 70)
        print(f"{analysis_name}")
        print("=" * 70)

        results = run_enrichr_analysis(
            gene_symbols, gene_sets, output_dir, prefix=prefix
        )
        all_results[analysis_name] = results

        if results is not None:
            create_barplot(
                results,
                analysis_name,
                os.path.join(output_dir, f"{prefix}_barplot.png"),
                pval_threshold=0.25,  # Use relaxed threshold for plotting
            )

    # Create Summary Report
    print("\n" + "=" * 70)
    print("CREATING SUMMARY REPORT")
    print("=" * 70)

    report = create_summary_report(all_results, gene_symbols, output_dir, gene_sources)
    print(report)

    # Combined significant results
    print("\n" + "=" * 70)
    print("COMBINED SIGNIFICANT RESULTS")
    print("=" * 70)

    combined_all = []
    combined_sig = []
    for name, df in all_results.items():
        if df is None or len(df) == 0:
            continue

        df_copy = df.copy()
        df_copy["Analysis"] = name
        combined_all.append(df_copy)

        sig = df_copy[df_copy["Adjusted P-value"] < 0.05]
        if len(sig) > 0:
            combined_sig.append(sig)

    if combined_all:
        combined_all_df = pd.concat(combined_all, ignore_index=True)
        combined_all_df = combined_all_df.sort_values("Adjusted P-value")
        combined_all_df.to_csv(
            os.path.join(output_dir, "all_results_combined.csv"), index=False
        )
        print(f"Saved {len(combined_all_df)} total terms to all_results_combined.csv")

    if combined_sig:
        combined_sig_df = pd.concat(combined_sig, ignore_index=True)
        combined_sig_df = combined_sig_df.sort_values("Adjusted P-value")
        combined_sig_df.to_csv(
            os.path.join(output_dir, "significant_terms_combined.csv"), index=False
        )
        print(f"Saved {len(combined_sig_df)} significant terms (adj.p < 0.05)")
    else:
        print("No terms with adj.p < 0.05 found")

        # Save nominally significant
        combined_nominal = []
        for name, df in all_results.items():
            if df is None or len(df) == 0:
                continue
            df_copy = df.copy()
            df_copy["Analysis"] = name
            sig = df_copy[df_copy["Adjusted P-value"] < 0.25]
            if len(sig) > 0:
                combined_nominal.append(sig)

        if combined_nominal:
            combined_nom_df = pd.concat(combined_nominal, ignore_index=True)
            combined_nom_df = combined_nom_df.sort_values("Adjusted P-value")
            combined_nom_df.to_csv(
                os.path.join(output_dir, "suggestive_terms_combined.csv"), index=False
            )
            print(f"Saved {len(combined_nom_df)} suggestive terms (adj.p < 0.25)")

    print(f"\n{'=' * 70}")
    print(f"ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
