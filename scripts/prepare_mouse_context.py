import argparse
import json
import os

import numpy as np
import scanpy as sc

from utils.context_utils import (
    build_gene_accessibility_from_peaks,
    read_bed_peaks_midpoints,
    read_gene_tss_from_gtf,
)
from utils.ortholog import align_mouse_vector_to_human_order, load_one2one_ortholog_map


def parse_args():
    p = argparse.ArgumentParser(description="Prepare mouse cross-species context (control RNA + ATAC token)")
    p.add_argument("--human_h5ad", type=str, required=True, help="Human training h5ad used to define gene order")
    p.add_argument("--mouse_rna_h5", type=str, required=True, help="Mouse 10x RNA h5")
    p.add_argument("--mouse_atac_peaks_bed", type=str, required=True, help="Mouse ATAC peaks bed(.gz)")
    p.add_argument("--mouse_gtf", type=str, required=True, help="Mouse GTF (mm10)")
    p.add_argument("--ortholog_tsv", type=str, required=True, help="One2one ortholog mapping TSV")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--promoter_upstream", type=int, default=2000)
    p.add_argument("--promoter_downstream", type=int, default=500)
    p.add_argument("--mouse_col", type=str, default="mouse_gene")
    p.add_argument("--human_col", type=str, default="human_gene")
    p.add_argument("--one2one_col", type=str, default="is_one2one")
    return p.parse_args()


def load_human_gene_order(human_h5ad):
    adata = sc.read_h5ad(human_h5ad)
    return [str(g) for g in adata.var_names.tolist()]


def load_mouse_rna_gene_means(mouse_rna_h5):
    adata = sc.read_10x_h5(mouse_rna_h5)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    gene_means = x.mean(axis=0)
    genes = [str(g) for g in adata.var_names.tolist()]
    return {g: float(v) for g, v in zip(genes, gene_means)}


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    human_gene_order = load_human_gene_order(args.human_h5ad)
    ortholog_map = load_one2one_ortholog_map(
        args.ortholog_tsv,
        mouse_col=args.mouse_col,
        human_col=args.human_col,
        one2one_col=args.one2one_col,
    )

    mouse_rna_means = load_mouse_rna_gene_means(args.mouse_rna_h5)
    mouse_control_expr = align_mouse_vector_to_human_order(mouse_rna_means, ortholog_map, human_gene_order)

    tss_df = read_gene_tss_from_gtf(args.mouse_gtf)
    peaks_mid_df = read_bed_peaks_midpoints(args.mouse_atac_peaks_bed)
    gene_access = build_gene_accessibility_from_peaks(
        tss_df,
        peaks_mid_df,
        upstream=args.promoter_upstream,
        downstream=args.promoter_downstream,
    )
    mouse_atac_token = align_mouse_vector_to_human_order(gene_access, ortholog_map, human_gene_order)
    mouse_atac_token = np.log1p(mouse_atac_token)

    assert len(mouse_control_expr) == len(human_gene_order)
    assert len(mouse_atac_token) == len(human_gene_order)
    assert np.isnan(mouse_control_expr).sum() == 0
    assert np.isnan(mouse_atac_token).sum() == 0

    np.save(os.path.join(args.out_dir, "mouse_control_expr.npy"), mouse_control_expr.astype(np.float32))
    np.save(os.path.join(args.out_dir, "mouse_atac_token.npy"), mouse_atac_token.astype(np.float32))
    with open(os.path.join(args.out_dir, "shared_gene_order.txt"), "w", encoding="utf-8") as f:
        for g in human_gene_order:
            f.write(g + "\n")

    meta = {
        "n_genes": len(human_gene_order),
        "human_h5ad": args.human_h5ad,
        "mouse_rna_h5": args.mouse_rna_h5,
        "mouse_atac_peaks_bed": args.mouse_atac_peaks_bed,
        "mouse_gtf": args.mouse_gtf,
        "ortholog_tsv": args.ortholog_tsv,
        "promoter_upstream": args.promoter_upstream,
        "promoter_downstream": args.promoter_downstream,
    }
    with open(os.path.join(args.out_dir, "mouse_context_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f">>> done. output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
