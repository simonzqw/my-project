#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build gene-level ATAC bank from peak tracks (bigBed/.bb) for background conditioning.

Outputs atac_bank.npz with:
  - genes: gene order aligned to input h5ad
  - <background_name_1>: vector shape (n_genes,)
  - <background_name_2>: vector shape (n_genes,)
"""

import os
import re
import json
import gzip
import argparse
import subprocess
from collections import OrderedDict

import anndata as ad
import numpy as np
import pandas as pd
import pyranges as pr


def run_cmd(cmd):
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom)
    if chrom.startswith("chr"):
        return chrom
    return "chr" + chrom


def load_h5ad_gene_order(h5ad_path):
    adata = ad.read_h5ad(h5ad_path)
    candidate_cols = ["gene_name", "gene_symbol", "symbol", "feature_name"]
    gene_order = None
    for col in candidate_cols:
        if col in adata.var.columns:
            vals = adata.var[col].astype(str).tolist()
            if len(set(vals)) > 10:
                gene_order = vals
                print(f"[INFO] gene order from adata.var['{col}']")
                break
    if gene_order is None:
        gene_order = adata.var_names.astype(str).tolist()
        print("[INFO] gene order from adata.var_names")
    dedup = list(OrderedDict.fromkeys(gene_order))
    print(f"[INFO] n_genes in model order = {len(dedup)}")
    return dedup


def parse_gtf_attributes(attr_str):
    out = {}
    for item in attr_str.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        m = re.match(r'([^ ]+) "(.*)"', item)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def load_gtf_genes(gtf_path):
    rows = []
    opener = gzip.open if gtf_path.endswith(".gz") else open
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attr = parts
            if feature != "gene":
                continue
            attrs = parse_gtf_attributes(attr)
            rows.append({
                "Chromosome": normalize_chrom(chrom),
                "Start": int(start) - 1,
                "End": int(end),
                "Strand": strand,
                "gene_id": attrs.get("gene_id", ""),
                "gene_name": attrs.get("gene_name", "")
            })
    df = pd.DataFrame(rows)
    print(f"[INFO] loaded {len(df)} gene records from GTF")
    return df


def build_promoters(gtf_df, gene_order, upstream=2000, downstream=500):
    gtf_df = gtf_df.copy()
    gtf_df = gtf_df[gtf_df["gene_name"].isin(gene_order)]
    gtf_df["gene_len"] = gtf_df["End"] - gtf_df["Start"]
    gtf_df = (
        gtf_df.sort_values(["gene_name", "gene_len"], ascending=[True, False])
        .drop_duplicates("gene_name", keep="first")
    )

    promoters, missing = [], []
    gtf_map = {r["gene_name"]: r for _, r in gtf_df.iterrows()}
    for gene in gene_order:
        if gene not in gtf_map:
            missing.append(gene)
            continue
        r = gtf_map[gene]
        if r["Strand"] == "+":
            tss = r["Start"]
            p_start = max(0, tss - upstream)
            p_end = tss + downstream
        else:
            tss = r["End"]
            p_start = max(0, tss - downstream)
            p_end = tss + upstream
        promoters.append({
            "Chromosome": r["Chromosome"],
            "Start": int(p_start),
            "End": int(p_end),
            "gene": gene,
            "Strand": r["Strand"]
        })

    promoters_df = pd.DataFrame(promoters)
    print(f"[INFO] promoters built for {len(promoters_df)} genes")
    print(f"[INFO] missing genes in GTF = {len(missing)}")
    return promoters_df, missing


def convert_bigbed_to_bed(bigbed_path, bed_path):
    run_cmd(["bigBedToBed", bigbed_path, bed_path])


def load_peaks_bed(bed_path):
    df = pd.read_csv(bed_path, sep="\t", header=None, comment="#")
    if df.shape[1] < 3:
        raise ValueError(f"BED file has <3 columns: {bed_path}")
    df = df.copy()
    df[0] = df[0].astype(str).map(normalize_chrom)
    df[1] = df[1].astype(int)
    df[2] = df[2].astype(int)
    if df.shape[1] >= 5:
        score = pd.to_numeric(df[4], errors="coerce").fillna(1.0).astype(float)
    else:
        score = pd.Series(np.ones(len(df)), index=df.index, dtype=float)
    return pd.DataFrame({
        "Chromosome": df[0].values,
        "Start": df[1].values,
        "End": df[2].values,
        "peak_score": score.values
    })


def compute_gene_atac_vector(promoters_df, peaks_df, gene_order, mode="binary"):
    pr_prom = pr.PyRanges(promoters_df)
    pr_peak = pr.PyRanges(peaks_df)
    ov = pr_prom.join(pr_peak).as_df()
    gene2val = {g: 0.0 for g in gene_order}
    if len(ov) == 0:
        print("[WARN] no overlaps found")
        return np.array([gene2val[g] for g in gene_order], dtype=np.float32)

    if mode == "binary":
        overlap_genes = set(ov["gene"].astype(str).tolist())
        for g in overlap_genes:
            gene2val[g] = 1.0
    elif mode == "count":
        cnt = ov.groupby("gene").size()
        for g, v in cnt.items():
            gene2val[str(g)] = float(v)
    elif mode == "max_score":
        mx = ov.groupby("gene")["peak_score"].max()
        for g, v in mx.items():
            gene2val[str(g)] = float(v)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    return np.array([gene2val[g] for g in gene_order], dtype=np.float32)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    gene_order = load_h5ad_gene_order(args.h5ad_path)
    gtf_df = load_gtf_genes(args.gtf_path)
    promoters_df, missing = build_promoters(
        gtf_df=gtf_df,
        gene_order=gene_order,
        upstream=args.upstream,
        downstream=args.downstream,
    )
    promoters_df.to_csv(os.path.join(args.out_dir, "promoters_used.tsv"), sep="\t", index=False)

    k562_bed = os.path.join(args.out_dir, "K562_peaks.bed")
    rpe1_bed = os.path.join(args.out_dir, "RPE1_peaks.bed")
    convert_bigbed_to_bed(args.k562_bigbed, k562_bed)
    convert_bigbed_to_bed(args.rpe1_bigbed, rpe1_bed)

    k562_peaks = load_peaks_bed(k562_bed)
    rpe1_peaks = load_peaks_bed(rpe1_bed)

    k562_vec = compute_gene_atac_vector(promoters_df, k562_peaks, gene_order, mode=args.mode)
    rpe1_vec = compute_gene_atac_vector(promoters_df, rpe1_peaks, gene_order, mode=args.mode)

    np.savez_compressed(
        os.path.join(args.out_dir, "atac_bank.npz"),
        genes=np.array(gene_order, dtype=object),
        K562=k562_vec.astype(np.float32),
        RPE1=rpe1_vec.astype(np.float32),
    )

    meta = {
        "h5ad_path": args.h5ad_path,
        "gtf_path": args.gtf_path,
        "k562_bigbed": args.k562_bigbed,
        "rpe1_bigbed": args.rpe1_bigbed,
        "feature_mode": args.mode,
        "promoter_upstream": args.upstream,
        "promoter_downstream": args.downstream,
        "n_genes": len(gene_order),
        "n_missing_genes_in_gtf": len(missing),
        "k562_nonzero_genes": int((k562_vec > 0).sum()),
        "rpe1_nonzero_genes": int((rpe1_vec > 0).sum()),
    }
    with open(os.path.join(args.out_dir, "atac_bank_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[DONE] atac_bank saved to:", os.path.join(args.out_dir, "atac_bank.npz"))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad_path", type=str, required=True)
    parser.add_argument("--gtf_path", type=str, required=True)
    parser.add_argument("--k562_bigbed", type=str, required=True)
    parser.add_argument("--rpe1_bigbed", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="binary", choices=["binary", "count", "max_score"])
    parser.add_argument("--upstream", type=int, default=2000)
    parser.add_argument("--downstream", type=int, default=500)
    args = parser.parse_args()
    main(args)
