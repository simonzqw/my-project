import gzip
import numpy as np
import pandas as pd


def read_gene_tss_from_gtf(gtf_path):
    records = []
    opener = gzip.open if gtf_path.endswith(".gz") else open
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            arr = line.strip().split("\t")
            if len(arr) < 9 or arr[2] != "gene":
                continue
            chrom, _, _, start, end, _, strand, _, attr = arr
            attrs = {}
            for item in attr.split(";"):
                item = item.strip()
                if not item:
                    continue
                kv = item.split(" ", 1)
                if len(kv) != 2:
                    continue
                attrs[kv[0]] = kv[1].replace('"', "").strip()
            gene_name = attrs.get("gene_name", None)
            if gene_name is None:
                continue
            start_i, end_i = int(start), int(end)
            tss = start_i if strand == "+" else end_i
            records.append((chrom, tss, gene_name))
    return pd.DataFrame(records, columns=["chrom", "tss", "gene_name"])


def read_bed_peaks_midpoints(peaks_bed_gz):
    peaks = pd.read_csv(peaks_bed_gz, sep="\t", header=None, usecols=[0, 1, 2], compression="infer")
    peaks.columns = ["chrom", "start", "end"]
    peaks["mid"] = ((peaks["start"].values + peaks["end"].values) // 2).astype(np.int64)
    return peaks[["chrom", "mid"]]


def build_gene_accessibility_from_peaks(tss_df, peaks_mid_df, upstream=2000, downstream=500):
    gene_to_access = {}
    peaks_by_chr = {
        c: np.sort(g["mid"].values.astype(np.int64))
        for c, g in peaks_mid_df.groupby("chrom")
    }
    for _, row in tss_df.iterrows():
        chrom = row["chrom"]
        tss = int(row["tss"])
        gname = str(row["gene_name"])
        mids = peaks_by_chr.get(chrom, None)
        if mids is None or len(mids) == 0:
            gene_to_access[gname] = 0.0
            continue
        left = max(0, tss - upstream)
        right = tss + downstream
        i = np.searchsorted(mids, left, side="left")
        j = np.searchsorted(mids, right, side="right")
        gene_to_access[gname] = float(max(j - i, 0))
    return gene_to_access
