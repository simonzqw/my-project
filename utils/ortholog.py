import numpy as np
import pandas as pd


def load_one2one_ortholog_map(path, mouse_col="mouse_gene", human_col="human_gene", one2one_col="is_one2one"):
    df = pd.read_csv(path, sep="\t")
    if one2one_col in df.columns:
        df = df[df[one2one_col] == 1].copy()
    df[mouse_col] = df[mouse_col].astype(str)
    df[human_col] = df[human_col].astype(str)
    return dict(zip(df[mouse_col], df[human_col]))


def align_mouse_vector_to_human_order(mouse_gene_to_val, ortholog_map, human_gene_order):
    out = np.zeros(len(human_gene_order), dtype=np.float32)
    human_index = {g: i for i, g in enumerate(human_gene_order)}
    for mouse_gene, val in mouse_gene_to_val.items():
        human_gene = ortholog_map.get(str(mouse_gene), None)
        if human_gene is None:
            continue
        idx = human_index.get(human_gene, None)
        if idx is None:
            continue
        out[idx] = float(val)
    return out
