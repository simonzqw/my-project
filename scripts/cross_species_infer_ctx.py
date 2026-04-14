import argparse
import json
import os

import numpy as np
import torch

from models.reasoning_mlp import PerturbationPredictorNoCellLine
from utils.data_processor import DataProcessor
from utils.emb_loader import GeneEmbeddingLoader


def parse_args():
    p = argparse.ArgumentParser(description="Cross-species inference with mouse context and human-trained model.")
    p.add_argument("--human_h5ad", type=str, required=True)
    p.add_argument("--context_dir", type=str, required=True, help="contains mouse_control_expr.npy and mouse_atac_token.npy")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--pretrained_emb", type=str, default=None)
    p.add_argument("--perturbations", type=str, nargs="*", default=None)
    p.add_argument("--split_strategy", type=str, default="perturbation")
    p.add_argument("--perturb_dim", type=int, default=200)
    p.add_argument("--drug_dim", type=int, default=2048)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dim_ff", type=int, default=1024)
    p.add_argument("--n_ctrl_tokens", type=int, default=8)
    return p.parse_args()


def load_mouse_context(context_dir, device):
    ctrl = np.load(os.path.join(context_dir, "mouse_control_expr.npy")).astype(np.float32)
    atac = np.load(os.path.join(context_dir, "mouse_atac_token.npy")).astype(np.float32)
    ctrl_t = torch.tensor(ctrl, dtype=torch.float32, device=device).unsqueeze(0)
    atac_t = torch.tensor(atac, dtype=torch.float32, device=device).unsqueeze(0)
    return ctrl_t, atac_t


def build_model(args, processor, ckpt, device):
    pretrained_weights = None
    if args.pretrained_emb:
        loader = GeneEmbeddingLoader(args.pretrained_emb, processor.id_to_perturb)
        pretrained_weights = loader.load_weights()

    atac_dim = int(np.load(os.path.join(args.context_dir, "mouse_atac_token.npy")).shape[0])
    model = PerturbationPredictorNoCellLine(
        n_genes=processor.adata.n_vars,
        n_perturbations=len(processor.perturb_categories),
        pretrained_weights=pretrained_weights,
        perturb_dim=args.perturb_dim,
        drug_dim=args.drug_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        n_ctrl_tokens=args.n_ctrl_tokens,
        atac_dim=atac_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict_one(model, processor, perturb_name, mouse_ctrl, mouse_atac, device):
    pert_id = processor.perturb_map[perturb_name]
    perturb = torch.tensor([pert_id], dtype=torch.long, device=device)
    pred = model(rna_control=mouse_ctrl, perturb=perturb, atac_feat=mouse_atac)
    return pred.squeeze(0).cpu().numpy()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    processor = DataProcessor(args.human_h5ad, split_strategy=args.split_strategy)
    processor.load_data()
    mouse_ctrl, mouse_atac = load_mouse_context(args.context_dir, device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(args, processor, ckpt, device)

    pert_list = args.perturbations
    if pert_list is None or len(pert_list) == 0:
        pert_list = [p for p in processor.perturb_categories if p != "control"]

    preds = {}
    for p in pert_list:
        if p not in processor.perturb_map:
            print(f"skip unknown perturbation: {p}")
            continue
        preds[p] = predict_one(model, processor, p, mouse_ctrl, mouse_atac, device).astype(np.float32)

    out_npz = os.path.join(args.out_dir, "mouse_cross_species_preds.npz")
    np.savez_compressed(out_npz, **preds)
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_preds": len(preds),
                "context_dir": args.context_dir,
                "human_h5ad": args.human_h5ad,
                "ckpt": args.ckpt,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f">>> saved: {out_npz}")


if __name__ == "__main__":
    main()
