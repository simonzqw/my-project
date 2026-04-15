import argparse
import json
import os

import numpy as np
import torch

from models.scerso_diffusion import PerturbationDiffusionPredictor
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
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--dose_dim", type=int, default=32)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--cond_dropout", type=float, default=0.0)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=1.2)
    return p.parse_args()


def load_mouse_context(context_dir, device):
    ctrl = np.load(os.path.join(context_dir, "mouse_control_expr.npy")).astype(np.float32)
    atac = np.load(os.path.join(context_dir, "mouse_atac_token.npy")).astype(np.float32)
    ctrl_t = torch.tensor(ctrl, dtype=torch.float32, device=device).unsqueeze(0)
    atac_t = torch.tensor(atac, dtype=torch.float32, device=device).unsqueeze(0)
    return ctrl_t, atac_t


def build_model(args, processor, ckpt, device):
    ckpt_args = ckpt.get("args", argparse.Namespace())
    pretrained_weights = None
    if args.pretrained_emb:
        loader = GeneEmbeddingLoader(args.pretrained_emb, processor.id_to_perturb)
        pretrained_weights = loader.load_weights()

    atac_dim = int(np.load(os.path.join(args.context_dir, "mouse_atac_token.npy")).shape[0])
    model = PerturbationDiffusionPredictor(
        n_genes=processor.adata.n_vars,
        n_perturbations=len(processor.perturb_categories),
        pretrained_weights=pretrained_weights,
        perturb_dim=getattr(ckpt_args, "perturb_dim", args.perturb_dim),
        hidden_dims=getattr(ckpt_args, "hidden_dims", args.hidden_dims),
        dropout=getattr(ckpt_args, "dropout", args.dropout),
        timesteps=getattr(ckpt_args, "timesteps", args.timesteps),
        dose_dim=getattr(ckpt_args, "dose_dim", args.dose_dim),
        time_dim=getattr(ckpt_args, "time_dim", args.time_dim),
        drug_dim=(processor.drug_embeddings.shape[1] if processor.drug_embeddings is not None else args.drug_dim),
        use_atac=True,
        atac_dim=atac_dim,
        cond_dropout=getattr(ckpt_args, "cond_dropout", args.cond_dropout),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_one(model, processor, perturb_name, mouse_ctrl, mouse_atac, device, sample_steps, guidance_scale):
    pert_id = processor.perturb_map[perturb_name]
    perturb = torch.tensor([pert_id], dtype=torch.long, device=device)
    pred = model.predict_single(
        rna_control=mouse_ctrl,
        perturb=perturb,
        atac_feat=mouse_atac,
        sample_steps=sample_steps,
        guidance_scale=guidance_scale,
    )
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
        preds[p] = predict_one(
            model,
            processor,
            p,
            mouse_ctrl,
            mouse_atac,
            device,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
        ).astype(np.float32)

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
