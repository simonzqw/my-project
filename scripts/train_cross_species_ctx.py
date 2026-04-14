import argparse
import os

import numpy as np
import torch
import torch.optim as optim

from models.reasoning_mlp import PerturbationPredictorNoCellLine
from utils.data_processor import DataProcessor
from utils.emb_loader import GeneEmbeddingLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train cross-species context model without cell_line token.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--pretrained_emb", type=str, default=None)
    p.add_argument("--split_strategy", type=str, default="perturbation", choices=["random", "perturbation"])
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--context_dropout", type=float, default=0.15)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dim_ff", type=int, default=1024)
    p.add_argument("--n_ctrl_tokens", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--perturb_dim", type=int, default=200)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--atac_key", type=str, default=None)
    p.add_argument("--atac_bank_path", type=str, default=None)
    p.add_argument("--background_key", type=str, default="cell_context")
    return p.parse_args()


def run_epoch(model, loader, optimizer, scaler, device, context_dropout=0.0, drug_embeddings=None, train=True):
    if train:
        model.train()
    else:
        model.eval()
    losses = []
    for batch in loader:
        ctrl = batch["rna_control"].to(device)
        target = batch["rna_target"].to(device)
        perturb = batch["perturb"].to(device)
        dose = batch["dose"].to(device) if "dose" in batch else None
        atac_feat = batch["atac_feat"].to(device) if "atac_feat" in batch else None
        drug_feat = drug_embeddings[perturb] if drug_embeddings is not None else None

        if train and atac_feat is not None and context_dropout > 0:
            keep = (torch.rand(atac_feat.shape[0], device=device) > context_dropout).float().unsqueeze(1)
            atac_feat = atac_feat * keep

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(train and scaler.is_enabled())):
                pred = model(
                    rna_control=ctrl,
                    perturb=perturb,
                    drug_feat=drug_feat,
                    dose=dose,
                    atac_feat=atac_feat,
                )
                loss = ((pred - target) ** 2).mean()

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        losses.append(float(loss.detach().item()))
    return float(np.mean(losses)) if len(losses) > 0 else 0.0


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    processor = DataProcessor(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        split_strategy=args.split_strategy,
        atac_key=args.atac_key,
        atac_bank_path=args.atac_bank_path,
        background_key=args.background_key,
    )
    n_genes, n_perts, _ = processor.load_data()
    train_loader, val_loader, _ = processor.prepare_loaders(
        batch_size=args.batch_size,
        rna_noise=0.0,
        atac_key=args.atac_key,
        atac_bank_path=args.atac_bank_path,
        background_key=args.background_key,
    )

    pretrained_weights = None
    if args.pretrained_emb:
        loader = GeneEmbeddingLoader(args.pretrained_emb, processor.id_to_perturb)
        pretrained_weights = loader.load_weights()

    atac_dim = processor.atac_dim if getattr(processor, "atac_features", None) is not None else 0
    model = PerturbationPredictorNoCellLine(
        n_genes=n_genes,
        n_perturbations=n_perts,
        pretrained_weights=pretrained_weights,
        perturb_dim=args.perturb_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        n_ctrl_tokens=args.n_ctrl_tokens,
        atac_dim=atac_dim,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    drug_embeddings = processor.drug_embeddings.to(device) if processor.drug_embeddings is not None else None

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, scaler, device,
            context_dropout=args.context_dropout, drug_embeddings=drug_embeddings, train=True
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, scaler, device,
            context_dropout=0.0, drug_embeddings=drug_embeddings, train=False
        )
        print(f"[E{epoch:03d}/{args.epochs:03d}] train={train_loss:.6f} val={val_loss:.6f}")

        ckpt = {
            "model_state_dict": model.state_dict(),
            "args": args,
            "n_genes": n_genes,
            "n_perts": n_perts,
            "perturb_categories": processor.perturb_categories,
            "cell_line_categories": processor.cell_line_categories,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "latest.pth"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, "best_model_ctx.pth"))
            print(f"  ↳ best updated: val={best_val:.6f}")

    print(f">>> done. checkpoints in {args.save_dir}")


if __name__ == "__main__":
    main()
