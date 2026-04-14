
import argparse
import os

omp_threads = os.environ.get("OMP_NUM_THREADS")
if not (omp_threads and omp_threads.isdigit() and int(omp_threads) > 0):
    os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from models.scerso_diffusion import PerturbationDiffusionPredictor
from utils.data_processor import DataProcessor


def get_args():
    parser = argparse.ArgumentParser(description='scERso Diffusion Visualization')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='diffusion_combo_report.png')
    parser.add_argument('--split_strategy', type=str, default='perturbation', choices=['random', 'perturbation'])
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--cell_line', type=str, default='0')
    parser.add_argument('--perturb_genes', type=str, nargs='+', required=True)
    parser.add_argument('--weights', type=float, nargs='*', default=None)
    parser.add_argument('--latent_mode', type=str, default='adaptive', choices=['sum', 'mean', 'adaptive'])
    parser.add_argument('--sample_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--atac_key', type=str, default=None)
    parser.add_argument('--atac_bank_path', type=str, default=None)
    parser.add_argument('--background_key', type=str, default='cell_context')
    return parser.parse_args()


def resolve_cell_line(processor, cell_line_arg):
    try:
        cl_id = int(cell_line_arg)
        if cl_id in processor.cell_line_baselines:
            return cl_id
    except ValueError:
        pass

    if cell_line_arg not in processor.cell_line_map:
        raise ValueError(f"未找到 cell line: {cell_line_arg}")
    return processor.cell_line_map[cell_line_arg]


def resolve_or_autopick_gene(processor, gene, cell_line_id):
    if gene in processor.perturb_map:
        return gene, False

    obs = processor.adata.obs
    cell_col = processor.cell_line_col
    cl_name = processor.cell_line_categories[cell_line_id]
    subset = obs[obs[cell_col].astype(str) == str(cl_name)]
    counts = subset['perturbation'].astype(str).value_counts()
    for g in counts.index.tolist():
        if g != 'control' and g in processor.perturb_map:
            return g, True

    for g in processor.perturb_categories:
        if g != 'control':
            return g, True
    raise ValueError("未找到可用的非 control 扰动基因。")


def load_model(checkpoint, processor, n_genes, n_perts, n_cell_lines, device):
    state_dict = checkpoint['model_state_dict']
    ckpt_args = checkpoint.get('args', argparse.Namespace())
    perturb_dim = int(state_dict['perturb_embedding.weight'].shape[1])
    cell_line_dim = int(state_dict['cell_line_embedding.weight'].shape[1])

    model = PerturbationDiffusionPredictor(
        n_genes=n_genes,
        n_perturbations=n_perts,
        n_cell_lines=n_cell_lines,
        perturb_dim=perturb_dim,
        cell_line_dim=cell_line_dim,
        hidden_dims=getattr(ckpt_args, 'hidden_dims', [512, 512, 512]),
        dropout=getattr(ckpt_args, 'dropout', 0.1),
        timesteps=getattr(ckpt_args, 'timesteps', 1000),
        dose_dim=getattr(ckpt_args, 'dose_dim', 32),
        time_dim=getattr(ckpt_args, 'time_dim', 128),
        drug_dim=(processor.drug_embeddings.shape[1] if processor.drug_embeddings is not None else 0),
        use_atac=(processor.atac_features is not None),
        atac_dim=(processor.atac_dim if processor.atac_features is not None else 0),
        cond_dropout=getattr(ckpt_args, 'cond_dropout', 0.0),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in checkpoint['ema_state_dict']:
                p.data.copy_(checkpoint['ema_state_dict'][name].to(device))
    model.eval()
    return model


def visualize():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = DataProcessor(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        split_strategy=args.split_strategy,
        atac_key=args.atac_key,
        atac_bank_path=args.atac_bank_path,
        background_key=args.background_key,
    )
    n_genes, n_perts, n_cls = processor.load_data()
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model = load_model(checkpoint, processor, n_genes, n_perts, n_cls, device)

    cell_line_id = resolve_cell_line(processor, args.cell_line)
    control = processor.get_cell_line_control(cell_line_id, device=device).unsqueeze(0)
    cell_line_tensor = torch.tensor([cell_line_id], dtype=torch.long, device=device)
    atac_feat = processor.get_cell_line_atac(cell_line_id, device=device)
    if atac_feat is not None:
        atac_feat = atac_feat.unsqueeze(0)
    if len(args.perturb_genes) > 1 and atac_feat is not None:
        print(">>> 提示: 可视化组合扰动时默认复用同一 cell-line baseline ATAC，未显式建模组合特异 ATAC 变化。")

    latents = []
    deltas_single = []
    perturb_ids = []
    resolved_genes = []
    for gene in args.perturb_genes:
        resolved_gene, auto_picked = resolve_or_autopick_gene(processor, gene, cell_line_id)
        if auto_picked:
            print(f">>> 提示: 扰动 {gene} 不存在，自动替换为 {resolved_gene}")
        resolved_genes.append(resolved_gene)
        pid = processor.perturb_map[resolved_gene]
        perturb_ids.append(pid)

        latent = model.get_latent(
            rna_control=control,
            perturb=torch.tensor([pid], dtype=torch.long, device=device),
            cell_line=cell_line_tensor,
            atac_feat=atac_feat,
        )
        latents.append(latent)

        single_pred = model.predict_single(
            rna_control=control,
            perturb=torch.tensor([pid], dtype=torch.long, device=device),
            cell_line=cell_line_tensor,
            atac_feat=atac_feat,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
        )
        deltas_single.append(single_pred.squeeze(0).detach().cpu().numpy() - control.squeeze(0).detach().cpu().numpy())

    if len(latents) == 1:
        combo_pred = model.predict_single(
            rna_control=control,
            perturb=torch.tensor([perturb_ids[0]], dtype=torch.long, device=device),
            cell_line=cell_line_tensor,
            atac_feat=atac_feat,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
        )
        delta_additive = deltas_single[0]
    else:
        combo_latent = model.combine_latents(latents, weights=args.weights, mode=args.latent_mode)
        combo_pred = model.predict_from_latent(
            rna_control=control,
            cell_line=cell_line_tensor,
            latent=combo_latent,
            perturb=torch.tensor([perturb_ids[0]], dtype=torch.long, device=device),
            atac_feat=atac_feat,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
        )
        delta_additive = np.sum(np.stack(deltas_single, axis=0), axis=0)

    baseline = control.squeeze(0).detach().cpu().numpy()
    delta_combo = combo_pred.squeeze(0).detach().cpu().numpy() - baseline

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.scatterplot(x=delta_additive, y=delta_combo, ax=axes[0], alpha=0.5, color='purple')
    min_val = float(min(delta_additive.min(), delta_combo.min()))
    max_val = float(max(delta_additive.max(), delta_combo.max()))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
    corr = np.corrcoef(delta_additive, delta_combo)[0, 1]
    axes[0].set_title(f"{' + '.join(resolved_genes)} | Pearson={corr:.4f}", fontsize=14)
    axes[0].set_xlabel("Additive expectation")
    axes[0].set_ylabel("Diffusion prediction")

    diff = np.abs(delta_combo - delta_additive)
    top_idx = np.argsort(diff)[-15:][::-1]
    plot_data = []
    gene_names = processor.adata.var_names.tolist()
    for idx in top_idx:
        g = gene_names[idx]
        plot_data.append({'Gene': g, 'Value': float(delta_additive[idx]), 'Type': 'Additive'})
        plot_data.append({'Gene': g, 'Value': float(delta_combo[idx]), 'Type': 'Diffusion'})

    df_bar = pd.DataFrame(plot_data)
    sns.barplot(data=df_bar, x='Gene', y='Value', hue='Type', ax=axes[1], palette={'Additive': 'gray', 'Diffusion': 'crimson'})
    axes[1].set_title("Top 15 non-additive genes", fontsize=14)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    axes[1].set_ylabel("Expression delta")

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=200)
    plt.close(fig)
    print(f">>> 图已保存: {args.save_path}")


if __name__ == "__main__":
    visualize()
