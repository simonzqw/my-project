
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
    parser.add_argument('--top_n', type=int, default=30)
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


def select_display_genes(delta_combo, delta_additive, gene_names, top_n=30):
    score = 0.6 * np.abs(delta_combo) + 0.4 * np.abs(delta_combo - delta_additive)
    top_idx = np.argsort(score)[-min(top_n, len(score)):][::-1]
    top_genes = [gene_names[i] for i in top_idx]
    return top_idx, top_genes


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
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    ax_scatter, ax_pair, ax_heatmap, ax_nonlin = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel A: additive vs diffusion scatter
    sns.scatterplot(x=delta_additive, y=delta_combo, ax=ax_scatter, alpha=0.45, color='#5b4db7', s=18)
    min_val = float(min(delta_additive.min(), delta_combo.min()))
    max_val = float(max(delta_additive.max(), delta_combo.max()))
    ax_scatter.plot([min_val, max_val], [min_val, max_val], '--', color='#d95f02', linewidth=1.8)
    corr = np.corrcoef(delta_additive, delta_combo)[0, 1]
    ax_scatter.set_title(f"Additive vs Diffusion Delta | {' + '.join(resolved_genes)} | Pearson={corr:.4f}", fontsize=13)
    ax_scatter.set_xlabel("Additive expectation (delta)")
    ax_scatter.set_ylabel("Diffusion prediction (delta)")

    # Gene selection for display
    gene_names = processor.adata.var_names.tolist()
    top_idx, top_genes = select_display_genes(delta_combo, delta_additive, gene_names, top_n=args.top_n)
    top_ctrl = baseline[top_idx]
    top_pred = combo_pred.squeeze(0).detach().cpu().numpy()[top_idx]
    top_add = delta_additive[top_idx]
    top_diff = delta_combo[top_idx]

    # Panel B: control vs predicted expression (paired comparison)
    y_pos = np.arange(len(top_genes))
    ax_pair.barh(y_pos - 0.18, top_ctrl, height=0.34, color='#7f8c8d', label='Control')
    ax_pair.barh(y_pos + 0.18, top_pred, height=0.34, color='#c0392b', label='Predicted')
    ax_pair.set_yticks(y_pos)
    ax_pair.set_yticklabels(top_genes, fontsize=9)
    ax_pair.invert_yaxis()
    ax_pair.set_title(f"Top {len(top_genes)} genes: Control vs Predicted expression", fontsize=13)
    ax_pair.set_xlabel("Expression")
    ax_pair.legend(frameon=False, loc='lower right')

    # Panel C: heatmap of additive/diffusion/nonlinearity deltas
    heat_df = pd.DataFrame(
        np.vstack([top_add, top_diff, top_diff - top_add]),
        index=['Additive delta', 'Diffusion delta', 'Nonlinear residual'],
        columns=top_genes,
    )
    sns.heatmap(
        heat_df,
        ax=ax_heatmap,
        cmap='coolwarm',
        center=0.0,
        cbar_kws={'shrink': 0.7},
    )
    ax_heatmap.set_title("Delta structure on selected genes", fontsize=13)
    ax_heatmap.set_xlabel("Genes")
    ax_heatmap.set_ylabel("")
    ax_heatmap.tick_params(axis='x', labelrotation=65, labelsize=8)

    # Panel D: strongest nonlinearity ranking
    nonlin = np.abs(top_diff - top_add)
    order = np.argsort(nonlin)[::-1]
    ax_nonlin.bar(np.arange(len(order)), nonlin[order], color='#8e44ad', alpha=0.9)
    ax_nonlin.set_xticks(np.arange(len(order)))
    ax_nonlin.set_xticklabels([top_genes[i] for i in order], rotation=65, ha='right', fontsize=8)
    ax_nonlin.set_title("Nonlinear effect ranking | |Diffusion - Additive|", fontsize=13)
    ax_nonlin.set_ylabel("Absolute residual")

    # Save a companion table for reproducibility
    report_df = pd.DataFrame({
        'gene': top_genes,
        'control_expr': top_ctrl,
        'pred_expr': top_pred,
        'additive_delta': top_add,
        'diffusion_delta': top_diff,
        'nonlinear_abs_residual': np.abs(top_diff - top_add),
    }).sort_values('nonlinear_abs_residual', ascending=False)
    report_csv = os.path.splitext(args.save_path)[0] + "_top_genes.csv"
    report_df.to_csv(report_csv, index=False)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=260)
    plt.close(fig)
    print(f">>> 图已保存: {args.save_path}")
    print(f">>> 基因报告: {report_csv}")


if __name__ == "__main__":
    visualize()
