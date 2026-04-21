
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
    parser = argparse.ArgumentParser(description="Predict single/combinatorial perturbations with diffusion model")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./diffusion_predictions')
    parser.add_argument('--cell_line', type=str, required=True, help='Cell line name or numeric id')
    parser.add_argument('--perturb_genes', type=str, nargs='+', required=True, help='One gene for single prediction, multiple genes for latent arithmetic')
    parser.add_argument('--weights', type=float, nargs='*', default=None, help='Optional weights for latent arithmetic')
    parser.add_argument('--latent_mode', type=str, default='adaptive', choices=['sum', 'mean', 'adaptive'])
    parser.add_argument('--sample_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--interpolate_to', type=str, default=None, help='Optional target perturb gene for latent interpolation trajectory')
    parser.add_argument('--interp_steps', type=int, default=8, help='Interpolation steps when --interpolate_to is set')
    parser.add_argument('--atac_key', type=str, default=None)
    parser.add_argument('--atac_bank_path', type=str, default=None)
    parser.add_argument('--background_key', type=str, default='cell_context')
    return parser.parse_args()


def infer_model_config(checkpoint, processor):
    ckpt_args = checkpoint.get('args', argparse.Namespace())
    state_dict = checkpoint['model_state_dict']
    perturb_dim = int(state_dict['perturb_embedding.weight'].shape[1])
    return dict(
        perturb_dim=perturb_dim,
        hidden_dims=getattr(ckpt_args, 'hidden_dims', [512, 512, 512]),
        dropout=getattr(ckpt_args, 'dropout', 0.1),
        timesteps=getattr(ckpt_args, 'timesteps', 1000),
        target_mode=getattr(ckpt_args, 'target_mode', 'target'),
        dose_dim=getattr(ckpt_args, 'dose_dim', 32),
        time_dim=getattr(ckpt_args, 'time_dim', 128),
        cond_dropout=getattr(ckpt_args, 'cond_dropout', 0.0),
        atac_dim=processor.atac_dim if getattr(processor, 'atac_features', None) is not None else 0,
    )


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


def get_observed_mean_expression(processor, cell_line_id, perturb_name):
    obs = processor.adata.obs
    cell_col = processor.cell_line_col
    cl_name = processor.cell_line_categories[cell_line_id]
    mask = (obs[cell_col].astype(str) == str(cl_name)) & (obs['perturbation'].astype(str) == str(perturb_name))
    idx = np.where(mask.values)[0]
    if len(idx) == 0:
        return None
    x = processor.adata.X[idx]
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x).mean(axis=0).astype(np.float32)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = DataProcessor(
        args.data_path,
        split_strategy='perturbation',
        atac_key=args.atac_key,
        atac_bank_path=args.atac_bank_path,
        background_key=args.background_key,
    )
    n_genes, n_perts, n_cell_lines = processor.load_data()
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    config = infer_model_config(checkpoint, processor)

    model = PerturbationDiffusionPredictor(
        n_genes=n_genes,
        n_perturbations=n_perts,
        perturb_dim=config['perturb_dim'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        timesteps=config['timesteps'],
        target_mode=config['target_mode'],
        dose_dim=config['dose_dim'],
        time_dim=config['time_dim'],
        drug_dim=(processor.drug_embeddings.shape[1] if processor.drug_embeddings is not None else 0),
        use_atac=(processor.atac_features is not None),
        atac_dim=config['atac_dim'],
        cond_dropout=config['cond_dropout'],
        perturb_gene_vocab_size=len(getattr(processor, 'perturb_gene_vocab', []) or []),
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if len(missing) > 0:
        print(f">>> 提示: checkpoint 缺少以下参数（已用随机初始化兼容）: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if len(unexpected) > 0:
        print(f">>> 提示: checkpoint 存在未使用参数: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in checkpoint['ema_state_dict']:
                p.data.copy_(checkpoint['ema_state_dict'][name].to(device))
    model.eval()

    cell_line_id = resolve_cell_line(processor, args.cell_line)
    control = processor.get_cell_line_control(cell_line_id, device=device).unsqueeze(0)
    cell_line_tensor = torch.tensor([cell_line_id], dtype=torch.long, device=device)
    atac_feat = processor.get_cell_line_atac(cell_line_id, device=device)
    if atac_feat is not None:
        atac_feat = atac_feat.unsqueeze(0)
    if len(args.perturb_genes) > 1 and atac_feat is not None:
        print(">>> 提示: 组合扰动当前默认复用同一 cell-line baseline ATAC；若组合引发显著染色质变化，建议外部构建组合特异 ATAC 条件。")

    latents = []
    perturb_ids = []
    resolved_genes = []
    for gene in args.perturb_genes:
        resolved_gene, auto_picked = resolve_or_autopick_gene(processor, gene, cell_line_id)
        if auto_picked:
            print(f">>> 提示: 扰动 {gene} 不存在，自动替换为 {resolved_gene}")
        resolved_genes.append(resolved_gene)
        pid = processor.perturb_map[resolved_gene]
        perturb_ids.append(pid)
        structured = processor.encode_structured_perturbation_names([resolved_gene])
        latent = model.get_latent(
            rna_control=control,
            perturb=torch.tensor([pid], dtype=torch.long, device=device),
            atac_feat=atac_feat,
            perturb_type=structured['perturb_type'].to(device),
            perturb_gene_a=structured['perturb_gene_a'].to(device),
            perturb_gene_b=structured['perturb_gene_b'].to(device),
            has_second_gene=structured['has_second_gene'].to(device),
        )
        latents.append(latent)

    primary_structured = processor.encode_structured_perturbation_names([resolved_genes[0]])
    primary_structured = {k: v.to(device) for k, v in primary_structured.items()}

    if len(latents) == 1:
        pred = model.predict_single(
            rna_control=control,
            perturb=torch.tensor([perturb_ids[0]], dtype=torch.long, device=device),
            atac_feat=atac_feat,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
            perturb_type=primary_structured['perturb_type'],
            perturb_gene_a=primary_structured['perturb_gene_a'],
            perturb_gene_b=primary_structured['perturb_gene_b'],
            has_second_gene=primary_structured['has_second_gene'],
        )
        latent_used = latents[0]
    else:
        latent_used = model.combine_latents(latents, weights=args.weights, mode=args.latent_mode)
        pred = model.predict_from_latent(
            rna_control=control,
            latent=latent_used,
            perturb=torch.tensor([perturb_ids[0]], dtype=torch.long, device=device),
            atac_feat=atac_feat,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
            perturb_type=primary_structured['perturb_type'],
            perturb_gene_a=primary_structured['perturb_gene_a'],
            perturb_gene_b=primary_structured['perturb_gene_b'],
            has_second_gene=primary_structured['has_second_gene'],
        )

    pred_np = pred.squeeze(0).detach().cpu().numpy()
    ctrl_np = control.squeeze(0).detach().cpu().numpy()
    delta_np = pred_np - ctrl_np
    true_np = None
    if len(resolved_genes) == 1:
        true_np = get_observed_mean_expression(processor, cell_line_id, resolved_genes[0])
        if true_np is not None:
            print(f">>> 已匹配真实均值样本: gene={resolved_genes[0]} | n={int(np.sum((processor.adata.obs['perturbation'].astype(str)==resolved_genes[0]).values))}")

    os.makedirs(args.save_dir, exist_ok=True)
    prefix = "__".join(resolved_genes)
    csv_path = os.path.join(args.save_dir, f"{prefix}_prediction.csv")
    fig_path = os.path.join(args.save_dir, f"{prefix}_top_genes.png")
    latent_path = os.path.join(args.save_dir, f"{prefix}_latent.npy")

    df = pd.DataFrame({
        'gene': processor.adata.var_names.tolist(),
        'control': ctrl_np,
        'prediction': pred_np,
        'delta': delta_np,
        'abs_delta': np.abs(delta_np),
    })
    if true_np is not None:
        df['true'] = true_np
        df['true_delta'] = true_np - ctrl_np
        df['abs_true_delta'] = np.abs(df['true_delta'])
        rank_score = 0.6 * df['abs_true_delta'] + 0.4 * df['abs_delta']
        df['rank_score'] = rank_score
        df = df.sort_values('rank_score', ascending=False)
    else:
        df = df.sort_values('abs_delta', ascending=False)
    df.to_csv(csv_path, index=False)
    np.save(latent_path, latent_used.squeeze(0).detach().cpu().numpy())

    if args.interpolate_to is not None:
        interp_gene, interp_auto = resolve_or_autopick_gene(processor, args.interpolate_to, cell_line_id)
        if interp_auto:
            print(f">>> 提示: interpolate_to={args.interpolate_to} 不存在，自动替换为 {interp_gene}")
        pid_to = processor.perturb_map[interp_gene]
        interp_structured = processor.encode_structured_perturbation_names([interp_gene])
        interp_structured = {k: v.to(device) for k, v in interp_structured.items()}
        latent_to = model.get_latent(
            rna_control=control,
            perturb=torch.tensor([pid_to], dtype=torch.long, device=device),
            atac_feat=atac_feat,
            perturb_type=interp_structured['perturb_type'],
            perturb_gene_a=interp_structured['perturb_gene_a'],
            perturb_gene_b=interp_structured['perturb_gene_b'],
            has_second_gene=interp_structured['has_second_gene'],
        )
        interp = model.interpolate_latents(latents[0], latent_to, steps=args.interp_steps)
        traj_preds = []
        for i in range(interp.shape[0]):
            p = model.predict_from_latent(
                rna_control=control,
                latent=interp[i],
                perturb=torch.tensor([perturb_ids[0]], dtype=torch.long, device=device),
                atac_feat=atac_feat,
                sample_steps=args.sample_steps,
                guidance_scale=args.guidance_scale,
                perturb_type=primary_structured['perturb_type'],
                perturb_gene_a=primary_structured['perturb_gene_a'],
                perturb_gene_b=primary_structured['perturb_gene_b'],
                has_second_gene=primary_structured['has_second_gene'],
            )
            traj_preds.append(p.squeeze(0).detach().cpu().numpy())
        traj_arr = np.stack(traj_preds, axis=0)
        traj_path = os.path.join(args.save_dir, f"{prefix}__to__{interp_gene}_trajectory.npy")
        np.save(traj_path, traj_arr)
        print(f">>> 插值轨迹保存: {traj_path}")

    top_df = df.head(20).copy()
    plt.style.use('seaborn-v0_8-whitegrid')
    if true_np is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        plot_df = top_df.iloc[::-1]
        y_pos = np.arange(len(plot_df))
        axes[0].barh(y_pos - 0.22, plot_df['control'].values, height=0.2, color='#7f8c8d', label='Control')
        axes[0].barh(y_pos, plot_df['prediction'].values, height=0.2, color='#c0392b', label='Predicted')
        axes[0].barh(y_pos + 0.22, plot_df['true'].values, height=0.2, color='#2980b9', label='True')
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(plot_df['gene'].tolist(), fontsize=8)
        axes[0].set_title("Control vs Predicted vs True (Top genes)")
        axes[0].legend(frameon=False)

        axes[1].scatter(plot_df['true_delta'].values, plot_df['delta'].values, alpha=0.8, color='#8e44ad')
        min_v = float(min(plot_df['true_delta'].min(), plot_df['delta'].min()))
        max_v = float(max(plot_df['true_delta'].max(), plot_df['delta'].max()))
        axes[1].plot([min_v, max_v], [min_v, max_v], '--', color='#d95f02', linewidth=1.5)
        corr = np.corrcoef(plot_df['true_delta'].values, plot_df['delta'].values)[0, 1]
        axes[1].set_title(f"True delta vs Pred delta | Pearson={corr:.4f}")
        axes[1].set_xlabel("True delta")
        axes[1].set_ylabel("Predicted delta")
        plt.suptitle(f"{' + '.join(resolved_genes)} | cell_line={args.cell_line}", fontsize=12)
        plt.tight_layout()
    else:
        plot_df = top_df.iloc[::-1]
        plt.figure(figsize=(10, 8))
        sns.barplot(data=plot_df, x='delta', y='gene')
        plt.title(f"Top 20 response genes | {' + '.join(resolved_genes)} | cell_line={args.cell_line}")
        plt.xlabel("Predicted delta")
        plt.ylabel("Gene")
        plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    print(f">>> 预测 CSV: {csv_path}")
    print(f">>> 可视化图: {fig_path}")
    print(f">>> latent 保存: {latent_path}")


if __name__ == '__main__':
    main()
