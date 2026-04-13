
import argparse
import os

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
    cell_line_dim = int(state_dict['cell_line_embedding.weight'].shape[1])
    return dict(
        perturb_dim=perturb_dim,
        cell_line_dim=cell_line_dim,
        hidden_dims=getattr(ckpt_args, 'hidden_dims', [512, 512, 512]),
        dropout=getattr(ckpt_args, 'dropout', 0.1),
        timesteps=getattr(ckpt_args, 'timesteps', 1000),
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
        n_cell_lines=n_cell_lines,
        perturb_dim=config['perturb_dim'],
        cell_line_dim=config['cell_line_dim'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        timesteps=config['timesteps'],
        dose_dim=config['dose_dim'],
        time_dim=config['time_dim'],
        drug_dim=(processor.drug_embeddings.shape[1] if processor.drug_embeddings is not None else 0),
        use_atac=(processor.atac_features is not None),
        atac_dim=config['atac_dim'],
        cond_dropout=config['cond_dropout'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
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
    for gene in args.perturb_genes:
        if gene not in processor.perturb_map:
            raise ValueError(f"扰动 {gene} 不在 perturbation 列表中。")
        pid = processor.perturb_map[gene]
        perturb_ids.append(pid)
        latent = model.get_latent(
            rna_control=control,
            perturb=torch.tensor([pid], dtype=torch.long, device=device),
            cell_line=cell_line_tensor,
            atac_feat=atac_feat,
        )
        latents.append(latent)

    if len(latents) == 1:
        pred = model.predict_single(
            rna_control=control,
            perturb=torch.tensor([perturb_ids[0]], dtype=torch.long, device=device),
            cell_line=cell_line_tensor,
            atac_feat=atac_feat,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
        )
        latent_used = latents[0]
    else:
        latent_used = model.combine_latents(latents, weights=args.weights, mode=args.latent_mode)
        pred = model.predict_from_latent(
            rna_control=control,
            cell_line=cell_line_tensor,
            latent=latent_used,
            perturb=torch.tensor([perturb_ids[0]], dtype=torch.long, device=device),
            atac_feat=atac_feat,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
        )

    pred_np = pred.squeeze(0).detach().cpu().numpy()
    ctrl_np = control.squeeze(0).detach().cpu().numpy()
    delta_np = pred_np - ctrl_np

    os.makedirs(args.save_dir, exist_ok=True)
    prefix = "__".join(args.perturb_genes)
    csv_path = os.path.join(args.save_dir, f"{prefix}_prediction.csv")
    fig_path = os.path.join(args.save_dir, f"{prefix}_top_genes.png")
    latent_path = os.path.join(args.save_dir, f"{prefix}_latent.npy")

    df = pd.DataFrame({
        'gene': processor.adata.var_names.tolist(),
        'control': ctrl_np,
        'prediction': pred_np,
        'delta': delta_np,
        'abs_delta': np.abs(delta_np),
    }).sort_values('abs_delta', ascending=False)
    df.to_csv(csv_path, index=False)
    np.save(latent_path, latent_used.squeeze(0).detach().cpu().numpy())

    if args.interpolate_to is not None:
        if args.interpolate_to not in processor.perturb_map:
            raise ValueError(f"interpolate_to 扰动 {args.interpolate_to} 不在 perturbation 列表中。")
        pid_to = processor.perturb_map[args.interpolate_to]
        latent_to = model.get_latent(
            rna_control=control,
            perturb=torch.tensor([pid_to], dtype=torch.long, device=device),
            cell_line=cell_line_tensor,
            atac_feat=atac_feat,
        )
        interp = model.interpolate_latents(latents[0], latent_to, steps=args.interp_steps)
        traj_preds = []
        for i in range(interp.shape[0]):
            p = model.predict_from_latent(
                rna_control=control,
                cell_line=cell_line_tensor,
                latent=interp[i],
                perturb=torch.tensor([perturb_ids[0]], dtype=torch.long, device=device),
                atac_feat=atac_feat,
                sample_steps=args.sample_steps,
                guidance_scale=args.guidance_scale,
            )
            traj_preds.append(p.squeeze(0).detach().cpu().numpy())
        traj_arr = np.stack(traj_preds, axis=0)
        traj_path = os.path.join(args.save_dir, f"{prefix}__to__{args.interpolate_to}_trajectory.npy")
        np.save(traj_path, traj_arr)
        print(f">>> 插值轨迹保存: {traj_path}")

    top_df = df.head(20).iloc[::-1]
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_df, x='delta', y='gene')
    plt.title(f"Top 20 response genes | {' + '.join(args.perturb_genes)} | cell_line={args.cell_line}")
    plt.xlabel("Predicted delta")
    plt.ylabel("Gene")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f">>> 预测 CSV: {csv_path}")
    print(f">>> 可视化图: {fig_path}")
    print(f">>> latent 保存: {latent_path}")


if __name__ == '__main__':
    main()
