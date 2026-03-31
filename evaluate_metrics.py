import argparse
import json
import numpy as np
import torch
from scipy.stats import pearsonr

from utils.data_processor import DataProcessor
from models.reasoning_mlp import PerturbationPredictor


def safe_pearson(x, y):
    if np.std(x) <= 1e-8 or np.std(y) <= 1e-8:
        return np.nan
    r, _ = pearsonr(x, y)
    return r if not np.isnan(r) else np.nan


def collect_metrics(pred, target, ctrl, top_ks=(10, 20, 50), dropout_eps=1e-8):
    metrics = {
        'all_mse': [],
        'all_pearson': [],
        'all_de_mse': [],
        'all_de_pearson': [],
        'delta_pearson': [],
        'frac_opposite_direction_top20_non_dropout': [],
        'frac_sigma_below_1_non_dropout': [],
        'mse_top20_de_non_dropout': [],
    }
    for k in top_ks:
        metrics[f'top{k}_mse'] = []
        metrics[f'top{k}_pearson'] = []
        metrics[f'top{k}_recall'] = []

    for i in range(pred.shape[0]):
        p, t, c = pred[i], target[i], ctrl[i]
        d_p = p - c
        d_t = t - c

        metrics['all_mse'].append(float(np.mean((p - t) ** 2)))
        r_all = safe_pearson(p, t)
        if not np.isnan(r_all):
            metrics['all_pearson'].append(r_all)

        r_delta = safe_pearson(d_p, d_t)
        if not np.isnan(r_delta):
            metrics['delta_pearson'].append(r_delta)

        non_dropout_mask = np.abs(d_t) > dropout_eps
        if np.any(non_dropout_mask):
            d_p_n = d_p[non_dropout_mask]
            d_t_n = d_t[non_dropout_mask]
            err_n = d_p_n - d_t_n

            metrics['all_de_mse'].append(float(np.mean(err_n ** 2)))
            r_de = safe_pearson(d_p_n, d_t_n)
            if not np.isnan(r_de):
                metrics['all_de_pearson'].append(r_de)

            sigma = np.std(d_t_n)
            if sigma > 1e-8:
                metrics['frac_sigma_below_1_non_dropout'].append(float(np.mean(np.abs(err_n) < sigma)))
            else:
                metrics['frac_sigma_below_1_non_dropout'].append(0.0)

        k20 = min(20, len(d_t))
        top20_idx = np.argsort(np.abs(d_t))[-k20:]
        top20_non_dropout = top20_idx[np.abs(d_t[top20_idx]) > dropout_eps]
        if len(top20_non_dropout) > 0:
            t20 = d_t[top20_non_dropout]
            p20 = d_p[top20_non_dropout]
            opp = np.mean(np.sign(t20) * np.sign(p20) < 0)
            mse20 = np.mean((p20 - t20) ** 2)
            metrics['frac_opposite_direction_top20_non_dropout'].append(float(opp))
            metrics['mse_top20_de_non_dropout'].append(float(mse20))
        else:
            metrics['frac_opposite_direction_top20_non_dropout'].append(0.0)
            metrics['mse_top20_de_non_dropout'].append(0.0)

        for k in top_ks:
            k_eff = min(k, len(d_t))
            top_true = np.argsort(np.abs(d_t))[-k_eff:]
            top_pred = np.argsort(np.abs(d_p))[-k_eff:]
            metrics[f'top{k}_mse'].append(float(np.mean((d_p[top_true] - d_t[top_true]) ** 2)))
            r_top = safe_pearson(d_p[top_true], d_t[top_true])
            if not np.isnan(r_top):
                metrics[f'top{k}_pearson'].append(r_top)
            metrics[f'top{k}_recall'].append(float(len(set(top_true) & set(top_pred)) / max(k_eff, 1)))

    return {k: (float(np.mean(v)) if len(v) > 0 else 0.0) for k, v in metrics.items()}


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate best model with Top10/20/50/all metrics.")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--split_strategy', type=str, default='perturbation', choices=['random', 'perturbation'])
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_ema', action='store_true', help='Evaluate with ema_state_dict if available')
    parser.add_argument('--output_json', type=str, default=None, help='Optional path to dump metrics json')
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = DataProcessor(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        split_strategy=args.split_strategy
    )
    n_genes, n_perts, n_cell_lines = processor.load_data()
    _, _, test_loader = processor.prepare_loaders(batch_size=args.batch_size, num_workers=args.num_workers, rna_noise=0.0)

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = ckpt['ema_state_dict'] if (args.use_ema and ('ema_state_dict' in ckpt) and ckpt['ema_state_dict'] is not None) else ckpt['model_state_dict']
    model_args = ckpt.get('args', None)

    model = PerturbationPredictor(
        n_genes=n_genes,
        n_perturbations=state_dict['perturb_embedding.weight'].shape[0],
        n_cell_lines=state_dict['cell_line_embedding.weight'].shape[0],
        perturb_dim=state_dict['perturb_embedding.weight'].shape[1],
        cell_line_dim=state_dict['cell_line_embedding.weight'].shape[1],
        drug_dim=getattr(model_args, 'drug_dim', 2048),
        hidden_dims=getattr(model_args, 'hidden_dims', [512, 1024, 2048]),
        dropout=getattr(model_args, 'dropout', 0.2)
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    drug_embeddings = processor.drug_embeddings.to(device) if processor.drug_embeddings is not None else None

    per_batch_metrics = []
    with torch.no_grad():
        for batch in test_loader:
            ctrl = batch['rna_control'].to(device)
            target = batch['rna_target'].to(device)
            perturb = batch['perturb'].to(device)
            cell_line = batch['cell_line'].to(device)
            dose = batch['dose'].to(device) if 'dose' in batch else None

            drug_feat = drug_embeddings[perturb] if drug_embeddings is not None else None
            pred = model(ctrl, perturb, cell_line, drug_feat=drug_feat, dose=dose)

            m = collect_metrics(
                pred.cpu().numpy(),
                target.cpu().numpy(),
                ctrl.cpu().numpy()
            )
            per_batch_metrics.append(m)

    final_m = {k: float(np.mean([x[k] for x in per_batch_metrics])) for k in per_batch_metrics[0].keys()}

    report = {
        "Best model": "Best performing model: Test Top 20 DE MSE",
        "test_unseen_single_mse": final_m["all_mse"],
        "test_unseen_single_pearson": final_m["all_pearson"],
        "test_unseen_single_mse_de": final_m["all_de_mse"],
        "test_unseen_single_pearson_de": final_m["all_de_pearson"],
        "test_unseen_single_pearson_delta": final_m["delta_pearson"],
        "test_unseen_single_frac_opposite_direction_top20_non_dropout": final_m["frac_opposite_direction_top20_non_dropout"],
        "test_unseen_single_frac_sigma_below_1_non_dropout": final_m["frac_sigma_below_1_non_dropout"],
        "test_unseen_single_mse_top20_de_non_dropout": final_m["mse_top20_de_non_dropout"],
        "top10_mse": final_m["top10_mse"],
        "top10_pearson": final_m["top10_pearson"],
        "top20_mse": final_m["top20_mse"],
        "top20_pearson": final_m["top20_pearson"],
        "top50_mse": final_m["top50_mse"],
        "top50_pearson": final_m["top50_pearson"],
        "all_mse": final_m["all_mse"],
        "all_pearson": final_m["all_pearson"]
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f">>> Metrics saved to: {args.output_json}")


if __name__ == "__main__":
    main()
