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


def _build_de_mask(delta_true, mode='threshold', dropout_eps=1e-3, de_topk=200, de_quantile=0.9):
    abs_dt = np.abs(delta_true)
    if mode == 'topk':
        k = min(de_topk, len(abs_dt))
        idx = np.argsort(abs_dt)[-k:]
        mask = np.zeros_like(abs_dt, dtype=bool)
        mask[idx] = True
        return mask
    if mode == 'quantile':
        q = np.quantile(abs_dt, de_quantile)
        return abs_dt >= q
    return abs_dt > dropout_eps


def collect_metrics(
    pred,
    target,
    ctrl,
    top_ks=(10, 20, 50),
    dropout_eps=1e-3,
    de_mode='threshold',
    de_topk=200,
    de_quantile=0.9
):
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

        non_dropout_mask = _build_de_mask(
            d_t,
            mode=de_mode,
            dropout_eps=dropout_eps,
            de_topk=de_topk,
            de_quantile=de_quantile
        )
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
        top20_non_dropout = top20_idx[non_dropout_mask[top20_idx]]
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


def init_metric_collector(top_ks=(10, 20, 50)):
    collector = {
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
        collector[f'top{k}_mse'] = []
        collector[f'top{k}_pearson'] = []
        collector[f'top{k}_recall'] = []
    return collector


def update_metric_collector(
    collector,
    p,
    t,
    c,
    top_ks=(10, 20, 50),
    dropout_eps=1e-3,
    de_mode='threshold',
    de_topk=200,
    de_quantile=0.9
):
    d_p = p - c
    d_t = t - c

    collector['all_mse'].append(float(np.mean((p - t) ** 2)))
    r_all = safe_pearson(p, t)
    if not np.isnan(r_all):
        collector['all_pearson'].append(r_all)

    r_delta = safe_pearson(d_p, d_t)
    if not np.isnan(r_delta):
        collector['delta_pearson'].append(r_delta)

    non_dropout_mask = _build_de_mask(
        d_t,
        mode=de_mode,
        dropout_eps=dropout_eps,
        de_topk=de_topk,
        de_quantile=de_quantile
    )
    if np.any(non_dropout_mask):
        d_p_n = d_p[non_dropout_mask]
        d_t_n = d_t[non_dropout_mask]
        err_n = d_p_n - d_t_n
        collector['all_de_mse'].append(float(np.mean(err_n ** 2)))
        r_de = safe_pearson(d_p_n, d_t_n)
        if not np.isnan(r_de):
            collector['all_de_pearson'].append(r_de)

        sigma = np.std(d_t_n)
        if sigma > 1e-8:
            collector['frac_sigma_below_1_non_dropout'].append(float(np.mean(np.abs(err_n) < sigma)))
        else:
            collector['frac_sigma_below_1_non_dropout'].append(0.0)

    k20 = min(20, len(d_t))
    top20_idx = np.argsort(np.abs(d_t))[-k20:]
    top20_non_dropout = top20_idx[non_dropout_mask[top20_idx]]
    if len(top20_non_dropout) > 0:
        t20 = d_t[top20_non_dropout]
        p20 = d_p[top20_non_dropout]
        opp = np.mean(np.sign(t20) * np.sign(p20) < 0)
        mse20 = np.mean((p20 - t20) ** 2)
        collector['frac_opposite_direction_top20_non_dropout'].append(float(opp))
        collector['mse_top20_de_non_dropout'].append(float(mse20))
    else:
        collector['frac_opposite_direction_top20_non_dropout'].append(0.0)
        collector['mse_top20_de_non_dropout'].append(0.0)

    for k in top_ks:
        k_eff = min(k, len(d_t))
        top_true = np.argsort(np.abs(d_t))[-k_eff:]
        top_pred = np.argsort(np.abs(d_p))[-k_eff:]
        collector[f'top{k}_mse'].append(float(np.mean((d_p[top_true] - d_t[top_true]) ** 2)))
        r_top = safe_pearson(d_p[top_true], d_t[top_true])
        if not np.isnan(r_top):
            collector[f'top{k}_pearson'].append(r_top)
        collector[f'top{k}_recall'].append(float(len(set(top_true) & set(top_pred)) / max(k_eff, 1)))


def finalize_metric_collector(collector):
    return {k: (float(np.mean(v)) if len(v) > 0 else 0.0) for k, v in collector.items()}


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate best model with Top10/20/50/all metrics.")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--split_strategy', type=str, default='perturbation', choices=['random', 'perturbation'])
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_ema', action='store_true', help='Evaluate with ema_state_dict if available')
    parser.add_argument('--output_json', type=str, default=None, help='Optional path to dump metrics json')
    parser.add_argument('--dropout_eps', type=float, default=1e-3, help='DE threshold when de_mode=threshold')
    parser.add_argument('--de_mode', type=str, default='threshold', choices=['threshold', 'topk', 'quantile'])
    parser.add_argument('--de_topk', type=int, default=200, help='DE topK when de_mode=topk')
    parser.add_argument('--de_quantile', type=float, default=0.9, help='DE quantile when de_mode=quantile')
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
    has_perturb_encoder = any(k.startswith('perturb_encoder.') for k in state_dict.keys())
    pretrained_weights = state_dict['perturb_feature_bank'] if 'perturb_feature_bank' in state_dict else None
    perturb_weight_for_shape = state_dict['perturb_embedding.weight'] if 'perturb_embedding.weight' in state_dict else None
    # 兼容旧 checkpoint：存在 perturb_encoder 参数但没有 perturb_feature_bank
    # 这时用 perturb_embedding.weight 作为 feature_bank 构建语义分支结构，确保能完整加载权重。
    if pretrained_weights is None and has_perturb_encoder and perturb_weight_for_shape is not None:
        pretrained_weights = perturb_weight_for_shape.detach().clone()
    missing_feature_bank_in_ckpt = has_perturb_encoder and ('perturb_feature_bank' not in state_dict)

    perturb_dim = int(perturb_weight_for_shape.shape[1]) if perturb_weight_for_shape is not None else int(pretrained_weights.shape[1])
    n_perturbations = int(perturb_weight_for_shape.shape[0]) if perturb_weight_for_shape is not None else int(pretrained_weights.shape[0])

    model = PerturbationPredictor(
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        n_cell_lines=state_dict['cell_line_embedding.weight'].shape[0],
        pretrained_weights=pretrained_weights,
        perturb_dim=perturb_dim,
        cell_line_dim=state_dict['cell_line_embedding.weight'].shape[1],
        drug_dim=getattr(model_args, 'drug_dim', 2048),
        hidden_dims=getattr(model_args, 'hidden_dims', [512, 1024, 2048]),
        dropout=getattr(model_args, 'dropout', 0.2),
        d_model=getattr(model_args, 'd_model', 256),
        nhead=getattr(model_args, 'nhead', 8),
        num_layers=getattr(model_args, 'num_layers', 4),
        dim_ff=getattr(model_args, 'dim_ff', 1024),
        n_ctrl_tokens=getattr(model_args, 'n_ctrl_tokens', 8)
    ).to(device)
    model.load_state_dict(state_dict, strict=not missing_feature_bank_in_ckpt)
    model.eval()

    drug_embeddings = processor.drug_embeddings.to(device) if processor.drug_embeddings is not None else None

    sample_collector = init_metric_collector(top_ks=(10, 20, 50))
    by_perturb = {}
    with torch.no_grad():
        for batch in test_loader:
            ctrl = batch['rna_control'].to(device)
            target = batch['rna_target'].to(device)
            perturb = batch['perturb'].to(device)
            cell_line = batch['cell_line'].to(device)
            dose = batch['dose'].to(device) if 'dose' in batch else None
            perturb_ids = batch['perturb'].cpu().numpy()

            drug_feat = drug_embeddings[perturb] if drug_embeddings is not None else None
            pred = model(ctrl, perturb, cell_line, drug_feat=drug_feat, dose=dose)

            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            ctrl_np = ctrl.cpu().numpy()

            for i in range(pred_np.shape[0]):
                update_metric_collector(
                    sample_collector,
                    pred_np[i],
                    target_np[i],
                    ctrl_np[i],
                    top_ks=(10, 20, 50),
                    dropout_eps=args.dropout_eps,
                    de_mode=args.de_mode,
                    de_topk=args.de_topk,
                    de_quantile=args.de_quantile
                )
                p_name = processor.id_to_perturb[int(perturb_ids[i])]
                if p_name not in by_perturb:
                    by_perturb[p_name] = {'p': [], 't': [], 'c': []}
                by_perturb[p_name]['p'].append(pred_np[i])
                by_perturb[p_name]['t'].append(target_np[i])
                by_perturb[p_name]['c'].append(ctrl_np[i])

    final_m = finalize_metric_collector(sample_collector)

    # Perturbation-level (group-level) aggregation
    perturb_collector = init_metric_collector(top_ks=(10, 20, 50))
    n_eval_perturbations = 0
    for p_name, data in by_perturb.items():
        if p_name == 'control':
            continue
        avg_p = np.mean(np.array(data['p']), axis=0)
        avg_t = np.mean(np.array(data['t']), axis=0)
        avg_c = np.mean(np.array(data['c']), axis=0)
        update_metric_collector(
            perturb_collector,
            avg_p,
            avg_t,
            avg_c,
            top_ks=(10, 20, 50),
            dropout_eps=args.dropout_eps,
            de_mode=args.de_mode,
            de_topk=args.de_topk,
            de_quantile=args.de_quantile
        )
        n_eval_perturbations += 1
    perturb_m = finalize_metric_collector(perturb_collector)

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
        "all_pearson": final_m["all_pearson"],
        "test_unseen_perturb_mse": perturb_m["all_mse"],
        "test_unseen_perturb_pearson": perturb_m["all_pearson"],
        "test_unseen_perturb_mse_de": perturb_m["all_de_mse"],
        "test_unseen_perturb_pearson_de": perturb_m["all_de_pearson"],
        "test_unseen_perturb_pearson_delta": perturb_m["delta_pearson"],
        "test_unseen_perturb_mse_top20_de_non_dropout": perturb_m["mse_top20_de_non_dropout"],
        "test_unseen_perturb_frac_opposite_direction_top20_non_dropout": perturb_m["frac_opposite_direction_top20_non_dropout"],
        "n_eval_perturbations": n_eval_perturbations
    }
    report["metric_config"] = {
        "de_mode": args.de_mode,
        "dropout_eps": args.dropout_eps,
        "de_topk": args.de_topk,
        "de_quantile": args.de_quantile,
        "sample_aggregation": "sample-weighted",
        "group_aggregation": "perturbation-mean-profile"
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f">>> Metrics saved to: {args.output_json}")


if __name__ == "__main__":
    main()
