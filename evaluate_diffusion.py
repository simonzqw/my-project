
import argparse
import json
import os

omp_threads = os.environ.get("OMP_NUM_THREADS")
if not (omp_threads and omp_threads.isdigit() and int(omp_threads) > 0):
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
from scipy.stats import pearsonr
from tqdm import tqdm

from models.scerso_diffusion import PerturbationDiffusionPredictor
from utils.data_processor import DataProcessor
from utils.emb_loader import GeneEmbeddingLoader


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


def init_metric_collector(top_ks=(10, 20, 50)):
    collector = {
        'all_mse': [],
        'all_pearson': [],
        'delta_pearson': [],
        'mse_top20_de_non_dropout': [],
        'frac_opposite_direction_top20_non_dropout': [],
        'rmse_top20': [],
    }
    for k in top_ks:
        collector[f'top{k}_mse'] = []
        collector[f'top{k}_pearson'] = []
        collector[f'top{k}_recall'] = []
        collector[f'pearson_delta_top{k}'] = []
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
    de_quantile=0.9,
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
        de_quantile=de_quantile,
    )
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
        collector['rmse_top20'].append(float(np.sqrt(max(mse20, 0.0))))
    else:
        collector['frac_opposite_direction_top20_non_dropout'].append(0.0)
        collector['mse_top20_de_non_dropout'].append(0.0)
        collector['rmse_top20'].append(0.0)

    for k in top_ks:
        k_eff = min(k, len(d_t))
        top_true = np.argsort(np.abs(d_t))[-k_eff:]
        top_pred = np.argsort(np.abs(d_p))[-k_eff:]
        collector[f'top{k}_mse'].append(float(np.mean((d_p[top_true] - d_t[top_true]) ** 2)))
        r_top = safe_pearson(d_p[top_true], d_t[top_true])
        if not np.isnan(r_top):
            collector[f'top{k}_pearson'].append(r_top)
        collector[f'top{k}_recall'].append(float(len(set(top_true) & set(top_pred)) / max(k_eff, 1)))
        r_delta_top = safe_pearson(d_p[top_true], d_t[top_true])
        if not np.isnan(r_delta_top):
            collector[f'pearson_delta_top{k}'].append(r_delta_top)


def finalize_metric_collector(collector):
    return {k: (float(np.mean(v)) if len(v) > 0 else 0.0) for k, v in collector.items()}


def calculate_metrics(pred, target, ctrl, top_k=(10, 20, 50)):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(ctrl, torch.Tensor):
        ctrl = ctrl.detach().cpu().numpy()

    out = {'all_pearson': [], 'delta_pearson': []}
    for k in top_k:
        out[f'top{k}_mse'] = []
        out[f'top{k}_pearson'] = []
        out[f'top{k}_recall'] = []

    for i in range(pred.shape[0]):
        p, t, c = pred[i], target[i], ctrl[i]
        d_p = p - c
        d_t = t - c

        r_all = safe_pearson(p, t)
        if not np.isnan(r_all):
            out['all_pearson'].append(r_all)

        r_delta = safe_pearson(d_p, d_t)
        if not np.isnan(r_delta):
            out['delta_pearson'].append(r_delta)

        for k in top_k:
            k_eff = min(k, len(d_t))
            top_true = np.argsort(np.abs(d_t))[-k_eff:]
            top_pred = np.argsort(np.abs(d_p))[-k_eff:]
            out[f'top{k}_mse'].append(float(np.mean((d_p[top_true] - d_t[top_true]) ** 2)))
            r_top = safe_pearson(d_p[top_true], d_t[top_true])
            if not np.isnan(r_top):
                out[f'top{k}_pearson'].append(r_top)
            out[f'top{k}_recall'].append(float(len(set(top_true) & set(top_pred)) / max(k_eff, 1)))

    return {k: (float(np.mean(v)) if len(v) > 0 else 0.0) for k, v in out.items()}


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate diffusion perturbation predictor")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--split_strategy', type=str, default='perturbation', choices=['random', 'perturbation', 'custom'])
    parser.add_argument('--split_col', type=str, default='split')
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--sample_steps', type=int, default=None)
    parser.add_argument('--guidance_scale', type=float, default=None)
    parser.add_argument('--target_mode', type=str, default=None, choices=['target', 'delta'])
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--output_json', type=str, default='diffusion_eval.json')
    parser.add_argument('--dropout_eps', type=float, default=1e-3)
    parser.add_argument('--de_mode', type=str, default='threshold', choices=['threshold', 'topk', 'quantile'])
    parser.add_argument('--de_topk', type=int, default=200)
    parser.add_argument('--de_quantile', type=float, default=0.9)
    parser.add_argument('--atac_key', type=str, default=None)
    parser.add_argument('--atac_bank_path', type=str, default=None)
    parser.add_argument('--background_key', type=str, default='cell_context')
    parser.add_argument('--control_match_mode', type=str, default='random', choices=['random', 'atac_knn'])
    parser.add_argument('--control_match_k', type=int, default=32)
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint, n_genes, n_perts, processor, device, target_mode_override=None):
    ckpt_args = checkpoint.get('args', argparse.Namespace())
    pretrained_weights = None

    # 尽量从 checkpoint 恢复语义 perturb embedding 模式
    if hasattr(ckpt_args, 'pretrained_emb') and ckpt_args.pretrained_emb:
        loader = GeneEmbeddingLoader(ckpt_args.pretrained_emb, processor.id_to_perturb)
        pretrained_weights = loader.load_weights()
    else:
        state_dict = checkpoint['model_state_dict']
        if 'perturb_embedding.weight' in state_dict:
            emb_weight = state_dict['perturb_embedding.weight']
            if emb_weight.shape[0] == n_perts:
                pretrained_weights = None

    atac_dim = processor.atac_dim if getattr(processor, 'atac_features', None) is not None else 0

    model = PerturbationDiffusionPredictor(
        n_genes=n_genes,
        n_perturbations=n_perts,
        pretrained_weights=pretrained_weights,
        perturb_dim=getattr(ckpt_args, 'perturb_dim', state_dict_dim(checkpoint, 'perturb_embedding.weight', default=200)),
        hidden_dims=getattr(ckpt_args, 'hidden_dims', [512, 512, 512]),
        dropout=getattr(ckpt_args, 'dropout', 0.1),
        timesteps=getattr(ckpt_args, 'timesteps', 1000),
        target_mode=(target_mode_override if target_mode_override is not None else getattr(ckpt_args, 'target_mode', 'target')),
        dose_dim=getattr(ckpt_args, 'dose_dim', 32),
        time_dim=getattr(ckpt_args, 'time_dim', 128),
        drug_dim=(processor.drug_embeddings.shape[1] if processor.drug_embeddings is not None else 0),
        use_atac=(processor.atac_features is not None),
        atac_dim=atac_dim,
        cond_dropout=getattr(ckpt_args, 'cond_dropout', 0.0),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if getattr(get_args_cache, 'use_ema', False) and ('ema_state_dict' in checkpoint) and (checkpoint['ema_state_dict'] is not None):
        for name, p in model.named_parameters():
            if p.requires_grad and name in checkpoint['ema_state_dict']:
                p.data.copy_(checkpoint['ema_state_dict'][name].to(device))
    model.eval()
    return model


def state_dict_dim(checkpoint, key, default=0):
    sd = checkpoint['model_state_dict']
    if key not in sd:
        return default
    return int(sd[key].shape[1])


def evaluate():
    global get_args_cache
    args = get_args()
    get_args_cache = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get('args', argparse.Namespace())

    processor = DataProcessor(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        split_strategy=args.split_strategy,
        split_col=args.split_col,
        atac_key=args.atac_key if args.atac_key is not None else getattr(ckpt_args, 'atac_key', None),
        atac_bank_path=args.atac_bank_path if args.atac_bank_path is not None else getattr(ckpt_args, 'atac_bank_path', None),
        background_key=args.background_key if args.background_key is not None else getattr(ckpt_args, 'background_key', 'cell_context'),
    )
    n_genes, n_perts, _ = processor.load_data()
    _, _, test_loader = processor.prepare_loaders(
        batch_size=args.batch_size,
        rna_noise=0.0,
        atac_key=args.atac_key if args.atac_key is not None else getattr(ckpt_args, 'atac_key', None),
        atac_bank_path=args.atac_bank_path if args.atac_bank_path is not None else getattr(ckpt_args, 'atac_bank_path', None),
        background_key=args.background_key if args.background_key is not None else getattr(ckpt_args, 'background_key', 'cell_context'),
        control_match_mode=args.control_match_mode,
        control_match_k=args.control_match_k,
    )

    model = load_model_from_checkpoint(checkpoint, n_genes, n_perts, processor, device, target_mode_override=args.target_mode)
    drug_embeddings = processor.drug_embeddings.to(device) if processor.drug_embeddings is not None else None

    sample_steps = args.sample_steps if args.sample_steps is not None else getattr(ckpt_args, 'sample_steps', 50)
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else getattr(ckpt_args, 'guidance_scale', 1.0)

    sample_collector = init_metric_collector(top_ks=(10, 20, 50))
    by_perturb = {}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            ctrl = batch['rna_control'].to(device)
            target = batch['rna_target'].to(device)
            perturb = batch['perturb'].to(device)
            dose = batch['dose'].to(device) if 'dose' in batch else None
            atac_feat = batch['atac_feat'].to(device) if 'atac_feat' in batch else None
            drug_feat = drug_embeddings[perturb] if drug_embeddings is not None else None

            pred = model.predict_single(
                rna_control=ctrl,
                perturb=perturb,
                dose=dose,
                atac_feat=atac_feat,
                drug_feat=drug_feat,
                sample_steps=sample_steps,
                guidance_scale=guidance_scale,
            )
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            ctrl_np = ctrl.cpu().numpy()
            perturb_ids = batch['perturb'].cpu().numpy()

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
                    de_quantile=args.de_quantile,
                )
                p_name = processor.id_to_perturb[int(perturb_ids[i])]
                if p_name not in by_perturb:
                    by_perturb[p_name] = {'p': [], 't': [], 'c': []}
                by_perturb[p_name]['p'].append(pred_np[i])
                by_perturb[p_name]['t'].append(target_np[i])
                by_perturb[p_name]['c'].append(ctrl_np[i])

    sample_m = finalize_metric_collector(sample_collector)

    perturb_collector = init_metric_collector(top_ks=(10, 20, 50))
    for p_name, data in by_perturb.items():
        if p_name == 'control':
            continue
        p_mean = np.mean(np.stack(data['p'], axis=0), axis=0)
        t_mean = np.mean(np.stack(data['t'], axis=0), axis=0)
        c_mean = np.mean(np.stack(data['c'], axis=0), axis=0)
        update_metric_collector(
            perturb_collector,
            p_mean,
            t_mean,
            c_mean,
            top_ks=(10, 20, 50),
            dropout_eps=args.dropout_eps,
            de_mode=args.de_mode,
            de_topk=args.de_topk,
            de_quantile=args.de_quantile,
        )
    perturb_m = finalize_metric_collector(perturb_collector)

    final_m = {
        "test_unseen_single_pearson": sample_m["all_pearson"],
        "test_unseen_single_mse": sample_m["all_mse"],
        "test_unseen_single_pearson_delta": sample_m["delta_pearson"],
        "test_unseen_single_pearson_delta_top20": sample_m["pearson_delta_top20"],
        "test_unseen_single_rmse_top20": sample_m["rmse_top20"],
        "test_unseen_single_mse_top20_de_non_dropout": sample_m["mse_top20_de_non_dropout"],
        "test_unseen_single_opposite_direction_top20_non_dropout": sample_m["frac_opposite_direction_top20_non_dropout"],
        "test_unseen_perturb_pearson": perturb_m["all_pearson"],
        "test_unseen_perturb_mse": perturb_m["all_mse"],
        "perturb_pearson_delta": perturb_m["delta_pearson"],
        "perturb_pearson_delta_top20": perturb_m["pearson_delta_top20"],
        "test_unseen_perturb_mse_top20_de_non_dropout": perturb_m["mse_top20_de_non_dropout"],
        "opposite_direction_top20_non_dropout": perturb_m["frac_opposite_direction_top20_non_dropout"],
        "perturb_rmse_top20": perturb_m["rmse_top20"],
        "sample_steps": sample_steps,
        "guidance_scale": guidance_scale,
        "model_path": args.model_path,
    }

    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(final_m, f, indent=2, ensure_ascii=False)

    print(json.dumps(final_m, indent=2, ensure_ascii=False))
    print(f">>> 指标已保存到: {args.output_json}")


if __name__ == "__main__":
    evaluate()
