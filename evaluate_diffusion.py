
import argparse
import json
import os

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
    parser.add_argument('--split_strategy', type=str, default='perturbation', choices=['random', 'perturbation'])
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--sample_steps', type=int, default=None)
    parser.add_argument('--guidance_scale', type=float, default=None)
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--output_json', type=str, default='diffusion_eval.json')
    parser.add_argument('--atac_key', type=str, default=None)
    parser.add_argument('--atac_bank_path', type=str, default=None)
    parser.add_argument('--background_key', type=str, default='cell_context')
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint, n_genes, n_perts, n_cell_lines, processor, device):
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
        n_cell_lines=n_cell_lines,
        pretrained_weights=pretrained_weights,
        perturb_dim=getattr(ckpt_args, 'perturb_dim', state_dict_dim(checkpoint, 'perturb_embedding.weight', default=200)),
        cell_line_dim=getattr(ckpt_args, 'cell_line_dim', state_dict_dim(checkpoint, 'cell_line_embedding.weight', default=32)),
        hidden_dims=getattr(ckpt_args, 'hidden_dims', [512, 512, 512]),
        dropout=getattr(ckpt_args, 'dropout', 0.1),
        timesteps=getattr(ckpt_args, 'timesteps', 1000),
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
        atac_key=args.atac_key if args.atac_key is not None else getattr(ckpt_args, 'atac_key', None),
        atac_bank_path=args.atac_bank_path if args.atac_bank_path is not None else getattr(ckpt_args, 'atac_bank_path', None),
        background_key=args.background_key if args.background_key is not None else getattr(ckpt_args, 'background_key', 'cell_context'),
    )
    n_genes, n_perts, n_cell_lines = processor.load_data()
    _, _, test_loader = processor.prepare_loaders(
        batch_size=args.batch_size,
        rna_noise=0.0,
        atac_key=args.atac_key if args.atac_key is not None else getattr(ckpt_args, 'atac_key', None),
        atac_bank_path=args.atac_bank_path if args.atac_bank_path is not None else getattr(ckpt_args, 'atac_bank_path', None),
        background_key=args.background_key if args.background_key is not None else getattr(ckpt_args, 'background_key', 'cell_context'),
    )

    model = load_model_from_checkpoint(checkpoint, n_genes, n_perts, n_cell_lines, processor, device)
    drug_embeddings = processor.drug_embeddings.to(device) if processor.drug_embeddings is not None else None

    sample_steps = args.sample_steps if args.sample_steps is not None else getattr(ckpt_args, 'sample_steps', 50)
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else getattr(ckpt_args, 'guidance_scale', 1.0)

    metrics = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            ctrl = batch['rna_control'].to(device)
            target = batch['rna_target'].to(device)
            perturb = batch['perturb'].to(device)
            cell_line = batch['cell_line'].to(device)
            dose = batch['dose'].to(device) if 'dose' in batch else None
            atac_feat = batch['atac_feat'].to(device) if 'atac_feat' in batch else None
            drug_feat = drug_embeddings[perturb] if drug_embeddings is not None else None

            pred = model.predict_single(
                rna_control=ctrl,
                perturb=perturb,
                cell_line=cell_line,
                dose=dose,
                atac_feat=atac_feat,
                drug_feat=drug_feat,
                sample_steps=sample_steps,
                guidance_scale=guidance_scale,
            )
            metrics.append(calculate_metrics(pred, target, ctrl))

    final_m = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0].keys()} if metrics else {}
    final_m['sample_steps'] = sample_steps
    final_m['guidance_scale'] = guidance_scale
    final_m['model_path'] = args.model_path

    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(final_m, f, indent=2, ensure_ascii=False)

    print(json.dumps(final_m, indent=2, ensure_ascii=False))
    print(f">>> 指标已保存到: {args.output_json}")


if __name__ == "__main__":
    evaluate()
