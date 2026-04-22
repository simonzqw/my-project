
import argparse
import glob
import os
import re

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from models.scerso_diffusion import PerturbationDiffusionPredictor
from utils.data_processor import DataProcessor
from utils.diffusion_schedule import LossSecondMomentResampler, UniformTimestepSampler
from utils.emb_loader import GeneEmbeddingLoader


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def print_run_header(args, device):
    print(f"\n┌{_hr('─', 70)}┐")
    print(f"│ {'scERso Diffusion Training':^68} │")
    print(f"├{_hr('─', 70)}┤")
    print(f"│ device: {str(device):<60} │")
    print(f"│ data:   {args.data_path[:60]:<60} │")
    print(f"│ split:  {args.split_strategy:<60} │")
    print(f"│ steps:  t={args.timesteps}, sample={args.sample_steps}, mode={args.target_mode:<6}, sampler={args.timestep_sampler:<15} │")
    print(f"│ cfg:    scale={args.guidance_scale:.2f}, cond_dropout={args.cond_dropout:.2f}, amp={str(args.amp):<15} │")
    print(f"│ early:  metric={args.early_stop_metric:<16} (w_d={args.score_w_delta:.2f}, w_p={args.score_w_top20p:.2f}, w_m={args.score_w_top20mse:.2f}) │")
    print(f"└{_hr('─', 70)}┘")


def print_epoch_summary(epoch, total_epochs, train_loss, val_loss, metrics):
    top20_p = metrics.get('top20_pearson', 0.0)
    delta_p = metrics.get('delta_pearson', 0.0)
    top20_m = metrics.get('top20_mse', 0.0)
    print(
        f"[E{epoch:03d}/{total_epochs:03d}] "
        f"train={train_loss:.4f}  val={val_loss:.4f}  "
        f"top20_p={top20_p:.4f}  delta_p={delta_p:.4f}  top20_mse={top20_m:.4f}"
    )


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.detach().clone()
                p.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.copy_(self.backup[name])
        self.backup = {}


def rotate_epoch_checkpoints(save_dir, keep_last_n):
    pattern = os.path.join(save_dir, "epoch_*.pth")
    files = glob.glob(pattern)
    if not files:
        return

    def _epoch_num(path):
        match = re.search(r"epoch_(\d+)\.pth$", os.path.basename(path))
        return int(match.group(1)) if match else -1

    files = sorted(files, key=_epoch_num)
    if len(files) > keep_last_n:
        for stale in files[:-keep_last_n]:
            if os.path.exists(stale):
                os.remove(stale)


def get_args():
    parser = argparse.ArgumentParser(description="scERso Diffusion Training (Squidiff-inspired latent injection)")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_diff')
    parser.add_argument('--pretrained_emb', type=str, default=None)
    parser.add_argument('--preset', type=str, default='none', choices=['none', 'vnext', 'smoke'])

    parser.add_argument('--split_strategy', type=str, default='perturbation', choices=['random', 'perturbation', 'custom'])
    parser.add_argument('--split_col', type=str, default='split')
    parser.add_argument('--perturb_parse_mode', type=str, default='raw', choices=['raw', 'single_gene_suffix_clean', 'double_gene_parse'])
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--accum_steps', type=int, default=1)

    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--target_mode', type=str, default='delta', choices=['target', 'delta'])
    parser.add_argument('--perturb_dim', type=int, default=200)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 512, 512])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dose_dim', type=int, default=32)
    parser.add_argument('--time_dim', type=int, default=128)
    parser.add_argument('--val_sample_batches', type=int, default=5)
    parser.add_argument('--sample_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--cond_dropout', type=float, default=0.0)
    parser.add_argument('--timestep_sampler', type=str, default='uniform', choices=['uniform', 'loss-second-moment'])
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--save_every_epoch', action='store_true')
    parser.add_argument('--keep_last_n', type=int, default=3)
    parser.add_argument('--early_stop_metric', type=str, default='composite', choices=['composite', 'delta_pearson', 'top20_pearson'])
    parser.add_argument('--score_w_delta', type=float, default=1.0)
    parser.add_argument('--score_w_top20p', type=float, default=0.2)
    parser.add_argument('--score_w_top20mse', type=float, default=0.02)
    parser.add_argument('--lambda_topde', type=float, default=0.0)
    parser.add_argument('--lambda_delta_corr', type=float, default=0.0)
    parser.add_argument('--lambda_centroid', type=float, default=0.0)
    parser.add_argument('--topde_k', type=int, default=50)

    parser.add_argument('--atac_key', type=str, default=None)
    parser.add_argument('--atac_bank_path', type=str, default=None)
    parser.add_argument('--background_key', type=str, default='cell_context')
    parser.add_argument('--control_match_mode', type=str, default='random', choices=['random', 'atac_knn'])
    parser.add_argument('--control_match_k', type=int, default=32)
    parser.add_argument('--control_match_scope', type=str, default='global', choices=['global', 'cell_line'])
    parser.add_argument('--control_prototype_mode', type=str, default='topk_weighted', choices=['single', 'topk_mean', 'topk_weighted'])
    parser.add_argument('--control_prototype_temp', type=float, default=1.0)

    return parser.parse_args()


def apply_preset(args):
    if args.preset == 'none':
        return args

    base_defaults = {
        'split_strategy': 'perturbation',
        'split_col': 'split',
        'perturb_parse_mode': 'raw',
        'batch_size': 512,
        'epochs': 50,
        'target_mode': 'delta',
        'timesteps': 1000,
        'sample_steps': 50,
        'timestep_sampler': 'uniform',
        'cond_dropout': 0.0,
        'val_sample_batches': 5,
        'early_stop_metric': 'composite',
        'score_w_delta': 1.0,
        'score_w_top20p': 0.2,
        'score_w_top20mse': 0.02,
        'lambda_topde': 0.0,
        'lambda_delta_corr': 0.0,
        'lambda_centroid': 0.0,
        'topde_k': 50,
        'control_match_mode': 'random',
        'control_match_k': 32,
        'control_match_scope': 'global',
        'control_prototype_mode': 'topk_weighted',
        'control_prototype_temp': 1.0,
    }

    preset_updates = {
        'vnext': {
            'split_strategy': 'custom',
            'timestep_sampler': 'loss-second-moment',
            'cond_dropout': 0.1,
            'val_sample_batches': 0,
            'control_match_mode': 'atac_knn',
            'control_match_k': 16,
            'control_match_scope': 'global',
            'control_prototype_mode': 'topk_weighted',
            'control_prototype_temp': 1.0,
            'lambda_topde': 0.5,
            'lambda_delta_corr': 0.2,
            'lambda_centroid': 0.2,
        },
        'smoke': {
            'split_strategy': 'custom',
            'epochs': 3,
            'timesteps': 200,
            'sample_steps': 20,
            'batch_size': 64,
            'timestep_sampler': 'uniform',
            'cond_dropout': 0.1,
            'val_sample_batches': 0,
            'control_match_mode': 'atac_knn',
            'control_match_k': 8,
            'control_match_scope': 'global',
            'control_prototype_mode': 'topk_weighted',
            'lambda_topde': 0.2,
            'lambda_delta_corr': 0.1,
            'lambda_centroid': 0.1,
        },
    }[args.preset]

    for k, v in preset_updates.items():
        if hasattr(args, k) and getattr(args, k) == base_defaults.get(k, None):
            setattr(args, k, v)

    return args


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


def train():
    args = apply_preset(get_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_run_header(args, device)

    processor = DataProcessor(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        split_strategy=args.split_strategy,
        split_col=args.split_col,
        perturb_parse_mode=args.perturb_parse_mode,
        atac_key=args.atac_key,
        atac_bank_path=args.atac_bank_path,
        background_key=args.background_key,
    )
    n_genes, n_perts, _ = processor.load_data()
    train_loader, val_loader, test_loader = processor.prepare_loaders(
        batch_size=args.batch_size,
        rna_noise=0.0,
        atac_key=args.atac_key,
        atac_bank_path=args.atac_bank_path,
        background_key=args.background_key,
        control_match_mode=args.control_match_mode,
        control_match_k=args.control_match_k,
        control_match_scope=args.control_match_scope,
        control_prototype_mode=args.control_prototype_mode,
        control_prototype_temp=args.control_prototype_temp,
    )

    pretrained_weights = None
    if args.pretrained_emb:
        loader = GeneEmbeddingLoader(args.pretrained_emb, processor.id_to_perturb)
        pretrained_weights = loader.load_weights()

    atac_dim = processor.atac_dim if getattr(processor, 'atac_features', None) is not None else 0

    model = PerturbationDiffusionPredictor(
        n_genes=n_genes,
        n_perturbations=n_perts,
        pretrained_weights=pretrained_weights,
        perturb_dim=args.perturb_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        timesteps=args.timesteps,
        target_mode=args.target_mode,
        dose_dim=args.dose_dim,
        time_dim=args.time_dim,
        drug_dim=(processor.drug_embeddings.shape[1] if processor.drug_embeddings is not None else 0),
        use_atac=(processor.atac_features is not None),
        atac_dim=atac_dim,
        cond_dropout=args.cond_dropout,
        n_perturb_genes=len(getattr(processor, 'perturb_gene_vocab', []) or []),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)

    if args.timestep_sampler == 'loss-second-moment':
        timestep_sampler = LossSecondMomentResampler(args.timesteps)
    else:
        timestep_sampler = UniformTimestepSampler(args.timesteps)

    drug_embeddings = processor.drug_embeddings.to(device) if processor.drug_embeddings is not None else None

    os.makedirs(args.save_dir, exist_ok=True)
    best_score = -float('inf')
    best_top20_p = -float('inf')
    best_delta_p = -float('inf')
    best_top20_mse = float('inf')
    early_stopper = EarlyStopping(patience=args.patience)
    start_epoch = 0

    if args.resume_path is not None:
        print(f"↺ Resume from: {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] is not None:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'ema_state_dict' in ckpt and ckpt['ema_state_dict'] is not None:
            ema.shadow = {k: v.to(device) for k, v in ckpt['ema_state_dict'].items()}
        best_score = ckpt.get('best_score', best_score)
        best_top20_p = ckpt.get('best_top20_p', best_top20_p)
        best_delta_p = ckpt.get('best_delta_p', best_delta_p)
        best_top20_mse = ckpt.get('best_top20_mse', best_top20_mse)
        start_epoch = ckpt.get('epoch', -1) + 1

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"E{epoch+1:03d}/{args.epochs:03d}", leave=False)

        for i, batch in enumerate(pbar):
            ctrl_rna = batch['rna_control'].to(device)
            target_rna = batch['rna_target'].to(device)
            perturb = batch['perturb'].to(device)
            perturb_type = batch['perturb_type'].to(device) if 'perturb_type' in batch else None
            perturb_gene_a = batch['perturb_gene_a'].to(device) if 'perturb_gene_a' in batch else None
            perturb_gene_b = batch['perturb_gene_b'].to(device) if 'perturb_gene_b' in batch else None
            has_second_gene = batch['has_second_gene'].to(device) if 'has_second_gene' in batch else None
            dose = batch['dose'].to(device) if 'dose' in batch else None
            atac_feat = batch['atac_feat'].to(device) if 'atac_feat' in batch else None
            drug_feat = drug_embeddings[perturb] if drug_embeddings is not None else None
            t, weights = timestep_sampler.sample(ctrl_rna.shape[0], device)

            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                use_aux = (args.lambda_topde > 0) or (args.lambda_delta_corr > 0) or (args.lambda_centroid > 0)
                if use_aux:
                    diff_loss, details = model(
                        ctrl_rna,
                        perturb,
                        target_rna=target_rna,
                        dose=dose,
                        atac_feat=atac_feat,
                        drug_feat=drug_feat,
                        t=t,
                        weights=weights,
                        return_details=True,
                        perturb_type=perturb_type,
                        perturb_gene_a=perturb_gene_a,
                        perturb_gene_b=perturb_gene_b,
                        has_second_gene=has_second_gene,
                    )
                    pred_target = details['pred_target']
                    true_target = details['target_target']
                    delta_pred = pred_target - ctrl_rna
                    delta_true = true_target - ctrl_rna

                    aux_loss = torch.tensor(0.0, device=ctrl_rna.device)
                    if args.lambda_topde > 0:
                        k = min(args.topde_k, delta_true.shape[1])
                        top_idx = torch.topk(delta_true.abs(), k=k, dim=1).indices
                        top_pred = torch.gather(delta_pred, 1, top_idx)
                        top_true = torch.gather(delta_true, 1, top_idx)
                        aux_loss = aux_loss + args.lambda_topde * torch.mean((top_pred - top_true) ** 2)

                    if args.lambda_delta_corr > 0:
                        cos = torch.nn.functional.cosine_similarity(delta_pred, delta_true, dim=1)
                        aux_loss = aux_loss + args.lambda_delta_corr * torch.mean(1.0 - cos)

                    if args.lambda_centroid > 0:
                        centroid_loss = torch.tensor(0.0, device=ctrl_rna.device)
                        uniq = torch.unique(perturb)
                        counted = 0
                        for pid in uniq:
                            mask = (perturb == pid)
                            if mask.sum() < 2:
                                continue
                            dpm = delta_pred[mask].mean(dim=0)
                            dtm = delta_true[mask].mean(dim=0)
                            centroid_loss = centroid_loss + torch.mean((dpm - dtm) ** 2)
                            counted += 1
                        if counted > 0:
                            centroid_loss = centroid_loss / counted
                        aux_loss = aux_loss + args.lambda_centroid * centroid_loss

                    loss = diff_loss + aux_loss
                else:
                    loss = model(
                        ctrl_rna,
                        perturb,
                        target_rna=target_rna,
                        dose=dose,
                        atac_feat=atac_feat,
                        drug_feat=drug_feat,
                        t=t,
                        weights=weights,
                        perturb_type=perturb_type,
                        perturb_gene_a=perturb_gene_a,
                        perturb_gene_b=perturb_gene_b,
                        has_second_gene=has_second_gene,
                    )

            scaler.scale(loss / args.accum_steps).backward()

            if (i + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            train_loss += float(loss.item())
            timestep_sampler.update_with_losses(t, torch.full_like(t, float(loss.detach().item()), dtype=torch.float32))
            if i % 200 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        if len(train_loader) % args.accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)

        avg_train_loss = train_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ctrl = batch['rna_control'].to(device)
                target = batch['rna_target'].to(device)
                perturb = batch['perturb'].to(device)
                perturb_type = batch['perturb_type'].to(device) if 'perturb_type' in batch else None
                perturb_gene_a = batch['perturb_gene_a'].to(device) if 'perturb_gene_a' in batch else None
                perturb_gene_b = batch['perturb_gene_b'].to(device) if 'perturb_gene_b' in batch else None
                has_second_gene = batch['has_second_gene'].to(device) if 'has_second_gene' in batch else None
                dose = batch['dose'].to(device) if 'dose' in batch else None
                atac_feat = batch['atac_feat'].to(device) if 'atac_feat' in batch else None
                drug_feat = drug_embeddings[perturb] if drug_embeddings is not None else None
                t, weights = timestep_sampler.sample(ctrl.shape[0], device)
                loss = model(
                    ctrl,
                    perturb,
                    target,
                    dose=dose,
                    atac_feat=atac_feat,
                    drug_feat=drug_feat,
                    t=t,
                    weights=weights,
                    perturb_type=perturb_type,
                    perturb_gene_a=perturb_gene_a,
                    perturb_gene_b=perturb_gene_b,
                    has_second_gene=has_second_gene,
                )
                val_loss += float(loss.item())

        val_metrics = []
        ema.apply_shadow(model)
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if args.val_sample_batches > 0 and i >= args.val_sample_batches:
                    break
                ctrl = batch['rna_control'].to(device)
                target = batch['rna_target'].to(device)
                perturb = batch['perturb'].to(device)
                perturb_type = batch['perturb_type'].to(device) if 'perturb_type' in batch else None
                perturb_gene_a = batch['perturb_gene_a'].to(device) if 'perturb_gene_a' in batch else None
                perturb_gene_b = batch['perturb_gene_b'].to(device) if 'perturb_gene_b' in batch else None
                has_second_gene = batch['has_second_gene'].to(device) if 'has_second_gene' in batch else None
                dose = batch['dose'].to(device) if 'dose' in batch else None
                atac_feat = batch['atac_feat'].to(device) if 'atac_feat' in batch else None
                drug_feat = drug_embeddings[perturb] if drug_embeddings is not None else None

                pred = model.predict_single(
                    rna_control=ctrl,
                    perturb=perturb,
                    dose=dose,
                    atac_feat=atac_feat,
                    drug_feat=drug_feat,
                    sample_steps=args.sample_steps,
                    guidance_scale=args.guidance_scale,
                    perturb_type=perturb_type,
                    perturb_gene_a=perturb_gene_a,
                    perturb_gene_b=perturb_gene_b,
                    has_second_gene=has_second_gene,
                )
                val_metrics.append(calculate_metrics(pred, target, ctrl))
        ema.restore(model)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        final_m = {k: float(np.mean([m[k] for m in val_metrics])) for k in val_metrics[0].keys()} if val_metrics else {}

        print_epoch_summary(epoch + 1, args.epochs, avg_train_loss, avg_val_loss, final_m)

        scheduler.step()
        top20_p = float(final_m.get('top20_pearson', 0.0))
        delta_p = float(final_m.get('delta_pearson', 0.0))
        top20_mse = float(final_m.get('top20_mse', 0.0))
        composite_score = (
            args.score_w_delta * delta_p
            + args.score_w_top20p * top20_p
            - args.score_w_top20mse * top20_mse
        )

        if args.early_stop_metric == 'delta_pearson':
            current_score = delta_p
        elif args.early_stop_metric == 'top20_pearson':
            current_score = top20_p
        else:
            current_score = composite_score

        ckpt = {
            'model_state_dict': model.state_dict(),
            'args': args,
            'n_genes': n_genes,
            'n_perts': n_perts,
            'perturb_categories': processor.perturb_categories,
            'atac_dim': atac_dim,
            'use_atac': bool(processor.atac_features is not None),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'ema_state_dict': ema.shadow,
            'best_score': best_score,
            'best_top20_p': best_top20_p,
            'best_delta_p': best_delta_p,
            'best_top20_mse': best_top20_mse,
            'composite_score': composite_score,
            'epoch': epoch,
        }

        if top20_p > best_top20_p:
            best_top20_p = top20_p
            ckpt['best_top20_p'] = best_top20_p
            torch.save(ckpt, os.path.join(args.save_dir, "best_model_top20p.pth"))
            print(f"  ↳ New best top20_p: {best_top20_p:.4f}  saved=best_model_top20p.pth")

        if delta_p > best_delta_p:
            best_delta_p = delta_p
            ckpt['best_delta_p'] = best_delta_p
            torch.save(ckpt, os.path.join(args.save_dir, "best_model_delta.pth"))
            print(f"  ↳ New best delta_p: {best_delta_p:.4f}  saved=best_model_delta.pth")

        if top20_mse < best_top20_mse:
            best_top20_mse = top20_mse
            ckpt['best_top20_mse'] = best_top20_mse
            torch.save(ckpt, os.path.join(args.save_dir, "best_model_mse.pth"))
            print(f"  ↳ New best top20_mse: {best_top20_mse:.4f}  saved=best_model_mse.pth")

        if current_score > best_score:
            best_score = current_score
            ckpt['best_score'] = best_score
            torch.save(ckpt, os.path.join(args.save_dir, "best_model_diff.pth"))
            print(f"  ↳ New best stop_metric({args.early_stop_metric})={best_score:.4f}  saved=best_model_diff.pth")

        torch.save(ckpt, os.path.join(args.save_dir, "latest.pth"))
        if args.save_every_epoch:
            torch.save(ckpt, os.path.join(args.save_dir, f"epoch_{epoch+1}.pth"))
            rotate_epoch_checkpoints(args.save_dir, args.keep_last_n)

        if early_stopper(current_score):
            print("  ↳ Early stopping triggered.")
            break

    print(f"\n✓ Training finished. Artifacts in: {args.save_dir}")


if __name__ == "__main__":
    train()
