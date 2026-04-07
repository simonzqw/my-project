import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import glob
import re
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils.data_processor import DataProcessor
from models.scerso_diffusion import PerturbationDiffusionPredictor
from utils.emb_loader import GeneEmbeddingLoader
from utils.diffusion_schedule import UniformTimestepSampler, LossSecondMomentResampler

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        # 监控 Top50 Pearson，越高越好
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
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters() if p.requires_grad
        }
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
    parser = argparse.ArgumentParser(description='scERso V9-Diff: Diffusion Training')
    parser.add_argument('--data_path', type=str, required=True, help='Path to .h5ad file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_diff', help='Directory to save models')
    parser.add_argument('--pretrained_emb', type=str, default=None, help='Path to Gene2Vec embedding file')
    
    # 数据划分参数
    parser.add_argument('--split_strategy', type=str, default='perturbation', choices=['random', 'perturbation'])
    parser.add_argument('--test_size', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set ratio')
    
    # 训练超参
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--accum_steps', type=int, default=1)
    
    # Diffusion & Model 参数
    parser.add_argument('--timesteps', type=int, default=1000, help='Diffusion steps')
    parser.add_argument('--perturb_dim', type=int, default=200)
    parser.add_argument('--cell_line_dim', type=int, default=32)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 512, 512]) # 默认参数同步简化
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dose_dim', type=int, default=32)
    parser.add_argument('--time_dim', type=int, default=128)
    parser.add_argument('--val_sample_batches', type=int, default=5, help='Number of val batches for sampling metrics')
    parser.add_argument('--sample_steps', type=int, default=50, help='DDIM steps used during validation/test sampling')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Classifier-free guidance scale for sampling')
    parser.add_argument('--cond_dropout', type=float, default=0.0, help='Condition dropout ratio for classifier-free guidance training')
    parser.add_argument('--timestep_sampler', type=str, default='uniform', choices=['uniform', 'loss-second-moment'])
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--save_every_epoch', action='store_true')
    parser.add_argument('--keep_last_n', type=int, default=3)
    parser.add_argument('--atac_key', type=str, default=None)
    parser.add_argument('--atac_bank_path', type=str, default=None)
    
    return parser.parse_args()

def calculate_metrics(pred, target, ctrl, n=50):
    """
    Diffusion 模型的评估指标：
    因为 Diffusion 生成的是分布，我们取采样结果的均值作为预测值。
    """
    global_rs, delta_rs, top_n_rs, top_n_recalls = [], [], [], []
    
    # 转换为 numpy
    if isinstance(pred, torch.Tensor): pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor): target = target.cpu().numpy()
    if isinstance(ctrl, torch.Tensor): ctrl = ctrl.cpu().numpy()

    for i in range(pred.shape[0]):
        p, t, c = pred[i], target[i], ctrl[i]
        d_p = p - c
        d_t = t - c
        
        # 1. Global Pearson
        r_g, _ = pearsonr(p, t)
        if not np.isnan(r_g): global_rs.append(r_g)
        
        # 2. Delta Pearson
        if np.std(d_p) > 1e-6 and np.std(d_t) > 1e-6:
            r_d, _ = pearsonr(d_p, d_t)
            if not np.isnan(r_d): delta_rs.append(r_d)
            
            # 3. TopN Pearson
            top_indices = np.argsort(np.abs(d_t))[-n:] 
            r_tn, _ = pearsonr(d_p[top_indices], d_t[top_indices])
            if not np.isnan(r_tn): top_n_rs.append(r_tn)
            
            # 4. TopN Recall
            pred_top_indices = np.argsort(np.abs(d_p))[-n:]
            recall = len(set(top_indices) & set(pred_top_indices)) / n
            top_n_recalls.append(recall)
            
    return {
        'global_r': np.mean(global_rs) if global_rs else 0.0,
        'delta_r': np.mean(delta_rs) if delta_rs else 0.0,
        'top_n_r': np.mean(top_n_rs) if top_n_rs else 0.0,
        'top_n_recall': np.mean(top_n_recalls) if top_n_recalls else 0.0
    }

def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> scERso V9-Diff 启动 | 任务: 扩散生成预测 | 策略: {args.split_strategy}")

    # 1. 数据准备
    processor = DataProcessor(
        args.data_path, 
        test_size=args.test_size, 
        val_size=args.val_size, 
        split_strategy=args.split_strategy,
    )
    n_genes, n_perts, n_cell_lines = processor.load_data()
    train_loader, val_loader, test_loader = processor.prepare_loaders(
        batch_size=args.batch_size,
        rna_noise=0.0,
        atac_key=args.atac_key,
        atac_bank_path=args.atac_bank_path,
    ) # Diffusion 自带加噪，无需额外数据增强
    
    # 2. 加载预训练向量
    pretrained_weights = None
    if args.pretrained_emb:
        loader = GeneEmbeddingLoader(args.pretrained_emb, processor.id_to_perturb)
        pretrained_weights = loader.load_weights()

    # 3. 初始化 Diffusion 模型
    model = PerturbationDiffusionPredictor(
        n_genes=n_genes,
        n_perturbations=n_perts,
        n_cell_lines=n_cell_lines,
        pretrained_weights=pretrained_weights,
        perturb_dim=args.perturb_dim,
        cell_line_dim=args.cell_line_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        timesteps=args.timesteps,
        dose_dim=args.dose_dim,
        time_dim=args.time_dim,
        drug_dim=(processor.drug_embeddings.shape[1] if processor.drug_embeddings is not None else 0),
        use_atac=(processor.atac_features is not None),
        cond_dropout=args.cond_dropout,
    ).to(device)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)
    if args.timestep_sampler == 'loss-second-moment':
        timestep_sampler = LossSecondMomentResampler(args.timesteps)
    else:
        timestep_sampler = UniformTimestepSampler(args.timesteps)
    drug_embeddings = processor.drug_embeddings.to(device) if processor.drug_embeddings is not None else None
    
    # 4. 训练循环
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    best_score = -float('inf') 
    early_stopper = EarlyStopping(patience=args.patience)
    start_epoch = 0

    if args.resume_path is not None:
        print(f">>> 检测到断点续训: {args.resume_path}")
        resume_ckpt = torch.load(args.resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt['model_state_dict'])
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in resume_ckpt and resume_ckpt['scaler_state_dict'] is not None:
            scaler.load_state_dict(resume_ckpt['scaler_state_dict'])
        if 'ema_state_dict' in resume_ckpt and resume_ckpt['ema_state_dict'] is not None:
            ema.shadow = {k: v.to(device) for k, v in resume_ckpt['ema_state_dict'].items()}
        best_score = resume_ckpt.get('best_score', best_score)
        start_epoch = resume_ckpt.get('epoch', -1) + 1
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        optimizer.zero_grad()
        for i, batch in enumerate(pbar):
            ctrl_rna = batch['rna_control'].to(device)
            target_rna = batch['rna_target'].to(device)
            perturb = batch['perturb'].to(device)
            cell_line = batch['cell_line'].to(device)
            dose = batch['dose'].to(device) if 'dose' in batch else None
            atac_feat = batch['atac_feat'].to(device) if 'atac_feat' in batch else None
            drug_feat = drug_embeddings[perturb] if drug_embeddings is not None else None
            t, weights = timestep_sampler.sample(ctrl_rna.shape[0], device)
            
            # Forward pass 计算 Diffusion Loss (MSE of Noise Prediction)
            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                loss = model(
                    ctrl_rna, perturb, cell_line, target_rna, dose=dose,
                    atac_feat=atac_feat, drug_feat=drug_feat, t=t, weights=weights
                )
            
            scaler.scale(loss / args.accum_steps).backward()
            
            if (i + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
            
            train_loss += loss.item()
            timestep_sampler.update_with_losses(t, torch.full_like(t, loss.detach().item(), dtype=torch.float32))
            if i % 100 == 0:
                pbar.set_postfix({'diff_loss': f"{loss.item():.6f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证集评估 (需要采样，比较慢，建议每隔几轮做一次全量采样，或者采样部分 batch)
        # 这里为了效率，每轮只评估验证集的前 5 个 batch
        model.eval()
        val_metrics = []
        val_loss = 0
        
        # 简单计算 Loss (不需要采样)
        with torch.no_grad():
            for batch in val_loader:
                ctrl = batch['rna_control'].to(device)
                target = batch['rna_target'].to(device)
                perturb = batch['perturb'].to(device)
                cell_line = batch['cell_line'].to(device)
                dose = batch['dose'].to(device) if 'dose' in batch else None
                atac_feat = batch['atac_feat'].to(device) if 'atac_feat' in batch else None
                drug_feat = drug_embeddings[perturb] if drug_embeddings is not None else None
                t, weights = timestep_sampler.sample(ctrl.shape[0], device)
                loss = model(ctrl, perturb, cell_line, target, dose=dose, atac_feat=atac_feat, drug_feat=drug_feat, t=t, weights=weights)
                val_loss += loss.item()
        
        # 采样评估 (只取第一个 batch 做快速指标参考)
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= args.val_sample_batches:
                    break
                ctrl = batch['rna_control'].to(device)
                target = batch['rna_target'].to(device)
                perturb = batch['perturb'].to(device)
                cell_line = batch['cell_line'].to(device)
                dose = batch['dose'].to(device) if 'dose' in batch else None
                atac_feat = batch['atac_feat'].to(device) if 'atac_feat' in batch else None
                drug_feat = drug_embeddings[perturb] if drug_embeddings is not None else None

                # Diffusion 采样生成预测值
                ema.apply_shadow(model)
                generated_rna = model.sample(
                    ctrl, perturb, cell_line, dose=dose, atac_feat=atac_feat, drug_feat=drug_feat,
                    sample_steps=args.sample_steps, guidance_scale=args.guidance_scale
                )
                ema.restore(model)

                # 计算指标
                batch_m = calculate_metrics(generated_rna, target, ctrl)
                val_metrics.append(batch_m)

        avg_val_loss = val_loss / len(val_loader)
        if val_metrics:
            final_m = {
                'global_r': float(np.mean([m['global_r'] for m in val_metrics])),
                'delta_r': float(np.mean([m['delta_r'] for m in val_metrics])),
                'top_n_r': float(np.mean([m['top_n_r'] for m in val_metrics])),
                'top_n_recall': float(np.mean([m['top_n_recall'] for m in val_metrics]))
            }
        else:
            final_m = {'global_r': 0.0, 'delta_r': 0.0, 'top_n_r': 0.0, 'top_n_recall': 0.0}

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   >> Val Metrics (Mean of {min(args.val_sample_batches, len(val_loader))} batches): Top50 R: {final_m['top_n_r']:.4f} | Delta R: {final_m['delta_r']:.4f}")
        
        scheduler.step()
        
        # 保存最佳模型 (基于 Top50 Pearson)
        current_score = final_m['top_n_r']
        if current_score > best_score:
            best_score = current_score
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': args,
                'n_genes': n_genes,
                'n_perts': n_perts,
                'n_cell_lines': n_cell_lines,
                'baselines': processor.cell_line_baselines
                ,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'ema_state_dict': ema.shadow,
                'best_score': best_score,
                'epoch': epoch,
            }, os.path.join(args.save_dir, "best_model_diff.pth"))
            print(f"*** 发现更优模型 (Top50 R: {best_score:.4f}), 已保存")
        latest_ckpt = {
            'model_state_dict': model.state_dict(),
            'args': args,
            'n_genes': n_genes,
            'n_perts': n_perts,
            'n_cell_lines': n_cell_lines,
            'baselines': processor.cell_line_baselines,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'ema_state_dict': ema.shadow,
            'best_score': best_score,
            'epoch': epoch,
        }
        torch.save(latest_ckpt, os.path.join(args.save_dir, "latest.pth"))
        if args.save_every_epoch:
            torch.save(latest_ckpt, os.path.join(args.save_dir, f"epoch_{epoch+1}.pth"))
            rotate_epoch_checkpoints(args.save_dir, args.keep_last_n)
        
        if early_stopper(current_score):
            print(f"!!! 早停触发")
            break

    print("\n>>> 训练结束")

if __name__ == "__main__":
    train()
