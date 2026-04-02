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

from utils.data_processor import DataProcessor
from models.reasoning_mlp import PerturbationPredictor
from utils.emb_loader import GeneEmbeddingLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2.0):
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha # GlobalMSE 权重
        self.gamma = gamma # DeltaMSE 放大系数

    def forward(self, pred, target, ctrl, is_control=None):
        delta_true = target - ctrl
        delta_pred = pred - ctrl
        
        # 1. DeltaMSE (带加权)
        # 根据真实变化量的绝对值进行加权，强迫模型拟合剧烈变化的基因
        weights = 1.0 + self.gamma * torch.abs(delta_true)
        weights = weights / weights.mean() # 归一化权重
        delta_loss = torch.mean(weights * (delta_pred - delta_true)**2)
        
        # 2. GlobalMSE (基础保底)
        global_loss = torch.mean((pred - target)**2)
        
        # 3. Control 正则化 (如果输入是 control 样本，则 delta 必须为 0)
        # 优先只在 control 样本上施加约束，避免过度抑制真实扰动信号
        if is_control is not None and torch.any(is_control):
            ctrl_reg = torch.mean((delta_pred[is_control])**2)
        else:
            ctrl_reg = torch.mean(delta_pred**2)
        
        return self.alpha * global_loss + (1 - self.alpha) * delta_loss + 0.01 * ctrl_reg

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        # 注意：现在监控的是 Top50 Pearson，越高越好
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
    """Simple EMA tracker for model parameters."""
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
    parser = argparse.ArgumentParser(description='scERso V7: Generative Perturbation Predictor')
    parser.add_argument('--data_path', type=str, required=True, help='Path to .h5ad file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--split_strategy', type=str, default='perturbation', choices=['random', 'perturbation'])
    parser.add_argument('--pretrained_emb', type=str, default=None, help='Path to Gene2Vec embedding file')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--freeze_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--accum_steps', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training on CUDA')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to resume checkpoint (latest.pth)')
    parser.add_argument('--keep_last_n', type=int, default=5, help='Keep only latest N epoch checkpoints')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay for parameter averaging')
    parser.add_argument('--eval_ema', action='store_true', help='Use EMA weights for final test evaluation')
    parser.add_argument('--test_size', type=float, default=0.1, help='Test set ratio (e.g. 0.1 for 10%)')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set ratio (e.g. 0.1 for 10%)')

    # 模型架构参数
    parser.add_argument('--perturb_dim', type=int, default=200)
    parser.add_argument('--cell_line_dim', type=int, default=32)
    parser.add_argument('--drug_dim', type=int, default=2048, help='Morgan Fingerprint dimension')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 1024, 2048])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dim_ff', type=int, default=1024)
    parser.add_argument('--n_ctrl_tokens', type=int, default=8)
    return parser.parse_args()

def _safe_pearson(x, y):
    if np.std(x) <= 1e-8 or np.std(y) <= 1e-8:
        return np.nan
    r, _ = pearsonr(x, y)
    return r if not np.isnan(r) else np.nan

def calculate_metrics(pred, target, ctrl, de_eps=1e-3):
    """
    返回更完整的指标:
    - 全谱: all_mse / all_pearson
    - Delta: delta_pearson
    - TopK (K=10/20/50): mse / pearson / recall
    - all_de: 对所有非dropout(真实delta!=0)基因统计 mse / pearson
    """
    top_ks = [10, 20, 50]
    collector = {
        'all_mse': [],
        'all_pearson': [],
        'delta_pearson': [],
        'all_de_mse': [],
        'all_de_pearson': []
    }
    for k in top_ks:
        collector[f'top{k}_mse'] = []
        collector[f'top{k}_pearson'] = []
        collector[f'top{k}_recall'] = []

    for i in range(pred.shape[0]):
        p, t, c = pred[i], target[i], ctrl[i]
        d_p = p - c
        d_t = t - c

        collector['all_mse'].append(float(np.mean((p - t) ** 2)))
        r_all = _safe_pearson(p, t)
        if not np.isnan(r_all):
            collector['all_pearson'].append(r_all)

        r_delta = _safe_pearson(d_p, d_t)
        if not np.isnan(r_delta):
            collector['delta_pearson'].append(r_delta)

        non_dropout_mask = np.abs(d_t) > de_eps
        if np.any(non_dropout_mask):
            collector['all_de_mse'].append(float(np.mean((d_p[non_dropout_mask] - d_t[non_dropout_mask]) ** 2)))
            r_de = _safe_pearson(d_p[non_dropout_mask], d_t[non_dropout_mask])
            if not np.isnan(r_de):
                collector['all_de_pearson'].append(r_de)

        for k in top_ks:
            k_eff = min(k, len(d_t))
            top_true = np.argsort(np.abs(d_t))[-k_eff:]
            top_pred = np.argsort(np.abs(d_p))[-k_eff:]
            collector[f'top{k}_mse'].append(float(np.mean((d_p[top_true] - d_t[top_true]) ** 2)))
            r_top = _safe_pearson(d_p[top_true], d_t[top_true])
            if not np.isnan(r_top):
                collector[f'top{k}_pearson'].append(r_top)
            collector[f'top{k}_recall'].append(float(len(set(top_true) & set(top_pred)) / max(k_eff, 1)))

    # 聚合
    return {
        k: (float(np.mean(v)) if len(v) > 0 else 0.0)
        for k, v in collector.items()
    }
def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> scERso V7 启动 | 任务: 生成式扰动预测 | 策略: {args.split_strategy}")

    # 1. 数据准备
    processor = DataProcessor(
        args.data_path, 
        test_size=args.test_size, 
        val_size=args.val_size, 
        split_strategy=args.split_strategy
    )
    n_genes, n_perts, n_cell_lines = processor.load_data()
    if args.split_strategy == 'perturbation' and args.pretrained_emb is None and processor.drug_embeddings is None:
        raise ValueError(
            "当前是 unseen perturbation 划分，必须提供 side information（如 --pretrained_emb 或 SMILES 药物特征），"
            "否则 perturbation 只能依赖 ID embedding，泛化会明显受限。"
        )
    train_loader, val_loader, test_loader = processor.prepare_loaders(
        batch_size=args.batch_size,
        rna_noise=args.noise,
        num_workers=args.num_workers
    )
    
    # 2. 加载预训练向量
    pretrained_weights = None
    if args.pretrained_emb:
        loader = GeneEmbeddingLoader(args.pretrained_emb, processor.id_to_perturb)
        pretrained_weights = loader.load_weights()
    elif args.resume_path is not None and os.path.exists(args.resume_path):
        resume_meta = torch.load(args.resume_path, map_location='cpu', weights_only=False)
        resume_state = resume_meta.get('model_state_dict', {})
        if 'perturb_feature_bank' in resume_state:
            pretrained_weights = resume_state['perturb_feature_bank'].float()

    # 3. 初始化模型
    model = PerturbationPredictor(
        n_genes, n_perts, n_cell_lines, 
        pretrained_weights=pretrained_weights,
        perturb_dim=args.perturb_dim,
        cell_line_dim=args.cell_line_dim,
        drug_dim=args.drug_dim, # 新增药物维度参数
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        n_ctrl_tokens=args.n_ctrl_tokens
    ).to(device)

    # 损失函数: 加权 MSE (根据 Delta 绝对值加权，抑制基线红利)
    criterion = WeightedMSELoss() 
    
    # 参数分组优化
    perturb_params = []
    if model.use_semantic_perturb and model.perturb_encoder is not None:
        perturb_params = list(model.perturb_encoder.parameters())
    else:
        perturb_params = list(model.perturb_embedding.parameters())

    perturb_param_ids = {id(p) for p in perturb_params}
    other_params = [p for p in model.parameters() if id(p) not in perturb_param_ids]
    param_groups = [
        {'params': perturb_params, 'lr': args.lr * 0.1, 'name': 'perturb_branch'},
        {'params': other_params, 'lr': args.lr, 'name': 'other'}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))
    control_id = processor.perturb_map.get('control', None)
    drug_embeddings = processor.drug_embeddings.to(device) if processor.drug_embeddings is not None else None
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)

    # 4. 训练循环
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    best_score = -float('inf') # 监控 -Top20 MSE，越大越好（等价于 Top20 MSE 越小越好）
    early_stopper = EarlyStopping(patience=args.patience)
    warmup_epochs = 5
    start_epoch = 0

    if args.resume_path is not None:
        print(f">>> 检测到断点续训: {args.resume_path}")
        resume_ckpt = torch.load(args.resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt['model_state_dict'])
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in resume_ckpt:
            main_scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in resume_ckpt and resume_ckpt['scaler_state_dict'] is not None:
            scaler.load_state_dict(resume_ckpt['scaler_state_dict'])
        if 'ema_state_dict' in resume_ckpt and resume_ckpt['ema_state_dict'] is not None:
            ema.shadow = {k: v.to(device) for k, v in resume_ckpt['ema_state_dict'].items()}
        best_score = resume_ckpt.get('best_score', best_score)
        start_epoch = resume_ckpt.get('epoch', -1) + 1
        early_stopper.best_score = resume_ckpt.get('early_stopping_best_score', early_stopper.best_score)
        early_stopper.counter = resume_ckpt.get('early_stopping_counter', early_stopper.counter)
        print(f">>> 从 epoch {start_epoch} 继续训练")
    
    for epoch in range(start_epoch, args.epochs):
        # Warmup
        if epoch < warmup_epochs:
            curr_lr_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                base_lr = args.lr * 0.1 if param_group['name'] == 'embedding' else args.lr
                if param_group['name'] == 'perturb_branch':
                    base_lr = args.lr * 0.1
                param_group['lr'] = base_lr * curr_lr_factor

        # 冻结策略
        if args.pretrained_emb:
            is_frozen = epoch < args.freeze_epochs
            model.freeze_perturbation_embedding(is_frozen)
            if is_frozen: print(f">>> Epoch {epoch+1}: 扰动 Embedding 层已 冻结")
            else: print(f">>> Epoch {epoch+1}: 扰动 Embedding 层已 解冻")

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
            is_control = (perturb == control_id) if control_id is not None else None
            
            # 药物特征处理 (如果有)
            drug_feat = None
            if drug_embeddings is not None:
                # 根据 perturb index 获取对应的 drug feature
                drug_feat = drug_embeddings[perturb]
            
            # 传递 drug_feat 和 dose 到 forward
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                outputs = model(ctrl_rna, perturb, cell_line, drug_feat=drug_feat, dose=dose)
                
                # 使用加权损失
                loss = criterion(outputs, target_rna, ctrl_rna, is_control=is_control)
            
            scaler.scale(loss / args.accum_steps).backward()
            
            if (i + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)
                optimizer.zero_grad()
            
            train_loss += loss.item()
            if i % 100 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        if len(train_loader) % args.accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            optimizer.zero_grad()

        # 验证集评估
        model.eval()
        val_loss = 0
        all_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                ctrl = batch['rna_control'].to(device)
                target = batch['rna_target'].to(device)
                perturb = batch['perturb'].to(device)
                cell_line = batch['cell_line'].to(device)
                dose = batch['dose'].to(device) if 'dose' in batch else None
                is_control = (perturb == control_id) if control_id is not None else None
                
                # 药物特征处理 (验证集)
                drug_feat = None
                if drug_embeddings is not None:
                    drug_feat = drug_embeddings[perturb]
                
                outputs = model(ctrl, perturb, cell_line, drug_feat=drug_feat, dose=dose)
                
                loss = criterion(outputs, target, ctrl, is_control=is_control)
                val_loss += loss.item()
                
                # 计算多维指标
                batch_m = calculate_metrics(outputs.cpu().numpy(), target.cpu().numpy(), ctrl.cpu().numpy())
                all_metrics.append(batch_m)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        # 聚合所有 batch 的指标
        final_m = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()} if all_metrics else {}
        
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"All Pearson: {final_m.get('all_pearson', 0.0):.4f} | Delta Pearson: {final_m.get('delta_pearson', 0.0):.4f} | "
            f"Top20 MSE: {final_m.get('top20_mse', 0.0):.4f} | Top20 Pearson: {final_m.get('top20_pearson', 0.0):.4f} | "
            f"Top50 Recall: {final_m.get('top50_recall', 0.0):.4f}"
        )
        
        if epoch >= warmup_epochs:
            main_scheduler.step()
        
        # 核心：以 Top20 MSE 作为保存和早停依据（越小越好）
        current_score = -final_m.get('top20_mse', float('inf'))  # 越小越好 -> 取负号后越大越好
        if current_score > best_score:
            best_score = current_score
            torch.save({
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.shadow,
                'args': args,
                'n_genes': n_genes,
                'n_perts': n_perts,
                'n_cell_lines': n_cell_lines,
                'baselines': processor.cell_line_baselines
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"*** 发现更优模型 (Val Top20 DE MSE: {-best_score:.6f}), 已保存")

        # 每个 epoch 保存 checkpoint，并仅保留最近 N 个
        epoch_ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': main_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if (args.amp and device.type == "cuda") else None,
            'ema_state_dict': ema.shadow,
            'best_score': best_score,
            'early_stopping_best_score': early_stopper.best_score,
            'early_stopping_counter': early_stopper.counter,
            'args': args,
            'n_genes': n_genes,
            'n_perts': n_perts,
            'n_cell_lines': n_cell_lines,
            'baselines': processor.cell_line_baselines
        }
        torch.save(epoch_ckpt, os.path.join(args.save_dir, f"epoch_{epoch+1:03d}.pth"))
        torch.save(epoch_ckpt, os.path.join(args.save_dir, "latest.pth"))
        rotate_epoch_checkpoints(args.save_dir, args.keep_last_n)
        
        if early_stopper(current_score):
            print(f"!!! 早停触发 (核心基因拟合已达瓶颈)")
            break

    # 5. 最终测试集评估 (使用保存的最佳权重)
    print("\n" + "="*30)
    print(">>> 正在进行最终测试集 (Test Set) 评估...")
    best_ckpt = torch.load(os.path.join(args.save_dir, "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    if args.eval_ema and ('ema_state_dict' in best_ckpt) and (best_ckpt['ema_state_dict'] is not None):
        print(">>> 使用 EMA 权重进行最终测试评估")
        ema.shadow = {k: v.to(device) for k, v in best_ckpt['ema_state_dict'].items()}
        ema.apply_shadow(model)
    model.eval()
    
    test_metrics = []
    with torch.no_grad():
        for batch in test_loader:
            ctrl = batch['rna_control'].to(device)
            target = batch['rna_target'].to(device)
            perturb = batch['perturb'].to(device)
            cell_line = batch['cell_line'].to(device)
            dose = batch['dose'].to(device) if 'dose' in batch else None
            
            # 药物特征处理 (测试集)
            drug_feat = None
            if drug_embeddings is not None:
                drug_feat = drug_embeddings[perturb]
            
            outputs = model(ctrl, perturb, cell_line, drug_feat=drug_feat, dose=dose)
            m = calculate_metrics(outputs.cpu().numpy(), target.cpu().numpy(), ctrl.cpu().numpy())
            test_metrics.append(m)
    
    final_test_m = {k: np.mean([m[k] for m in test_metrics]) for k in test_metrics[0].keys()} if test_metrics else {}
    print(f"!!! 最终评估结果 (Test Set) !!!")
    print(f"All Pearson: {final_test_m.get('all_pearson', 0.0):.4f}")
    print(f"Delta Pearson: {final_test_m.get('delta_pearson', 0.0):.4f}")
    print(f"Top20 MSE: {final_test_m.get('top20_mse', 0.0):.4f}")
    print(f"Top20 Pearson: {final_test_m.get('top20_pearson', 0.0):.4f}")
    print(f"Top50 Recall: {final_test_m.get('top50_recall', 0.0):.4f}")
    print("="*30)
    if args.eval_ema and ('ema_state_dict' in best_ckpt) and (best_ckpt['ema_state_dict'] is not None):
        ema.restore(model)

if __name__ == "__main__":
    train()
