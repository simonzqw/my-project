import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
import argparse
import os
from utils.data_processor import DataProcessor
from models.scerso_diffusion import PerturbationDiffusionPredictor

def get_args():
    parser = argparse.ArgumentParser(description='scERso V9-Diff Visualization')
    parser.add_argument('--data_path', type=str, required=True, help='Path to .h5ad file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model_diff.pth')
    parser.add_argument('--save_path', type=str, default='v9_diff_combo_report.png', help='Path to save the plot')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--split_strategy', type=str, default='perturbation', choices=['random', 'perturbation'])
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--combo_genes', type=str, nargs='+', help='Pair of genes for combinatorial prediction, e.g. "GTPBP4" "POLR3K"')
    return parser.parse_args()

def predict_combination_latent_sum(model, processor, gene_pair, cell_line_idx, device):
    """
    双基因预测 (Latent Space Arithmetic 模式)
    逻辑: z_combo = z_A + z_B (在 Latent Space 中叠加特征)
    """
    g1_name, g2_name = gene_pair
    
    if g1_name not in processor.perturb_map or g2_name not in processor.perturb_map:
        print(f"!!! 警告: 基因 {g1_name} 或 {g2_name} 不在扰动列表中。")
        return None, None, None

    g1_id = processor.perturb_map[g1_name]
    g2_id = processor.perturb_map[g2_name]
    
    # 构造输入
    if cell_line_idx not in processor.cell_line_baselines:
        print(f"!!! 警告: 细胞系 ID {cell_line_idx} 没有 Baseline 数据。")
        return None, None, None
        
    rna_control = processor.cell_line_baselines[cell_line_idx].unsqueeze(0).to(device) # [1, 2000]
    cell_line = torch.tensor([cell_line_idx], dtype=torch.long).to(device)
    
    # 1. 获取单基因 A 的 Latent
    p1 = torch.tensor([g1_id], dtype=torch.long).to(device)
    z_a = model.get_latent(rna_control, p1, cell_line) # [1, D]
    
    # 2. 获取单基因 B 的 Latent
    p2 = torch.tensor([g2_id], dtype=torch.long).to(device)
    z_b = model.get_latent(rna_control, p2, cell_line) # [1, D]
    
    # 3. Latent Space 融合 (简单的加法，或加权)
    z_combo = z_a + z_b 
    
    # 4. 基于融合的 Latent 生成预测结果
    # 注意：这里我们传入 None 作为 perturb，因为我们提供了 custom_latent
    pred_ab = model.sample(rna_control, perturb=None, cell_line=cell_line, custom_latent=z_combo)
    
    # 5. 同时生成单基因预测结果用于对比
    pred_a = model.sample(rna_control, perturb=None, cell_line=cell_line, custom_latent=z_a)
    pred_b = model.sample(rna_control, perturb=None, cell_line=cell_line, custom_latent=z_b)
    
    return pred_ab.cpu().numpy().flatten(), pred_a.cpu().numpy().flatten(), pred_b.cpu().numpy().flatten()

def visualize():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> scERso V9-Diff 组合预测启动 (Latent Fusion Mode)")
    
    # 1. 加载 Checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_args = checkpoint['args']
    state_dict = checkpoint['model_state_dict']
    
    # 2. 加载数据
    processor = DataProcessor(
        args.data_path, 
        test_size=args.test_size, 
        val_size=args.val_size, 
        split_strategy=args.split_strategy
    )
    n_genes, n_perts, n_cls = processor.load_data()
    gene_names = processor.adata.var_names.tolist()
    
    # 3. 初始化 Diffusion 模型
    model = PerturbationDiffusionPredictor(
        n_genes=n_genes,
        n_perturbations=n_perts,
        n_cell_lines=n_cls,
        perturb_dim=state_dict['perturb_embedding.weight'].shape[1],
        cell_line_dim=state_dict['cell_line_embedding.weight'].shape[1],
        hidden_dims=[512, 1024, 2048],
        dropout=getattr(model_args, 'dropout', 0.1),
        timesteps=getattr(model_args, 'timesteps', 1000)
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # --- 双基因组合预测 ---
    if args.combo_genes and len(args.combo_genes) == 2:
        print(f">>> 正在预测组合: {args.combo_genes[0]} + {args.combo_genes[1]}")
        
        target_cell_line = 0 # 演示用
        
        # 使用 Latent Fusion 逻辑预测
        pred_ab, pred_a, pred_b = predict_combination_latent_sum(
            model, processor, args.combo_genes, target_cell_line, device
        )
        
        if pred_ab is not None:
            baseline = processor.cell_line_baselines[target_cell_line].numpy()
            
            # 计算 Delta
            delta_ab = pred_ab - baseline
            delta_a = pred_a - baseline
            delta_b = pred_b - baseline
            
            # 理论加性模型
            delta_additive = delta_a + delta_b
            
            # 绘图逻辑 (保持不变)
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            
            # 散点图
            sns.scatterplot(x=delta_additive, y=delta_ab, ax=axes[0], alpha=0.5, color='purple')
            min_val = min(delta_additive.min(), delta_ab.min())
            max_val = max(delta_additive.max(), delta_ab.max())
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
            axes[0].set_title(f"Combinatorial Effect (Latent Sum): {args.combo_genes[0]}+{args.combo_genes[1]}", fontsize=14)
            axes[0].set_xlabel("Additive Expectation (Delta A + Delta B)")
            axes[0].set_ylabel("Model Prediction (Latent Fusion)")
            
            # 柱状图
            diff = np.abs(delta_ab - delta_additive)
            top_idx = np.argsort(diff)[-15:][::-1]
            
            plot_data = []
            for idx in top_idx:
                g = gene_names[idx]
                plot_data.append({'Gene': g, 'Value': delta_additive[idx], 'Type': 'Additive (A+B)'})
                plot_data.append({'Gene': g, 'Value': delta_ab[idx], 'Type': 'Model Prediction'})
            
            df_bar = pd.DataFrame(plot_data)
            sns.barplot(data=df_bar, x='Gene', y='Value', hue='Type', ax=axes[1], palette={'Additive (A+B)': 'gray', 'Model Prediction': 'crimson'})
            axes[1].set_title("Top 15 Non-Additive Responsive Genes", fontsize=14)
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
            axes[1].set_ylabel("Expression Delta")
            
            plt.tight_layout()
            plt.savefig(args.save_path, dpi=200)
            print(f">>> 组合预测报告已生成: {args.save_path}")

if __name__ == "__main__":
    visualize()
