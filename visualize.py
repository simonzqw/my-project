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
from models.reasoning_mlp import PerturbationPredictor

def get_args():
    parser = argparse.ArgumentParser(description='scERso V9 Visualization: Professional Scouter Report')
    parser.add_argument('--data_path', type=str, required=True, help='Path to .h5ad file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--save_path', type=str, default='v9_professional_report.png', help='Path to save the plot')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--split_strategy', type=str, default='perturbation', choices=['random', 'perturbation'])
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--top_n', type=int, default=50, help='Number of top DE genes to consider for ROC')
    parser.add_argument('--heatmap_gene', type=str, default='POLR3K', help='Gene to show in Top-20 bar plot')
    return parser.parse_args()

def visualize():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 当前请求的可视化基因: {args.heatmap_gene}")
    
    # 1. 加载 Checkpoint
    print(f">>> 正在加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_args = checkpoint['args']
    state_dict = checkpoint['model_state_dict']
    
    # 2. 加载数据
    print(f">>> 正在加载数据: {args.data_path}")
    processor = DataProcessor(
        args.data_path, 
        test_size=args.test_size, 
        val_size=args.val_size, 
        split_strategy=args.split_strategy
    )
    n_genes, n_perts, n_cls = processor.load_data()
    _, _, test_loader = processor.prepare_loaders(batch_size=args.batch_size)
    gene_names = processor.adata.var_names.tolist()
    
    # 3. 还原模型 (V9 架构)
    model = PerturbationPredictor(
        n_genes=n_genes,
        n_perturbations=state_dict['perturb_embedding.weight'].shape[0],
        n_cell_lines=state_dict['cell_line_embedding.weight'].shape[0],
        perturb_dim=state_dict['perturb_embedding.weight'].shape[1],
        cell_line_dim=state_dict['cell_line_embedding.weight'].shape[1],
        drug_dim=getattr(model_args, 'drug_dim', 2048),
        hidden_dims=[512, 1024, 2048],
        dropout=getattr(model_args, 'dropout', 0.2)
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 4. 执行推理并按扰动分组
    results_by_pert = {} # {pert_name: {preds: [], targets: [], ctrls: []}}
    
    print(">>> 正在进行测试集推理评估...")
    with torch.no_grad():
        for batch in test_loader:
            ctrl = batch['rna_control'].to(device)
            target = batch['rna_target'].to(device)
            p_ids = batch['perturb'].cpu().numpy()
            
            perturb = batch['perturb'].to(device)
            cell_line = batch['cell_line'].to(device)
            dose = batch['dose'].to(device) if 'dose' in batch else None
            
            drug_feat = None
            if processor.drug_embeddings is not None:
                drug_feat = processor.drug_embeddings[perturb].to(device)
            
            res = model(ctrl, perturb, cell_line, drug_feat=drug_feat, dose=dose)
            
            res_np = res.cpu().numpy()
            target_np = target.cpu().numpy()
            ctrl_np = ctrl.cpu().numpy()
            
            for i in range(len(p_ids)):
                p_name = processor.id_to_perturb[p_ids[i]]
                if p_name not in results_by_pert:
                    results_by_pert[p_name] = {'p': [], 't': [], 'c': []}
                results_by_pert[p_name]['p'].append(res_np[i])
                results_by_pert[p_name]['t'].append(target_np[i])
                results_by_pert[p_name]['c'].append(ctrl_np[i])

    # 5. 计算指标与准备绘图数据
    print(">>> 正在计算多维度评估指标...")
    pert_metrics = []
    roc_curves_data = {} 
    bar_plot_data = None
    
    # 用户指定的 ROC 基因列表
    target_roc_genes = ['POLR3K', 'ZNF687', 'EIF1AX', 'DSN1', 'GTPBP4', 'MRPS18A', 'METTL16', 'HOXC10']
    target_heatmap_gene_upper = args.heatmap_gene.upper()

    for p_name, data in results_by_pert.items():
        avg_p = np.mean(data['p'], axis=0)
        avg_t = np.mean(data['t'], axis=0)
        avg_c = np.mean(data['c'], axis=0)
        
        d_p = avg_p - avg_c
        d_t = avg_t - avg_c
        
        # ROC AUC
        top_idx = np.argsort(np.abs(d_t))[-args.top_n:]
        binary_labels = np.zeros_like(d_t)
        binary_labels[top_idx] = 1
        fpr, tpr, _ = roc_curve(binary_labels, np.abs(d_p))
        gene_auc = auc(fpr, tpr)
        
        r_g, _ = pearsonr(avg_p, avg_t)
        
        # 计算 NormMSE (Normalized Mean Squared Error)
        # NormMSE = MSE / Var(Target)
        mse = np.mean((avg_p - avg_t)**2)
        var_t = np.var(avg_t)
        norm_mse = mse / (var_t + 1e-8)
        
        pert_metrics.append({'perturb': p_name, 'auc': gene_auc, 'pearson': r_g, 'norm_mse': norm_mse})
        roc_curves_data[p_name] = (fpr, tpr, gene_auc)
        
        # 柱状图数据准备 (针对指定的基因)
        if target_heatmap_gene_upper in p_name.upper():
            # 找到真实变化最大的 Top 20
            top_20_idx = np.argsort(np.abs(d_t))[-20:][::-1]
            display_genes = [gene_names[idx] for idx in top_20_idx]
            
            # 计算 Top 20 重叠率
            pred_top_20_idx = np.argsort(np.abs(d_p))[-20:]
            overlap = len(set(top_20_idx) & set(pred_top_20_idx))
            
            # 构建“长表”数据 (Tidy Data)
            plot_list = []
            for idx in top_20_idx:
                g = gene_names[idx]
                plot_list.append({'Gene': g, 'Expression': avg_c[idx], 'Group': 'Control (Baseline)'})
                plot_list.append({'Gene': g, 'Expression': avg_t[idx], 'Group': 'Real Perturbation (GT)'})
                plot_list.append({'Gene': g, 'Expression': avg_p[idx], 'Group': 'Predicted Perturbation (Model)'})
            
            df_plot = pd.DataFrame(plot_list)
            bar_plot_data = (df_plot, overlap, p_name)

    df_pert = pd.DataFrame(pert_metrics)

    # --- 新增: 打印详细的测试指标表格 ---
    print("\n" + "="*50)
    print(f"正在为 {len(df_pert)} 个测试扰动计算 AUC...")
    print("--- AUC 计算完成 ---")
    
    # 格式化输出表格 (类似于用户提供的格式)
    print(f"{'Perturbation':<15} {'NormMSE':<10} {'Pearson':<10} {'AUC':<10}")
    print("-" * 50)
    
    for _, row in df_pert.iterrows():
        p_name = row['perturb']
        p_display = p_name if '+ctrl' in p_name else f"{p_name}+ctrl"
        print(f"{p_display:<15} {row['norm_mse']:.6f}   {row['pearson']:.6f}   {row['auc']:.6f}")
        
    print("-" * 50)
    print(f"{'平均值':<15} {df_pert['norm_mse'].mean():.6f}   {df_pert['pearson'].mean():.6f}   {df_pert['auc'].mean():.6f}")
    print("="*50 + "\n")

    # 保存测试集基因列表
    test_genes_list = sorted(df_pert['perturb'].tolist())
    with open("test_genes_list.txt", "w") as f:
        f.write("\n".join(test_genes_list))
    print(f">>> 已将 {len(test_genes_list)} 个测试集基因保存至: test_genes_list.txt")

    # 如果没找到指定的基因，自动选择测试集中 AUC 最高的基因作为备份
    if bar_plot_data is None:
        print(f"!!! 警告: 基因 {args.heatmap_gene} 不在测试集中。")
        best_gene_row = df_pert.sort_values('auc', ascending=False).iloc[0]
        best_gene_name = best_gene_row['perturb']
        print(f">>> 自动选择测试集表现最佳基因进行展示: {best_gene_name}")
        
        data = results_by_pert[best_gene_name]
        avg_p, avg_t, avg_c = np.mean(data['p'], axis=0), np.mean(data['t'], axis=0), np.mean(data['c'], axis=0)
        d_t, d_p = avg_t - avg_c, avg_p - avg_c
        
        top_20_idx = np.argsort(np.abs(d_t))[-20:][::-1]
        pred_top_20_idx = np.argsort(np.abs(d_p))[-20:]
        overlap = len(set(top_20_idx) & set(pred_top_20_idx))
        
        plot_list = []
        for idx in top_20_idx:
            g = gene_names[idx]
            plot_list.append({'Gene': g, 'Expression': avg_c[idx], 'Group': 'Control (Baseline)'})
            plot_list.append({'Gene': g, 'Expression': avg_t[idx], 'Group': 'Real Perturbation (GT)'})
            plot_list.append({'Gene': g, 'Expression': avg_p[idx], 'Group': 'Predicted Perturbation (Model)'})
        df_plot = pd.DataFrame(plot_list)
        bar_plot_data = (df_plot, overlap, best_gene_name)
    
    # 6. 绘图 (2x2 布局)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # (1) AUC 分布直方图
    sns.histplot(df_pert['auc'], bins=20, kde=True, ax=axes[0, 0], color='teal', alpha=0.6)
    axes[0, 0].axvline(df_pert['auc'].mean(), color='red', linestyle='--', label=f'Mean AUC: {df_pert["auc"].mean():.3f}')
    axes[0, 0].set_title('Overall Distribution of AUC Scores', fontsize=16, fontweight='bold')
    axes[0, 0].set_xlabel('Target Discovery AUC')
    axes[0, 0].legend()
    
    # (2) 多线 ROC 曲线 (用户指定基因)
    available_targets = []
    for tg in target_roc_genes:
        match = [p for p in roc_curves_data.keys() if tg in p]
        if match: available_targets.append(match[0])
    
    if len(available_targets) < 5:
        top_performers = df_pert.sort_values('auc', ascending=False)['perturb'].head(8).tolist()
        for tp in top_performers:
            if tp not in available_targets: available_targets.append(tp)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(available_targets[:8])))
    for p_name, color in zip(available_targets[:8], colors):
        fpr, tpr, g_auc = roc_curves_data[p_name]
        label_name = p_name if '+ctrl' in p_name else f"{p_name}+ctrl"
        axes[0, 1].plot(fpr, tpr, color=color, lw=3, label=f"{label_name} (AUC={g_auc:.3f})")
    
    axes[0, 1].plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1.5)
    axes[0, 1].set_title('ROC Curves: Targeted Perturbations', fontsize=16, fontweight='bold')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].legend(loc="lower right", fontsize=11, frameon=True)
    
    # (3) Top-20 DE 柱状图 (改为热图)
    if bar_plot_data:
        df_p, overlap, b_pname = bar_plot_data
        b_pname_display = b_pname if '+ctrl' in b_pname else f"{b_pname}+ctrl"
        
        # 转换数据格式为 Heatmap 所需的 Matrix
        # df_p 结构: Gene | Expression | Group
        
        # 1. 提取 Control 基线
        # 注意: set_index 后可能有重复的 Gene (因为 df_p 是 long format)，需要先 filter
        ctrl_expr = df_p[df_p['Group'] == 'Control (Baseline)'].set_index('Gene')['Expression']
        real_expr = df_p[df_p['Group'] == 'Real Perturbation (GT)'].set_index('Gene')['Expression']
        pred_expr = df_p[df_p['Group'] == 'Predicted Perturbation (Model)'].set_index('Gene')['Expression']
        
        # 2. 计算 Delta (Real - Ctrl, Pred - Ctrl) 并取绝对值
        # 确保索引对齐
        genes = ctrl_expr.index
        real_delta = (real_expr.loc[genes] - ctrl_expr.loc[genes]).abs()
        pred_delta = (pred_expr.loc[genes] - ctrl_expr.loc[genes]).abs()
        
        # 3. 合并为 DataFrame
        heatmap_df = pd.DataFrame({'Real_Delta': real_delta, 'Pred_Delta': pred_delta})
        
        # 4. 排序：按 Real_Delta 降序排列
        heatmap_df = heatmap_df.sort_values('Real_Delta', ascending=False)
        
        # 5. 绘制热图 (使用 YlGnBu 颜色，类似提供的示例)
        sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[1, 0], cbar_kws={'label': 'Abs(Delta Expression)'})
        
        axes[1, 0].set_title(f'Comparison of Top 20 DE Genes for {b_pname_display}\n(Overlap: {overlap}/20)', fontsize=16, fontweight='bold')
        axes[1, 0].set_ylabel('Gene Symbol')
        axes[1, 0].set_xlabel('') # 移除 X 轴标签
    
    # (4) 性能汇总统计
    axes[1, 1].axis('off')
    
    # --- 新增: 绘制第二个基因的柱状图 (替代纯文本区域的一部分) ---
    # 我们在 axes[1, 1] 上方绘制柱状图，下方放统计文本
    
    # 分割 axes[1, 1] 区域
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_bar = inset_axes(axes[1, 1], width="100%", height="60%", loc='upper center', borderpad=0)
    
    # 准备第二个基因的数据 (硬编码为 AARS，或者如果 args.heatmap_gene 不是 AARS 则展示 AARS)
    second_gene = "AARS"
    if args.heatmap_gene.upper() == "AARS": 
        # 如果用户已经选了 AARS，那我们选个别的，比如 AUC 最高的
        candidates = df_pert.sort_values('auc', ascending=False)['perturb'].tolist()
        for c in candidates:
            if c != args.heatmap_gene:
                second_gene = c
                break
    
    # 获取第二个基因的数据
    # 注意：需要检查 second_gene 是否在测试集中
    sec_data_found = False
    for p_name in results_by_pert.keys():
        if second_gene in p_name: # 模糊匹配
            sec_p_name = p_name
            sec_data_found = True
            break
            
    if sec_data_found:
        data = results_by_pert[sec_p_name]
        avg_p, avg_t, avg_c = np.mean(data['p'], axis=0), np.mean(data['t'], axis=0), np.mean(data['c'], axis=0)
        d_t = avg_t - avg_c
        
        # Top 20 DE
        top_20_idx = np.argsort(np.abs(d_t))[-20:][::-1]
        
        plot_list = []
        for idx in top_20_idx:
            g = gene_names[idx]
            plot_list.append({'Gene': g, 'Expression': avg_c[idx], 'Group': 'Control (Baseline)'})
            plot_list.append({'Gene': g, 'Expression': avg_t[idx], 'Group': 'Real Perturbation (GT)'})
            plot_list.append({'Gene': g, 'Expression': avg_p[idx], 'Group': 'Predicted Perturbation (Model)'})
        df_sec = pd.DataFrame(plot_list)
        
        palette = {'Control (Baseline)': '#959595', 'Real Perturbation (GT)': '#e67e22', 'Predicted Perturbation (Model)': '#27ae60'}
        sns.barplot(data=df_sec, x='Gene', y='Expression', hue='Group', palette=palette, ax=ax_bar)
        
        sec_display = sec_p_name if '+ctrl' in sec_p_name else f"{sec_p_name}+ctrl"
        ax_bar.set_title(f'Comparison of Top 20 DE Genes ({sec_display})', fontsize=14, fontweight='bold')
        ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax_bar.set_xlabel('')
        ax_bar.set_ylabel('Expression')
        ax_bar.legend(fontsize=8)
    else:
        ax_bar.text(0.5, 0.5, f"Gene {second_gene} not in test set", ha='center')

    summary_text = (
        f"scERso V9 Report\n"
        f"Mean AUC:       {df_pert['auc'].mean():.4f}\n"
        f"Mean Pearson:   {df_pert['pearson'].mean():.4f}\n"
        f"Mean NormMSE:   {df_pert['norm_mse'].mean():.4f}"
    )
    # 将文本放在 axes[1, 1] 的底部
    axes[1, 1].text(0.05, 0.05, summary_text, transform=axes[1, 1].transAxes, fontsize=14, family='monospace', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='grey'))

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=200)
    print(f">>> 最终专业评估报告已生成: {args.save_path}")

if __name__ == "__main__":
    visualize()
