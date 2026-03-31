import numpy as np
import pandas as pd
import torch

def generate_synthetic_data(n_cells=1000, n_genes=2000, n_perturbations=10, n_cell_types=5):
    """
    生成模拟的单细胞扰动数据。
    
    逻辑：
    1. 为每个细胞随机分配一个 Cell Type。
    2. 生成基础表达矩阵 (HVGs)。
    3. 生成正样本 (Label=1): 扰动与细胞类型匹配的真实生物学反应。
    4. 生成负样本 (Label=0): 随机替换扰动，模拟不匹配的情况。
    """
    print(f"Generating synthetic data: {n_cells} cells, {n_genes} genes...")
    
    # 1. 随机分配细胞类型
    cell_types = np.random.randint(0, n_cell_types, size=n_cells)
    
    # 2. 生成基础表达矩阵 (模拟 HVGs)
    # 为不同 Cell Type 生成不同的基础表达分布
    rna_data = np.zeros((n_cells, n_genes), dtype=np.float32)
    for ct in range(n_cell_types):
        mask = (cell_types == ct)
        if np.any(mask):
            # 每个细胞类型有一个独特的“特征均值向量”
            ct_mean = np.random.randn(n_genes) * 2.0
            rna_data[mask] = ct_mean + np.random.randn(np.sum(mask), n_genes)
    
    # 3. 定义扰动
    # 扰动 ID 0-9
    perturbations = np.random.randint(0, n_perturbations, size=n_cells)
    
    # 4. 生成标签 (Label)
    # 模拟某种规律：只有当 (perturbation_id + cell_type) 是偶数时，才产生反应 (Label=1)
    # 这是一个简单的非线性规律，供 MLP 学习
    labels = ((perturbations + cell_types) % 2 == 0).astype(np.float32)
    
    # 为了增加难度，给标签加一点噪音
    noise = np.random.choice([0, 1], size=n_cells, p=[0.95, 0.05])
    labels = np.abs(labels - noise) # 翻转 5% 的标签
    
    # 5. 转换为 DataFrame 或 Tensor 格式
    data = {
        'rna': rna_data,
        'perturb': perturbations,
        'cell_type': cell_types,
        'label': labels
    }
    
    return data

def prepare_tensors(data, test_size=0.2):
    """将数据划分为训练集和验证集，并转换为 PyTorch Tensor"""
    rna = data['rna']
    perturb = data['perturb']
    labels = data['label']
    
    # 划分数据集 (手动实现)
    n_samples = len(labels)
    idx = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(idx)
    
    val_size = int(n_samples * test_size)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    
    train_data = {
        'rna': torch.tensor(rna[train_idx]),
        'perturb': torch.tensor(perturb[train_idx]),
        'label': torch.tensor(labels[train_idx])
    }
    
    val_data = {
        'rna': torch.tensor(rna[val_idx]),
        'perturb': torch.tensor(perturb[val_idx]),
        'label': torch.tensor(labels[val_idx])
    }
    
    return train_data, val_data

if __name__ == "__main__":
    data = generate_synthetic_data()
    train_data, val_data = prepare_tensors(data)
    print(f"Train samples: {len(train_data['label'])}")
    print(f"Val samples: {len(val_data['label'])}")
    print(f"Label distribution (Train): {np.bincount(train_data['label'].numpy().astype(int))}")
