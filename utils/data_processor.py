import scanpy as sc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse
from rdkit import Chem
from rdkit.Chem import AllChem

class DataProcessor:
    def __init__(self, h5ad_path, test_size=0.1, val_size=0.1, split_strategy='random'):
        self.h5ad_path = h5ad_path
        self.test_size = test_size
        self.val_size = val_size
        self.split_strategy = split_strategy
        self.adata = None
        self.perturb_map = None
        self.id_to_perturb = None
        self.cell_line_map = None
        self.gene_to_idx = None

    def load_data(self):
        print(f">>> 正在加载数据: {self.h5ad_path}")
        self.adata = sc.read_h5ad(self.h5ad_path)
        
        # --- 核心新增: SCI-Plex 数据适配 ---
        if 'product_name' in self.adata.obs and 'perturbation' not in self.adata.obs:
            print(">>> 检测到 SCI-Plex 格式，正在适配列名...")
            self.adata.obs['perturbation'] = self.adata.obs['product_name'].astype(str)
            # 标准化 Control 名称
            self.adata.obs.loc[self.adata.obs['perturbation'].isin(['Vehicle', 'control']), 'perturbation'] = 'control'
            
        if 'cell_type' in self.adata.obs and 'cell_line' not in self.adata.obs:
            self.adata.obs['cell_line'] = self.adata.obs['cell_type']
            
        # --- 核心新增: Adamson 数据集适配 ---
        if 'condition' in self.adata.obs and 'perturbation' not in self.adata.obs:
             print(">>> 检测到 Adamson 格式，正在适配列名...")
             self.adata.obs['perturbation'] = self.adata.obs['condition'].astype(str)
             
             # 清洗后缀: 将 "GENE+ctrl" 清洗为 "GENE"
             # 同时处理 "ctrl" -> "control"
             def clean_adamson_pert(x):
                 if x == 'ctrl': return 'control'
                 if '+ctrl' in x: return x.split('+')[0]
                 return x
                 
             self.adata.obs['perturbation'] = self.adata.obs['perturbation'].apply(clean_adamson_pert)
             print(f">>> Adamson 格式清洗示例: {self.adata.obs['perturbation'].unique()[:5]}")
            
        if 'SMILES' in self.adata.obs and 'smiles' not in self.adata.obs:
            self.adata.obs['smiles'] = self.adata.obs['SMILES']
        
        # 40 扰动标签处理 (兼容 Drug 和 Gene)
        # 如果是药物数据，perturbation 列通常是药物名
        self.adata.obs['perturbation'] = self.adata.obs['perturbation'].astype('category')
        
        # --- 核心修复: Datlinger 数据集清洗 ---
        # 去除后缀 (如 ZAP70_1 -> ZAP70) 以匹配 Gene2Vec
        # 但要注意: 某些数据集可能真的有两个不同的 gRNA 针对同一基因，如果需要合并:
        if self.adata.obs['perturbation'].str.contains('_').any():
             print(">>> 检测到扰动名称包含下划线 (如 Gene_1)，正在尝试清洗以匹配 Gene2Vec...")
             # 仅清洗非 control 的项
             # 使用 apply lambda 处理，如果是 control 则保持不变，否则 split('_')[0]
             self.adata.obs['perturbation'] = self.adata.obs['perturbation'].apply(
                 lambda x: x.split('_')[0] if x != 'control' and '_' in str(x) else x
             )
             print(f">>> 清洗后的扰动类别示例: {self.adata.obs['perturbation'].unique()[:5]}")
        
        # --- 核心修复: 确保 apply 后重新转回 category ---
        self.adata.obs['perturbation'] = self.adata.obs['perturbation'].astype('category')
        
        self.perturb_categories = self.adata.obs['perturbation'].cat.categories.tolist()
        self.perturb_map = {name: i for i, name in enumerate(self.perturb_categories)}
        self.id_to_perturb = {i: name for name, i in self.perturb_map.items()}
        
        # --- 核心新增: 药物化学特征提取 (Morgan Fingerprint) ---
        self.drug_embeddings = None
        if 'smiles' in self.adata.obs:
            print(">>> 检测到 SMILES 信息，正在提取药物化学特征 (Morgan Fingerprint)...")
            # 去重提取 SMILES
            unique_perts = self.adata.obs[['perturbation', 'smiles']].drop_duplicates('perturbation')
            # 建立 Perturbation Name -> SMILES 的映射
            pert_to_smiles = dict(zip(unique_perts['perturbation'], unique_perts['smiles']))
            
            drug_feats = []
            valid_drug_indices = []
            
            for i, name in enumerate(self.perturb_categories):
                if name == 'control' or name not in pert_to_smiles:
                    # Control 或无 SMILES 的药物用全 0 向量
                    drug_feats.append(np.zeros(2048))
                else:
                    mol = Chem.MolFromSmiles(pert_to_smiles[name])
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        drug_feats.append(np.array(fp))
                    else:
                        print(f"!!! 警告: 无法解析 SMILES: {name}")
                        drug_feats.append(np.zeros(2048))
            
            self.drug_embeddings = torch.tensor(np.array(drug_feats), dtype=torch.float32)
            print(f">>> 药物特征提取完成，维度: {self.drug_embeddings.shape}")
        
        # 细胞系标签处理
        cell_line_col = 'cell_line' if 'cell_line' in self.adata.obs else 'source_batch'
        self.adata.obs[cell_line_col] = self.adata.obs[cell_line_col].astype('category')
        self.cell_line_categories = self.adata.obs[cell_line_col].cat.categories.tolist()
        self.cell_line_map = {name: i for i, name in enumerate(self.cell_line_categories)}
        
        # --- 核心新增: 计算每个细胞系的平均控制组表达谱 (Baseline) ---
        print(">>> 正在计算细胞系基线表达谱...")
        self.cell_line_baselines = {}
        for cl_name, cl_id in self.cell_line_map.items():
            ctrl_mask = (self.adata.obs[cell_line_col] == cl_name) & (self.adata.obs['perturbation'] == 'control')
            ctrl_adata = self.adata[ctrl_mask]
            if ctrl_adata.n_obs > 0:
                avg_expr = ctrl_adata.X.mean(axis=0)
                if issparse(avg_expr): avg_expr = avg_expr.toarray()
                # 确保转换为 1D numpy array
                avg_expr = np.asarray(avg_expr).flatten()
                self.cell_line_baselines[cl_id] = torch.tensor(avg_expr, dtype=torch.float32)
            else:
                print(f"!!! 警告: 细胞系 {cl_name} 缺失控制组数据")

        # 基因名与索引映射
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.adata.var_names)}
        
        print(f">>> 数据加载完成: {self.adata.n_obs} 细胞, {self.adata.n_vars} 基因")
        return self.adata.n_vars, len(self.perturb_categories), len(self.cell_line_categories)

    def prepare_loaders(self, batch_size=2048, rna_noise=0.1, gene_mask_rate=0.05, scale_rate=0.05, num_workers=4):
        X = self.adata.X
        perturb_ids = self.adata.obs['perturbation'].cat.codes.values
        cell_line_col = 'cell_line' if 'cell_line' in self.adata.obs else 'source_batch'
        cell_line_ids = self.adata.obs[cell_line_col].cat.codes.values
        
        # --- 核心新增: 剂量信息处理 ---
        # 1. 提取剂量
        # 2. Log1p 变换 (压缩长尾分布)
        # 3. MinMax 归一化到 [0, 1]
        if 'dose' in self.adata.obs:
            print(">>> 检测到剂量信息，正在处理 Dose 特征...")
            doses = self.adata.obs['dose'].values.astype(np.float32)
            doses = np.log1p(doses) # Log1p 变换
            
            d_min, d_max = doses.min(), doses.max()
            if d_max > d_min:
                doses = (doses - d_min) / (d_max - d_min)
            else:
                doses = np.zeros_like(doses) # 如果只有一个剂量，归一化为 0
            
            self.doses = torch.tensor(doses, dtype=torch.float32)
        else:
            self.doses = None
        
        indices = np.arange(self.adata.n_obs)
        
        if self.split_strategy == 'perturbation':
            print(">>> 采用按扰动基因划分策略 (Zero-shot 分层模式)...")
            real_perts = [p for p in self.perturb_categories if p != 'control']
            np.random.seed(42)
            np.random.shuffle(real_perts)
            
            n_test = int(len(real_perts) * self.test_size)
            n_val = int(len(real_perts) * self.val_size)
            
            test_p = set(real_perts[:n_test])
            val_p = set(real_perts[n_test:n_test+n_val])
            train_p = set(real_perts[n_test+n_val:])
            
            # 核心：分层获取索引，确保每个集合内细胞系分布与全局一致
            def get_stratified_idx(pert_set):
                mask = self.adata.obs['perturbation'].isin(pert_set)
                return np.where(mask)[0]

            train_idx_p = get_stratified_idx(train_p)
            val_idx = get_stratified_idx(val_p)
            test_idx = get_stratified_idx(test_p)
            
            # Control 组全部留在训练集作为基线支撑
            ctrl_idx = np.where(self.adata.obs['perturbation'] == 'control')[0]
            train_idx = np.concatenate([train_idx_p, ctrl_idx])
            
            print(f">>> 划分结果: 训练集 {len(train_p)} 基因(+Control), 验证集 {len(val_p)} 基因, 测试集 {len(test_p)} 基因")
        else:
            print(">>> 采用随机划分策略...")
            train_idx, temp = train_test_split(indices, test_size=(self.val_size+self.test_size), random_state=42)
            val_idx, test_idx = train_test_split(temp, test_size=0.5, random_state=42)

        class GenerativeDataset(Dataset):
            def __init__(self, rna, p_ids, c_ids, doses, baselines, rna_noise=0.0, gene_mask_rate=0.0, scale_rate=0.0, is_train=True):
                self.rna = rna
                self.p_ids = p_ids
                self.c_ids = c_ids
                self.doses = doses # 新增 dose
                self.baselines = baselines
                self.rna_noise = rna_noise
                self.gene_mask_rate = gene_mask_rate
                self.scale_rate = scale_rate
                self.is_train = is_train

            def __len__(self):
                return len(self.p_ids)

            def __getitem__(self, idx):
                # 目标：真实扰动后的表达谱
                target_rna = self._get_rna(idx)
                c_id = self.c_ids[idx]
                p_id = self.p_ids[idx]
                
                # 剂量信息
                dose_val = self.doses[idx] if self.doses is not None else torch.tensor(0.0)
                
                # 输入：对应细胞系的平均控制组表达谱 (Baseline)
                input_rna = self.baselines[c_id].clone()
                
                # --- 数据增强 (仅训练集，作用于目标或输入) ---
                if self.is_train:
                    if self.rna_noise > 0:
                        target_rna += torch.randn_like(target_rna) * self.rna_noise
                    if self.gene_mask_rate > 0:
                        mask = torch.rand_like(target_rna) > self.gene_mask_rate
                        target_rna *= mask
                
                return {
                    'rna_control': input_rna,
                    'rna_target': target_rna,
                    'perturb': torch.tensor(p_id, dtype=torch.long),
                    'cell_line': torch.tensor(c_id, dtype=torch.long),
                    'dose': dose_val # 返回 dose
                }

            def _get_rna(self, idx):
                row = self.rna[idx]
                if issparse(row): row = row.toarray().flatten()
                return torch.tensor(row, dtype=torch.float32)

        # 获取对应的 doses 切片
        train_doses = self.doses[train_idx] if self.doses is not None else None
        val_doses = self.doses[val_idx] if self.doses is not None else None
        test_doses = self.doses[test_idx] if self.doses is not None else None

        train_ds = GenerativeDataset(X[train_idx], perturb_ids[train_idx], cell_line_ids[train_idx], train_doses,
                                    self.cell_line_baselines, rna_noise, gene_mask_rate, scale_rate, True)
        val_ds = GenerativeDataset(X[val_idx], perturb_ids[val_idx], cell_line_ids[val_idx], val_doses,
                                  self.cell_line_baselines, 0.0, 0.0, 0.0, False)
        test_ds = GenerativeDataset(X[test_idx], perturb_ids[test_idx], cell_line_ids[test_idx], test_doses,
                                   self.cell_line_baselines, 0.0, 0.0, 0.0, False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        return train_loader, val_loader, test_loader
