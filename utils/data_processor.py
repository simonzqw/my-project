import scanpy as sc
import pandas as pd
import numpy as np
import torch
import os
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
        cell_line_values = self.adata.obs[cell_line_col].values
        pert_values = self.adata.obs['perturbation'].values
        X = self.adata.X
        for cl_name, cl_id in self.cell_line_map.items():
            ctrl_mask = (cell_line_values == cl_name) & (pert_values == 'control')
            ctrl_idx = np.where(ctrl_mask)[0]
            if len(ctrl_idx) > 0:
                # 直接对 X 子矩阵求均值，避免频繁构建 AnnData 切片带来的额外内存开销
                avg_expr = X[ctrl_idx].mean(axis=0)
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

    def prepare_loaders(
        self,
        batch_size=2048,
        rna_noise=0.1,
        gene_mask_rate=0.05,
        scale_rate=0.05,
        num_workers=4,
        atac_key=None,
        atac_bank_path=None,
        background_key='cell_context'
    ):
        X = self.adata.X
        perturb_ids = self.adata.obs['perturbation'].cat.codes.values
        cell_line_col = 'cell_line' if 'cell_line' in self.adata.obs else 'source_batch'
        cell_line_ids = self.adata.obs[cell_line_col].cat.codes.values
        control_id = self.perturb_map.get('control', None)

        batch_ids = None
        if 'batch' in self.adata.obs:
            self.adata.obs['batch'] = self.adata.obs['batch'].astype('category')
            batch_ids = self.adata.obs['batch'].cat.codes.values
        
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

        # --- 可选: ATAC 特征 ---
        self.atac_features = None
        candidate_keys = [atac_key] if atac_key is not None else ['X_atac', 'atac', 'atac_feat']
        candidate_keys = [k for k in candidate_keys if k is not None]
        for key in candidate_keys:
            if key in self.adata.obsm:
                print(f">>> 检测到 ATAC 特征: obsm['{key}']")
                atac = self.adata.obsm[key]
                if issparse(atac):
                    atac = atac.toarray()
                atac = np.asarray(atac, dtype=np.float32)
                self.atac_features = torch.tensor(atac, dtype=torch.float32)
                print(f">>> ATAC 维度: {self.atac_features.shape}")
                break

        # --- 可选: 从 atac_bank.npz + 背景标签映射 ATAC ---
        if self.atac_features is None and atac_bank_path is not None:
            if not os.path.exists(atac_bank_path):
                raise FileNotFoundError(f"ATAC bank 不存在: {atac_bank_path}")
            print(f">>> 从 atac_bank 加载 ATAC 特征: {atac_bank_path}")
            bank = np.load(atac_bank_path, allow_pickle=True)
            if 'genes' in bank:
                bank_genes = [str(x) for x in bank['genes'].tolist()]
                adata_genes = [str(x) for x in self.adata.var_names.tolist()]
                if len(bank_genes) != len(adata_genes) or any(g1 != g2 for g1, g2 in zip(bank_genes, adata_genes)):
                    raise ValueError(
                        "atac_bank['genes'] 与当前 h5ad.var_names 不一致，请先按同一基因顺序构建 atac_bank。"
                    )
            bank_map = {k: bank[k].astype(np.float32) for k in bank.files if k != 'genes'}
            if len(bank_map) == 0:
                raise ValueError("atac_bank 中未找到背景向量键（除 'genes' 外）。")

            if background_key in self.adata.obs:
                bg_values = self.adata.obs[background_key].astype(str).values
                print(f">>> 使用 obs['{background_key}'] 映射 ATAC 背景。")
            else:
                fallback_col = cell_line_col
                bg_values = self.adata.obs[fallback_col].astype(str).values
                print(f">>> 未找到 obs['{background_key}']，回退使用 obs['{fallback_col}']。")

            sample_vec = next(iter(bank_map.values()))
            atac_arr = np.zeros((self.adata.n_obs, sample_vec.shape[0]), dtype=np.float32)
            missing_bg = set()
            for i, bg in enumerate(bg_values):
                if bg in bank_map:
                    atac_arr[i] = bank_map[bg]
                else:
                    missing_bg.add(bg)
            if len(missing_bg) > 0:
                print(f"!!! 警告: 以下背景在 atac_bank 中缺失，已用全0向量替代: {sorted(list(missing_bg))[:10]}")
            self.atac_features = torch.tensor(atac_arr, dtype=torch.float32)
            print(f">>> 从 atac_bank 构建样本级 ATAC 完成，维度: {self.atac_features.shape}")
        
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

        # 构造 control pool: 同 cell_line 优先；若存在 batch，则优先同 (cell_line, batch)
        control_pool_coarse = {}
        control_pool_fine = {}
        all_ctrl_idx = np.where(perturb_ids == control_id)[0] if control_id is not None else np.array([], dtype=np.int64)

        for gidx in all_ctrl_idx:
            c_id = int(cell_line_ids[gidx])
            control_pool_coarse.setdefault(c_id, []).append(int(gidx))
            if batch_ids is not None:
                b_id = int(batch_ids[gidx])
                control_pool_fine.setdefault((c_id, b_id), []).append(int(gidx))

        if len(all_ctrl_idx) == 0:
            raise ValueError("未找到 control 样本，无法构建 control pool。")

        class GenerativeDataset(Dataset):
            def __init__(
                self,
                full_rna,
                sample_indices,
                p_ids,
                c_ids,
                doses,
                atac_feats,
                control_id,
                control_pool_coarse,
                control_pool_fine=None,
                local_batch_ids=None,
                rna_noise=0.0,
                gene_mask_rate=0.0,
                scale_rate=0.0,
                is_train=True,
                seed=42
            ):
                self.full_rna = full_rna
                self.sample_indices = sample_indices
                self.p_ids = p_ids
                self.c_ids = c_ids
                self.doses = doses
                self.atac_feats = atac_feats
                self.control_id = control_id
                self.control_pool_coarse = control_pool_coarse
                self.control_pool_fine = control_pool_fine
                self.local_batch_ids = local_batch_ids
                self.rna_noise = rna_noise
                self.gene_mask_rate = gene_mask_rate
                self.scale_rate = scale_rate
                self.is_train = is_train
                self.rng = np.random.RandomState(seed)
                self.global_control_fallback = np.concatenate([np.array(v, dtype=np.int64) for v in self.control_pool_coarse.values()])
                self.fixed_ctrl_idx = []

                if not self.is_train:
                    for i in range(len(self.p_ids)):
                        p_id = int(self.p_ids[i])
                        c_id = int(self.c_ids[i])
                        if p_id == self.control_id:
                            self.fixed_ctrl_idx.append(None)
                            continue
                        candidates = self._get_control_candidates(i, c_id)
                        self.fixed_ctrl_idx.append(int(self.rng.choice(candidates)))

            def __len__(self):
                return len(self.p_ids)

            def __getitem__(self, idx):
                # 目标：真实扰动后的表达谱
                target_rna = self._get_rna_from_global(self.sample_indices[idx])
                c_id = int(self.c_ids[idx])
                p_id = int(self.p_ids[idx])
                
                # 剂量信息
                dose_val = self.doses[idx] if self.doses is not None else torch.tensor(0.0)
                atac_val = self.atac_feats[idx] if self.atac_feats is not None else None
                
                # 输入 control：同背景 control 细胞采样（control 样本本身则使用自身）
                if p_id == self.control_id:
                    input_rna = target_rna.clone()
                else:
                    if self.is_train:
                        candidates = self._get_control_candidates(idx, c_id)
                        ctrl_gidx = int(self.rng.choice(candidates))
                    else:
                        ctrl_gidx = self.fixed_ctrl_idx[idx]
                    input_rna = self._get_rna_from_global(ctrl_gidx)
                
                # --- 数据增强 (仅训练集，作用于目标或输入) ---
                if self.is_train:
                    if self.rna_noise > 0:
                        target_rna += torch.randn_like(target_rna) * self.rna_noise
                    if self.gene_mask_rate > 0:
                        mask = torch.rand_like(target_rna) > self.gene_mask_rate
                        target_rna *= mask
                
                item = {
                    'rna_control': input_rna,
                    'rna_target': target_rna,
                    'perturb': torch.tensor(p_id, dtype=torch.long),
                    'cell_line': torch.tensor(c_id, dtype=torch.long),
                    'dose': dose_val # 返回 dose
                }
                if atac_val is not None:
                    item['atac_feat'] = atac_val
                return item

            def _get_control_candidates(self, local_idx, c_id):
                if self.local_batch_ids is not None and self.control_pool_fine is not None:
                    b_id = int(self.local_batch_ids[local_idx])
                    key = (c_id, b_id)
                    if key in self.control_pool_fine and len(self.control_pool_fine[key]) > 0:
                        return self.control_pool_fine[key]
                if c_id in self.control_pool_coarse and len(self.control_pool_coarse[c_id]) > 0:
                    return self.control_pool_coarse[c_id]
                return self.global_control_fallback

            def _get_rna_from_global(self, global_idx):
                row = self.full_rna[global_idx]
                if issparse(row): row = row.toarray().flatten()
                return torch.tensor(row, dtype=torch.float32)

        # 获取对应的 doses 切片
        train_doses = self.doses[train_idx] if self.doses is not None else None
        val_doses = self.doses[val_idx] if self.doses is not None else None
        test_doses = self.doses[test_idx] if self.doses is not None else None
        train_atac = self.atac_features[train_idx] if self.atac_features is not None else None
        val_atac = self.atac_features[val_idx] if self.atac_features is not None else None
        test_atac = self.atac_features[test_idx] if self.atac_features is not None else None

        train_local_batch = batch_ids[train_idx] if batch_ids is not None else None
        val_local_batch = batch_ids[val_idx] if batch_ids is not None else None
        test_local_batch = batch_ids[test_idx] if batch_ids is not None else None

        train_ds = GenerativeDataset(
            full_rna=X,
            sample_indices=train_idx,
            p_ids=perturb_ids[train_idx],
            c_ids=cell_line_ids[train_idx],
            doses=train_doses,
            atac_feats=train_atac,
            control_id=control_id,
            control_pool_coarse=control_pool_coarse,
            control_pool_fine=control_pool_fine if batch_ids is not None else None,
            local_batch_ids=train_local_batch,
            rna_noise=rna_noise,
            gene_mask_rate=gene_mask_rate,
            scale_rate=scale_rate,
            is_train=True,
            seed=42
        )
        val_ds = GenerativeDataset(
            full_rna=X,
            sample_indices=val_idx,
            p_ids=perturb_ids[val_idx],
            c_ids=cell_line_ids[val_idx],
            doses=val_doses,
            atac_feats=val_atac,
            control_id=control_id,
            control_pool_coarse=control_pool_coarse,
            control_pool_fine=control_pool_fine if batch_ids is not None else None,
            local_batch_ids=val_local_batch,
            is_train=False,
            seed=42
        )
        test_ds = GenerativeDataset(
            full_rna=X,
            sample_indices=test_idx,
            p_ids=perturb_ids[test_idx],
            c_ids=cell_line_ids[test_idx],
            doses=test_doses,
            atac_feats=test_atac,
            control_id=control_id,
            control_pool_coarse=control_pool_coarse,
            control_pool_fine=control_pool_fine if batch_ids is not None else None,
            local_batch_ids=test_local_batch,
            is_train=False,
            seed=42
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        return train_loader, val_loader, test_loader
