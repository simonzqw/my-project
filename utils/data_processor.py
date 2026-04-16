
import os
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class DataProcessor:
    def __init__(
        self,
        h5ad_path,
        test_size=0.1,
        val_size=0.1,
        split_strategy='random',
        split_col: str = 'split',
        atac_key: Optional[str] = None,
        atac_bank_path: Optional[str] = None,
        background_key: str = 'cell_context',
    ):
        self.h5ad_path = h5ad_path
        self.test_size = test_size
        self.val_size = val_size
        self.split_strategy = split_strategy
        self.split_col = split_col
        self.atac_key = atac_key
        self.atac_bank_path = atac_bank_path
        self.background_key = background_key

        self.adata = None
        self.perturb_map = None
        self.id_to_perturb = None
        self.cell_line_map = None
        self.gene_to_idx = None

        self.atac_features = None
        self.atac_dim = 0
        self.cell_line_atac_baselines = {}
        self.doses = None
        self.drug_embeddings = None

    def _ensure_dense_numpy(self, x):
        if issparse(x):
            x = x.toarray()
        return np.asarray(x)

    def _prepare_metadata(self):
        if 'product_name' in self.adata.obs and 'perturbation' not in self.adata.obs:
            print(">>> 检测到 SCI-Plex 格式，正在适配列名...")
            self.adata.obs['perturbation'] = self.adata.obs['product_name'].astype(str)
            self.adata.obs.loc[self.adata.obs['perturbation'].isin(['Vehicle', 'control']), 'perturbation'] = 'control'

        if 'cell_type' in self.adata.obs and 'cell_line' not in self.adata.obs:
            self.adata.obs['cell_line'] = self.adata.obs['cell_type']

        if 'condition' in self.adata.obs and 'perturbation' not in self.adata.obs:
            print(">>> 检测到 Adamson 格式，正在适配列名...")
            self.adata.obs['perturbation'] = self.adata.obs['condition'].astype(str)

            def clean_adamson_pert(x):
                if x == 'ctrl':
                    return 'control'
                if '+ctrl' in x:
                    return x.split('+')[0]
                return x

            self.adata.obs['perturbation'] = self.adata.obs['perturbation'].apply(clean_adamson_pert)
            print(f">>> Adamson 格式清洗示例: {self.adata.obs['perturbation'].unique()[:5]}")

        if 'SMILES' in self.adata.obs and 'smiles' not in self.adata.obs:
            self.adata.obs['smiles'] = self.adata.obs['SMILES']

        self.adata.obs['perturbation'] = self.adata.obs['perturbation'].astype(str)
        if self.adata.obs['perturbation'].str.contains('_').any():
            print(">>> 检测到扰动名称包含下划线，尝试清洗以匹配 side information...")
            self.adata.obs['perturbation'] = self.adata.obs['perturbation'].apply(
                lambda x: x.split('_')[0] if x != 'control' and '_' in str(x) else x
            )
            print(f">>> 清洗后的扰动类别示例: {self.adata.obs['perturbation'].unique()[:5]}")

        self.adata.obs['perturbation'] = self.adata.obs['perturbation'].astype('category')
        self.perturb_categories = self.adata.obs['perturbation'].cat.categories.tolist()
        self.perturb_map = {name: i for i, name in enumerate(self.perturb_categories)}
        self.id_to_perturb = {i: name for name, i in self.perturb_map.items()}

        cell_line_col = 'cell_line' if 'cell_line' in self.adata.obs else 'source_batch'
        self.cell_line_col = cell_line_col
        self.adata.obs[cell_line_col] = self.adata.obs[cell_line_col].astype('category')
        self.cell_line_categories = self.adata.obs[cell_line_col].cat.categories.tolist()
        self.cell_line_map = {name: i for i, name in enumerate(self.cell_line_categories)}

    def _prepare_drug_features(self):
        self.drug_embeddings = None
        if 'smiles' not in self.adata.obs:
            return

        print(">>> 检测到 SMILES 信息，正在提取药物化学特征 (Morgan Fingerprint)...")
        unique_perts = self.adata.obs[['perturbation', 'smiles']].drop_duplicates('perturbation')
        pert_to_smiles = dict(zip(unique_perts['perturbation'], unique_perts['smiles']))

        drug_feats = []
        for name in self.perturb_categories:
            if name == 'control' or name not in pert_to_smiles:
                drug_feats.append(np.zeros(2048, dtype=np.float32))
                continue

            mol = Chem.MolFromSmiles(pert_to_smiles[name])
            if mol is None:
                print(f"!!! 警告: 无法解析 SMILES: {name}")
                drug_feats.append(np.zeros(2048, dtype=np.float32))
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            drug_feats.append(np.asarray(fp, dtype=np.float32))

        self.drug_embeddings = torch.tensor(np.stack(drug_feats, axis=0), dtype=torch.float32)
        print(f">>> 药物特征提取完成，维度: {self.drug_embeddings.shape}")

    def _prepare_dose_features(self):
        if 'dose' not in self.adata.obs:
            self.doses = None
            return

        print(">>> 检测到剂量信息，正在处理 Dose 特征...")
        doses = self.adata.obs['dose'].values.astype(np.float32)
        doses = np.log1p(doses)
        d_min, d_max = doses.min(), doses.max()
        if d_max > d_min:
            doses = (doses - d_min) / (d_max - d_min)
        else:
            doses = np.zeros_like(doses)
        self.doses = torch.tensor(doses, dtype=torch.float32)

    def _prepare_atac_features(self, atac_key=None, atac_bank_path=None, background_key='cell_context'):
        if self.atac_features is not None:
            return

        self.atac_features = None
        candidate_keys = [atac_key] if atac_key is not None else ['X_atac', 'atac', 'atac_feat']
        candidate_keys = [k for k in candidate_keys if k is not None]

        for key in candidate_keys:
            if key in self.adata.obsm:
                print(f">>> 检测到 ATAC 特征: obsm['{key}']")
                atac = self._ensure_dense_numpy(self.adata.obsm[key]).astype(np.float32)
                self.atac_features = torch.tensor(atac, dtype=torch.float32)
                self.atac_dim = int(self.atac_features.shape[1])
                print(f">>> ATAC 维度: {self.atac_features.shape}")
                break

        if self.atac_features is None and atac_bank_path is not None:
            if not os.path.exists(atac_bank_path):
                raise FileNotFoundError(f"ATAC bank 不存在: {atac_bank_path}")

            print(f">>> 从 atac_bank 加载 ATAC 特征: {atac_bank_path}")
            bank = np.load(atac_bank_path, allow_pickle=True)

            if 'genes' in bank:
                bank_genes = [str(x) for x in bank['genes'].tolist()]
                adata_genes = [str(x) for x in self.adata.var_names.tolist()]
                if len(bank_genes) != len(adata_genes) or any(g1 != g2 for g1, g2 in zip(bank_genes, adata_genes)):
                    raise ValueError("atac_bank['genes'] 与当前 h5ad.var_names 不一致，请先按同一基因顺序构建 atac_bank。")

            bank_map = {k: bank[k].astype(np.float32) for k in bank.files if k != 'genes'}
            if len(bank_map) == 0:
                raise ValueError("atac_bank 中未找到背景向量键（除 'genes' 外）。")

            if background_key in self.adata.obs:
                bg_values = self.adata.obs[background_key].astype(str).values
                print(f">>> 使用 obs['{background_key}'] 映射 ATAC 背景。")
            else:
                bg_values = self.adata.obs[self.cell_line_col].astype(str).values
                print(f">>> 未找到 obs['{background_key}']，回退使用 obs['{self.cell_line_col}']。")

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
            self.atac_dim = int(self.atac_features.shape[1])
            print(f">>> 从 atac_bank 构建样本级 ATAC 完成，维度: {self.atac_features.shape}")

        if self.atac_features is not None:
            self.cell_line_atac_baselines = {}
            cell_line_ids = self.adata.obs[self.cell_line_col].cat.codes.values
            atac_np = self.atac_features.numpy()
            for cl_id in np.unique(cell_line_ids):
                idx = np.where(cell_line_ids == cl_id)[0]
                if len(idx) == 0:
                    continue
                self.cell_line_atac_baselines[int(cl_id)] = torch.tensor(atac_np[idx].mean(axis=0), dtype=torch.float32)

    def get_cell_line_control(self, cell_line_idx: int, device=None):
        x = self.cell_line_baselines[cell_line_idx]
        if device is not None:
            x = x.to(device)
        return x

    def get_cell_line_atac(self, cell_line_idx: int, device=None):
        if not self.cell_line_atac_baselines or cell_line_idx not in self.cell_line_atac_baselines:
            return None
        x = self.cell_line_atac_baselines[cell_line_idx]
        if device is not None:
            x = x.to(device)
        return x

    def load_data(self):
        print(f">>> 正在加载数据: {self.h5ad_path}")
        self.adata = sc.read_h5ad(self.h5ad_path)

        self._prepare_metadata()
        self._prepare_drug_features()
        self._prepare_dose_features()
        self._prepare_atac_features(
            atac_key=self.atac_key,
            atac_bank_path=self.atac_bank_path,
            background_key=self.background_key,
        )

        print(">>> 正在计算细胞系基线表达谱...")
        self.cell_line_baselines = {}
        cell_line_values = self.adata.obs[self.cell_line_col].values
        pert_values = self.adata.obs['perturbation'].values
        X = self.adata.X

        for cl_name, cl_id in self.cell_line_map.items():
            ctrl_mask = (cell_line_values == cl_name) & (pert_values == 'control')
            ctrl_idx = np.where(ctrl_mask)[0]
            if len(ctrl_idx) > 0:
                avg_expr = X[ctrl_idx].mean(axis=0)
                if issparse(avg_expr):
                    avg_expr = avg_expr.toarray()
                avg_expr = np.asarray(avg_expr).flatten()
                self.cell_line_baselines[cl_id] = torch.tensor(avg_expr, dtype=torch.float32)
            else:
                print(f"!!! 警告: 细胞系 {cl_name} 缺失控制组数据")

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
        if self.atac_features is None and (atac_key is not None or atac_bank_path is not None):
            self._prepare_atac_features(
                atac_key=atac_key,
                atac_bank_path=atac_bank_path,
                background_key=background_key,
            )

        X = self.adata.X
        perturb_ids = self.adata.obs['perturbation'].cat.codes.values
        cell_line_ids = self.adata.obs[self.cell_line_col].cat.codes.values
        control_id = self.perturb_map.get('control', None)

        batch_ids = None
        if 'batch' in self.adata.obs:
            self.adata.obs['batch'] = self.adata.obs['batch'].astype('category')
            batch_ids = self.adata.obs['batch'].cat.codes.values

        indices = np.arange(self.adata.n_obs)

        if self.split_strategy == 'custom':
            if self.split_col not in self.adata.obs:
                raise ValueError(f"adata.obs 缺少自定义划分列: {self.split_col}")
            split_values = self.adata.obs[self.split_col].astype(str).values
            train_idx = np.where(split_values == 'train')[0]
            val_idx = np.where(split_values == 'val')[0]
            test_idx = np.where(split_values == 'test')[0]
            print(f">>> 采用自定义划分策略: obs['{self.split_col}']")
            print(f">>> 划分结果: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
            if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
                raise ValueError(f"自定义划分列 {self.split_col} 中 train/val/test 至少有一个为空。")
        elif self.split_strategy == 'perturbation':
            print(">>> 采用按扰动基因划分策略 (Zero-shot 分层模式)...")
            real_perts = [p for p in self.perturb_categories if p != 'control']
            np.random.seed(42)
            np.random.shuffle(real_perts)

            n_test = int(len(real_perts) * self.test_size)
            n_val = int(len(real_perts) * self.val_size)

            test_p = set(real_perts[:n_test])
            val_p = set(real_perts[n_test:n_test + n_val])
            train_p = set(real_perts[n_test + n_val:])

            def get_idx(pert_set):
                mask = self.adata.obs['perturbation'].isin(pert_set)
                return np.where(mask)[0]

            train_idx_p = get_idx(train_p)
            val_idx = get_idx(val_p)
            test_idx = get_idx(test_p)
            ctrl_idx = np.where(self.adata.obs['perturbation'] == 'control')[0]
            train_idx = np.concatenate([train_idx_p, ctrl_idx])

            print(f">>> 划分结果: 训练集 {len(train_p)} 基因(+Control), 验证集 {len(val_p)} 基因, 测试集 {len(test_p)} 基因")
        else:
            print(">>> 采用随机划分策略...")
            train_idx, temp = train_test_split(indices, test_size=(self.val_size + self.test_size), random_state=42)
            val_idx, test_idx = train_test_split(temp, test_size=0.5, random_state=42)

        def build_control_pools(ctrl_indices):
            control_pool_coarse = {}
            control_pool_fine = {} if batch_ids is not None else None
            for gidx in ctrl_indices:
                c_id = int(cell_line_ids[gidx])
                control_pool_coarse.setdefault(c_id, []).append(int(gidx))
                if batch_ids is not None:
                    b_id = int(batch_ids[gidx])
                    control_pool_fine.setdefault((c_id, b_id), []).append(int(gidx))
            return control_pool_coarse, control_pool_fine

        all_ctrl_idx = np.where(perturb_ids == control_id)[0] if control_id is not None else np.array([], dtype=np.int64)
        train_ctrl_idx = np.intersect1d(train_idx, all_ctrl_idx)
        val_ctrl_idx = np.intersect1d(val_idx, all_ctrl_idx)
        test_ctrl_idx = np.intersect1d(test_idx, all_ctrl_idx)

        train_control_pool_coarse, train_control_pool_fine = build_control_pools(train_ctrl_idx)
        val_control_pool_coarse, val_control_pool_fine = build_control_pools(val_ctrl_idx)
        test_control_pool_coarse, test_control_pool_fine = build_control_pools(test_ctrl_idx)

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
                seed=42,
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
                if len(self.control_pool_coarse) > 0:
                    self.global_control_fallback = np.concatenate(
                        [np.array(v, dtype=np.int64) for v in self.control_pool_coarse.values()]
                    )
                else:
                    self.global_control_fallback = np.array([], dtype=np.int64)
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
                target_rna = self._get_rna_from_global(self.sample_indices[idx])
                c_id = int(self.c_ids[idx])
                p_id = int(self.p_ids[idx])

                dose_val = self.doses[idx] if self.doses is not None else torch.tensor(0.0)
                atac_val = self.atac_feats[idx] if self.atac_feats is not None else None

                if p_id == self.control_id:
                    input_rna = target_rna.clone()
                else:
                    if self.is_train:
                        candidates = self._get_control_candidates(idx, c_id)
                        ctrl_gidx = int(self.rng.choice(candidates))
                    else:
                        ctrl_gidx = self.fixed_ctrl_idx[idx]
                    input_rna = self._get_rna_from_global(ctrl_gidx)

                if self.is_train:
                    if self.rna_noise > 0:
                        target_rna = target_rna + torch.randn_like(target_rna) * self.rna_noise
                    if self.gene_mask_rate > 0:
                        mask = torch.rand_like(target_rna) > self.gene_mask_rate
                        target_rna = target_rna * mask

                item = {
                    'rna_control': input_rna,
                    'rna_target': target_rna,
                    'perturb': torch.tensor(p_id, dtype=torch.long),
                    'cell_line': torch.tensor(c_id, dtype=torch.long),
                    'dose': dose_val,
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
                if self.global_control_fallback.size == 0:
                    raise ValueError("当前 split 内不存在可用 control 样本，无法为非-control 样本匹配输入 control。")
                return self.global_control_fallback

            def _get_rna_from_global(self, global_idx):
                row = self.full_rna[global_idx]
                if issparse(row):
                    row = row.toarray().flatten()
                return torch.tensor(row, dtype=torch.float32)

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
            control_pool_coarse=train_control_pool_coarse,
            control_pool_fine=train_control_pool_fine if batch_ids is not None else None,
            local_batch_ids=train_local_batch,
            rna_noise=rna_noise,
            gene_mask_rate=gene_mask_rate,
            scale_rate=scale_rate,
            is_train=True,
            seed=42,
        )
        val_ds = GenerativeDataset(
            full_rna=X,
            sample_indices=val_idx,
            p_ids=perturb_ids[val_idx],
            c_ids=cell_line_ids[val_idx],
            doses=val_doses,
            atac_feats=val_atac,
            control_id=control_id,
            control_pool_coarse=val_control_pool_coarse,
            control_pool_fine=val_control_pool_fine if batch_ids is not None else None,
            local_batch_ids=val_local_batch,
            is_train=False,
            seed=42,
        )
        test_ds = GenerativeDataset(
            full_rna=X,
            sample_indices=test_idx,
            p_ids=perturb_ids[test_idx],
            c_ids=cell_line_ids[test_idx],
            doses=test_doses,
            atac_feats=test_atac,
            control_id=control_id,
            control_pool_coarse=test_control_pool_coarse,
            control_pool_fine=test_control_pool_fine if batch_ids is not None else None,
            local_batch_ids=test_local_batch,
            is_train=False,
            seed=42,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader, test_loader
