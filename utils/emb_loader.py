import pandas as pd
import numpy as np
import torch
import os

class GeneEmbeddingLoader:
    """
    负责加载外部预训练基因向量并与当前数据的扰动 ID 对齐
    """
    def __init__(self, embedding_path, perturb_id_to_name_map):
        self.embedding_path = embedding_path
        self.perturb_map = perturb_id_to_name_map
        
    def load_weights(self, default_dim=200):
        if not os.path.exists(self.embedding_path):
            print(f"!!! 警告: 预训练文件 {self.embedding_path} 不存在。将使用随机初始化。")
            return None
        
        print(f">>> 正在加载预训练向量: {self.embedding_path}")
        
        # 假设文件是 CSV (gene_name, dim1, dim2...)，无 header 或 index 为基因名
        # 根据实际下载格式可能需要微调 (例如 gene2vec 通常是 txt 或 bin)
        try:
            # Gene2Vec .txt 格式通常是: 第一行是 (n_genes, dim)，或者直接每行 gene val1 val2...
            # 我们直接使用 pd.read_csv 并根据内容自动判断
            df = pd.read_csv(self.embedding_path, sep='\s+', index_col=0, header=None, engine='python')
            
            # 如果第一行是元数据 (例如 24447 200)，我们需要过滤掉
            if isinstance(df.index[0], (int, float)) or (isinstance(df.index[0], str) and df.index[0].isdigit()):
                print(">>> 检测到文件首行包含元数据，已跳过。")
                df = df.iloc[1:]
                
            print(f">>> 已读取 {len(df)} 个基因的预训练向量，维度: {df.shape[1]}")
            
            # 构建权重矩阵
            n_perturbations = len(self.perturb_map)
            emb_dim = df.shape[1]
            weights = np.random.normal(scale=0.02, size=(n_perturbations, emb_dim))
            mean_vec = df.values.mean(axis=0)
            
            hit_count = 0
            # 遍历我们数据中的所有扰动 ID
            for idx, gene_name in self.perturb_map.items():
                if gene_name == 'control':
                    # Control 可以初始化为全 0 或特定向量
                    weights[idx] = np.zeros(emb_dim)
                    continue
                    
                # 尝试匹配 (处理大小写不一致等问题)
                if gene_name in df.index:
                    weights[idx] = df.loc[gene_name].values
                    hit_count += 1
                elif gene_name.upper() in df.index:
                    weights[idx] = df.loc[gene_name.upper()].values
                    hit_count += 1
                else:
                    weights[idx] = mean_vec
                    
            print(f">>> 预训练向量匹配成功率: {hit_count}/{n_perturbations} ({hit_count/n_perturbations:.1%})")
            return torch.FloatTensor(weights)
            
        except Exception as e:
            print(f"!!! 加载预训练向量失败: {e}")
            return None
