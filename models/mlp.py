import torch
import torch.nn as nn

class SpecificityMLP(nn.Module):
    """
    单细胞扰动特异性判别模型 (MLP) - 增强版
    
    支持:
    1. RNA-seq 特征 (HVGs)
    2. 扰动基因 Embedding
    3. 细胞系/组织上下文 Embedding (实现老师要求的特异性建模)
    """
    def __init__(self, n_genes, n_perturbations, n_cell_lines, 
                 perturb_dim=128, cell_line_dim=32, 
                 hidden_dims=[512, 256, 128], dropout=0.2):
        super(SpecificityMLP, self).__init__()
        
        # 1. 嵌入层
        self.perturb_embedding = nn.Embedding(n_perturbations, perturb_dim)
        self.cell_line_embedding = nn.Embedding(n_cell_lines, cell_line_dim)
        
        # 2. 特征融合层
        # 输入维度 = RNA维度 + 扰动维度 + 细胞系维度
        input_dim = n_genes + perturb_dim + cell_line_dim
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
            
        # 3. 输出层 (二分类: 匹配/响应 vs 不匹配/正常)
        layers.append(nn.Linear(curr_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, rna, perturb, cell_line):
        """
        Args:
            rna: [batch_size, n_genes]
            perturb: [batch_size] 扰动 ID
            cell_line: [batch_size] 细胞系 ID
        """
        p_emb = self.perturb_embedding(perturb)
        c_emb = self.cell_line_embedding(cell_line)
        
        # 拼接所有特征
        x = torch.cat([rna, p_emb, c_emb], dim=1)
        
        return self.network(x)
