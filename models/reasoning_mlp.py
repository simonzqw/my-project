import torch
import torch.nn as nn

class PerturbationPredictor(nn.Module):
    """
    scERso V10: 药物/基因双模态扰动预测模型
    兼容：
    1. Gene ID -> Gene2Vec Embedding
    2. Drug Feature -> Linear Projection
    """
    def __init__(self, n_genes, n_perturbations, n_cell_lines, 
                 pretrained_weights=None,
                 perturb_dim=200, cell_line_dim=32, drug_dim=2048,
                 hidden_dims=[512, 1024, 2048], dropout=0.2):
        super(PerturbationPredictor, self).__init__()
        
        self.n_genes = n_genes
        self.perturb_dim = perturb_dim
        
        # 1. 嵌入层 (基因)
        if pretrained_weights is not None:
            self.perturb_embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
            perturb_dim = pretrained_weights.shape[1]
        else:
            self.perturb_embedding = nn.Embedding(n_perturbations, perturb_dim)
            
        # 1.5 投影层 (药物)
        # 将高维药物特征 (2048) 映射到统一的 Latent 空间 (200)
        self.drug_projection = nn.Sequential(
            nn.Linear(drug_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, perturb_dim),
            nn.LayerNorm(perturb_dim)
        )
        
        # 1.6 剂量调制模块 (Dose Modulator)
        # 输入: 标量 Dose [0, 1] -> 输出: 缩放因子 [B, perturb_dim]
        # 逻辑: 剂量越大，缩放因子越大 (Sigmoid * 2 允许放大到 2 倍)
        self.dose_scaler = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, perturb_dim),
            nn.Sigmoid() 
        )
            
        self.cell_line_embedding = nn.Embedding(n_cell_lines, cell_line_dim)
        
        # 2. 投影层 (用于 MHA)
        self.rna_projection = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, perturb_dim),
            nn.LayerNorm(perturb_dim)
        )
        self.cell_line_projection = nn.Linear(cell_line_dim, perturb_dim)
        
        # 3. 多头注意力融合 (提取扰动对全局的影响权重)
        self.feature_fusion = nn.MultiheadAttention(embed_dim=perturb_dim, num_heads=4, batch_first=True)
        
        # 4. 解码网络 (预测 Delta 变化量)
        # 输入：Control_RNA + 融合特征 + 扰动向量 + 细胞系向量
        input_dim = n_genes + (perturb_dim * 3)
        
        self.decoder = nn.Sequential()
        curr_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            self.decoder.add_module(f"linear_{i}", nn.Linear(curr_dim, h_dim))
            self.decoder.add_module(f"norm_{i}", nn.LayerNorm(h_dim))
            self.decoder.add_module(f"act_{i}", nn.LeakyReLU(0.2))
            self.decoder.add_module(f"dropout_{i}", nn.Dropout(dropout))
            curr_dim = h_dim
            
        # 最终输出层：输出 2000 维的 Delta 变化量
        self.decoder.add_module("delta_out", nn.Linear(curr_dim, n_genes))
        
    def forward(self, rna_control, perturb, cell_line, drug_feat=None, dose=None):
        """
        Args:
            rna_control: 控制组平均表达谱 [B, 2000]
            perturb: 扰动基因 ID [B] (如果是药物任务，此参数可能为 None 或 dummy)
            cell_line: 细胞系 ID [B]
            drug_feat: 药物特征向量 [B, 2048] (可选)
            dose: 剂量强度 [B] (可选, 0-1之间)
        Returns:
            rna_predicted: 预测的扰动后表达谱 [B, 2000]
        """
        # 兼容性处理: 如果 rna_control 是 [B, 1, G], squeeze 掉中间维度
        if rna_control.dim() == 3 and rna_control.shape[1] == 1:
            rna_control = rna_control.squeeze(1)
            
        # 获取基础特征 (双模态切换)
        if drug_feat is not None:
            # 药物模式: 使用 Drug Projector
            p_emb = self.drug_projection(drug_feat)
            
            # --- 核心: 剂量调制 ---
            if dose is not None:
                # dose: [B] -> [B, 1]
                if dose.dim() == 1:
                    dose = dose.unsqueeze(1)
                
                # 计算缩放因子 (0.0 - 2.0)
                # 逻辑: 
                # 1. 基础缩放: dose_scaler 输出 (0, 1)
                # 2. 强度调整: * 2.0 允许放大
                # 3. 零点约束: 如果 dose=0 (Control), 强制 scale=0 (物理约束)
                scale = self.dose_scaler(dose) * 2.0
                
                # 物理约束: Control (dose=0) 时不应有扰动效果
                # 虽然 ReLU(Linear) 理论上可以学到，但显式乘上 dose 更稳健
                # p_emb = p_emb * scale * dose  <-- 这种太强硬，可能导致梯度消失
                
                # 采用柔性调制: p_emb * scale
                # 由于 dose 已经在 scaler 输入里了，网络会学到 dose=0 -> scale=0
                p_emb = p_emb * scale
                
        else:
            # 基因模式: 使用 Gene Embedding
            p_emb = self.perturb_embedding(perturb)
            
        c_emb = self.cell_line_embedding(cell_line)
        
        # 投影与融合
        rna_feat = self.rna_projection(rna_control)
        c_feat = self.cell_line_projection(c_emb)
        
        tokens = torch.stack([rna_feat, p_emb, c_feat], dim=1)
        attn_out, _ = self.feature_fusion(tokens, tokens, tokens)
        fused_feat = attn_out.mean(dim=1)
        
        # 拼接特征
        combined = torch.cat([rna_control, fused_feat, p_emb, c_feat], dim=1)
        
        # 预测 Delta (残差变化)
        delta = self.decoder(combined)
        
        # 残差相加: Predicted = Control + Delta
        rna_predicted = rna_control + delta
        
        return rna_predicted

    def freeze_perturbation_embedding(self, freeze=True):
        """控制 Embedding 层的更新，用于训练初期的稳定性"""
        self.perturb_embedding.weight.requires_grad = not freeze
        state = "冻结" if freeze else "解冻"
        print(f">>> 扰动 Embedding 层已 {state}")
