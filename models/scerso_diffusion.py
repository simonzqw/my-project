import torch
import torch.nn as nn
import math
from models.diffusion_core import GaussianDiffusion

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionDecoder(nn.Module):
    """
    MLP that predicts x0 given x_t, t, and context.
    """
    def __init__(self, input_dim, context_dim, hidden_dims=[512, 512, 512], dropout=0.1): # 简化网络
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = context_dim 
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.layers = nn.ModuleList()
        
        # Initial projection: x_t + context + time
        # input: x_t (2000) + context (2000+600) + time (200) -> hidden
        total_input_dim = input_dim + context_dim + self.time_dim
        
        curr_dim = total_input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(curr_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ))
            curr_dim = h_dim
            
        self.final = nn.Linear(curr_dim, input_dim)

    def forward(self, x, t, context):
        t_emb = self.time_mlp(t)
        
        # Concatenate inputs: Noisy Data + Context + Time
        # x: [B, input_dim], context: [B, context_dim], t_emb: [B, time_dim]
        combined = torch.cat([x, context, t_emb], dim=1)
        
        h = combined
        for layer in self.layers:
            h = layer(h)
            
        return self.final(h)

class PerturbationDiffusionPredictor(nn.Module):
    """
    scERso V9-Diff: Diffusion-based Generative Perturbation Predictor
    
    Structure:
    1. Encoder (Attention-based): Extracts 'context' from (RNA_control, Perturbations, CellLine).
    2. Diffusion Decoder: Generates RNA_target (or Delta) using the context.
    
    Innovation:
    - Supports Multi-Gene Perturbation (via variable length Attention tokens).
    - Uses Diffusion for high-quality distribution modeling.
    """
    def __init__(self, n_genes, n_perturbations, n_cell_lines, 
                 pretrained_weights=None,
                 perturb_dim=200, cell_line_dim=32, 
                 hidden_dims=[512, 1024, 2048], dropout=0.1,
                 timesteps=1000):
        super().__init__()
        
        self.n_genes = n_genes
        self.perturb_dim = perturb_dim
        self.cell_line_dim = cell_line_dim
        
        # --- 1. Encoder Components (Same as V7/V9 MLP) ---
        if pretrained_weights is not None:
            self.perturb_embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
            perturb_dim = pretrained_weights.shape[1]
        else:
            self.perturb_embedding = nn.Embedding(n_perturbations, perturb_dim)
            
        self.cell_line_embedding = nn.Embedding(n_cell_lines, cell_line_dim)
        
        # Projections
        self.rna_projection = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, perturb_dim),
            nn.LayerNorm(perturb_dim)
        )
        self.cell_line_projection = nn.Linear(cell_line_dim, perturb_dim)
        
        # Attention Fusion
        self.feature_fusion = nn.MultiheadAttention(embed_dim=perturb_dim, num_heads=4, batch_first=True)
        
        # --- 2. Diffusion Components ---
        # Context dim = RNA_control (n_genes) + Fused_Attention (perturb_dim) + Cell_Emb (perturb_dim)
        # Note: We pass raw RNA_control as part of context to guide the generation
        self.context_dim = n_genes + perturb_dim * 2
        
        self.denoise_fn = DiffusionDecoder(
            input_dim=n_genes, # Predicting the full expression profile (or delta)
            context_dim=self.context_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        self.diffusion = GaussianDiffusion(
            model=self.denoise_fn,
            input_dim=n_genes,
            timesteps=timesteps,
            objective='pred_x0' # 核心修改：预测原始数据 x0，而非噪声 epsilon
        )

    def encode_context(self, rna_control, perturb, cell_line, custom_latent=None):
        """
        Generates the conditioning vector for diffusion.
        
        Args:
            rna_control: [B, Genes]
            perturb: [B] or [B, K] (Indices of perturbations)
            cell_line: [B]
            custom_latent: [B, D] (Optional) Directly provide the fused latent vector.
                           If provided, 'perturb' is ignored for the latent part.
                           This allows 'Sum of Latents' logic for combo prediction.
        """
        batch_size = rna_control.shape[0]
        
        # Embeddings
        c_emb = self.cell_line_embedding(cell_line)
        c_feat = self.cell_line_projection(c_emb).unsqueeze(1) # [B, 1, D]
        rna_feat = self.rna_projection(rna_control).unsqueeze(1) # [B, 1, D]
        
        if custom_latent is not None:
            # Case 1: Use provided latent directly (e.g. z_A + z_B)
            # custom_latent expected shape: [B, D]
            fused_feat = custom_latent
        else:
            # Case 2: Standard Attention Fusion
            if perturb.dim() == 1:
                p_emb = self.perturb_embedding(perturb).unsqueeze(1) # [B, 1, D]
            else:
                p_emb = self.perturb_embedding(perturb) # [B, K, D]
                
            # Stack tokens: [RNA, P1, P2..., Cell]
            tokens = torch.cat([rna_feat, p_emb, c_feat], dim=1)
            
            # Self-Attention
            attn_out, _ = self.feature_fusion(tokens, tokens, tokens)
            
            # Pool attention output (Mean pooling over sequence)
            fused_feat = attn_out.mean(dim=1) # [B, D]
        
        # Construct Context: [RNA_Control, Fused_Feat, Cell_Feat_Squeezed]
        context = torch.cat([rna_control, fused_feat, c_feat.squeeze(1)], dim=1)
        
        return context

    def get_latent(self, rna_control, perturb, cell_line):
        """
        Helper to extract ONLY the fused latent vector (z_sem equivalent).
        Useful for combinatorial arithmetic (z_combo = z_A + z_B).
        """
        # Embeddings
        c_emb = self.cell_line_embedding(cell_line)
        c_feat = self.cell_line_projection(c_emb).unsqueeze(1)
        rna_feat = self.rna_projection(rna_control).unsqueeze(1)
        
        if perturb.dim() == 1:
            p_emb = self.perturb_embedding(perturb).unsqueeze(1)
        else:
            p_emb = self.perturb_embedding(perturb)
            
        tokens = torch.cat([rna_feat, p_emb, c_feat], dim=1)
        attn_out, _ = self.feature_fusion(tokens, tokens, tokens)
        fused_feat = attn_out.mean(dim=1) # [B, D]
        return fused_feat

    def forward(self, rna_control, perturb, cell_line, target_rna=None):
        """
        Training forward pass: Calculates diffusion loss.
        """
        context = self.encode_context(rna_control, perturb, cell_line)
        
        # If target is provided, calculate loss
        if target_rna is not None:
            t = torch.randint(0, self.diffusion.timesteps, (target_rna.shape[0],), device=target_rna.device).long()
            loss = self.diffusion.p_losses(x_start=target_rna, t=t, context=context)
            return loss
        else:
            return None

    @torch.no_grad()
    def sample(self, rna_control, perturb, cell_line, custom_latent=None):
        """
        Inference: Generate prediction
        Supports custom_latent for combinatorial prediction.
        """
        context = self.encode_context(rna_control, perturb, cell_line, custom_latent=custom_latent)
        generated_rna = self.diffusion.sample(context)
        return generated_rna
