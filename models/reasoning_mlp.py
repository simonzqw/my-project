import torch
import torch.nn as nn


class PerturbationPredictor(nn.Module):
    """
    Transformer-based generative perturbation predictor.

    Core idea:
    - Build token sequence from control state + perturbation/cell/dose conditions.
    - Use TransformerEncoder as the main interaction backbone.
    - Decode delta expression and add residual to control input.
    """
    def __init__(
        self,
        n_genes,
        n_perturbations,
        n_cell_lines,
        pretrained_weights=None,
        perturb_dim=200,
        cell_line_dim=32,
        drug_dim=2048,
        hidden_dims=[512, 1024, 2048],
        dropout=0.2,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_ff=1024,
        n_ctrl_tokens=8,
        atac_dim=0
    ):
        super(PerturbationPredictor, self).__init__()
        self.n_genes = n_genes
        self.perturb_dim = perturb_dim
        self.use_semantic_perturb = pretrained_weights is not None
        self.n_ctrl_tokens = n_ctrl_tokens
        self.d_model = d_model
        self.atac_dim = atac_dim

        # ===== 1) Perturbation branch =====
        self.perturb_embedding = nn.Embedding(n_perturbations, perturb_dim)
        if self.use_semantic_perturb:
            self.register_buffer("perturb_feature_bank", pretrained_weights.float())
            in_dim = pretrained_weights.shape[1]
            self.perturb_encoder = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Linear(512, perturb_dim),
                nn.LayerNorm(perturb_dim)
            )
        else:
            self.perturb_encoder = None

        self.drug_projection = nn.Sequential(
            nn.Linear(drug_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, perturb_dim),
            nn.LayerNorm(perturb_dim)
        )

        self.dose_scaler = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, perturb_dim),
            nn.Sigmoid()
        )

        self.cell_line_embedding = nn.Embedding(n_cell_lines, cell_line_dim)
        self.cell_line_projection = nn.Sequential(
            nn.Linear(cell_line_dim, perturb_dim),
            nn.LayerNorm(perturb_dim),
            nn.Dropout(0.2)
        )

        # ===== 2) Control RNA tokenization =====
        # [B, G] -> [B, T, d_model] via learned linear tokenizer.
        self.ctrl_tokenizer = nn.Linear(n_genes, n_ctrl_tokens * d_model)

        # Condition projections to d_model
        self.perturb_to_dmodel = nn.Sequential(
            nn.Linear(perturb_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.cell_to_dmodel = nn.Sequential(
            nn.Linear(perturb_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.dose_to_dmodel = nn.Sequential(
            nn.Linear(perturb_dim, d_model),
            nn.LayerNorm(d_model)
        )
        if atac_dim is not None and atac_dim > 0:
            self.atac_encoder = nn.Sequential(
                nn.Linear(atac_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, perturb_dim),
                nn.LayerNorm(perturb_dim)
            )
            self.atac_to_dmodel = nn.Sequential(
                nn.Linear(perturb_dim, d_model),
                nn.LayerNorm(d_model)
            )
            self.use_atac = True
        else:
            self.atac_encoder = None
            self.atac_to_dmodel = None
            self.use_atac = False
        self.null_atac_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learned special tokens + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        max_seq_len = n_ctrl_tokens + 5  # CLS + ctrl_tokens + pert + cell + dose + atac
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

        # ===== 3) Delta decoder =====
        # Use CLS summary + mean ctrl token summary to predict delta.
        decoder_in = d_model * 2
        layers = []
        curr = decoder_in
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(curr, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ]
            curr = h_dim
        layers.append(nn.Linear(curr, n_genes))
        self.delta_head = nn.Sequential(*layers)

    def _build_perturb_feature(self, perturb, drug_feat=None, dose=None):
        if drug_feat is not None:
            p_feat = self.drug_projection(drug_feat)
            if dose is not None:
                if dose.dim() == 1:
                    dose = dose.unsqueeze(1)
                scale = self.dose_scaler(dose) * 2.0
                p_feat = p_feat * scale
            return p_feat

        if self.use_semantic_perturb:
            p_raw = self.perturb_feature_bank[perturb]
            return self.perturb_encoder(p_raw)
        return self.perturb_embedding(perturb)

    def forward(self, rna_control, perturb, cell_line, drug_feat=None, dose=None, atac_feat=None):
        if rna_control.dim() == 3 and rna_control.shape[1] == 1:
            rna_control = rna_control.squeeze(1)

        bsz = rna_control.size(0)

        # Condition features
        p_feat = self._build_perturb_feature(perturb, drug_feat=drug_feat, dose=dose)
        c_emb = self.cell_line_embedding(cell_line)
        c_feat = self.cell_line_projection(c_emb)

        if dose is None:
            dose = torch.zeros((bsz, 1), device=rna_control.device, dtype=rna_control.dtype)
        elif dose.dim() == 1:
            dose = dose.unsqueeze(1)
        dose_feat = self.dose_scaler(dose)
        if self.use_atac and atac_feat is not None:
            atac_latent = self.atac_encoder(atac_feat)
            atac_token = self.atac_to_dmodel(atac_latent).unsqueeze(1)
        else:
            atac_token = self.null_atac_token.expand(bsz, -1, -1)

        # Control tokens
        ctrl_tokens = self.ctrl_tokenizer(rna_control).view(bsz, self.n_ctrl_tokens, self.d_model)
        p_token = self.perturb_to_dmodel(p_feat).unsqueeze(1)
        c_token = self.cell_to_dmodel(c_feat).unsqueeze(1)
        d_token = self.dose_to_dmodel(dose_feat).unsqueeze(1)
        cls_token = self.cls_token.expand(bsz, -1, -1)

        seq = torch.cat([cls_token, ctrl_tokens, p_token, c_token, d_token, atac_token], dim=1)
        seq = seq + self.pos_embed[:, :seq.size(1), :]

        h = self.transformer(seq)
        h = self.final_norm(h)

        cls_h = h[:, 0, :]
        ctrl_h = h[:, 1:1 + self.n_ctrl_tokens, :].mean(dim=1)
        fused = torch.cat([cls_h, ctrl_h], dim=1)

        delta = self.delta_head(fused)
        rna_predicted = rna_control + delta
        return rna_predicted

    def freeze_perturbation_embedding(self, freeze=True):
        """Keep compatibility with training script."""
        if self.use_semantic_perturb:
            # Semantic mode trains perturb_encoder instead of raw ID embedding.
            return
        self.perturb_embedding.weight.requires_grad = not freeze
        state = "冻结" if freeze else "解冻"
        print(f">>> 扰动 Embedding 层已 {state}")
