
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.diffusion_core import GaussianDiffusion


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        if half_dim == 0:
            return time[:, None]
        factor = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -factor)
        emb = time[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ConditionalResidualBlock(nn.Module):
    """
    Squidiff-inspired denoise block:
    - project x
    - inject timestep embedding
    - inject semantic latent z_sem
    - residual refinement
    """
    def __init__(self, in_dim: int, out_dim: int, time_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.time_proj = nn.Linear(time_dim, out_dim)
        self.latent_proj = nn.Linear(latent_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = h + self.time_proj(t_emb) + self.latent_proj(z_sem)

        h = self.fc2(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        return h + self.skip(x)


class SquidiffStyleDecoder(nn.Module):
    """
    Denoiser conditioned on:
    - x_t
    - background latent z_bg
    - perturbation-effect latent z_eff

    context layout:
    [z_bg (bg_dim), z_eff (eff_dim)]
    """
    def __init__(
        self,
        input_dim: int,
        bg_dim: int,
        eff_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 512),
        dropout: float = 0.1,
        time_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bg_dim = bg_dim
        self.eff_dim = eff_dim
        self.cond_dim = bg_dim + eff_dim
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        first_dim = hidden_dims[0]
        self.x_proj = nn.Sequential(
            nn.Linear(input_dim, first_dim),
            nn.LayerNorm(first_dim),
            nn.SiLU(),
        )
        self.bg_proj = nn.Sequential(
            nn.Linear(bg_dim, first_dim),
            nn.LayerNorm(first_dim),
            nn.SiLU(),
        )
        self.eff_proj = nn.Sequential(
            nn.Linear(eff_dim, first_dim),
            nn.LayerNorm(first_dim),
            nn.SiLU(),
        )

        blocks = []
        curr_dim = first_dim
        for h_dim in hidden_dims:
            blocks.append(ConditionalResidualBlock(curr_dim, h_dim, time_dim=time_dim, latent_dim=self.cond_dim, dropout=dropout))
            curr_dim = h_dim
        self.blocks = nn.ModuleList(blocks)

        self.final = nn.Linear(curr_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z_bg = context[:, : self.bg_dim]
        z_eff = context[:, self.bg_dim : self.bg_dim + self.eff_dim]
        cond = torch.cat([z_bg, z_eff], dim=1)

        t_emb = self.time_mlp(t)
        h = self.x_proj(x) + self.bg_proj(z_bg) + self.eff_proj(z_eff)

        for block in self.blocks:
            h = block(h, t_emb=t_emb, z_sem=cond)

        return self.final(h)


class PerturbationDiffusionPredictor(nn.Module):
    """
    Diffusion perturbation predictor with Squidiff-style latent injection.

    Core flow:
    1) encode semantic latent z_sem from control + perturb + cell line + optional dose/ATAC/drug
    2) condition diffusion model on [rna_control, z_sem]
    3) support single-gene prediction and latent arithmetic for combinatorial prediction
    """
    def __init__(
        self,
        n_genes: int,
        n_perturbations: int,
        pretrained_weights: Optional[torch.Tensor] = None,
        pretrained_gene_weights: Optional[torch.Tensor] = None,
        output_gene_weights: Optional[torch.Tensor] = None,
        gene_prior_scale: float = 0.1,
        perturb_dim: int = 200,
        hidden_dims: Sequence[int] = (512, 512, 512),
        dropout: float = 0.1,
        timesteps: int = 1000,
        dose_dim: int = 32,
        time_dim: int = 128,
        drug_dim: int = 0,
        use_atac: bool = False,
        atac_dim: int = 0,
        cond_dropout: float = 0.0,
        target_mode: str = "target",
        n_perturb_genes: int = 0,
        task_mode: str = "single_gene",
        n_conditions: int = 0,
    ):
        super().__init__()

        self.n_genes = n_genes
        self.perturb_dim = perturb_dim
        self.dose_dim = dose_dim
        self.drug_dim = drug_dim
        self.use_atac = use_atac and (atac_dim > 0)
        self.atac_dim = atac_dim
        self.cond_dropout = cond_dropout
        self.n_perturb_genes = int(n_perturb_genes) if n_perturb_genes is not None else 0
        if task_mode not in {"single_gene", "translation"}:
            raise ValueError("task_mode must be 'single_gene' or 'translation'")
        self.task_mode = task_mode
        self.n_conditions = int(n_conditions) if n_conditions is not None else 0
        if target_mode not in {"target", "delta"}:
            raise ValueError("target_mode 必须是 'target' 或 'delta'")
        self.target_mode = target_mode

        if pretrained_weights is not None:
            self.perturb_embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
            perturb_dim = int(pretrained_weights.shape[1])
            self.perturb_dim = perturb_dim
        else:
            self.perturb_embedding = nn.Embedding(n_perturbations, perturb_dim)

        self.perturb_gene_embedding = None
        if self.n_perturb_genes > 0:
            if pretrained_gene_weights is not None:
                self.perturb_gene_embedding = nn.Embedding.from_pretrained(
                    pretrained_gene_weights,
                    freeze=False,
                )
                perturb_dim = int(pretrained_gene_weights.shape[1])
                self.perturb_dim = perturb_dim
            else:
                self.perturb_gene_embedding = nn.Embedding(self.n_perturb_genes, perturb_dim)
        self.control_flag_embedding = nn.Embedding(2, perturb_dim)
        self.condition_embedding = None
        if self.task_mode == "translation" and self.n_conditions > 0:
            self.condition_embedding = nn.Embedding(self.n_conditions, perturb_dim)
        self.source_flag_embedding = nn.Embedding(2, perturb_dim)

        self.rna_projection = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, perturb_dim),
            nn.LayerNorm(perturb_dim),
        )
        self.dose_projection = nn.Sequential(
            nn.Linear(1, dose_dim),
            nn.SiLU(),
            nn.Linear(dose_dim, perturb_dim),
            nn.LayerNorm(perturb_dim),
        )

        self.drug_projection = None
        if self.drug_dim > 0:
            self.drug_projection = nn.Sequential(
                nn.Linear(self.drug_dim, perturb_dim),
                nn.LayerNorm(perturb_dim),
                nn.SiLU(),
                nn.Linear(perturb_dim, perturb_dim),
                nn.LayerNorm(perturb_dim),
            )

        self.atac_projection = None
        if self.use_atac:
            self.atac_projection = nn.Sequential(
                nn.Linear(self.atac_dim, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Linear(512, perturb_dim),
                nn.LayerNorm(perturb_dim),
            )

        self.feature_fusion = nn.MultiheadAttention(embed_dim=perturb_dim, num_heads=4, batch_first=True)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(perturb_dim, perturb_dim),
            nn.LayerNorm(perturb_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(perturb_dim, perturb_dim),
        )
        self.fusion_norm = nn.LayerNorm(perturb_dim)
        self.semantic_joint_encoder = nn.Sequential(
            nn.Linear(perturb_dim * 4, perturb_dim),
            nn.LayerNorm(perturb_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(perturb_dim, perturb_dim),
            nn.LayerNorm(perturb_dim),
        )
        self.semantic_blend_gate = nn.Sequential(
            nn.Linear(perturb_dim * 2, perturb_dim),
            nn.SiLU(),
            nn.Linear(perturb_dim, perturb_dim),
            nn.Sigmoid(),
        )
        self.latent_composer = nn.Sequential(
            nn.Linear(perturb_dim * 4, perturb_dim),
            nn.LayerNorm(perturb_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(perturb_dim, perturb_dim),
        )
        self.latent_gate = nn.Sequential(
            nn.Linear(perturb_dim * 2, perturb_dim),
            nn.SiLU(),
            nn.Linear(perturb_dim, perturb_dim),
            nn.Sigmoid(),
        )

        self.background_encoder = nn.Sequential(
            nn.Linear(perturb_dim * 2, perturb_dim),
            nn.LayerNorm(perturb_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(perturb_dim, perturb_dim),
            nn.LayerNorm(perturb_dim),
        )
        self.perturbation_encoder = nn.Sequential(
            nn.Linear(perturb_dim * 3, perturb_dim),
            nn.LayerNorm(perturb_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(perturb_dim, perturb_dim),
            nn.LayerNorm(perturb_dim),
        )
        self.effect_composer = nn.Sequential(
            nn.Linear(perturb_dim * 2, perturb_dim),
            nn.LayerNorm(perturb_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(perturb_dim, perturb_dim),
            nn.LayerNorm(perturb_dim),
        )
        self.effect_gate = nn.Sequential(
            nn.Linear(perturb_dim * 2, perturb_dim),
            nn.SiLU(),
            nn.Linear(perturb_dim, perturb_dim),
            nn.Sigmoid(),
        )

        self.context_dim = perturb_dim * 2
        self.use_output_gene_prior = output_gene_weights is not None
        self.gene_prior_scale = float(gene_prior_scale)
        if self.use_output_gene_prior:
            if output_gene_weights.dim() != 2:
                raise ValueError("output_gene_weights must be a 2D tensor: [n_genes, gene_emb_dim]")
            if output_gene_weights.shape[0] != n_genes:
                raise ValueError(
                    f"output_gene_weights rows must match n_genes: "
                    f"{output_gene_weights.shape[0]} vs {n_genes}"
                )

            gene_emb_dim = int(output_gene_weights.shape[1])
            self.register_buffer(
                "output_gene_embedding",
                output_gene_weights.detach().float().clone(),
            )
            self.output_gene_proj = nn.Sequential(
                nn.Linear(gene_emb_dim, hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                nn.SiLU(),
                nn.Linear(hidden_dims[0], hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
            )
            self.context_gene_query = nn.Sequential(
                nn.Linear(self.context_dim, hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                nn.SiLU(),
                nn.Linear(hidden_dims[0], hidden_dims[0]),
            )
            self.gene_prior_norm = nn.LayerNorm(n_genes)
        else:
            self.output_gene_embedding = None
            self.output_gene_proj = None
            self.context_gene_query = None
            self.gene_prior_norm = None
        self.mean_delta_head = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], n_genes),
        )
        self.residual_sample_scale = 0.0
        self.denoise_fn = SquidiffStyleDecoder(
            input_dim=n_genes,
            bg_dim=perturb_dim,
            eff_dim=perturb_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            time_dim=time_dim,
        )
        self.diffusion = GaussianDiffusion(
            model=self.denoise_fn,
            input_dim=n_genes,
            timesteps=timesteps,
            objective="pred_x0",
        )

    def _prepare_dose(self, batch_size: int, dose: Optional[torch.Tensor], ref_tensor: torch.Tensor) -> torch.Tensor:
        if dose is None:
            dose = torch.zeros((batch_size, 1), device=ref_tensor.device, dtype=ref_tensor.dtype)
        elif dose.dim() == 1:
            dose = dose.unsqueeze(1)
        return dose

    def _perturb_tokens(
        self,
        perturb: torch.Tensor,
        dose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if perturb.dim() == 1:
            p_emb = self.perturb_embedding(perturb).unsqueeze(1)
        else:
            p_emb = self.perturb_embedding(perturb)

        if dose is not None:
            if dose.dim() == 1:
                dose = dose.unsqueeze(1)
            if dose.dim() == 2 and dose.shape[1] == 1:
                p_emb = p_emb * (1.0 + dose.unsqueeze(-1))
            elif dose.dim() == 2 and dose.shape[1] == p_emb.shape[1]:
                p_emb = p_emb * (1.0 + dose.unsqueeze(-1))
        return p_emb

    def encode_background(
        self,
        rna_control: torch.Tensor,
        atac_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rna_feat = self.rna_projection(rna_control)
        if self.atac_projection is not None and atac_feat is not None:
            if atac_feat.dim() == 1:
                atac_feat = atac_feat.unsqueeze(0)
            atac_token = self.atac_projection(atac_feat)
        else:
            atac_token = torch.zeros_like(rna_feat)
        z_bg = self.background_encoder(torch.cat([rna_feat, atac_token], dim=1))
        return z_bg

    def encode_perturbation(
        self,
        perturb: torch.Tensor,
        perturb_gene_idx: Optional[torch.Tensor] = None,
        is_control: Optional[torch.Tensor] = None,
        condition_id: Optional[torch.Tensor] = None,
        source_flag: Optional[torch.Tensor] = None,
        dose: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = perturb.shape[0]
        if self.task_mode == "translation" and self.condition_embedding is not None and condition_id is not None:
            cond_emb = self.condition_embedding(condition_id)
            if source_flag is None:
                source_flag = torch.zeros_like(condition_id)
            src_emb = self.source_flag_embedding(source_flag.long())
            base_feat = cond_emb + src_emb
            ref = base_feat
        elif self.perturb_gene_embedding is not None and perturb_gene_idx is not None:
            gene_emb = self.perturb_gene_embedding(perturb_gene_idx)
            if is_control is None:
                is_control = torch.zeros_like(perturb_gene_idx)
            ctrl_flag_emb = self.control_flag_embedding(is_control.long())
            base_feat = gene_emb + ctrl_flag_emb
            ref = base_feat
        else:
            p_emb = self._perturb_tokens(perturb, dose=dose)
            base_feat = p_emb.mean(dim=1)
            ref = base_feat

        dose = self._prepare_dose(batch_size, dose, ref)
        dose_feat = self.dose_projection(dose)
        if self.drug_projection is not None and drug_feat is not None:
            if drug_feat.dim() == 1:
                drug_feat = drug_feat.unsqueeze(0)
            drug_token = self.drug_projection(drug_feat)
        else:
            drug_token = torch.zeros_like(base_feat)
        z_pert = self.perturbation_encoder(torch.cat([base_feat, dose_feat, drug_token], dim=1))
        return z_pert

    def compose_effect(self, z_bg: torch.Tensor, z_pert: torch.Tensor) -> torch.Tensor:
        composed = self.effect_composer(torch.cat([z_bg, z_pert], dim=1))
        gate = self.effect_gate(torch.cat([z_bg, z_pert], dim=1))
        z_eff = gate * composed + (1.0 - gate) * z_pert
        return z_eff

    def output_gene_prior(self, context: torch.Tensor) -> torch.Tensor:
        if not self.use_output_gene_prior:
            return torch.zeros(
                context.shape[0],
                self.n_genes,
                device=context.device,
                dtype=context.dtype,
            )

        gene_emb = self.output_gene_embedding.to(device=context.device, dtype=context.dtype)
        gene_feat = self.output_gene_proj(gene_emb)
        query = self.context_gene_query(context)
        prior = torch.matmul(query, gene_feat.t())
        prior = prior / (gene_feat.shape[1] ** 0.5)
        prior = self.gene_prior_norm(prior)
        return self.gene_prior_scale * prior

    def encode_semantic_latent(
        self,
        rna_control: torch.Tensor,
        perturb: torch.Tensor,
        dose: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
        custom_latent: Optional[torch.Tensor] = None,
        perturb_gene_idx: Optional[torch.Tensor] = None,
        is_control: Optional[torch.Tensor] = None,
        condition_id: Optional[torch.Tensor] = None,
        source_flag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if custom_latent is not None:
            return custom_latent
        z_bg = self.encode_background(rna_control=rna_control, atac_feat=atac_feat)
        z_pert = self.encode_perturbation(
            perturb=perturb,
            perturb_gene_idx=perturb_gene_idx,
            is_control=is_control,
            condition_id=condition_id,
            source_flag=source_flag,
            dose=dose,
            drug_feat=drug_feat,
        )
        z_eff = self.compose_effect(z_bg=z_bg, z_pert=z_pert)
        return z_eff

    def encode_context(
        self,
        rna_control: torch.Tensor,
        perturb: torch.Tensor,
        dose: Optional[torch.Tensor] = None,
        custom_latent: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
        force_uncond: bool = False,
        perturb_gene_idx: Optional[torch.Tensor] = None,
        is_control: Optional[torch.Tensor] = None,
        condition_id: Optional[torch.Tensor] = None,
        source_flag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = rna_control.shape[0]
        z_bg = self.encode_background(
            rna_control=rna_control,
            atac_feat=atac_feat,
        )
        z_sem = self.encode_semantic_latent(
            rna_control=rna_control,
            perturb=perturb,
            dose=dose,
            atac_feat=atac_feat,
            drug_feat=drug_feat,
            custom_latent=custom_latent,
            perturb_gene_idx=perturb_gene_idx,
            is_control=is_control,
            condition_id=condition_id,
            source_flag=source_flag,
        )

        if force_uncond:
            z_sem = torch.zeros_like(z_sem)
            z_bg = torch.zeros_like(z_bg)
        elif self.training and self.cond_dropout > 0:
            keep = (torch.rand(batch_size, 1, device=rna_control.device) > self.cond_dropout).to(rna_control.dtype)
            z_sem = z_sem * keep
            z_bg = z_bg * keep

        context = torch.cat([z_bg, z_sem], dim=1)
        return context

    def get_latent(
        self,
        rna_control: torch.Tensor,
        perturb: torch.Tensor,
        dose: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
        perturb_gene_idx: Optional[torch.Tensor] = None,
        is_control: Optional[torch.Tensor] = None,
        condition_id: Optional[torch.Tensor] = None,
        source_flag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encode_semantic_latent(
            rna_control=rna_control,
            perturb=perturb,
            dose=dose,
            atac_feat=atac_feat,
            drug_feat=drug_feat,
            perturb_gene_idx=perturb_gene_idx,
            is_control=is_control,
            condition_id=condition_id,
            source_flag=source_flag,
        )

    def combine_latents(
        self,
        latents: Sequence[torch.Tensor],
        weights: Optional[Sequence[float]] = None,
        mode: str = "sum",
    ) -> torch.Tensor:
        if len(latents) == 0:
            raise ValueError("latents 不能为空。")
        if weights is None:
            weights = [1.0] * len(latents)
        if len(weights) != len(latents):
            raise ValueError("weights 长度必须和 latents 一致。")

        stacked = torch.stack(latents, dim=1)  # [B, K, D]
        weight_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
        weight_sum = weight_tensor.sum().clamp(min=1e-8)
        norm_weight = weight_tensor / weight_sum

        weighted = stacked * norm_weight.view(1, -1, 1)
        out = weighted.sum(dim=1)

        if mode == "mean":
            out = stacked.mean(dim=1)
        elif mode == "adaptive":
            if stacked.shape[1] == 1:
                return out

            pair_terms = []
            k = stacked.shape[1]
            for i in range(k):
                for j in range(i + 1, k):
                    li = stacked[:, i, :]
                    lj = stacked[:, j, :]
                    pair_input = torch.cat([li, lj, li * lj, torch.abs(li - lj)], dim=1)
                    pair_terms.append(self.latent_composer(pair_input))

            pair_agg = torch.stack(pair_terms, dim=1).mean(dim=1)
            avg_latent = stacked.mean(dim=1)
            gate = self.latent_gate(torch.cat([out, avg_latent], dim=1))
            out = gate * out + (1.0 - gate) * (out + pair_agg)
        elif mode != "sum":
            raise ValueError(f"未知组合模式: {mode}")
        return out

    @staticmethod
    def interpolate_latents(
        z_start: torch.Tensor,
        z_end: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        if steps < 2:
            raise ValueError("steps 必须 >= 2。")
        alphas = torch.linspace(0.0, 1.0, steps=steps, device=z_start.device, dtype=z_start.dtype).view(steps, 1, 1)
        return (1.0 - alphas) * z_start.unsqueeze(0) + alphas * z_end.unsqueeze(0)

    def forward(
        self,
        rna_control: torch.Tensor,
        perturb: torch.Tensor,
        target_rna: Optional[torch.Tensor] = None,
        dose: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        return_details: bool = False,
        perturb_gene_idx: Optional[torch.Tensor] = None,
        is_control: Optional[torch.Tensor] = None,
        condition_id: Optional[torch.Tensor] = None,
        source_flag: Optional[torch.Tensor] = None,
        mean_loss_weight: float = 10.0,
        diff_loss_weight: float = 0.1,
        mean_only: bool = False,
    ) -> Optional[torch.Tensor]:
        context = self.encode_context(
            rna_control=rna_control,
            perturb=perturb,
            dose=dose,
            atac_feat=atac_feat,
            drug_feat=drug_feat,
            perturb_gene_idx=perturb_gene_idx,
            is_control=is_control,
            condition_id=condition_id,
            source_flag=source_flag,
        )

        if target_rna is None:
            return None

        target_for_diffusion = target_rna
        if self.target_mode == "delta":
            target_for_diffusion = target_rna - rna_control

        mean_delta = self.mean_delta_head(context) + self.output_gene_prior(context)
        if self.target_mode == "delta":
            residual_target = target_for_diffusion - mean_delta
        else:
            residual_target = target_for_diffusion - mean_delta

        if t is None:
            t = torch.randint(0, self.diffusion.timesteps, (target_rna.shape[0],), device=target_rna.device).long()
        diff_out = self.diffusion.p_losses(
            x_start=residual_target,
            t=t,
            context=context,
            weights=weights,
            return_details=return_details,
        )
        mean_loss = torch.mean((mean_delta - target_for_diffusion) ** 2)
        if return_details:
            diff_loss, details = diff_out
            residual_pred = details["pred_x0"]
            pred_x0 = mean_delta + residual_pred
            if mean_only:
                loss = mean_loss
            else:
                loss = diff_loss_weight * diff_loss + mean_loss_weight * mean_loss
            if self.target_mode == "delta":
                details["pred_target"] = pred_x0 + rna_control
                details["target_target"] = target_rna
                details["pred_delta"] = pred_x0
                details["target_delta"] = target_for_diffusion
            else:
                details["pred_target"] = pred_x0
                details["target_target"] = target_rna
            details["mean_delta"] = mean_delta
            details["residual_pred"] = residual_pred
            return loss, details
        diff_loss = diff_out
        if mean_only:
            return mean_loss
        return diff_loss_weight * diff_loss + mean_loss_weight * mean_loss

    @torch.no_grad()
    def predict_single(
        self,
        rna_control: torch.Tensor,
        perturb: torch.Tensor,
        dose: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
        sample_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        perturb_gene_idx: Optional[torch.Tensor] = None,
        is_control: Optional[torch.Tensor] = None,
        condition_id: Optional[torch.Tensor] = None,
        source_flag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.sample(
            rna_control=rna_control,
            perturb=perturb,
            dose=dose,
            atac_feat=atac_feat,
            drug_feat=drug_feat,
            sample_steps=sample_steps,
            guidance_scale=guidance_scale,
            perturb_gene_idx=perturb_gene_idx,
            is_control=is_control,
            condition_id=condition_id,
            source_flag=source_flag,
        )

    @torch.no_grad()
    def predict_from_latent(
        self,
        rna_control: torch.Tensor,
        latent: torch.Tensor,
        perturb: Optional[torch.Tensor] = None,
        dose: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
        sample_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        perturb_gene_idx: Optional[torch.Tensor] = None,
        is_control: Optional[torch.Tensor] = None,
        condition_id: Optional[torch.Tensor] = None,
        source_flag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if perturb is None:
            perturb = torch.zeros((rna_control.shape[0],), dtype=torch.long, device=rna_control.device)
        return self.sample(
            rna_control=rna_control,
            perturb=perturb,
            dose=dose,
            custom_latent=latent,
            atac_feat=atac_feat,
            drug_feat=drug_feat,
            sample_steps=sample_steps,
            guidance_scale=guidance_scale,
            perturb_gene_idx=perturb_gene_idx,
            is_control=is_control,
            condition_id=condition_id,
            source_flag=source_flag,
        )

    @torch.no_grad()
    def sample(
        self,
        rna_control: torch.Tensor,
        perturb: torch.Tensor,
        dose: Optional[torch.Tensor] = None,
        custom_latent: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
        sample_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        perturb_gene_idx: Optional[torch.Tensor] = None,
        is_control: Optional[torch.Tensor] = None,
        condition_id: Optional[torch.Tensor] = None,
        source_flag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context = self.encode_context(
            rna_control=rna_control,
            perturb=perturb,
            dose=dose,
            custom_latent=custom_latent,
            atac_feat=atac_feat,
            drug_feat=drug_feat,
            perturb_gene_idx=perturb_gene_idx,
            is_control=is_control,
            condition_id=condition_id,
            source_flag=source_flag,
        )

        uncond_context = None
        mean_delta = self.mean_delta_head(context) + self.output_gene_prior(context)
        if guidance_scale != 1.0:
            uncond_context = self.encode_context(
                rna_control=rna_control,
                perturb=perturb,
                dose=dose,
                custom_latent=custom_latent,
                atac_feat=atac_feat,
                drug_feat=drug_feat,
                force_uncond=True,
                perturb_gene_idx=perturb_gene_idx,
                is_control=is_control,
                condition_id=condition_id,
                source_flag=source_flag,
            )

        residual = self.diffusion.sample(
            context=context,
            sampling_timesteps=sample_steps,
            guidance_scale=guidance_scale,
            uncond_context=uncond_context,
        )
        if self.target_mode == "delta":
            generated_rna = rna_control + mean_delta + self.residual_sample_scale * residual
        else:
            generated_rna = mean_delta + self.residual_sample_scale * residual
        return generated_rna
