
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
    - control expression profile
    - semantic latent z_sem

    context layout:
    [rna_control (n_genes), z_sem (latent_dim)]
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 512),
        dropout: float = 0.1,
        time_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
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
        self.ctrl_proj = nn.Sequential(
            nn.Linear(input_dim, first_dim),
            nn.LayerNorm(first_dim),
            nn.SiLU(),
        )

        blocks = []
        curr_dim = first_dim
        for h_dim in hidden_dims:
            blocks.append(ConditionalResidualBlock(curr_dim, h_dim, time_dim=time_dim, latent_dim=latent_dim, dropout=dropout))
            curr_dim = h_dim
        self.blocks = nn.ModuleList(blocks)

        self.final = nn.Linear(curr_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        rna_control = context[:, : self.input_dim]
        z_sem = context[:, self.input_dim : self.input_dim + self.latent_dim]

        t_emb = self.time_mlp(t)
        h = self.x_proj(x) + self.ctrl_proj(rna_control)

        for block in self.blocks:
            h = block(h, t_emb=t_emb, z_sem=z_sem)

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
        n_cell_lines: int,
        pretrained_weights: Optional[torch.Tensor] = None,
        perturb_dim: int = 200,
        cell_line_dim: int = 32,
        hidden_dims: Sequence[int] = (512, 512, 512),
        dropout: float = 0.1,
        timesteps: int = 1000,
        dose_dim: int = 32,
        time_dim: int = 128,
        drug_dim: int = 0,
        use_atac: bool = False,
        atac_dim: int = 0,
        cond_dropout: float = 0.0,
    ):
        super().__init__()

        self.n_genes = n_genes
        self.perturb_dim = perturb_dim
        self.cell_line_dim = cell_line_dim
        self.dose_dim = dose_dim
        self.drug_dim = drug_dim
        self.use_atac = use_atac and (atac_dim > 0)
        self.atac_dim = atac_dim
        self.cond_dropout = cond_dropout

        if pretrained_weights is not None:
            self.perturb_embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
            perturb_dim = int(pretrained_weights.shape[1])
            self.perturb_dim = perturb_dim
        else:
            self.perturb_embedding = nn.Embedding(n_perturbations, perturb_dim)

        self.cell_line_embedding = nn.Embedding(n_cell_lines, cell_line_dim)

        self.rna_projection = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, perturb_dim),
            nn.LayerNorm(perturb_dim),
        )
        self.cell_line_projection = nn.Sequential(
            nn.Linear(cell_line_dim, perturb_dim),
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

        self.context_dim = n_genes + perturb_dim
        self.denoise_fn = SquidiffStyleDecoder(
            input_dim=n_genes,
            latent_dim=perturb_dim,
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

    def encode_semantic_latent(
        self,
        rna_control: torch.Tensor,
        perturb: torch.Tensor,
        cell_line: torch.Tensor,
        dose: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
        custom_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = rna_control.shape[0]

        if custom_latent is not None:
            return custom_latent

        c_emb = self.cell_line_embedding(cell_line)
        c_feat = self.cell_line_projection(c_emb).unsqueeze(1)
        rna_feat = self.rna_projection(rna_control).unsqueeze(1)
        dose = self._prepare_dose(batch_size, dose, rna_control)
        dose_feat = self.dose_projection(dose).unsqueeze(1)

        p_emb = self._perturb_tokens(perturb, dose=dose)

        tokens = [rna_feat, p_emb, c_feat, dose_feat]

        if self.atac_projection is not None and atac_feat is not None:
            if atac_feat.dim() == 1:
                atac_feat = atac_feat.unsqueeze(0)
            atac_token = self.atac_projection(atac_feat).unsqueeze(1)
            tokens.append(atac_token)

        if self.drug_projection is not None and drug_feat is not None:
            if drug_feat.dim() == 1:
                drug_feat = drug_feat.unsqueeze(0)
            drug_token = self.drug_projection(drug_feat).unsqueeze(1)
            tokens.append(drug_token)

        tokens = torch.cat(tokens, dim=1)
        attn_out, _ = self.feature_fusion(tokens, tokens, tokens)
        fused_feat = attn_out.mean(dim=1)
        fused_feat = self.fusion_norm(fused_feat + self.fusion_mlp(fused_feat))
        return fused_feat

    def encode_context(
        self,
        rna_control: torch.Tensor,
        perturb: torch.Tensor,
        cell_line: torch.Tensor,
        dose: Optional[torch.Tensor] = None,
        custom_latent: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        batch_size = rna_control.shape[0]
        z_sem = self.encode_semantic_latent(
            rna_control=rna_control,
            perturb=perturb,
            cell_line=cell_line,
            dose=dose,
            atac_feat=atac_feat,
            drug_feat=drug_feat,
            custom_latent=custom_latent,
        )

        if force_uncond:
            z_sem = torch.zeros_like(z_sem)
        elif self.training and self.cond_dropout > 0:
            keep = (torch.rand(batch_size, 1, device=rna_control.device) > self.cond_dropout).to(rna_control.dtype)
            z_sem = z_sem * keep

        context = torch.cat([rna_control, z_sem], dim=1)
        return context

    def get_latent(
        self,
        rna_control: torch.Tensor,
        perturb: torch.Tensor,
        cell_line: torch.Tensor,
        dose: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encode_semantic_latent(
            rna_control=rna_control,
            perturb=perturb,
            cell_line=cell_line,
            dose=dose,
            atac_feat=atac_feat,
            drug_feat=drug_feat,
        )

    @staticmethod
    def combine_latents(
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

        out = None
        total_weight = 0.0
        for latent, weight in zip(latents, weights):
            if out is None:
                out = latent * float(weight)
            else:
                out = out + latent * float(weight)
            total_weight += float(weight)

        if mode == "mean":
            out = out / max(total_weight, 1e-8)
        elif mode != "sum":
            raise ValueError(f"未知组合模式: {mode}")
        return out

    def forward(
        self,
        rna_control: torch.Tensor,
        perturb: torch.Tensor,
        cell_line: torch.Tensor,
        target_rna: Optional[torch.Tensor] = None,
        dose: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        context = self.encode_context(
            rna_control=rna_control,
            perturb=perturb,
            cell_line=cell_line,
            dose=dose,
            atac_feat=atac_feat,
            drug_feat=drug_feat,
        )

        if target_rna is None:
            return None

        if t is None:
            t = torch.randint(0, self.diffusion.timesteps, (target_rna.shape[0],), device=target_rna.device).long()
        loss = self.diffusion.p_losses(x_start=target_rna, t=t, context=context, weights=weights)
        return loss

    @torch.no_grad()
    def predict_single(
        self,
        rna_control: torch.Tensor,
        perturb: torch.Tensor,
        cell_line: torch.Tensor,
        dose: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
        sample_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        return self.sample(
            rna_control=rna_control,
            perturb=perturb,
            cell_line=cell_line,
            dose=dose,
            atac_feat=atac_feat,
            drug_feat=drug_feat,
            sample_steps=sample_steps,
            guidance_scale=guidance_scale,
        )

    @torch.no_grad()
    def predict_from_latent(
        self,
        rna_control: torch.Tensor,
        cell_line: torch.Tensor,
        latent: torch.Tensor,
        perturb: Optional[torch.Tensor] = None,
        dose: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
        sample_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        if perturb is None:
            perturb = torch.zeros((rna_control.shape[0],), dtype=torch.long, device=rna_control.device)
        return self.sample(
            rna_control=rna_control,
            perturb=perturb,
            cell_line=cell_line,
            dose=dose,
            custom_latent=latent,
            atac_feat=atac_feat,
            drug_feat=drug_feat,
            sample_steps=sample_steps,
            guidance_scale=guidance_scale,
        )

    @torch.no_grad()
    def sample(
        self,
        rna_control: torch.Tensor,
        perturb: torch.Tensor,
        cell_line: torch.Tensor,
        dose: Optional[torch.Tensor] = None,
        custom_latent: Optional[torch.Tensor] = None,
        atac_feat: Optional[torch.Tensor] = None,
        drug_feat: Optional[torch.Tensor] = None,
        sample_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        context = self.encode_context(
            rna_control=rna_control,
            perturb=perturb,
            cell_line=cell_line,
            dose=dose,
            custom_latent=custom_latent,
            atac_feat=atac_feat,
            drug_feat=drug_feat,
        )

        uncond_context = None
        if guidance_scale != 1.0:
            uncond_context = self.encode_context(
                rna_control=rna_control,
                perturb=perturb,
                cell_line=cell_line,
                dose=dose,
                custom_latent=custom_latent,
                atac_feat=atac_feat,
                drug_feat=drug_feat,
                force_uncond=True,
            )

        generated_rna = self.diffusion.sample(
            context=context,
            sampling_timesteps=sample_steps,
            guidance_scale=guidance_scale,
            uncond_context=uncond_context,
        )
        return generated_rna
