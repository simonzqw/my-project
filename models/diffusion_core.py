import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class GaussianDiffusion(nn.Module):
    """
    1D Gaussian Diffusion Model core logic (DDPM/DDIM)
    Adapted for single-cell transcriptomics data (continuous vectors).
    """
    def __init__(self, model, input_dim, timesteps=1000, beta_schedule='cosine', objective='pred_noise'):
        super().__init__()
        self.model = model  # The denoising network (e.g., MLP)
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.objective = objective  # 'pred_noise' or 'pred_x0' or 'pred_v'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # Helper function to register buffer (automatically moves to device)
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (forward process)
        x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, context, noise=None, weights=None, return_details=False):
        """
        Calculate loss for training
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Forward process: generate noisy sample x_t
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Reverse process: predict noise (or x0) from x_t and context
        model_out = self.model(x_noisy, t, context)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # Simple MSE loss
        loss = F.mse_loss(model_out, target, reduction='none').mean(dim=1)
        if weights is not None:
            loss = loss * weights
        loss = loss.mean()
        if not return_details:
            return loss

        if self.objective == 'pred_x0':
            pred_x0 = model_out
        elif self.objective == 'pred_noise':
            pred_x0 = self.predict_start_from_noise(x_noisy, t, model_out)
        else:
            pred_x0 = model_out
        return loss, {'pred_x0': pred_x0, 'target_x0': x_start, 'x_noisy': x_noisy, 't': t}

    def model_predictions(self, x, t, context, guidance_scale=1.0, uncond_context=None):
        model_out = self.model(x, t, context)
        if uncond_context is not None and guidance_scale != 1.0:
            uncond_out = self.model(x, t, uncond_context)
            model_out = uncond_out + guidance_scale * (model_out - uncond_out)
        return model_out

    def p_sample(self, x, t, context, t_index, guidance_scale=1.0, uncond_context=None):
        """
        Sample from the model (reverse process) - Single step
        """
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape) # Note: this might need check
        
        # We use the equation: x_{t-1} = 1/sqrt(alpha) * (x_t - beta/sqrt(1-alpha_bar) * epsilon) + sigma * z
        # But first, let's get the model prediction (epsilon or x0)
        model_out = self.model_predictions(
            x, t, context, guidance_scale=guidance_scale, uncond_context=uncond_context
        )
        
        if self.objective == 'pred_noise':
            pred_noise = model_out
            x_recon = self.predict_start_from_noise(x, t, pred_noise)
        elif self.objective == 'pred_x0':
            x_recon = model_out
        
        x_recon.clamp_(-10., 10.) # Clip for stability (optional but recommended for gene expr)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        
        noise = torch.randn_like(x) if t_index > 0 else 0.
        return model_mean + torch.exp(0.5 * posterior_log_variance) * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    @torch.no_grad()
    def ddim_sample(self, context, batch_size=None, sampling_timesteps=50, eta=0.0, guidance_scale=1.0, uncond_context=None):
        if batch_size is None:
            batch_size = context.shape[0]
        device = next(self.parameters()).device
        x = torch.randn((batch_size, self.input_dim), device=device)
        times = torch.linspace(self.timesteps - 1, 0, steps=sampling_timesteps, device=device).long()
        for i, t_curr in enumerate(times):
            t = torch.full((batch_size,), int(t_curr.item()), device=device, dtype=torch.long)
            model_out = self.model_predictions(
                x, t, context, guidance_scale=guidance_scale, uncond_context=uncond_context
            )
            if self.objective == 'pred_noise':
                pred_x0 = self.predict_start_from_noise(x, t, model_out)
            else:
                pred_x0 = model_out
            pred_x0 = pred_x0.clamp(-10.0, 10.0)
            if i == len(times) - 1:
                x = pred_x0
                break
            t_next = torch.full((batch_size,), int(times[i + 1].item()), device=device, dtype=torch.long)
            alpha = self._extract(self.alphas_cumprod, t, x.shape)
            alpha_next = self._extract(self.alphas_cumprod, t_next, x.shape)
            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next))
            noise = torch.randn_like(x)
            pred_noise = (x - torch.sqrt(alpha) * pred_x0) / torch.sqrt((1 - alpha).clamp(min=1e-8))
            x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt((1 - alpha_next - sigma ** 2).clamp(min=0.0)) * pred_noise + sigma * noise
        return x

    @torch.no_grad()
    def sample(self, context, batch_size=None, sampling_timesteps=None, guidance_scale=1.0, uncond_context=None):
        """
        Generate samples from pure noise
        """
        if batch_size is None:
            batch_size = context.shape[0]
            
        device = next(self.parameters()).device
        
        # Start from pure noise
        img = torch.randn((batch_size, self.input_dim), device=device)
        
        if sampling_timesteps is not None and sampling_timesteps < self.timesteps:
            return self.ddim_sample(
                context,
                batch_size=batch_size,
                sampling_timesteps=sampling_timesteps,
                guidance_scale=guidance_scale,
                uncond_context=uncond_context,
            )

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(
                img, t, context, i, guidance_scale=guidance_scale, uncond_context=uncond_context
            )
            
        return img
