import numpy as np
import torch


class UniformTimestepSampler:
    def __init__(self, diffusion_steps: int):
        self.diffusion_steps = diffusion_steps

    def sample(self, batch_size: int, device):
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=device).long()
        weights = torch.ones(batch_size, device=device, dtype=torch.float32)
        return t, weights

    def update_with_losses(self, timesteps, losses):
        return None


class LossSecondMomentResampler:
    """
    Lightweight loss-aware sampler inspired by Squidiff/OpenAI diffusion training:
    sample timesteps proportional to sqrt(E[loss^2]_t).
    """
    def __init__(self, diffusion_steps: int, history_per_term: int = 10, uniform_prob: float = 0.001):
        self.diffusion_steps = diffusion_steps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self.loss_history = [[] for _ in range(diffusion_steps)]

    def weights(self):
        second_moment = np.ones(self.diffusion_steps, dtype=np.float64)
        for i, hist in enumerate(self.loss_history):
            if hist:
                vals = np.array(hist, dtype=np.float64)
                second_moment[i] = np.sqrt(np.mean(vals ** 2) + 1e-12)
        probs = second_moment / second_moment.sum()
        probs = probs * (1.0 - self.uniform_prob) + self.uniform_prob / self.diffusion_steps
        return probs

    def sample(self, batch_size: int, device):
        probs = self.weights()
        t_np = np.random.choice(self.diffusion_steps, size=batch_size, p=probs)
        t = torch.tensor(t_np, device=device, dtype=torch.long)
        p = torch.tensor(probs[t_np], device=device, dtype=torch.float32)
        weights = 1.0 / (self.diffusion_steps * p)
        return t, weights

    def update_with_losses(self, timesteps, losses):
        ts = timesteps.detach().cpu().numpy().tolist()
        ls = losses.detach().cpu().numpy().tolist()
        for t, l in zip(ts, ls):
            h = self.loss_history[t]
            h.append(float(l))
            if len(h) > self.history_per_term:
                h.pop(0)
