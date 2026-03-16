import torch
import torch.nn as nn
import math


class HierarchicalNoiseSchedule(nn.Module):

    def __init__(self, num_levels=2, num_timesteps=1000):
        super().__init__()

        self.num_levels = num_levels
        self.num_timesteps = num_timesteps

        schedules = []

        for k in range(num_levels):
            t = torch.linspace(0, 1, num_timesteps + 1)
            offset = 0.15 * (num_levels - 1 - k)
            t_eff = torch.clamp(t - offset, min=0)
            mask_rate = torch.cos(math.pi / 2 * t_eff) ** 2
            schedules.append(mask_rate)

        self.register_buffer("masking_rates", torch.stack(schedules))

    def get_masking_rate(self, t, hierarchy_labels):

        t = t.long()
        rates = self.masking_rates[:, t].T  # (B, K)

        B, L = hierarchy_labels.shape

        rates = torch.gather(
            rates.unsqueeze(2).expand(-1, -1, L),
            dim=1,
            index=hierarchy_labels.unsqueeze(1)
        ).squeeze(1)

        return rates

    def sample_masks(self, t, hierarchy_labels):
        rates = self.get_masking_rate(t, hierarchy_labels)
        return torch.rand_like(rates) < rates

    def get_hazard_rate(self, t):
        """Returns lambda_k(t) for all hierarchy levels. Shape: (K,)."""

        t = int(t)

        if t == 0:
            return torch.zeros(self.num_levels, device=self.masking_rates.device)

        alpha_t = self.masking_rates[:, t]
        alpha_prev = self.masking_rates[:, t - 1]

        hazard = (alpha_prev - alpha_t) / (1 - alpha_t + 1e-8)

        return torch.clamp(hazard, 0, 1)
