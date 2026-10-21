"""Dorsolateral Prefrontal Cortex (dlPFC) Working Memory Slot Buffer."""

import torch
import torch.nn as nn
from continual_brain.config import BrainConfig


class WorkingMemoryBuffer(nn.Module):
    """
    Active, persistent firing state of the dlPFC (Patricia Goldman-Rakic model).
    Maintains a rolling buffer of N_slots slots in native dimension d_frontal.
    """

    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config
        self.n_slots = config.n_slots
        self.d_frontal = config.d_frontal

        # Initial learnable slot prior distribution (mu, sigma)
        self.slot_mu = nn.Parameter(torch.randn(1, 1, config.d_frontal))
        self.slot_sigma = nn.Parameter(torch.rand(1, 1, config.d_frontal) + 0.1)

    def init_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample initial working memory state for a new task or sequence.
        Returns:
            slots: (batch_size, n_slots, d_frontal)
        """
        mu = self.slot_mu.expand(batch_size, self.n_slots, -1)
        sigma = self.slot_sigma.expand(batch_size, self.n_slots, -1)
        slots = mu + torch.randn_like(mu) * sigma
        return slots
