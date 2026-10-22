"""Salience Gate on the Memory Learning Bus.

Decides which sensory/linguistic token representations contain factual/relational
content worth committing to Neocortical associative memory, filtering out
grammatical filler words, prepositions, and delimiters.
"""

import torch
import torch.nn as nn
from continual_brain.config import BrainConfig


class SalienceGate(nn.Module):
    """
    Dedicated 2-layer MLP classifier operating on Wernicke language latents H_lang.
    Outputs a continuous salience probability s_t in [0, 1] for every token position.
    """

    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config
        d_in = config.d_lang
        d_hidden = getattr(config, "d_gate_hidden", 128)

        self.net = nn.Sequential(

            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, 1)
        )

    def forward(self, h_lang: torch.Tensor) -> torch.Tensor:
        """
        Compute salience probabilities.

        Args:
            h_lang: Tensor of shape (B, L, d_lang)
        Returns:
            salience_probs: Tensor of shape (B, L) with values in (0, 1)
        """
        h_lang = h_lang.to(dtype=self.net[0].weight.dtype)
        logits = self.net(h_lang).squeeze(-1)  # (B, L)
        return torch.sigmoid(logits)

    def get_salience_logits(self, h_lang: torch.Tensor) -> torch.Tensor:
        """Raw logits for binary cross-entropy calibration loss."""
        h_lang = h_lang.to(dtype=self.net[0].weight.dtype)
        return self.net(h_lang).squeeze(-1)
