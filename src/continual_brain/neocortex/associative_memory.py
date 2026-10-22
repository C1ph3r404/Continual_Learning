"""Neocortical Knowledge System (Decoupled Parametric Key-Value Associative Memory)."""

import math
import torch
import torch.nn as nn
from continual_brain.config import BrainConfig


class NeocortexMemory(nn.Module):
    """
    Continuous Key-Value Associative Memory Substrate.
    Zero discrete graphs or integer tables: strictly parameterized by continuous
    matrices W_key and W_value. Performs forward associative lookup in O(1).
    """

    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config
        self.d_in = config.d_parietal
        self.d_out = config.d_know
        self.n_units = config.n_memory_units
        self.scale = 1.0 / math.sqrt(config.d_parietal)

        # W_key: projects sensory latent space to memory columns
        self.W_key = nn.Parameter(torch.empty(config.d_parietal, config.n_memory_units))
        # W_value: emits associated factual/affordance latents
        self.W_value = nn.Parameter(torch.empty(config.n_memory_units, config.d_know))

        # Semantic field projection head (used in Stage 1 self-supervised reading learning)
        self.semantic_field_head = nn.Linear(config.d_know, config.vocab_size, bias=False)

        # Salience Gate on the Memory Learning Bus (Filters fillers from factual concepts)
        from continual_brain.neocortex.salience_gate import SalienceGate
        self.salience_gate = SalienceGate(config)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.W_key)
        nn.init.orthogonal_(self.W_value)

    def forward(self, query: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Forward associative lookup:
            Recall = Softmax((query * W_key) / (sqrt(d) * temp)) * W_value

        Args:
            query: Tensor of shape (..., d_parietal)
            temperature: Softmax smoothing factor
        Returns:
            recall: Tensor of shape (..., d_know)
        """
        # (..., n_units)
        scores = torch.matmul(query, self.W_key) * self.scale
        attn = torch.softmax(scores / temperature, dim=-1)
        # (..., d_know)
        recall = torch.matmul(attn, self.W_value)
        return recall

    def reconstruct_semantic_field(self, recall: torch.Tensor) -> torch.Tensor:
        """
        Predict semantic co-occurrences / bag-of-words affordances from recalled latent.
        Used during Stage 1 self-supervised learning while reading.
        """
        return self.semantic_field_head(recall)

    def freeze(self):
        """Freeze Neocortex parameters permanently during downstream reasoning (Stage 2)."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self):
        """Unfreeze Neocortex parameters (only during Stage 1 Memory Learning or Sleep Replay)."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()
