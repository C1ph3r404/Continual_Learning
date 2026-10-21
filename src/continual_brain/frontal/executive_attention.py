"""Executive Attention Engine (Competitive Inverted Slot Routing)."""

import math
import torch
import torch.nn as nn
from continual_brain.config import BrainConfig


class ExecutiveAttentionEngine(nn.Module):
    """
    Attention as an executive reasoning process.
    The Frontal Lobe projects endogenous queries Q_Frontal down onto the Parietal
    Key-Value bank. Slots compete for finite perceptual tokens through inverted slot
    attention (normalizing Softmax across slots), binding distinct entities without slot collapse.
    """

    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config
        self.d_frontal = config.d_frontal
        self.d_parietal = config.d_parietal
        self.scale = 1.0 / math.sqrt(config.d_parietal)
        self.num_iterations = config.slot_iterations

        # Project Frontal slots to query dimension
        self.W_qfront = nn.Linear(config.d_frontal, config.d_parietal, bias=False)

        # Slot update GRU & MLP (recurrent integration of resolved packets)
        self.gru = nn.GRUCell(config.d_parietal, config.d_frontal)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_frontal, config.d_frontal * 2),
            nn.GELU(),
            nn.Linear(config.d_frontal * 2, config.d_frontal)
        )
        self.norm_slots = nn.LayerNorm(config.d_frontal)
        self.norm_input = nn.LayerNorm(config.d_parietal)

    def forward(self, slots: torch.Tensor, K_parietal: torch.Tensor, V_parietal: torch.Tensor):
        """
        Execute competitive inverted slot attention over the Parietal canvas.

        Args:
            slots: (batch_size, n_slots, d_frontal)
            K_parietal: (batch_size, total_tokens, d_parietal)
            V_parietal: (batch_size, total_tokens, d_parietal)
        Returns:
            updated_slots: (batch_size, n_slots, d_frontal)
            attn_weights: (batch_size, n_slots, total_tokens)
        """
        B, N_slots, D_f = slots.shape
        _, T_tokens, D_p = K_parietal.shape

        K_parietal = self.norm_input(K_parietal)
        V_parietal = self.norm_input(V_parietal)

        for _ in range(self.num_iterations):
            slots_norm = self.norm_slots(slots)

            # Endogenous queries from Frontal slots: (B, N_slots, d_parietal)
            Q_frontal = self.W_qfront(slots_norm)

            # Compute affinity dot products: (B, N_slots, T_tokens)
            dots = torch.matmul(Q_frontal, K_parietal.transpose(1, 2)) * self.scale

            # INVERTED SLOT ATTENTION: Normalize Softmax ACROSS SLOTS (dim=1)
            # Mathematically forces tokens to be distributed among slots without collapse
            attn = torch.softmax(dots, dim=1)  # (B, N_slots, T_tokens)

            # Weighted sum of perceptual values (Packets_i)
            # Add small epsilon to denominator to prevent division by zero
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            packets = torch.matmul(attn_norm, V_parietal)  # (B, N_slots, d_parietal)

            # Recurrent GRU step for each slot
            slots_flat = slots.view(B * N_slots, D_f)
            packets_flat = packets.view(B * N_slots, D_p)
            slots_flat = self.gru(packets_flat, slots_flat)

            # Residual MLP refinement
            slots = slots_flat.view(B, N_slots, D_f)
            slots = slots + self.mlp(slots)

        return slots, attn
