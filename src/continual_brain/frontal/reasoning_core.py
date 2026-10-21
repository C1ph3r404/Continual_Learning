"""Frontal Lobe Decision Core (Motor-Linguistic Planning & Broca Next-Token Generator)."""

import torch
import torch.nn as nn
from continual_brain.config import BrainConfig
from continual_brain.frontal.working_memory import WorkingMemoryBuffer
from continual_brain.frontal.executive_attention import ExecutiveAttentionEngine


class FrontalDecisionCore(nn.Module):
    """
    Higher Function Network / dlPFC Decision Core.
    Acts as the executive reasoning center:
      1. Holds rolling working memory slots.
      2. Directs the executive attention spotlight down onto the Parietal Hub.
      3. Binds variables and predicts downstream reasoning tokens (Broca engine).
    """

    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config
        self.memory_buffer = WorkingMemoryBuffer(config)
        self.attention_engine = ExecutiveAttentionEngine(config)

        # Slot pooling and decision head
        self.slot_attention_pool = nn.MultiheadAttention(
            embed_dim=config.d_frontal,
            num_heads=config.wernicke_heads,
            batch_first=True
        )
        self.reasoning_mlp = nn.Sequential(
            nn.Linear(config.d_frontal, config.d_frontal * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_frontal * 2, config.d_frontal)
        )
        self.token_head = nn.Linear(config.d_frontal, config.vocab_size)

    def forward(
        self,
        K_parietal: torch.Tensor,
        V_parietal: torch.Tensor,
        slots: torch.Tensor = None
    ):
        """
        Execute frontal executive attention over Parietal KV substrate and emit logits.

        Args:
            K_parietal: (batch_size, seq_len, d_parietal)
            V_parietal: (batch_size, seq_len, d_parietal)
            slots: Optional existing working memory state (batch_size, n_slots, d_frontal).
        Returns:
            logits: (batch_size, vocab_size) next-token prediction logits.
            slots: (batch_size, n_slots, d_frontal) updated working memory state.
            attn_weights: (batch_size, n_slots, seq_len) inverted slot attention map.
        """
        B = K_parietal.shape[0]
        if slots is None:
            slots = self.memory_buffer.init_slots(B, K_parietal.device)

        # 1. Executive attention spotlight (inverted slot routing)
        slots, attn_weights = self.attention_engine(slots, K_parietal, V_parietal)

        # 2. Pool slots into executive thought summary
        # Summary query attends over slots
        query = slots.mean(dim=1, keepdim=True)  # (B, 1, d_frontal)
        pooled_thought, _ = self.slot_attention_pool(query, slots, slots)
        pooled_thought = pooled_thought.squeeze(1)  # (B, d_frontal)

        # 3. Reasoning MLP and next-token prediction
        thought_repr = self.reasoning_mlp(pooled_thought)
        logits = self.token_head(thought_repr)  # (B, vocab_size)

        return logits, slots, attn_weights
