"""Wernicke Language Processing Network (Receptive Language Core)."""

import math
import torch
import torch.nn as nn
from continual_brain.config import BrainConfig


class WernickeEngine(nn.Module):
    """
    Dedicated processor for symbolic language sequences, syntactic structure,
    and lexical relationships without cross-modal binding or decision logic.
    """

    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_lang

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_lang, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_lang)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_lang,
            nhead=config.wernicke_heads,
            dim_feedforward=config.d_lang * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.wernicke_layers,
            enable_nested_tensor=False
        )
        self.layer_norm = nn.LayerNorm(config.d_lang)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode token IDs into structured language latents H_lang.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) with 1 for valid tokens, 0 for padding.
        Returns:
            H_lang: (batch_size, seq_len, d_lang)
        """
        B, L = input_ids.shape
        positions = torch.arange(0, L, device=input_ids.device).unsqueeze(0).expand(B, L)

        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.layer_norm(x)

        src_key_padding_mask = None
        if attention_mask is not None:
            # Transformer expects True for masked (ignored) positions
            src_key_padding_mask = (attention_mask == 0)

        h_lang = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return h_lang

    def freeze(self):
        """Permanently freeze Wernicke parameters to enforce the Strict Gradient Wall."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self):
        """Unfreeze Wernicke parameters (only during Stage 0 syntax pretraining)."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()
