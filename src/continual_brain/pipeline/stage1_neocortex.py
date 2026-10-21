"""Stage 1: Neocortical Memory Learning Pipeline (Memory Learning Bus)."""

import torch
import torch.nn as nn
from continual_brain.model import ContinualBrain


def train_neocortex_memory_step(
    model: ContinualBrain,
    passage_ids: torch.Tensor,
    target_semantic_bag: torch.Tensor,
    optimizer: torch.optim.Optimizer
) -> float:
    """
    Stage 1: Self-Supervised Learning While Reading (CLS Theory).
    
    1. Unannotated text streams through the FROZEN Wernicke encoder.
    2. Encoders emit latents onto the Memory Learning Bus.
    3. Neocortical Key-Value weights (W_key, W_value) are optimized to reconstruct
       the bag of salient semantic co-occurrences without next-token syntax overfitting.
    4. Wernicke, Parietal, and Frontal remain 100% FROZEN.
    """
    model.set_training_stage(1)
    optimizer.zero_grad()

    with torch.no_grad():
        # Frozen Wernicke extracts structured language latents
        h_lang = model.wernicke(passage_ids)  # (B, L, d_lang)
        # Average / pool concept occurrence across passage
        concept_latent = h_lang.mean(dim=1)   # (B, d_lang)

    # Memory Learning Bus: Stream latents into Neocortex associative memory
    recalled_latent = model.neocortex(concept_latent)  # (B, d_know)
    predicted_field_logits = model.neocortex.reconstruct_semantic_field(recalled_latent)  # (B, vocab_size)

    # Multi-label cross entropy / BCE loss over target semantic bag
    loss = nn.functional.binary_cross_entropy_with_logits(
        predicted_field_logits,
        target_semantic_bag
    )

    loss.backward()
    optimizer.step()

    return loss.item()
