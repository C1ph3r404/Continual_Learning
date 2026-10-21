"""Stage 2: Joint Executive-Perceptual Reasoning Pipeline (Strict Gradient Wall)."""

import torch
import torch.nn as nn
from continual_brain.model import ContinualBrain


def train_frontal_reasoning_step(
    model: ContinualBrain,
    input_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer
) -> float:
    """
    Stage 2: Supervised Learning (Teacher Forcing) on Downstream Reasoning.
    
    Strict Gradient Wall:
      - Wernicke Core and Neocortex are 100% FROZEN.
      - Gradients from reasoning cross-entropy ONLY update theta_frontal and theta_parietal.
      - Parietal Hub learns what queries to pose to the Neocortex to minimize executive error.
    """
    model.set_training_stage(2)
    optimizer.zero_grad()

    # Forward pass through complete brain architecture
    logits, _, _ = model(input_ids)  # (B, vocab_size)

    # Next-token loss
    loss = nn.functional.cross_entropy(logits, target_token_ids)
    loss.backward()

    # Assert Strict Gradient Wall integrity: Wernicke and Neocortex must receive zero gradients!
    for name, param in model.wernicke.named_parameters():
        assert param.grad is None, f"Gradient Wall breach: Wernicke param '{name}' received gradient!"
    for name, param in model.neocortex.named_parameters():
        assert param.grad is None, f"Gradient Wall breach: Neocortex param '{name}' received gradient!"

    optimizer.step()
    return loss.item()
