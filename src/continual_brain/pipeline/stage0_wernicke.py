"""Stage 0: Wernicke Syntax Pretraining Pipeline."""

import torch
import torch.nn as nn
from continual_brain.model import ContinualBrain


def train_wernicke_syntax_step(
    model: ContinualBrain,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer
) -> float:
    """
    Train Wernicke on syntax & masked token reconstruction.
    Strictly trains ONLY Wernicke; all other brain modules are frozen.
    """
    model.set_training_stage(0)
    optimizer.zero_grad()

    # Pass through Wernicke
    h_lang = model.wernicke(input_ids)
    
    # Reconstruct token logits from Wernicke embedding space
    logits = torch.matmul(h_lang, model.wernicke.token_embedding.weight.t())
    
    loss = nn.functional.cross_entropy(logits.view(-1, model.config.vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()

    return loss.item()
