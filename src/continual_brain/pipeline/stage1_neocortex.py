"""Stage 1: Neocortical Memory Learning Pipeline (Multi-Directional Self-Supervised Memory Bus)."""

import torch
import torch.nn as nn
from typing import Optional, Dict
from continual_brain.model import ContinualBrain


def extract_salient_token_mask(
    input_ids: torch.Tensor,
    special_token_ids: Optional[torch.Tensor] = None,
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Identifies high-salience candidate tokens for factual memory binding.
    Excludes padding, control tokens, and optionally common grammatical stopwords.
    
    Args:
        input_ids: (B, L) token ID tensor
        special_token_ids: Set of token IDs to explicitly ignore (e.g., [CLS], [SEP], stopwords)
        pad_token_id: Padding token ID
    Returns:
        salient_mask: (B, L) boolean mask where True indicates a valid content entity
    """
    mask = (input_ids != pad_token_id)
    if special_token_ids is not None:
        for tid in special_token_ids:
            mask = mask & (input_ids != tid)
    return mask


def train_neocortex_memory_step(
    model: ContinualBrain,
    passage_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    special_token_ids: Optional[torch.Tensor] = None,
    ortho_reg_weight: float = 1e-4
) -> Dict[str, float]:
    """
    Stage 1: Self-Supervised Multi-Directional Associative Memory Training.

    1. Unannotated text streams through the FROZEN Wernicke encoder (Strict Gradient Wall).
    2. Identifies salient content tokens in the sequence (excluding stopwords / syntax fillers).
    3. Randomly selects ONE salient concept token per sample as the Target Value (V),
       while the remaining salient context tokens form the Query Key (K).
       (Enables Multi-Directional Holographic Recall: any entity in a relational fact
        can act as query or answer).
    4. Neocortex performs associative lookup M(K) -> V_recalled and predicts the masked
       concept identity through the semantic field head.
    5. Updates ONLY Neocortex parameters (W_key, W_value, semantic_field_head).
       Wernicke, Parietal, and Frontal remain 100% FROZEN.

    Args:
        model: ContinualBrain instance
        passage_ids: (B, L) input token IDs
        optimizer: Optimizer bound to model.neocortex parameters
        special_token_ids: Optional token IDs to exclude from being chosen as values
        ortho_reg_weight: Weight for slot orthogonality regularization (prevents memory collapse)
    Returns:
        metrics: Dictionary containing loss components {'loss', 'loss_concept', 'loss_ortho'}
    """
    model.set_training_stage(1)
    optimizer.zero_grad()

    B, L = passage_ids.shape
    device = passage_ids.device

    # 1. Strict Gradient Wall: Extract frozen Wernicke syntax latents
    with torch.no_grad():
        h_lang = model.wernicke(passage_ids)  # (B, L, d_lang)

    # 2. Build Salience Mask (filter filler / padding tokens)
    salient_mask = extract_salient_token_mask(
        passage_ids,
        special_token_ids=special_token_ids,
        pad_token_id=model.config.pad_token_id
    )

    # 3. Dynamic Multi-Directional Key-Value Split per sequence
    query_keys = []
    target_values = []
    target_token_ids = []

    for b in range(B):
        salient_indices = torch.nonzero(salient_mask[b], as_tuple=True)[0]
        if len(salient_indices) < 2:
            # Fallback if passage has insufficient salient tokens: pool all tokens as key
            target_idx = 0 if len(salient_indices) == 0 else salient_indices[0].item()
            target_val = h_lang[b, target_idx]
            query_k = h_lang[b].mean(dim=0)
            target_id = passage_ids[b, target_idx]
        else:
            # Randomly pick ONE salient token as the Target Value (V)
            rand_pick = torch.randint(0, len(salient_indices), (1,)).item()
            val_idx = salient_indices[rand_pick].item()
            
            # The remaining salient tokens aggregate to form the Query Key (K)
            key_indices = salient_indices[salient_indices != val_idx]
            query_k = h_lang[b, key_indices].mean(dim=0)
            target_val = h_lang[b, val_idx]
            target_id = passage_ids[b, val_idx]

        query_keys.append(query_k)
        target_values.append(target_val)
        target_token_ids.append(target_id)

    query_keys = torch.stack(query_keys, dim=0)        # (B, d_lang)
    target_values = torch.stack(target_values, dim=0)  # (B, d_lang)
    target_token_ids = torch.stack(target_token_ids, dim=0)  # (B,)

    # 4. Neocortex Associative Recall
    # Query associative memory with context Key (K)
    recalled_latent = model.neocortex(query_keys)  # (B, d_know)

    # 5. Concept Prediction Loss
    # Reconstruct the token vocabulary ID of the masked target concept
    predicted_logits = model.neocortex.reconstruct_semantic_field(recalled_latent)  # (B, vocab_size)
    loss_concept = nn.functional.cross_entropy(predicted_logits, target_token_ids)

    # 6. Orthogonality Regularization (prevents memory slot collapse across n_memory_units)
    # W_key shape: (d_parietal, n_memory_units)
    W_k_norm = nn.functional.normalize(model.neocortex.W_key, dim=0)
    gram = torch.matmul(W_k_norm.T, W_k_norm)  # (n_units, n_units)
    identity = torch.eye(gram.shape[0], device=device)
    loss_ortho = nn.functional.mse_loss(gram, identity)

    # Total Stage 1 Loss
    total_loss = loss_concept + (ortho_reg_weight * loss_ortho)

    # Backward pass - strictly updates Neocortex
    total_loss.backward()

    # Safety assertion: Strict Gradient Wall check
    for p in model.wernicke.parameters():
        assert p.grad is None, "Gradient Wall violated: Wernicke received gradients during Stage 1!"
    for p in model.frontal.parameters():
        assert p.grad is None, "Gradient Wall violated: Frontal Core received gradients during Stage 1!"

    optimizer.step()

    return {
        "loss": total_loss.item(),
        "loss_concept": loss_concept.item(),
        "loss_ortho": loss_ortho.item()
    }

