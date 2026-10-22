"""Stage 1: Neocortical Memory Learning Pipeline (Two-Phase Architecture).

Phase 1A: Calibrate Salience Gate (Filter filler tokens, determiners, prepositions).
Phase 1B: Train Neocortical Associative Memory Matrix (Multi-directional Key-Value factual storage).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from continual_brain.model import ContinualBrain


# ---------------------------------------------------------------------------
# Phase 1A: Salience Gate Calibration
# ---------------------------------------------------------------------------

def train_salience_gate_step(
    model: ContinualBrain,
    passage_ids: torch.Tensor,
    target_salience_labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    sparsity_weight: float = 1e-3
) -> Dict[str, float]:
    """
    Phase 1A: Fast Calibration of the Memory Learning Bus Salience Gate.

    Teaches the dedicated SalienceGate MLP to discriminate content/concept tokens
    from grammatical filler words (e.g. articles, prepositions, punctuation).
    
    Wernicke, Parietal, Frontal, and Neocortex KV Memory matrices remain locked.
    Only the model.neocortex.salience_gate is optimized.

    Args:
        model: ContinualBrain instance
        passage_ids: (B, L) input token IDs
        target_salience_labels: (B, L) binary labels (1.0 = content concept, 0.0 = filler)
        optimizer: Optimizer bound to model.neocortex.salience_gate.parameters()
        sparsity_weight: Weight encouraging sparsity / selective gating
    Returns:
        metrics: {'loss', 'loss_bce', 'loss_sparsity'}
    """
    model.set_training_stage(1)
    optimizer.zero_grad()

    # 1. Strict Gradient Wall: Extract frozen Wernicke latents
    with torch.no_grad():
        h_lang = model.wernicke(passage_ids)  # (B, L, d_lang)

    # 2. Gate forward pass
    gate_logits = model.neocortex.salience_gate.get_salience_logits(h_lang)  # (B, L)
    salience_probs = torch.sigmoid(gate_logits)

    # 3. Supervised BCE calibration loss
    loss_bce = nn.functional.binary_cross_entropy_with_logits(
        gate_logits,
        target_salience_labels.float()
    )

    # 4. Sparsity penalty (encourages only ~20-30% of tokens to be passed to memory)
    loss_sparsity = salience_probs.mean()

    total_loss = loss_bce + (sparsity_weight * loss_sparsity)
    total_loss.backward()

    # Verify Gradient Wall
    for p in model.wernicke.parameters():
        assert p.grad is None, "Gradient Wall violated: Wernicke received gradients during Phase 1A!"
    for p in model.frontal.parameters():
        assert p.grad is None, "Gradient Wall violated: Frontal Core received gradients during Phase 1A!"
    assert model.neocortex.W_key.grad is None, "W_key should remain frozen during Phase 1A gate calibration!"
    assert model.neocortex.W_value.grad is None, "W_value should remain frozen during Phase 1A gate calibration!"

    optimizer.step()

    return {
        "loss": total_loss.item(),
        "loss_bce": loss_bce.item(),
        "loss_sparsity": loss_sparsity.item()
    }


# ---------------------------------------------------------------------------
# Phase 1B: Neocortex Associative Memory Training
# ---------------------------------------------------------------------------

def train_neocortex_memory_step(
    model: ContinualBrain,
    passage_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    salience_threshold: float = 0.5,
    ortho_reg_weight: float = 1e-4
) -> Dict[str, float]:
    """
    Phase 1B: Self-Supervised Multi-Directional Associative Memory Training.

    1. Unannotated text streams through FROZEN Wernicke (Strict Gradient Wall).
    2. Calibrated Salience Gate dynamically scores all tokens s_t in [0, 1].
       Tokens with s_t >= salience_threshold are selected as Concept Candidates.
    3. Randomly picks ONE concept token as Target Value (V), while remaining
       concept tokens aggregate to form Query Key (K) -> Holographic Recall.
    4. Neocortex performs associative lookup M(K) -> V_recalled and predicts
       the masked concept identity through semantic_field_head.
    5. Optimizes Neocortex associative weights (W_key, W_value, semantic_field_head).
       Wernicke, Parietal, and Frontal remain 100% FROZEN.

    Args:
        model: ContinualBrain instance
        passage_ids: (B, L) input token IDs
        optimizer: Optimizer bound to model.neocortex parameters (W_key, W_value, semantic_field_head)
        salience_threshold: Threshold above which tokens are admitted into memory
        ortho_reg_weight: Weight for slot orthogonality regularization
    Returns:
        metrics: {'loss', 'loss_concept', 'loss_ortho', 'n_salient_tokens'}
    """
    model.set_training_stage(1)
    optimizer.zero_grad()

    B, L = passage_ids.shape
    device = passage_ids.device

    # 1. Strict Gradient Wall: Extract frozen Wernicke syntax latents
    with torch.no_grad():
        h_lang = model.wernicke(passage_ids)  # (B, L, d_lang)
        # 2. Gate evaluation: Get salience scores without updating gate during Phase 1B
        salience_probs = model.neocortex.salience_gate(h_lang)  # (B, L)

    # Filter out padding tokens explicitly
    valid_mask = (passage_ids != model.config.pad_token_id)
    # Binary candidate selection via calibrated gate
    candidate_mask = (salience_probs >= salience_threshold) & valid_mask

    query_keys = []
    target_values = []
    target_token_ids = []
    total_salient_count = 0

    for b in range(B):
        salient_indices = torch.nonzero(candidate_mask[b], as_tuple=True)[0]
        total_salient_count += len(salient_indices)

        if len(salient_indices) < 2:
            # Fallback if passage has insufficient salient tokens: pool valid tokens
            valid_indices = torch.nonzero(valid_mask[b], as_tuple=True)[0]
            target_idx = 0 if len(valid_indices) == 0 else valid_indices[0].item()
            target_val = h_lang[b, target_idx]
            query_k = h_lang[b, valid_indices].mean(dim=0) if len(valid_indices) > 0 else h_lang[b].mean(dim=0)
            target_id = passage_ids[b, target_idx]
        else:
            # Randomly pick ONE salient token as Target Value (V)
            rand_pick = torch.randint(0, len(salient_indices), (1,)).item()
            val_idx = salient_indices[rand_pick].item()
            
            # Remaining salient tokens aggregate into Query Key (K)
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

    # 4. Neocortex Associative Recall M(K)
    recalled_latent = model.neocortex(query_keys)  # (B, d_know)

    # 5. Concept Prediction Loss
    predicted_logits = model.neocortex.reconstruct_semantic_field(recalled_latent)  # (B, vocab_size)
    loss_concept = nn.functional.cross_entropy(predicted_logits, target_token_ids)

    # 6. Orthogonality Regularization (prevents memory slot collapse across n_memory_units)
    W_k_norm = nn.functional.normalize(model.neocortex.W_key, dim=0)
    gram = torch.matmul(W_k_norm.T, W_k_norm)  # (n_units, n_units)
    identity = torch.eye(gram.shape[0], device=device)
    loss_ortho = nn.functional.mse_loss(gram, identity)

    # Total Stage 1B Loss
    total_loss = loss_concept + (ortho_reg_weight * loss_ortho)
    total_loss.backward()

    # Strict Gradient Wall check
    for p in model.wernicke.parameters():
        assert p.grad is None, "Gradient Wall violated: Wernicke received gradients during Phase 1B!"
    for p in model.frontal.parameters():
        assert p.grad is None, "Gradient Wall violated: Frontal Core received gradients during Phase 1B!"

    optimizer.step()

    return {
        "loss": total_loss.item(),
        "loss_concept": loss_concept.item(),
        "loss_ortho": loss_ortho.item(),
        "avg_salient_tokens": total_salient_count / max(1, B)
    }


