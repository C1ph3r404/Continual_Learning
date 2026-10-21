"""Unit tests verifying the Text-First Modular Continual Brain Architecture."""

import torch
import pytest
from continual_brain.config import BrainConfig
from continual_brain.wernicke.encoder import WernickeEngine
from continual_brain.neocortex.associative_memory import NeocortexMemory
from continual_brain.parietal.hub import ParietalHub
from continual_brain.frontal.reasoning_core import FrontalDecisionCore
from continual_brain.model import ContinualBrain
from continual_brain.pipeline import (
    train_wernicke_syntax_step,
    train_neocortex_memory_step,
    train_frontal_reasoning_step
)


@pytest.fixture
def test_config():
    return BrainConfig(
        vocab_size=128,
        max_seq_len=64,
        d_lang=64,
        wernicke_layers=2,
        wernicke_heads=2,
        d_parietal=64,
        d_know=64,
        n_memory_units=128,
        d_frontal=64,
        n_slots=4,
        slot_iterations=2
    )


def test_wernicke_engine(test_config):
    wernicke = WernickeEngine(test_config)
    input_ids = torch.randint(0, test_config.vocab_size, (2, 16))
    h_lang = wernicke(input_ids)

    assert h_lang.shape == (2, 16, test_config.d_lang), f"Unexpected shape {h_lang.shape}"
    
    # Test freezing
    wernicke.freeze()
    for param in wernicke.parameters():
        assert not param.requires_grad


def test_neocortex_associative_memory(test_config):
    neocortex = NeocortexMemory(test_config)
    query = torch.randn(2, 16, test_config.d_parietal)
    recall = neocortex(query)

    assert recall.shape == (2, 16, test_config.d_know), f"Unexpected shape {recall.shape}"
    
    # Semantic field reconstruction
    field_logits = neocortex.reconstruct_semantic_field(recall)
    assert field_logits.shape == (2, 16, test_config.vocab_size)


def test_parietal_hub_assembly(test_config):
    neocortex = NeocortexMemory(test_config)
    parietal = ParietalHub(test_config, neocortex)
    h_lang = torch.randn(2, 16, test_config.d_lang)

    K_parietal, V_parietal, saliency = parietal(h_lang, query_neocortex=True)

    # Combined tokens = 16 sensory + 16 recalled knowledge = 32
    assert K_parietal.shape == (2, 32, test_config.d_parietal)
    assert V_parietal.shape == (2, 32, test_config.d_parietal)
    assert saliency.shape == (2, 32)


def test_frontal_decision_core(test_config):
    frontal = FrontalDecisionCore(test_config)
    K = torch.randn(2, 32, test_config.d_parietal)
    V = torch.randn(2, 32, test_config.d_parietal)

    logits, slots, attn_weights = frontal(K, V)

    assert logits.shape == (2, test_config.vocab_size)
    assert slots.shape == (2, test_config.n_slots, test_config.d_frontal)
    assert attn_weights.shape == (2, test_config.n_slots, 32)


def test_end_to_end_brain_and_generation(test_config):
    brain = ContinualBrain(test_config)
    prompt_ids = torch.randint(2, test_config.vocab_size, (1, 8))

    # Test forward pass
    logits, slots, _ = brain(prompt_ids)
    assert logits.shape == (1, test_config.vocab_size)

    # Test autoregressive re-entrant generation loop
    generated = brain.generate(prompt_ids, max_new_tokens=5, temperature=0.0)
    assert generated.shape == (1, 13)
    assert (generated[:, :8] == prompt_ids).all()


def test_strict_gradient_wall(test_config):
    brain = ContinualBrain(test_config)
    input_ids = torch.randint(0, test_config.vocab_size, (2, 10))
    targets = torch.randint(0, test_config.vocab_size, (2,))

    optimizer = torch.optim.AdamW(
        [p for p in brain.parameters() if p.requires_grad],
        lr=1e-3
    )

    # Stage 2 reasoning step asserts the gradient wall internally
    loss = train_frontal_reasoning_step(brain, input_ids, targets, optimizer)
    assert isinstance(loss, float)
    assert loss > 0.0

    # Explicit check: Wernicke and Neocortex received zero gradients
    for name, p in brain.wernicke.named_parameters():
        assert p.grad is None, f"Wernicke {name} leaked gradient!"
    for name, p in brain.neocortex.named_parameters():
        assert p.grad is None, f"Neocortex {name} leaked gradient!"

    # Frontal and Parietal MUST have gradients
    has_frontal_grad = any(p.grad is not None for p in brain.frontal.parameters())
    has_parietal_grad = any(p.grad is not None for p in brain.parietal.parameters())
    assert has_frontal_grad, "Frontal Core did not receive gradients!"
    assert has_parietal_grad, "Parietal Hub did not receive gradients!"
