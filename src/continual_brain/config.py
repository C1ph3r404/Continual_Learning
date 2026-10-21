"""Configuration and hyperparameters for Text-First Continual Brain Architecture."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BrainConfig:
    # Vocabulary & Sequence
    vocab_size: int = 512       # Byte-level / tokenized vocabulary capacity
    max_seq_len: int = 256      # Maximum token context window
    pad_token_id: int = 0
    eos_token_id: int = 1

    # Wernicke Language Engine (theta_lang)
    d_lang: int = 128           # Native dimension of language latents (H_lang)
    wernicke_layers: int = 2    # Transformer / syntactic encoder depth
    wernicke_heads: int = 4     # Number of attention heads in Wernicke

    # Parietal Hub (theta_parietal)
    d_parietal: int = 128       # Working sensory Key-Value dimension

    # Neocortical Knowledge System (theta_know)
    d_know: int = 128           # Emitted factual attribute dimension
    n_memory_units: int = 512   # Continuous associative memory units (columns)
    memory_temperature: float = 1.0

    # Frontal Core & dlPFC Working Memory (theta_frontal)
    d_frontal: int = 128        # Native dimension of Frontal slots
    n_slots: int = 8            # Rolling active working memory slot capacity
    slot_iterations: int = 3    # Inverted slot routing refinement iterations
    temperature_slot: float = 1.0

    # Training & Regularization
    dropout: float = 0.1
    learning_rate: float = 1e-3
