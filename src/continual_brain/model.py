"""Integrated Text-First Modular Continual Brain Model."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from continual_brain.config import BrainConfig
from continual_brain.wernicke.encoder import WernickeEngine
from continual_brain.neocortex.associative_memory import NeocortexMemory
from continual_brain.parietal.hub import ParietalHub
from continual_brain.frontal.reasoning_core import FrontalDecisionCore


class ContinualBrain(nn.Module):
    """
    Modular Brain-Inspired Architecture (Text-First Core).

    Information Flow:
      Text -> Wernicke (Syntax) -> H_lang
           -> Parietal Hub (+ Neocortex Associative Recall) -> (K_Parietal, V_Parietal)
           -> Frontal dlPFC (Inverted Slot Attention & Reasoning) -> Next Token / Action
    """

    def __init__(self, config: Optional[BrainConfig] = None):
        super().__init__()
        self.config = config or BrainConfig()

        # 1. Sensory Language Core
        self.wernicke = WernickeEngine(self.config)

        # 2. Decoupled Parametric Associative Memory
        self.neocortex = NeocortexMemory(self.config)

        # 3. Active Sensory Blackboard
        self.parietal = ParietalHub(self.config, self.neocortex)

        # 4. Executive Reasoning & Decision Core
        self.frontal = FrontalDecisionCore(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        slots: Optional[torch.Tensor] = None,
        query_neocortex: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standard forward pass through the modular brain architecture.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: Optional padding mask
            slots: Optional existing dlPFC working memory slots
            query_neocortex: Whether Parietal Hub queries Neocortex for context
        Returns:
            logits: (batch_size, vocab_size) next token logits
            slots: (batch_size, n_slots, d_frontal) updated working memory state
            attn_weights: (batch_size, n_slots, total_parietal_tokens)
        """
        # Step 1: Wernicke receptive syntax processing
        h_lang = self.wernicke(input_ids, attention_mask=attention_mask)

        # Step 2: Parietal Hub gathers perception + queries Neocortex associative memory
        K_parietal, V_parietal, _ = self.parietal(h_lang, query_neocortex=query_neocortex)

        # Step 3: Frontal Lobe executive attention & motor next-token planning
        logits, slots, attn_weights = self.frontal(K_parietal, V_parietal, slots=slots)

        return logits, slots, attn_weights

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7
    ) -> torch.Tensor:
        """
        Re-Entrant Language Generation Loop (Broca-Parietal-Wernicke Circuit).
        Each emitted token is fed back into the working memory buffer and
        dynamically re-queries the Neocortex for updated factual context.
        """
        self.eval()
        curr_ids = prompt_ids.clone()
        slots = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, slots, _ = self.forward(curr_ids, slots=slots)
                
                # Sample or greedy next token
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                curr_ids = torch.cat([curr_ids, next_token], dim=1)
                
                # Stop if EOS token generated
                if (next_token == self.config.eos_token_id).all():
                    break

        return curr_ids

    def set_training_stage(self, stage: int):
        """
        Strict Gradient Wall Manager:
          Stage 0: Only Wernicke trains; Neocortex, Parietal, Frontal frozen.
          Stage 1: Only Neocortex trains (via Memory Learning Bus); Wernicke, Parietal, Frontal frozen.
          Stage 2: Only Parietal & Frontal train on downstream reasoning; Wernicke & Neocortex are 100% frozen.
        """
        if stage == 0:
            # Wernicke syntax pretraining
            self.wernicke.unfreeze()
            self.neocortex.freeze()
            self._set_requires_grad(self.parietal, False)
            self._set_requires_grad(self.frontal, False)
        elif stage == 1:
            # Neocortex memory learning
            self.wernicke.freeze()
            self.neocortex.unfreeze()
            self._set_requires_grad(self.parietal, False)
            self._set_requires_grad(self.frontal, False)
        elif stage == 2:
            # Downstream reasoning loop (Strict Gradient Wall enforced)
            self.wernicke.freeze()
            self.neocortex.freeze()
            self._set_requires_grad(self.parietal, True)
            self._set_requires_grad(self.frontal, True)
        else:
            raise ValueError(f"Unknown training stage: {stage}")

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool):
        for param in module.parameters():
            param.requires_grad = requires_grad
        if requires_grad:
            module.train()
        else:
            module.eval()
