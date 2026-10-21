"""Intelligence Gathering Network (Parietal Hub - Active Sensory Blackboard)."""

import torch
import torch.nn as nn
from continual_brain.config import BrainConfig
from continual_brain.neocortex.associative_memory import NeocortexMemory


class ParietalHub(nn.Module):
    """
    Active working sensory workspace.
    Combines incoming real-time perception (H_lang) with instant factual associations
    queried from the Neocortex into an assembled Key-Value substrate (K_Parietal, V_Parietal).
    Does NOT store long-term factual memories itself.
    """

    def __init__(self, config: BrainConfig, neocortex: NeocortexMemory):
        super().__init__()
        self.config = config
        # Store neocortex as an unmanaged reference so PyTorch module traversal
        # does not descend into Neocortex when unfreezing ParietalHub parameters.
        object.__setattr__(self, "neocortex", neocortex)

        # Projections for incoming language latents
        self.W_pk_sensory = nn.Linear(config.d_lang, config.d_parietal, bias=False)
        self.W_pv_sensory = nn.Linear(config.d_lang, config.d_parietal, bias=False)

        # Projections for recalled Neocortex knowledge
        self.W_pk_know = nn.Linear(config.d_know, config.d_parietal, bias=False)
        self.W_pv_know = nn.Linear(config.d_know, config.d_parietal, bias=False)

        # Exogenous saliency detector (bottom-up anomaly / novelty alert)
        self.saliency_head = nn.Linear(config.d_parietal, 1)

    def forward(self, h_lang: torch.Tensor, query_neocortex: bool = True):
        """
        Assemble the Parietal Key-Value bank.

        Args:
            h_lang: (batch_size, seq_len, d_lang) incoming sensory latents from Wernicke.
            query_neocortex: If True, query Neocortex for forward associative recall.
        Returns:
            K_parietal: (batch_size, total_tokens, d_parietal)
            V_parietal: (batch_size, total_tokens, d_parietal)
            saliency: (batch_size, total_tokens) bottom-up saliency scores
        """
        k_sensory = self.W_pk_sensory(h_lang)
        v_sensory = self.W_pv_sensory(h_lang)

        if query_neocortex and getattr(self, "neocortex", None) is not None:
            # Associative recall: Zero backprop into Neocortex (Strict Gradient Wall)
            with torch.no_grad():
                recalled_context = self.neocortex(h_lang.detach())  # (batch_size, seq_len, d_know)
            
            k_know = self.W_pk_know(recalled_context)
            v_know = self.W_pv_know(recalled_context)

            # Assemble multimodal key-value bank [Sensory ; Recalled Knowledge]
            K_parietal = torch.cat([k_sensory, k_know], dim=1)
            V_parietal = torch.cat([v_sensory, v_know], dim=1)
        else:
            K_parietal = k_sensory
            V_parietal = v_sensory

        # Compute exogenous saliency
        saliency = torch.sigmoid(self.saliency_head(K_parietal)).squeeze(-1)

        return K_parietal, V_parietal, saliency
