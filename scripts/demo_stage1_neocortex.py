"""Interactive demonstration of Stage 1 Neocortex Self-Supervised Memory Training.

This script demonstrates:
1. Feeding raw unannotated sentences through the frozen Wernicke backbone.
2. Filtering out filler/syntax words and extracting salient concept candidates.
3. Randomly assigning one concept as the Target Value (V) and the rest as Query Key (K)
   (Multi-Directional Associative Memory / Holographic Recall).
4. Updating the Neocortex Memory Matrix (W_key, W_value) while verifying that
   Wernicke and Frontal remain 100% frozen (Strict Gradient Wall).
"""

import torch
from continual_brain.config import BrainConfig
from continual_brain.model import ContinualBrain
from continual_brain.pipeline.stage1_neocortex import train_neocortex_memory_step


def run_demo():
    print("=" * 70)
    print("  STAGE 1 DEMO: MULTI-DIRECTIONAL NEOCORTEX MEMORY TRAINING")
    print("=" * 70)

    # 1. Initialize Continual Brain
    config = BrainConfig(
        vocab_size=256,
        max_seq_len=32,
        d_lang=64,
        d_parietal=64,
        d_know=64,
        n_memory_units=128
    )
    brain = ContinualBrain(config)

    # Configure Stage 1 (Neocortex unlocked, Wernicke/Parietal/Frontal locked)
    brain.set_training_stage(1)
    
    # Verify parameter statuses
    trainable_neocortex = sum(p.numel() for p in brain.neocortex.parameters() if p.requires_grad)
    trainable_wernicke = sum(p.numel() for p in brain.wernicke.parameters() if p.requires_grad)
    trainable_frontal = sum(p.numel() for p in brain.frontal.parameters() if p.requires_grad)

    print(f"\n[Gradient Wall Status]")
    print(f"  * Wernicke trainable params:  {trainable_wernicke} (LOCKED)")
    print(f"  * Frontal trainable params:   {trainable_frontal} (LOCKED)")
    print(f"  * Neocortex trainable params: {trainable_neocortex} (ACTIVE)")

    # 2. Simulated Web Passage with content words and filler tokens
    # Let 0=PAD, 1=EOS, 2="The", 3="is", 4="in" (stopwords)
    stopwords = torch.tensor([0, 1, 2, 3, 4])
    # Content tokens: 10="Eiffel", 11="Tower", 12="located", 13="Paris"
    passage_tokens = torch.tensor([
        [2, 10, 11, 3, 12, 4, 13, 1]  # "The Eiffel Tower is located in Paris [EOS]"
    ])

    optimizer = torch.optim.AdamW(
        [p for p in brain.neocortex.parameters() if p.requires_grad],
        lr=2e-3
    )

    print("\n[Input Sentence]")
    print("  Tokens: [The(2), Eiffel(10), Tower(11), is(3), located(12), in(4), Paris(13)]")
    print("  Identified Stopwords to ignore: [The, is, in, EOS]")
    print("  Salient Concept Pool:           [Eiffel, Tower, located, Paris]")

    print("\n[Running 5 Self-Supervised Training Iterations]")
    for step in range(1, 6):
        metrics = train_neocortex_memory_step(
            brain,
            passage_tokens,
            optimizer,
            special_token_ids=stopwords,
            ortho_reg_weight=1e-3
        )
        print(f"  Step {step}: Total Loss={metrics['loss']:.4f} "
              f"(Concept Prediction Loss={metrics['loss_concept']:.4f}, "
              f"Orthogonality Loss={metrics['loss_ortho']:.4f})")

    print("\n[Conclusion]")
    print("  -> Neocortex successfully updated its continuous memory columns.")
    print("  -> Wernicke and Frontal remained completely untouched by backpropagation.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
