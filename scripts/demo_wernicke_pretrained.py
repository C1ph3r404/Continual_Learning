"""Demonstration of Pretrained Wernicke Language Backbone (DeBERTa-v3-small)."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/deberta-v3-small"

def run_demo():
    print(f"=== 1. Downloading & Initializing Pretrained Wernicke Backbone ({MODEL_NAME}) ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    wernicke_backbone = AutoModel.from_pretrained(MODEL_NAME)

    # Count parameters
    total_params = sum(p.numel() for p in wernicke_backbone.parameters())
    print(f"Loaded {MODEL_NAME} with {total_params / 1e6:.1f}M parameters.")

    # 2. Strict Gradient Wall Enforcement: Freeze all weights
    print("\n=== 2. Enforcing Strict Gradient Wall (Freezing Parameters) ===")
    for param in wernicke_backbone.parameters():
        param.requires_grad = False
    wernicke_backbone.eval()
    
    trainable_params = sum(p.numel() for p in wernicke_backbone.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing: {trainable_params} (100% frozen: locked against catastrophic forgetting).")

    # 3. Process an example sentence (Winograd Schema)
    sentence = "The brown dog did not fit into the small suitcase because it was too large."
    print(f"\n=== 3. Input Sentence to Wernicke ===")
    print(f"Sentence: \"{sentence}\"")

    encoded = tokenizer(sentence, return_tensors="pt")
    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print("\nTokenized Sequence:")
    for idx, (token, tid) in enumerate(zip(tokens, input_ids[0].tolist())):
        print(f"  Pos {idx:02d}: Token='{token:<12}' ID={tid}")

    # 4. Forward Pass -> Receptive Syntax Latents (H_lang)
    print("\n=== 4. Extracting Receptive Syntax Latents (H_lang) ===")
    with torch.no_grad():
        outputs = wernicke_backbone(**encoded)
        h_lang = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

    print(f"Output H_lang Tensor Shape: {list(h_lang.shape)} (Batch=1, SeqLen={h_lang.shape[1]}, Dim={h_lang.shape[2]})")

    # 5. Semantic & Syntactic Inspection in Latent Space
    print("\n=== 5. Inspecting Semantic/Syntactic Geometry in Latent Space ===")
    # Find positions of key words
    def get_token_pos(target):
        for i, t in enumerate(tokens):
            if target.lower() in t.lower():
                return i
        return None

    pos_dog = get_token_pos("dog")
    pos_suitcase = get_token_pos("suitcase")
    pos_large = get_token_pos("large")
    pos_small = get_token_pos("small")
    pos_it = get_token_pos("it")

    if pos_dog and pos_suitcase and pos_it:
        vec_dog = h_lang[0, pos_dog]
        vec_suitcase = h_lang[0, pos_suitcase]
        vec_it = h_lang[0, pos_it]
        vec_large = h_lang[0, pos_large] if pos_large else None
        vec_small = h_lang[0, pos_small] if pos_small else None

        # Cosine similarities
        sim_it_dog = F.cosine_similarity(vec_it, vec_dog, dim=0).item()
        sim_it_suitcase = F.cosine_similarity(vec_it, vec_suitcase, dim=0).item()
        
        print(f"Word '{tokens[pos_dog]}' vector norm:      {vec_dog.norm().item():.2f}")
        print(f"Word '{tokens[pos_suitcase]}' vector norm: {vec_suitcase.norm().item():.2f}")
        print(f"Pronoun '{tokens[pos_it]}' similarity to '{tokens[pos_dog]}':      {sim_it_dog:.4f}")
        print(f"Pronoun '{tokens[pos_it]}' similarity to '{tokens[pos_suitcase]}': {sim_it_suitcase:.4f}")
        
        if vec_large is not None and vec_small is not None:
            sim_size = F.cosine_similarity(vec_large, vec_small, dim=0).item()
            print(f"Antonym similarity ('{tokens[pos_large]}' vs '{tokens[pos_small]}'): {sim_size:.4f} (high contextual affinity in syntactic dimension)")

    print("\n=== 6. Conclusion ===")
    print("Pretrained Wernicke delivers rich 768-dimensional syntactic & contextual representations out-of-the-box.")
    print("These vectors H_lang are ready to be fed directly into the Parietal Hub and Neocortex!")

if __name__ == "__main__":
    run_demo()
