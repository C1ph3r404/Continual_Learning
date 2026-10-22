"""Experiment: Testing Method 1 (Attention Entropy / Centrality) for Automatic Filler Token Filtering.

We test with real DeBERTa-v3 on complex sentences with:
- Grammatical fillers: 'the', 'is', 'a', 'of', 'in', 'because', 'which', 'that'
- Ambiguous words: 'can' (modal verb vs noun), 'book' (noun vs verb)
- Content entities: 'Eiffel Tower', 'quantum mechanics', 'neurotransmitter', 'photosynthesis'
"""

import torch
from transformers import AutoTokenizer, AutoModel


def analyze_salience_with_attention(model, tokenizer, sentence: str, top_k_ratio: float = 0.4):
    """
    Computes token salience from Wernicke self-attention matrices:
    
    1. Extract attention weights from all layers and heads: shape (num_layers, num_heads, L, L)
    2. Attention Centrality (Inflow / Authority):
       Tokens that receive high attention from other tokens are central semantic hubs.
       Inflow(j) = sum_i(Attention[i, j]) - Attention[j, j]
    3. Attention Dispersion / Entropy:
       Tokens that distribute attention uniformly across all words are diffuse grammatical glue.
       Tokens with sharp, localized attention to specific arguments are content words.
    """
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    L = len(input_ids)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # outputs.attentions is a tuple of length num_layers, each (1, num_heads, L, L)
    # Stack across layers: (num_layers, num_heads, L, L)
    all_attentions = torch.stack(outputs.attentions, dim=0).squeeze(1)
    
    # Focus on upper layers (syntax & semantic binding happen in upper half)
    num_layers = all_attentions.shape[0]
    upper_layers = all_attentions[num_layers // 2:]  # top half layers
    
    # Average across upper layers and heads: (L, L)
    # A[i, j] = how much token i pays attention to token j
    A_mean = upper_layers.mean(dim=(0, 1))

    # Metric 1: Inflow Attention (Authority Score)
    # How much do other tokens reference token j?
    # Ignore [CLS] and [SEP] special tokens at index 0 and L-1
    inflow = A_mean.sum(dim=0)  # column sum

    # Metric 2: Outflow Entropy
    # How selective is token i when looking at the rest of the sentence?
    # Low entropy = highly selective (content word)
    eps = 1e-9
    A_clamped = torch.clamp(A_mean, min=eps)
    entropy = - (A_clamped * torch.log(A_clamped)).sum(dim=-1)
    # Inverse entropy normalized (higher = more selective / salient)
    selectivity = 1.0 / (entropy + eps)

    # Combined Salience Score
    # We zero out special tokens [CLS] and [SEP]
    salience = inflow * selectivity
    salience[0] = 0.0
    salience[L - 1] = 0.0

    # Rank tokens
    ranked_indices = torch.argsort(salience, descending=True).tolist()
    
    # Determine top-k cutoff
    k = max(2, int(L * top_k_ratio))
    top_indices = set(ranked_indices[:k])

    results = []
    for i, tok in enumerate(tokens):
        if tok in ['[CLS]', '[SEP]']:
            continue
        is_salient = i in top_indices
        results.append({
            "idx": i,
            "token": tok,
            "inflow": inflow[i].item(),
            "entropy": entropy[i].item(),
            "salience": salience[i].item(),
            "classified_as": "CONTENT (Keep for Memory)" if is_salient else "FILLER (Ignore)"
        })

    return results


def main():
    print("=" * 80)
    print("  TESTING METHOD 1: ATTENTION-BASED AUTOMATIC FILLER TOKEN FILTERING")
    print("=" * 80)

    model_name = "microsoft/deberta-v3-small"
    print(f"Loading pretrained backbone: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    test_sentences = [
        # Sentence 1: Historical / Geographic fact with multiple prepositions and articles
        "The Eiffel Tower is a famous wrought-iron lattice monument located in Paris, France.",

        # Sentence 2: Scientific cause-and-effect with complex connectives
        "Photosynthesis allows green plants to synthesize carbohydrates from carbon dioxide and water using sunlight.",

        # Sentence 3: Ambiguous homonym 'can' (modal verb vs noun container)
        "The worker can pack the tin can into the cardboard box because it is durable.",

        # Sentence 4: Deep clause subordination with relative pronouns
        "The doctor who treated the patient said that the rare medication would cure the viral infection."
    ]

    for s_idx, sentence in enumerate(test_sentences, 1):
        print("\n" + "-" * 80)
        print(f"Sentence {s_idx}: \"{sentence}\"")
        print("-" * 80)
        results = analyze_salience_with_attention(model, tokenizer, sentence, top_k_ratio=0.45)
        
        print(f"{'Token':<18} | {'Inflow':<8} | {'Entropy':<8} | {'Salience':<10} | {'Classification'}")
        print("-" * 80)
        for r in results:
            tag = "[CONCEPT]" if "CONTENT" in r["classified_as"] else "  filler "
            print(f"{r['token']:<18} | {r['inflow']:<8.3f} | {r['entropy']:<8.3f} | {r['salience']:<10.4f} | {tag}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
