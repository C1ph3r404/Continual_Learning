"""Test and evaluate the trained salience_gate_weights.pt on complex test sentences."""

import torch
from transformers import AutoTokenizer, AutoModel
from continual_brain.config import BrainConfig
from continual_brain.neocortex.salience_gate import SalienceGate


def test_trained_salience_gate():
    print("=" * 80)
    print("  TESTING TRAINED SALIENCE GATE (colab/salience_gate_weights.pt)")
    print("=" * 80)

    device = torch.device("cpu")
    model_name = "microsoft/deberta-v3-small"
    print(f"Loading Wernicke backbone: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    wernicke = AutoModel.from_pretrained(model_name).to(device)
    wernicke.eval()

    # Initialize Salience Gate with d_in=768 (DeBERTa dimension)
    config = BrainConfig(d_lang=768)
    gate = SalienceGate(config).to(device)

    weights_path = "colab/salience_gate_weights.pt"
    print(f"Loading checkpoint: {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    gate.load_state_dict(state_dict)
    gate.eval()
    print("Checkpoint loaded successfully!\n")

    test_sentences = [
        "The Eiffel Tower is a famous wrought-iron lattice monument located in Paris, France.",
        "Photosynthesis allows green plants to synthesize carbohydrates from carbon dioxide and water using sunlight.",
        "The worker can pack the tin can into the cardboard box because it is durable.",
        "The doctor who treated the patient said that the rare medication would cure the viral infection."
    ]

    for sent_idx, sentence in enumerate(test_sentences, 1):
        print("-" * 80)
        print(f"Sentence {sent_idx}: \"{sentence}\"")
        print("-" * 80)

        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            h_lang = wernicke(**inputs).last_hidden_state
            probs = gate(h_lang)[0].cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        print(f"{'Token':<18} | {'Prob':<8} | {'Decision'}")
        print("-" * 55)

        kept_concepts = []
        ignored_fillers = []

        for tok, p in zip(tokens, probs):
            if tok in ['[CLS]', '[SEP]']:
                continue
            if p >= 0.5:
                kept_concepts.append(tok)
                decision = "FACT CONCEPT (Admit to Neocortex)"
            else:
                ignored_fillers.append(tok)
                decision = "filler (blocked at gate)"
            print(f"{tok:<18} | {p:<8.3f} | {decision}")

        print(f"\n  -> Kept Concepts ({len(kept_concepts)}): {kept_concepts}")
        print(f"  -> Blocked Fillers ({len(ignored_fillers)}): {ignored_fillers}\n")

    print("=" * 80)


if __name__ == "__main__":
    test_trained_salience_gate()
