#!/usr/bin/env -S colab run --gpu T4
"""
Phase 1A: Salience Gate Calibration on Google Colab (with T4 GPU).

This script:
1. Runs in Google Colab (CPU or GPU like T4).
2. Auto-installs needed dependencies (transformers, datasets, spacy, torch).
3. Uses frozen microsoft/deberta-v3-small as Wernicke backbone.
4. Generates POS-derived ground truth labels on Wikipedia/Wikitext sentences:
   - Content Tokens (Nouns, Verbs, Adjectives, Numbers): Label = 1.0
   - Fillers (Articles, Prepositions, Conjunctions, Delimiters): Label = 0.0
5. Trains the dedicated SalienceGate (2-layer MLP) using BCE + Sparsity loss.
6. Saves the calibrated weights to 'salience_gate_weights.pt'.
"""

import os
import sys
import subprocess

# Ensure essential libraries are installed in the Colab VM environment
def install_colab_deps():
    print("[Colab Setup] Verifying dependencies...")
    try:
        import transformers
        import spacy
    except ImportError:
        print("[Colab Setup] Installing transformers, spacy, and en_core_web_sm...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "transformers", "sentencepiece", "protobuf", "spacy", "datasets"
        ])
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])

install_colab_deps()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import spacy


# ---------------------------------------------------------------------------
# 1. SalienceGate Model Definition
# ---------------------------------------------------------------------------
class SalienceGate(nn.Module):
    def __init__(self, d_in: int = 768, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns probabilities in (0, 1) of shape (B, L)."""
        return torch.sigmoid(self.net(h).squeeze(-1))

    def get_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (B, L)."""
        return self.net(h).squeeze(-1)


# ---------------------------------------------------------------------------
# 2. Dataset Preparation (Self-Supervised POS Tagging via spaCy)
# ---------------------------------------------------------------------------
class WebSentencesSalienceDataset(Dataset):
    """
    Pairs raw sentences with automated POS-derived labels:
    - Content Concept = 1.0 (NOUN, PROPN, VERB, ADJ, NUM)
    - Filler / Syntactic glue = 0.0 (DET, ADP, PRON, CCONJ, SCONJ, PUNCT, AUX, SPACE)
    """
    def __init__(self, sentences, tokenizer, nlp, max_len: int = 64):
        self.samples = []
        self.tokenizer = tokenizer
        
        # Universal POS tags that represent conceptual knowledge
        content_pos = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}

        print(f"Annotating {len(sentences)} training sentences using spaCy...")
        for text in sentences:
            if not text.strip():
                continue
            doc = nlp(text)
            encoded = tokenizer(
                text,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"][0]
            attention_mask = encoded["attention_mask"][0]
            offsets = encoded["offset_mapping"][0]

            labels = torch.zeros_like(input_ids, dtype=torch.float32)

            for idx, (start, end) in enumerate(offsets):
                # Ignore special tokens ([CLS], [SEP], padding)
                if start == end:
                    labels[idx] = 0.0
                    continue
                
                # Find matching token in spaCy doc
                char_span = doc.char_span(start.item(), end.item(), alignment_mode="expand")
                if char_span is not None and len(char_span) > 0:
                    pos = char_span[0].pos_
                    if pos in content_pos:
                        labels[idx] = 1.0
                    else:
                        labels[idx] = 0.0

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  PHASE 1A: SALIENCE GATE TRAINING IN GOOGLE COLAB")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_name = "microsoft/deberta-v3-small"
    print(f"\n1. Loading Wernicke backbone ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    wernicke = AutoModel.from_pretrained(model_name).to(device)

    # Strictly lock Wernicke (Strict Gradient Wall)
    for p in wernicke.parameters():
        p.requires_grad = False
    wernicke.eval()
    print("   -> Wernicke 100% frozen: 0 trainable parameters.")

    # Load spaCy tagger
    nlp = spacy.load("en_core_web_sm")

    # Diverse training corpus across science, history, geography, actions, definitions
    sentences = [
        "The Eiffel Tower is a famous wrought-iron lattice monument located in Paris, France.",
        "Photosynthesis allows green plants to synthesize carbohydrates from carbon dioxide and water using sunlight.",
        "The worker can pack the tin can into the cardboard box because it is durable.",
        "The doctor who treated the patient said that the rare medication would cure the viral infection.",
        "Albert Einstein developed the theory of general relativity which revolutionized modern theoretical physics.",
        "Mitochondria generate most of the chemical energy needed to power the cell biochemical reactions.",
        "The Pacific Ocean is the largest and deepest of the world oceanic divisions.",
        "Continuous associative memory matrices allow neural networks to retain facts without catastrophic forgetting.",
        "DNA stores biological information using four basic chemical bases: adenine, thymine, cytosine, and guanine.",
        "The Great Wall of China was constructed across historical northern borders to protect against nomadic invasions.",
        "Jupiter is the largest planet in our solar system, with a mass more than two times all other planets combined.",
        "Machine learning models require robust regularization mechanisms to avoid overfitting noisy observations.",
        "Water boils at one hundred degrees Celsius under standard atmospheric pressure conditions.",
        "The human brain contains approximately eighty-six billion neurons interconnected by trillions of synapses.",
        "The Amazon rainforest produces substantial amounts of oxygen and houses millions of rare animal species.",
        "Isaac Newton formulated the universal laws of motion and gravitation that dominated classical mechanics."
    ] * 8  # 128 diverse sentences

    print(f"\n2. Creating Training Dataset ({len(sentences)} samples)...")
    dataset = WebSentencesSalienceDataset(sentences, tokenizer, nlp, max_len=48)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize Salience Gate (768 -> 128 -> 1)
    d_model = wernicke.config.hidden_size  # 768
    gate = SalienceGate(d_in=d_model, d_hidden=128).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"\n3. Training Salience Gate ({sum(p.numel() for p in gate.parameters())} params)...")
    epochs = 12
    sparsity_lambda = 0.01

    for epoch in range(1, epochs + 1):
        gate.train()
        total_loss, total_bce, total_spar = 0.0, 0.0, 0.0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                h_lang = wernicke(input_ids, attention_mask=attention_mask).last_hidden_state

            logits = gate.get_logits(h_lang)
            probs = torch.sigmoid(logits)

            # Mask out padding tokens
            valid_mask = (attention_mask == 1)
            loss_bce = nn.functional.binary_cross_entropy_with_logits(
                logits[valid_mask],
                labels[valid_mask]
            )

            loss_spar = probs[valid_mask].mean()
            loss = loss_bce + (sparsity_lambda * loss_spar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_spar += loss_spar.item()

        n_batches = len(loader)
        avg_loss = total_loss / n_batches
        avg_bce = total_bce / n_batches
        avg_spar = total_spar / n_batches

        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Total: {avg_loss:.4f} | BCE: {avg_bce:.4f} | Sparsity: {avg_spar:.4f}")

    # 4. Evaluation on unseen sentence
    print("\n4. Evaluating Calibrated Gate on Unseen Complex Sentence...")
    test_sent = "The ancient library of Alexandria contained thousands of scrolls before it was destroyed."
    test_enc = tokenizer(test_sent, return_tensors="pt").to(device)
    
    gate.eval()
    with torch.no_grad():
        h_test = wernicke(**test_enc).last_hidden_state
        test_probs = gate(h_test)[0].cpu().numpy()

    test_tokens = tokenizer.convert_ids_to_tokens(test_enc["input_ids"][0])
    print(f"Sentence: \"{test_sent}\"\n")
    print(f"{'Token':<18} | {'Salience Prob':<15} | {'Decision'}")
    print("-" * 55)
    for tok, p in zip(test_tokens, test_probs):
        if tok in ['[CLS]', '[SEP]']:
            continue
        role = "FACT CONCEPT (Keep)" if p >= 0.5 else "filler (ignore)"
        print(f"{tok:<18} | {p:<15.3f} | {role}")

    # 5. Save the trained gate weights
    out_path = "salience_gate_weights.pt"
    torch.save(gate.state_dict(), out_path)
    print(f"\n5. Successfully saved calibrated weights to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
