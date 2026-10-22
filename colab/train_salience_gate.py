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
        import datasets
    except ImportError:
        print("[Colab Setup] Installing transformers, datasets, sentencepiece...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "transformers", "sentencepiece", "protobuf", "datasets"
        ])

install_colab_deps()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel


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
        h = h.to(dtype=self.net[0].weight.dtype)
        return torch.sigmoid(self.net(h).squeeze(-1))

    def get_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (B, L)."""
        h = h.to(dtype=self.net[0].weight.dtype)
        return self.net(h).squeeze(-1)


# ---------------------------------------------------------------------------
# 2. Training Loop
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  PHASE 1A: SALIENCE GATE TRAINING ON REAL WEB DATA (UD en_ewt)")
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

    # -----------------------------------------------------------------------
    # 2. Real Web Dataset: Universal Dependencies (English Web Treebank: en_ewt)
    # -----------------------------------------------------------------------
    print("\n2. Downloading Universal Dependencies (en_ewt) from Hugging Face...")
    from datasets import load_dataset
    
    # Load standardized web corpus (12,543 real sentences from blogs, news, reviews, web text)
    ud_dataset = load_dataset("universal-dependencies/universal_dependencies", "en_ewt", split="train")
    print(f"   -> Loaded {len(ud_dataset)} authentic web sentences with gold-standard UPOS tags.")

    # Universal POS tags that represent conceptual / factual knowledge
    # NOUN, PROPN (Proper Noun), VERB, ADJ (Adjective), NUM (Number)
    content_upos = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}

    # Filter out empty or trivial sequences, limit to subset for fast calibration
    max_train_samples = 4000
    tokenized_samples = []
    
    print(f"3. Aligning Hugging Face subwords with UPOS labels (first {max_train_samples} samples)...")
    for row in ud_dataset:
        if len(tokenized_samples) >= max_train_samples:
            break
        
        words = row["tokens"]
        upos_tags = row["upos"]
        if len(words) < 3 or len(words) > 64:
            continue

        # Reconstruct sentence and map word-level UPOS to subword tokens
        encoded = tokenizer(
            words,
            is_split_into_words=True,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        word_ids = encoded.word_ids(batch_index=0)

        labels = torch.zeros_like(input_ids, dtype=torch.float32)

        for token_idx, word_idx in enumerate(word_ids):
            # Special tokens ([CLS], [SEP], padding) have word_idx == None
            if word_idx is None:
                labels[token_idx] = 0.0
            else:
                tag = upos_tags[word_idx]
                labels[token_idx] = 1.0 if tag in content_upos else 0.0

        tokenized_samples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        })

    class UPOSDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    train_loader = DataLoader(UPOSDataset(tokenized_samples), batch_size=32, shuffle=True)
    print(f"   -> Prepared {len(tokenized_samples)} web passages across {len(train_loader)} batches.")

    # Initialize Salience Gate (768 -> 128 -> 1)
    d_model = wernicke.config.hidden_size  # 768
    gate = SalienceGate(d_in=d_model, d_hidden=128).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"\n4. Training Salience Gate ({sum(p.numel() for p in gate.parameters())} params)...")
    epochs = 8
    sparsity_lambda = 0.01

    for epoch in range(1, epochs + 1):
        gate.train()
        total_loss, total_bce, total_spar = 0.0, 0.0, 0.0

        for batch in train_loader:
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

        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_bce = total_bce / n_batches
        avg_spar = total_spar / n_batches

        print(f"  Epoch {epoch:02d}/{epochs:02d} | Total Loss: {avg_loss:.4f} | BCE: {avg_bce:.4f} | Sparsity: {avg_spar:.4f}")

    # 5. Evaluation on complex unseen passages
    print("\n5. Evaluating Calibrated Gate on Unseen Real-World Test Sentences...")
    test_sentences = [
        "The ancient library of Alexandria contained thousands of scrolls before it was destroyed.",
        "Photosynthesis allows green plants to synthesize carbohydrates from carbon dioxide and water using sunlight.",
        "The worker can pack the tin can into the cardboard box because it is durable."
    ]

    gate.eval()
    for test_sent in test_sentences:
        test_enc = tokenizer(test_sent, return_tensors="pt").to(device)
        with torch.no_grad():
            h_test = wernicke(**test_enc).last_hidden_state
            test_probs = gate(h_test)[0].cpu().numpy()

        test_tokens = tokenizer.convert_ids_to_tokens(test_enc["input_ids"][0])
        print(f"\nSentence: \"{test_sent}\"")
        print(f"{'Token':<18} | {'Salience Prob':<15} | {'Decision'}")
        print("-" * 55)
        for tok, p in zip(test_tokens, test_probs):
            if tok in ['[CLS]', '[SEP]']:
                continue
            role = "FACT CONCEPT (Keep)" if p >= 0.5 else "filler (ignore)"
            print(f"{tok:<18} | {p:<15.3f} | {role}")

    # 6. Save the trained gate weights
    out_path = "salience_gate_weights.pt"
    torch.save(gate.state_dict(), out_path)
    print(f"\n6. Successfully saved calibrated weights to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

