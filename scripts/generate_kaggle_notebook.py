"""Script to generate the Kaggle Dual-GPU (2x T4) Distributed Training Notebook."""

import json
import os

NOTEBOOK_PATH = "/home/nate/Projects/Continual_Learning/notebooks/kaggle_dual_gpu_training.ipynb"
os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# Cell 1: Header Markdown
add_md("""# Modular Brain-Inspired Continual Learning: Dual-GPU (2x T4) Distributed Training
This Kaggle notebook implements end-to-end distributed multi-GPU training on **Kaggle's Dual Tesla T4 GPUs** (`2x T4`) using PyTorch `DistributedDataParallel` (DDP).

### Core Cognitive Modules:
1. **Wernicke Language Engine (`theta_lang`):** Frozen foundational receptive syntax processor.
2. **Neocortical Knowledge System (`theta_know`):** Continuous Key-Value FFN Associative Memory ($W_{\\text{key}}, W_{\\text{value}}$), zero discrete graph tables.
3. **Parietal Hub (`theta_parietal`):** Active sensory blackboard uniting perception and $O(1)$ associative recall.
4. **Frontal Decision Core (`theta_frontal`):** dlPFC active slot buffer, competitive inverted slot routing, and re-entrant Broca reasoning loop.
5. **Strict Gradient Wall:** Backpropagation from reasoning loss is mathematically prevented from updating Wernicke or Neocortex.""")

# Cell 2: System & GPU Verification
add_code("""# Cell 1: Environment & GPU Verification (Kaggle Dual T4)
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
device_count = torch.cuda.device_count()
print(f"Detected GPU Count: {device_count}")

for i in range(device_count):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GB")

if device_count < 2:
    print("WARNING: Less than 2 GPUs detected. Set Kaggle accelerator to 'GPU T4 x2'.")
else:
    print("SUCCESS: Dual-GPU environment verified and ready for DDP.")""")

# Cell 3: Self-Contained Architecture Definition
add_code("""# Cell 2: Modular Architecture Configuration & Modules
import math
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class BrainConfig:
    vocab_size: int = 512
    max_seq_len: int = 128
    pad_token_id: int = 0
    eos_token_id: int = 1
    d_lang: int = 128
    wernicke_layers: int = 2
    wernicke_heads: int = 4
    d_parietal: int = 128
    d_know: int = 128
    n_memory_units: int = 512
    d_frontal: int = 128
    n_slots: int = 8
    slot_iterations: int = 3
    dropout: float = 0.1
    learning_rate: float = 1e-3

class WernickeEngine(nn.Module):
    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_lang, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_lang)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_lang, nhead=config.wernicke_heads,
            dim_feedforward=config.d_lang * 4, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.wernicke_layers, enable_nested_tensor=False)
        self.layer_norm = nn.LayerNorm(config.d_lang)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(0, L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.layer_norm(self.token_embedding(input_ids) + self.pos_embedding(pos))
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        return self.transformer(x, src_key_padding_mask=key_padding_mask)

class NeocortexMemory(nn.Module):
    def __init__(self, config: BrainConfig):
        super().__init__()
        self.scale = 1.0 / math.sqrt(config.d_parietal)
        self.W_key = nn.Parameter(torch.empty(config.d_parietal, config.n_memory_units))
        self.W_value = nn.Parameter(torch.empty(config.n_memory_units, config.d_know))
        self.semantic_field_head = nn.Linear(config.d_know, config.vocab_size, bias=False)
        nn.init.orthogonal_(self.W_key)
        nn.init.orthogonal_(self.W_value)

    def forward(self, query: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        scores = torch.matmul(query, self.W_key) * self.scale
        attn = torch.softmax(scores / temperature, dim=-1)
        return torch.matmul(attn, self.W_value)

    def reconstruct_semantic_field(self, recall: torch.Tensor) -> torch.Tensor:
        return self.semantic_field_head(recall)

class ParietalHub(nn.Module):
    def __init__(self, config: BrainConfig, neocortex: NeocortexMemory):
        super().__init__()
        self.config = config
        object.__setattr__(self, "neocortex", neocortex)
        self.W_pk_sensory = nn.Linear(config.d_lang, config.d_parietal, bias=False)
        self.W_pv_sensory = nn.Linear(config.d_lang, config.d_parietal, bias=False)
        self.W_pk_know = nn.Linear(config.d_know, config.d_parietal, bias=False)
        self.W_pv_know = nn.Linear(config.d_know, config.d_parietal, bias=False)
        self.saliency_head = nn.Linear(config.d_parietal, 1)

    def forward(self, h_lang: torch.Tensor, query_neocortex: bool = True):
        k_sensory = self.W_pk_sensory(h_lang)
        v_sensory = self.W_pv_sensory(h_lang)
        if query_neocortex and getattr(self, "neocortex", None) is not None:
            with torch.no_grad():
                recalled = self.neocortex(h_lang.detach())
            k_know = self.W_pk_know(recalled)
            v_know = self.W_pv_know(recalled)
            K_p = torch.cat([k_sensory, k_know], dim=1)
            V_p = torch.cat([v_sensory, v_know], dim=1)
        else:
            K_p, V_p = k_sensory, v_sensory
        saliency = torch.sigmoid(self.saliency_head(K_p)).squeeze(-1)
        return K_p, V_p, saliency

class WorkingMemoryBuffer(nn.Module):
    def __init__(self, config: BrainConfig):
        super().__init__()
        self.n_slots = config.n_slots
        self.slot_mu = nn.Parameter(torch.randn(1, 1, config.d_frontal))
        self.slot_sigma = nn.Parameter(torch.rand(1, 1, config.d_frontal) + 0.1)

    def init_slots(self, B: int, device: torch.device) -> torch.Tensor:
        mu = self.slot_mu.expand(B, self.n_slots, -1)
        sigma = self.slot_sigma.expand(B, self.n_slots, -1)
        return mu + torch.randn_like(mu) * sigma

class ExecutiveAttentionEngine(nn.Module):
    def __init__(self, config: BrainConfig):
        super().__init__()
        self.scale = 1.0 / math.sqrt(config.d_parietal)
        self.num_iterations = config.slot_iterations
        self.W_qfront = nn.Linear(config.d_frontal, config.d_parietal, bias=False)
        self.gru = nn.GRUCell(config.d_parietal, config.d_frontal)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_frontal, config.d_frontal * 2), nn.GELU(),
            nn.Linear(config.d_frontal * 2, config.d_frontal)
        )
        self.norm_slots = nn.LayerNorm(config.d_frontal)
        self.norm_input = nn.LayerNorm(config.d_parietal)

    def forward(self, slots: torch.Tensor, K_p: torch.Tensor, V_p: torch.Tensor):
        B, N_s, D_f = slots.shape
        _, _, D_p = K_p.shape
        K_p = self.norm_input(K_p)
        V_p = self.norm_input(V_p)
        for _ in range(self.num_iterations):
            slots_norm = self.norm_slots(slots)
            Q_f = self.W_qfront(slots_norm)
            dots = torch.matmul(Q_f, K_p.transpose(1, 2)) * self.scale
            attn = torch.softmax(dots, dim=1) # INVERTED: normalizes across slots
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            packets = torch.matmul(attn_norm, V_p)
            s_flat = self.gru(packets.view(B * N_s, D_p), slots.view(B * N_s, D_f))
            slots = s_flat.view(B, N_s, D_f) + self.mlp(s_flat.view(B, N_s, D_f))
        return slots, attn

class FrontalDecisionCore(nn.Module):
    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config
        self.memory_buffer = WorkingMemoryBuffer(config)
        self.attention_engine = ExecutiveAttentionEngine(config)
        self.slot_attention_pool = nn.MultiheadAttention(config.d_frontal, config.wernicke_heads, batch_first=True)
        self.reasoning_mlp = nn.Sequential(
            nn.Linear(config.d_frontal, config.d_frontal * 2), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.d_frontal * 2, config.d_frontal)
        )
        self.token_head = nn.Linear(config.d_frontal, config.vocab_size)

    def forward(self, K_p: torch.Tensor, V_p: torch.Tensor, slots: torch.Tensor = None):
        B = K_p.shape[0]
        if slots is None:
            slots = self.memory_buffer.init_slots(B, K_p.device)
        slots, attn = self.attention_engine(slots, K_p, V_p)
        query = slots.mean(dim=1, keepdim=True)
        pooled, _ = self.slot_attention_pool(query, slots, slots)
        thought = self.reasoning_mlp(pooled.squeeze(1))
        logits = self.token_head(thought)
        return logits, slots, attn

class ContinualBrain(nn.Module):
    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config
        self.wernicke = WernickeEngine(config)
        self.neocortex = NeocortexMemory(config)
        self.parietal = ParietalHub(config, self.neocortex)
        self.frontal = FrontalDecisionCore(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, slots: torch.Tensor = None):
        h_lang = self.wernicke(input_ids, attention_mask=attention_mask)
        K_p, V_p, _ = self.parietal(h_lang, query_neocortex=True)
        logits, slots, attn = self.frontal(K_p, V_p, slots=slots)
        return logits, slots, attn

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 16, temperature: float = 0.0):
        self.eval()
        curr_ids = prompt_ids.clone()
        slots = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, slots, _ = self.forward(curr_ids, slots=slots)
                next_tok = torch.argmax(logits, dim=-1, keepdim=True) if temperature <= 0 else torch.multinomial(torch.softmax(logits/temperature, dim=-1), 1)
                curr_ids = torch.cat([curr_ids, next_tok], dim=1)
                if (next_tok == self.config.eos_token_id).all():
                    break
        return curr_ids""")

# Cell 4: Datasets
add_code("""# Cell 3: Datasets & Synthetic Memory/Reasoning Generators
class SyntheticMemoryDataset(Dataset):
    \"\"\"Stage 1: Generates synthetic text passages and salient bag-of-words targets for Neocortex.\"\"\"
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000):
        self.data = torch.randint(2, vocab_size, (num_samples, seq_len))
        self.targets = torch.zeros((num_samples, vocab_size), dtype=torch.float32)
        for i in range(num_samples):
            unique_tokens = torch.unique(self.data[i])
            self.targets[i, unique_tokens] = 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class SyntheticReasoningDataset(Dataset):
    \"\"\"Stage 2: Generates multi-step prompt sequences and target continuation tokens.\"\"\"
    def __init__(self, vocab_size: int, prompt_len: int, num_samples: int = 1000):
        self.prompts = torch.randint(2, vocab_size, (num_samples, prompt_len))
        # Deterministic reasoning target based on sum mod vocab_size
        self.targets = (self.prompts.sum(dim=-1) % vocab_size).long()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.targets[idx]""")

# Cell 5: DDP Setup & Teardown Utilities
add_code("""# Cell 4: DDP Process Group Setup & Cleanup (NCCL Backend)
def setup_ddp(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["NCCL_DEBUG"] = "INFO"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()""")

# Cell 6: Distributed Stage 1 & Stage 2 Training Worker
add_code("""# Cell 5: Distributed Multi-GPU Training Logic (Dual-GPU DDP)
def distributed_train_worker(rank: int, world_size: int, config: BrainConfig, epochs_s1: int = 2, epochs_s2: int = 3):
    setup_ddp(rank, world_size)
    print(f"[Rank {rank}/{world_size}] Initialized on GPU {rank}: {torch.cuda.get_device_name(rank)}")

    # 1. Initialize Full Brain Model
    brain = ContinualBrain(config).to(rank)

    # -------------------------------------------------------------
    # STAGE 1: Neocortical Memory Learning (Memory Learning Bus)
    # Wernicke is 100% frozen; Only Neocortex trains across both GPUs
    # -------------------------------------------------------------
    for p in brain.wernicke.parameters():
        p.requires_grad = False
    for p in brain.parietal.parameters():
        p.requires_grad = False
    for p in brain.frontal.parameters():
        p.requires_grad = False
    for p in brain.neocortex.parameters():
        p.requires_grad = True

    # Wrap Neocortex in DDP for distributed memory learning
    neocortex_ddp = DDP(brain.neocortex, device_ids=[rank])
    s1_dataset = SyntheticMemoryDataset(config.vocab_size, config.max_seq_len, num_samples=800)
    s1_sampler = DistributedSampler(s1_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    s1_loader = DataLoader(s1_dataset, batch_size=16, sampler=s1_sampler)
    s1_optimizer = torch.optim.AdamW(neocortex_ddp.parameters(), lr=1e-3)

    if rank == 0:
        print("\\n=== Starting Stage 1: Neocortex Memory Learning (Dual-GPU DDP) ===")

    for epoch in range(epochs_s1):
        s1_sampler.set_epoch(epoch)
        total_loss = 0.0
        for passages, target_bags in s1_loader:
            passages = passages.to(rank)
            target_bags = target_bags.to(rank)
            s1_optimizer.zero_grad()

            with torch.no_grad():
                h_lang = brain.wernicke(passages)
                concept = h_lang.mean(dim=1)

            recall = neocortex_ddp(concept)
            logits = neocortex_ddp.module.reconstruct_semantic_field(recall)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target_bags)
            loss.backward()
            s1_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(s1_loader)
        if rank == 0:
            print(f"[Stage 1 Epoch {epoch+1}/{epochs_s1}] Neocortex Field BCE Loss: {avg_loss:.4f}")

    dist.barrier()

    # -------------------------------------------------------------
    # STAGE 2: Joint Executive-Perceptual Reasoning Loop
    # STRICT GRADIENT WALL: Wernicke and Neocortex are 100% FROZEN!
    # Gradients flow ONLY into theta_parietal and theta_frontal.
    # -------------------------------------------------------------
    for p in brain.wernicke.parameters():
        p.requires_grad = False
    for p in brain.neocortex.parameters():
        p.requires_grad = False
    for p in brain.parietal.parameters():
        p.requires_grad = True
    for p in brain.frontal.parameters():
        p.requires_grad = True

    # Wrap Parietal & Frontal under a trainable composite module for DDP
    class ReasoningDDPModule(nn.Module):
        def __init__(self, parietal, frontal):
            super().__init__()
            self.parietal = parietal
            self.frontal = frontal
        def forward(self, h_lang):
            K_p, V_p, _ = self.parietal(h_lang, query_neocortex=True)
            logits, _, attn = self.frontal(K_p, V_p)
            return logits, attn

    reasoning_module = ReasoningDDPModule(brain.parietal, brain.frontal).to(rank)
    reasoning_ddp = DDP(reasoning_module, device_ids=[rank])

    s2_dataset = SyntheticReasoningDataset(config.vocab_size, config.max_seq_len // 2, num_samples=800)
    s2_sampler = DistributedSampler(s2_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    s2_loader = DataLoader(s2_dataset, batch_size=16, sampler=s2_sampler)
    s2_optimizer = torch.optim.AdamW(reasoning_ddp.parameters(), lr=1e-3)

    if rank == 0:
        print("\\n=== Starting Stage 2: Frontal Reasoning Loop (Strict Gradient Wall Enforced) ===")

    for epoch in range(epochs_s2):
        s2_sampler.set_epoch(epoch)
        total_loss = 0.0
        for prompts, targets in s2_loader:
            prompts = prompts.to(rank)
            targets = targets.to(rank)
            s2_optimizer.zero_grad()

            with torch.no_grad():
                h_lang = brain.wernicke(prompts)

            logits, _ = reasoning_ddp(h_lang)
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()

            # Assert Strict Gradient Wall integrity across both GPUs
            for name, param in brain.wernicke.named_parameters():
                assert param.grad is None, f"[Rank {rank}] Gradient Wall breach in Wernicke {name}!"
            for name, param in brain.neocortex.named_parameters():
                assert param.grad is None, f"[Rank {rank}] Gradient Wall breach in Neocortex {name}!"

            s2_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(s2_loader)
        if rank == 0:
            print(f"[Stage 2 Epoch {epoch+1}/{epochs_s2}] Frontal Reasoning Loss: {avg_loss:.4f}")

    dist.barrier()

    # Save checkpoint from Rank 0
    if rank == 0:
        os.makedirs("/kaggle/working", exist_ok=True)
        ckpt_path = "/kaggle/working/continual_brain_dual_gpu.pt"
        torch.save({
            "config": config,
            "brain_state": brain.state_dict(),
        }, ckpt_path)
        print(f"\\n[Rank 0] Successfully saved Dual-GPU checkpoint to {ckpt_path}")

        # Re-Entrant Generation Test
        print("\\n=== Re-Entrant Broca Language Generation Demonstration ===")
        sample_prompt = torch.randint(2, config.vocab_size, (1, 8)).to(rank)
        generated_seq = brain.generate(sample_prompt, max_new_tokens=10, temperature=0.0)
        print(f"Input Prompt Token IDs:  {sample_prompt[0].tolist()}")
        print(f"Generated Full Sequence: {generated_seq[0].tolist()}")
        print("Dual-GPU training completed with 100% success!")

    cleanup_ddp()""")

# Cell 7: Launch Cell
add_code("""# Cell 6: Launch Distributed Execution via torch.multiprocessing.spawn
if __name__ == "__main__" or "__file__" not in globals():
    world_size = min(2, torch.cuda.device_count())
    if world_size < 2:
        print("Note: Dual-GPU training requires 2 GPUs. Running on available device count.")
    
    config = BrainConfig(
        vocab_size=256,
        max_seq_len=64,
        d_lang=128,
        d_parietal=128,
        d_know=128,
        n_memory_units=256,
        d_frontal=128,
        n_slots=6
    )

    if torch.cuda.is_available() and world_size >= 2:
        print(f"Spawning DDP across {world_size} Kaggle T4 GPUs...")
        mp.spawn(
            distributed_train_worker,
            args=(world_size, config, 2, 3),
            nprocs=world_size,
            join=True
        )
    else:
        print("Running fallback single-device simulation since 2 CUDA GPUs were not detected in local container.")
        # Single-device test run
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        brain = ContinualBrain(config).to(device)
        sample_input = torch.randint(0, config.vocab_size, (2, 16)).to(device)
        logits, slots, attn = brain(sample_input)
        print(f"Forward Pass Output Logits Shape: {logits.shape}")
        print(f"Active Frontal Slots Shape:      {slots.shape}")
        print("Local verification passed. Upload this notebook to Kaggle with 'GPU T4 x2' enabled.")""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated Kaggle Dual-GPU Notebook at: {NOTEBOOK_PATH}")
