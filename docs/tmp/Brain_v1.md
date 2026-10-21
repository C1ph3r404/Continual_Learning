# The Modular Brain-Inspired Architecture Breakdown (Version 1.0)

```text
                                 ┌────────────────────────────────────────────────────────┐
                                 │                 HIGHER FUNCTION NETWORK                │
                                 │         (Executive Control / Frontal Decision Core)    │
                                 │                                                        │
                                 │  • Slot-Based Working Memory (Object-Centric Core)     │
                                 │  • Vector Symbolic Binding Engine (B = S ⊛ r_role)     │
                                 │  • Causal World Model & Predicate Energy Evaluator     │
                                 │  • Differentiable Foveation Generator (μ, σ ∈ R^4)     │
                                 │  • Gumbel-Softmax Adapter Router (Categorical r)       │
                                 │  • Top-Down Predictive Generative Prior (Ĥ_parietal)   │
                                 └───────┬───────────────────────▲────────────────────────┘
                                         │ Top-Down Expectation  │ Resolved Bound Slots
                                         │ Ĥ_parietal & Fovea    │ & Causal Inferences
                                         ▼                       │
                                 ┌───────────────────────────────┴────────────────────────┐
                                 │             INTELLIGENCE GATHERING NETWORK             │
                                 │         (Parietal Hub: Normalized Latent Substrate)    │
                                 │                                                        │
                                 │  • Modality Normalization & Metric Hypersphere Bridge  │
                                 │  • Multimodal Perceiver Resampler (Z_parietal Latents) │
                                 │  • Predictive Error Comparator: ε = H - Ĥ_parietal     │
                                 │  • Centroid-Tracking Token Merging (CT-ToMe)           │
                                 └──▲──────────────▲──────────────▲──────────────▲────────┘
                                    │               │              │              │
                       Lang Latents │  Vis Latents  │ Aud Latents  │ Long-Term    │ Retrieved
                       with (t_i)   │  with (x,y,t) │ with (t_i)   │ Key/Values   │ Associative
                                    │               │              │              │ Memories
                           ┌────────┴───────┐ ┌─────┴────────┐ ┌───┴──────────┐   │
                           │    LANGUAGE    │ │    VISUAL    │ │     AUDIO    │   │
                           │   PROCESSING   │ │  PROCESSING  │ │  PROCESSING  │   │
                           │(Wernicke Eng.) │ │(Occipital E.)│ │(Temporal E.) │   │
                           │[Ring Buffer 1D]│ │[Ring Buf 2D] │ │[Ring Buf 1D] │   │
                           └────────▲───────┘ └─────▲────────┘ └───▲──────────┘   │
                                    │               │              │              │
                            [Text / Code]    [Images / Video] [Audio / Speech]    │
                                    │               │              │              │
                                    ▼               ▼              ▼              ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │                LONG-TERM MEMORY SYSTEM                 │
                                 │       (Momentum-Stabilized Gated Titans Neural Memory) │
                                 │                                                        │
                                 │  • Data-Dependent Gated Decay: α_t = σ(W_α k_t)        │
                                 │  • Associative Surprise Gradient: S_t = ∇ ℓ_assoc      │
                                 │  • Momentum Update Buffer: P_t = β P_{t-1} + (1-β) S_t │
                                 │  • Anchor-Constrained Sleep Replay (ACER Core-Sets)    │
                                 └────────────────────────────────────────────────────────┘
```

---

## 1. Information Flow & Component Breakdown

### 1. Dedicated Sensory Encoders with Continuous Coordinate Framing
* **Occipital Engine (Vision):**
  * **Role:** Dedicated spatial-temporal feature extractor for high-resolution images, video frames, and geometric structures.
  * **Coordinate Grounding:** Attaches continuous spatial-temporal coordinates $c_i = (x_i, y_i, t_i) \in [0, 1]^3$ mapped via Random Fourier Features $\gamma(c_i)$.
  * **Local High-Resolution Ring Buffer:** Retains unpooled feature maps in a cyclic RAM cache for selective foveal re-inspection during Pass 2.
  * **Modularity:** Pretrained, frozen vision transformer backbone with discrete LoRA adapter pools routed dynamically.
* **Temporal Engine (Audio):**
  * **Role:** Dedicated acoustic processor for continuous waveforms, phonetics, environmental soundscapes, and prosody.
  * **Coordinate Grounding:** Attaches continuous time coordinates $t_i \in [0, 1]$ mapped via 1D Fourier temporal embeddings.
  * **Local Acoustic Ring Buffer:** Caches high-rate spectrogram features for acoustic re-examination upon phonetic ambiguity.
* **Wernicke Engine (Language):**
  * **Role:** Symbolic sequence extractor for natural language, source code, and mathematical notation.
  * **Coordinate Grounding:** Attaches continuous sequence position $t_i \in [0, 1]$ with continuous RoPE.
  * **Local Lexical Ring Buffer:** Retains tokenized character/subword activations.

---

### 2. The Intelligence Gathering Network (Parietal Hub - Multimodal Resampler Substrate)
* **Role:** Cross-modal metric unification, dimensionality harmonization, and adaptive token condensation.
* **Multimodal Metric Hypersphere Alignment:**
  Raw encoder latents $H_m \in \mathbb{R}^{L_m \times d_m}$ ($m \in \{\text{lang}, \text{vis}, \text{aud}\}$) are projected into a common dimension $d$, centered, and normalized:
  $$\tilde{H}_m = \text{LayerNorm}\left(H_m W_{\text{proj}}^{(m)}\right) + E_{\text{mod}}^{(m)}, \quad \hat{H}_m = \frac{\tilde{H}_m}{\|\tilde{H}_m\|_2}$$
  where $E_{\text{mod}}^{(m)}$ is a learned modality identification embedding. This eliminates modality cone divergence and feature norm domination.
* **Multimodal Perceiver Resampler:**
  Instead of raw concatenation of arbitrary sequence lengths, the Parietal Hub deploys $N_{\text{parietal}}$ learnable query latents $Z_{\text{parietal}} \in \mathbb{R}^{N_{\text{parietal}} \times d}$. These query the combined normalized representations $[\hat{H}_{\text{lang}}; \hat{H}_{\text{vis}}; \hat{H}_{\text{aud}}]$ via multi-head cross-attention:
  $$K_{\text{Parietal}} = Z_{\text{parietal}} + \text{CrossAttention}(Q = Z_{\text{parietal}}, K = \hat{H}, V = \hat{H})$$
  $$V_{\text{Parietal}} = \text{LayerNorm}\left(\text{FFN}(K_{\text{Parietal}})\right)$$
* **Predictive Error Comparator:**
  Compares incoming Parietal representations against the Frontal Core's top-down expectation prior $\hat{H}_{\text{Parietal}}$:
  $$\epsilon_t = K_{\text{Parietal}}^{(t)} - \hat{H}_{\text{Parietal}}^{(t)}$$
  If $\|\epsilon_t\|_2 > \tau_{\text{ambiguity}}$, an endogenous interrupt is raised to activate Pass 2 scrutiny.

---

### 3. The Higher Function Network (Frontal Core / dlPFC - The Cognitive Engine)
* **Role:** Discrete object-centric reasoning, variable-role binding, causal world modeling, and top-down foveation control.
* **Slot-Based Working Memory (Object-Centric Core):**
  * Working memory maintains $K_{\text{slots}}$ distinct cognitive slots $S = [S_1, S_2, \dots, S_K]^T \in \mathbb{R}^{K_{\text{slots}} \times d}$.
  * Tokens from the Parietal Hub are assigned to slots via **Iterative Slot Attention** (Locatello et al., 2020), preventing representation superposition and convex combination blurring.
* **Vector Symbolic Variable Binding (Smolensky TPR / Plate HRR):**
  * Symbolic role vectors (e.g., $r_{\text{SUBJ}}, r_{\text{OBJ}}, r_{\text{CONTAINER}}, r_{\text{FAILED\_CONTAINER}}$) are bound to entity slots via circular convolution:
    $$B_k = S_k \circledast r_{\text{role}}$$
* **Causal World Model & Predicate Evaluator:**
  * Evaluates physical feasibility and counterfactual consistency by computing an energy metric $\mathcal{E}$ over bound slots and schema predicates:
    $$\mathcal{E}(B_k, \text{Predicate}) = \text{MLP}_{\text{causal}}([B_k; \text{Predicate}])$$
* **Differentiable Top-Down Controllers:**
  * **Categorical Adapter Selection:** Emits discrete LoRA adapter weights $r$ via differentiable Gumbel-Softmax with reparameterization:
    $$r = \text{Gumbel-Softmax}(W_{\text{route}} h_{\text{frontal}}, \tau)$$
  * **Spatial Foveation Gaze:** Emits continuous bounding parameters $\theta = [\mu_x, \mu_y, \log \sigma_x, \log \sigma_y]^T \in \mathbb{R}^4$, unmasking targeted coordinate patches from the sensory ring buffers without action-space explosion.

---

### 4. Long-Term Memory System (Momentum-Stabilized Titans Neural Memory)
* **Role:** Permanent, interference-free retention of episodic experiences and fast retrieval of past world states.
* **Associative Surprise Gradient:**
  Surprise is calculated with an explicit associative reconstruction loss and orthogonality regularizer:
  $$\ell_{\text{mem}}(M_t; k_t, v_t) = \frac{1}{2} \|M_t(k_t) - v_t\|_2^2 + \lambda_{\text{ortho}} \|M_t^T M_t - I\|_F^2$$
  $$S_t = \nabla_{M_t} \ell_{\text{mem}}(M_t; k_t, v_t)$$
* **Momentum-Stabilized Plasticity & Gated Retention:**
  To eliminate exponential erasure from static decay, forgetting is conditioned on key identity:
  $$\alpha_t = \sigma(W_\alpha k_t) \in [0, \alpha_{\max}], \quad P_t = \beta_t P_{t-1} + (1 - \beta_t) S_t$$
  $$M_t = (1 - \alpha_t) M_{t-1} - \eta_t P_t$$
* **Dual-State Offline Consolidation (ACER):**
  * **Core-Set Anchor Buffer:** High-surprise keys ($S_t > \tau_{\text{surprise}}$) are committed to a physical anchor buffer $\mathcal{B}_{\text{anchor}}$.
  * **Replay Inversion:** Hippocampal generation is constrained by an MMD distribution matching loss against real anchor points:
    $$\mathcal{L}_{\text{anchor}} = \text{MMD}^2(\mathcal{B}_{\text{anchor}}, \{\tilde{x}\})$$
  * **Functional Knowledge Distillation:** Updates slow neocortical weights via teacher-student KL divergence, eliminating recursive model collapse.

---

## 2. Attention as Reasoning: Anaphora Resolution via Slot-Based Variable Binding

### Why Standard Attention Fails on Winograd Schemas
Standard self-attention computes:
$$\text{Attention}(Q, K, V) = \sum_i \alpha_i v_i$$
In *"The trophy didn't fit into the brown suitcase because it was too large"*, standard attention produces an entangled vector $\alpha_{\text{trophy}} v_{\text{trophy}} + \alpha_{\text{suitcase}} v_{\text{suitcase}}$, losing discrete structural identity.

### The Upgraded Resolution Pipeline in Brain_v1
```text
1. PARIETAL HUB (Harmonized Sensory Latents)
   └── Feeds aligned tokens: [trophy], [suitcase], [fit], [large], [small]

2. FRONTAL dlPFC (Slot Extraction & Variable Binding)
   ├── Step 1 (Slot Attention Extraction):
   │     Competitive routing partitions entities into discrete orthogonal slots:
   │     Slot 1 (S_1) ──► trophy representation
   │     Slot 2 (S_2) ──► suitcase representation
   │
   ├── Step 2 (Vector Symbolic Role Binding):
   │     Binds candidate antecedents to functional roles:
   │     B_1 = S_1 ⊛ r_CONTAINER  (Hypothesis 1: trophy is container)
   │     B_2 = S_1 ⊛ r_OBJECT     (Hypothesis 2: trophy is object)
   │     B_3 = S_2 ⊛ r_CONTAINER  (Hypothesis 3: suitcase is container)
   │     B_4 = S_2 ⊛ r_OBJECT     (Hypothesis 4: suitcase is object)
   │
   ├── Step 3 (Causal World Physics Simulation):
   │     Predicate P_fit = "prevent_fit_condition(it, too_large)"
   │     Evaluator computes energy E = MLP_causal([B_k ; P_fit]):
   │     • E(B_1, P_fit) = HIGH  (A container being too large DOES NOT prevent fitting)
   │     • E(B_2, P_fit) = LOW   (An object being too large PREVENTS fitting → Valid!)
   │     • E(B_3, P_fit) = HIGH  (Suitcase as container being large fits more, not less)
   │
   └── Step 4 (Deterministic Slot Assignment):
         Variable "it" is assigned to S_1 (trophy) with minimal energy.
         Bound entity is committed to the Active Scratchpad without representation blurring.
```

---

## 3. Core Architectural Mechanisms: Detailed Specification

### A. Memory Hierarchy Comparison

| Memory Tier | Anatomical Locus | Implementation | Retention Dynamic | Read/Write Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Working Memory** | Frontal dlPFC | **Slot-Based Object-Centric Core** | Dynamic persistent recurrent state ($K_{\text{slots}}$) | Competitive Slot Attention (Locatello et al.) + GRU update |
| **Episodic Fast Memory** | Hippocampus | **Fast-Weight Titans Neural Memory** | Test-time gradient momentum updates ($P_t$) | Key query forward pass: $v = M_t(q)$, data-dependent decay $\alpha(k)$ |
| **Semantic Slow Memory** | Neocortex | **Slow Base Network Weights** | Offline sleep-cycle distillation | Soft-target KL distillation with anchor core-sets |

---

### B. Predictive Coding & Adaptive Dynamic Gating

```text
Pass 1: Default Predictive Fast Path
┌────────────────────────────────────────────────────────┐
│ 1. Frontal Core projects expectation Ĥ_parietal        │
│ 2. Parietal Hub computes prediction error:             │
│    ε = K_parietal - Ĥ_parietal                         │
│ 3. If ||ε||_2 < τ_ambiguity:                           │
│    • Executive decision emitted immediately [DONE]     │
│    • Zero re-computation overhead                      │
└────────────────────────────────────────────────────────┘
                           │
                           │ If ||ε||_2 ≥ τ_ambiguity (Ambiguity/Surprise Detected)
                           ▼
Pass 2: Targeted Focal Scrutiny via Differentiable Foveation
┌────────────────────────────────────────────────────────┐
│ 1. Frontal Core emits Gumbel-Softmax adapter choice r  │
│ 2. Frontal Core emits 4D foveation gaze:               │
│    θ = [μ_x, μ_y, log σ_x, log σ_y]                    │
│ 3. Spatial Transformer extracts high-res patch from    │
│    the sensory encoder ring buffer                     │
│ 4. Refined foveal tokens update Parietal Resampler     │
│ 5. Slot Attention resolves ambiguity with minimal error│
└────────────────────────────────────────────────────────┘
```

---

### C. Scalability via Centroid-Tracking Token Merging (CT-ToMe)

To ensure unbounded sequence throughput without coordinate destruction:
1. **Centroid Tracking:** Similar tokens within complexity regions are merged using bipartite soft matching.
2. **Coordinate Preservation:** Merged tokens compute weighted spatial-temporal centroids:
   $$c_{\text{merged}} = \frac{\sum_{i \in \mathcal{C}} w_i c_i}{\sum_{i \in \mathcal{C}} w_i}$$
3. **Continuous Fourier Positional Mapping:** Parietal tokens retain continuous coordinate attachments, enabling exact mathematical correspondence with the Frontal Core's 4D foveation bounding box $(\mu, \sigma)$.

---

### D. The 6-Layer Catastrophic Forgetting Defense Stack

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        CATASTROPHIC FORGETTING DEFENSE STACK (v1)      │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 6: Offline Sleep Consolidation (ACER Core-Sets)                  │
│   • Core-set real exemplars regularize generative replay via MMD       │
│   • Knowledge distillation into slow neocortical weights               │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 5: Data-Dependent Gated Titans Neural Memory                     │
│   • Input-dependent decay α_t = σ(W_α k_t) stops past erasure          │
│   • Momentum buffer P_t stabilizes test-time plastic updates           │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 4: Orthogonality Regularization on Memory Subspaces              │
│   • Loss penalty λ_ortho || W^T W - I ||_F prevents memory crosstalk   │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Differentiable Gumbel-Softmax Adapter Routing                 │
│   • Low-rank LoRA pools isolated by categorical gate                   │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Hierarchical Predictive Coding with Differentiable Foveation  │
│   • Top-down feedback constrained to 4D coordinate foveal window       │
│   • Local error minimization without violating backbone freezing       │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Frozen Foundational Multimodal Backbones                      │
│   • Universal perceptual representations permanently protected         │
└────────────────────────────────────────────────────────────────────────┘
```
