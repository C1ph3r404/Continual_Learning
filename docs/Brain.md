# The Modular Brain-Inspired Architecture Breakdown

```text
                                ┌────────────────────────────────────────────────────────┐
                                │                 HIGHER FUNCTION NETWORK                │
                                │         (Executive Control / Frontal Decision Core)    │
                                │                                                        │
                                │  • Working Memory Scratchpad (Rolling Active Buffer)   │
                                │  • Executive Attention Engine (Endogenous Queries: Q)   │
                                │  • Causal World Model & Variable-Binding Logic         │
                                └───────┬────────────────────────────────▲───────────────┘
                                        │ Executive Attention Spotlight  │ Resolved Concept Packets
                                        │ (Top-Down Queries: Q_Frontal)  │ & Bound Representations
                                        ▼                                │
                                ┌────────────────────────────────────────┴───────────────┐
                                │             INTELLIGENCE GATHERING NETWORK             │
                                │         (Parietal Hub: Sensory Key-Value Substrate)    │
                                │                                                        │
                                │  • Multimodal Sensory Bank (K_Parietal, V_Parietal)    │
                                │  • Exogenous Saliency (Bottom-up alerts)               │
                                └──▲──────────────▲──────────────▲──────────────▲───────┘
                                   │               │              │              │
                              Lang Latents    Visual Latents  Audio Latents  Neocortical Recall
                                   │               │              │          (Forward Associative)
                          ┌────────┴───────┐ ┌─────┴────────┐ ┌───┴──────────┐      │
                          │    LANGUAGE    │ │    VISUAL    │ │     AUDIO    │      │
                          │   PROCESSING  │ │  PROCESSING  │ │   PROCESSING │      │
                          │    NETWORK    │ │   NETWORK    │ │    NETWORK   │      │
                          │(Wernicke Eng.)│ │(Occipital E.)│ │ (Temporal E.)│      │
                          └────────▲──────┘ └─────▲────────┘ └───▲──────────┘      │
                                   │              │              │                  │
                           [Text / Code]   [Images / Video] [Audio / Speech]       │
                                   │              │              │                  │
                                   ▼              ▼              ▼                  ▼
                                ┌────────────────────────────────────────────────────────┐
                                │              NEOCORTICAL KNOWLEDGE SYSTEM              │
                                │        (Decoupled Parametric Key-Value FFN Memory)     │
                                │                                                        │
                                │  • Zero-Backprop Forward Associative Lookup (O(1))     │
                                │  • Asynchronous Offline Consolidation (Sleep Phase)    │
                                └────────────────────────────────────────▲───────────────┘
                                                                         │ Offline Replay
                                                                         └─ (Episodic Traces)
```

---

## 1. Information Flow & Component Breakdown

### 1. Dedicated Sensory Encoders (Wernicke, Occipital & Temporal Engines)
* **Visual Processing Network (Occipital Engine):**
  * **Role:** Dedicated processor for raw visual frames, video sequences, and spatial geometry.
  * **Mechanism:** Extracts invariant visual tokens (edges, textures, spatial relations, optical flow) without attempting higher-level reasoning.
  * **Anti-Forgetting Strategy:** Frozen foundational feature extractors with modular residual adapters (LoRA pools) for novel sensor distributions.
* **Audio Processing Network (Temporal Engine):**
  * **Role:** Dedicated processor for acoustic waveforms, phonetics, environmental sounds, and temporal rhythm.
  * **Mechanism:** Maps continuous acoustic frequencies into structured audio embeddings.
  * **Anti-Forgetting Strategy:** Structural stability for acoustic primitives; high-level semantic binding is offloaded to the associative networks.
* **Language Processing Network (Wernicke Engine):**
  * **Role:** Dedicated processor for raw text, code, symbolic sequences, and natural language syntax.
  * **Mechanism:** Encodes tokenized text streams into structured language embeddings ($H_{\text{lang}}$), capturing syntactic structure, semantic roles, and lexical relationships without attempting higher-level reasoning or cross-modal binding.
  * **Anti-Forgetting Strategy:** Frozen pretrained language backbone with orthogonal LoRA adapter pools for domain-specific vocabulary and linguistic patterns. Core linguistic primitives (grammar, syntax, morphology) are universal and do not need continual overwriting.

### 2. The Intelligence Gathering Network (Parietal Hub - The Active Sensory Canvas)
* **Role:** The active perceptual blackboard and working sensory workspace for the current scene.
* **Non-Memory Nature:** The Parietal Hub **does not store long-term factual associations or world knowledge**. Instead, it acts as the short-term sensory stage where incoming real-time perception is assembled alongside recalled facts from the Neocortex.
* **Direct Inputs:** Ingests active sensory latents from the Wernicke Engine ($H_{\text{lang}}$), Occipital Engine ($H_{\text{vis}}$), and Temporal Engine ($H_{\text{aud}}$) for whichever modalities are currently active in the perceptual window.
* **Automatic Neocortical Association Query:** Incoming sensory tokens immediately query the pre-trained Neocortical Knowledge System via fast forward-pass lookup:
  $$\text{Recall}_{\text{Neocortex}} = \text{Softmax}\left(\frac{H_{\text{sensory}} \cdot W_{\text{key}}}{\sqrt{d}}\right) W_{\text{value}}$$
* **The Assembled Key-Value Bank ($K, V$):** Combines the active sensory tokens with their recalled multimodal affordances into a unified working representation:
  $$K_{\text{Parietal}} = [H_{\text{incoming}} W_{pk}^{(\text{sensory})} \;;\; \text{Recall}_{\text{Neocortex}} W_{pk}^{(\text{know})}]$$
  $$V_{\text{Parietal}} = [H_{\text{incoming}} W_{pv}^{(\text{sensory})} \;;\; \text{Recall}_{\text{Neocortex}} W_{pv}^{(\text{know})}]$$
* **Exogenous Saliency (Bottom-Up):** Flags sudden, intense sensory anomalies (e.g. sudden loud noises, flashing lights) and signals an alert up to the Frontal Core.
* **Passive Nature:** Awaits top-down executive spotlight queries from the Frontal Lobe.


### 3. The Higher Function Network (Frontal Lobe / dlPFC - The Attention & Decision Core)
* **Role:** The executive control center responsible for reasoning, planning, causal deduction, and **directing the attention spotlight**.
* **Working Memory Scratchpad (dlPFC):**
  * As proved by neurobiologist Patricia Goldman-Rakic, Working Memory is not a passive hard drive; it is the **active, persistent firing state of the Dorsolateral Prefrontal Cortex (dlPFC)**.
  * Houses the **Rolling Active Slot Buffer** ($N_{\text{slots}}$ slots in native dimension $d_{\text{frontal}}$) where entities, variables, and ongoing task goals are maintained in active computation.
* **Executive Attention Engine (Competitive Inverted Slot Routing):**
  * **Attention is a form of reasoning.** The Frontal Lobe projects endogenous queries ($Q_{\text{Frontal}} = S \cdot W_{\text{qfront}} \in \mathbb{R}^{N_{\text{slots}} \times d_{\text{parietal}}}$) down onto the Parietal Key-Value bank.
  * Slots compete for finite perceptual tokens through inverted slot attention (normalizing Softmax across slots):
    $$A_{i, j} = \frac{\exp(Q_i \cdot K_j^T / \sqrt{d_{\text{parietal}}})}{\sum_{s=1}^{N_{\text{slots}}} \exp(Q_s \cdot K_j^T / \sqrt{d_{\text{parietal}}})}, \quad \text{Packets}_i = \sum_{j} A_{i, j} V_j$$
    Mathematically forces distinct entities to bind to separate slots without slot collapse.
* **Variable Binding & Anaphora Resolution (e.g., Winograd Schemas):**
  * Resolves referential ambiguities (e.g., binding the pronoun `"it"` to the correct entity) using internal world model physics simulations rather than passive word co-occurrences.
* **Re-Entrant Language Generation Loop (Broca-Parietal-Wernicke Circuit):**
  * **Motor-Linguistic Planning:** To generate coherent, grammatically sound language, the Frontal Core (acting as the executive Broca engine) combines working memory prompt states with active Parietal context to emit the next token $t_{k+1}$.
  * **Efference Copy & Closed Feedback Loop:** The emitted token is immediately fed back into the **Parietal Hub** alongside the current prompt history held in the rolling working memory buffer.
  * **Dynamic Neocortical Re-Interrogation:** The newly appended token immediately triggers a fresh Neocortical associative lookup through the Parietal Hub, refreshing factual context before emitting token $t_{k+2}$.
  * **Supervised Training Protocol (Gradient Wall Preserved):**
    * The Frontal Core and Parietal Hub can be initially trained via standard **Supervised Autoregressive Language Modeling (Teacher Forcing)** on text generation and dialogue datasets.
    * **Strict Gradient Wall:** **Wernicke Core and Neocortex remain 100% FROZEN.** Gradients from the cross-entropy next-token loss only update $\theta_{\text{frontal}}$ and $\theta_{\text{parietal}}$.
    * **Zero Linguistic/Factual Drift:** Wernicke's universal syntax and Neocortex's long-term factual memories are completely protected from degradation during text generation training.
* **Isolation Advantage:** Completely shielded from raw sensory backpropagation noise; only processes structured concepts and emits top-down queries.


### 4. The Neocortical Knowledge System (Universal Multimodal Associative Memory)
* **Role:** Permanent, scalable retention of cross-sensory associations, declarative facts, and physical affordances, completely decoupled from language syntax and reasoning parameters.
* **Cross-Modal Association Hub:**
  * The Neocortex is **not** a text-only system. It is the central repository where **sights, sounds, words, and physical affordances are bound together**.
  * Absorbs associations across multiple modalities during offline learning:
    1. **Text Documents (via Wernicke):** Reads natural text to absorb semantic co-occurrences, functional roles, and declarative knowledge.
    2. **Visuolinguistic Grounding (via Occipital + Wernicke):** Associates visual patch features (e.g., shape, color, texture of an apple) with linguistic tokens and physical affordances.
    3. **Audiolinguistic Grounding (via Temporal + Wernicke):** Associates acoustic signatures (e.g., crunch sound, engine hum) with concepts.
* **Pure Continuous Neural Weights (No Discrete Graphs):**
  * The neural network contains **zero discrete graphs or hardcoded integer entity tables**.
  * Factual memory is strictly parameterized by continuous **Key-Value Associative Memory weights**:
    $$W_{\text{key}} \in \mathbb{R}^{d_{\text{parietal}} \times N_{\text{memory\_units}}}, \quad W_{\text{value}} \in \mathbb{R}^{N_{\text{memory\_units}} \times d_{\text{know}}}$$
  * An incoming concept query in continuous latent space activates memory columns via Softmax and emits associated factual/physical attributes in a single forward pass ($O(1)$ compute).
  * If external discrete knowledge graphs (e.g. Wikidata) are ever used, they are external software databases queried via tools, **not** neural weights inside the model.
* **Zero Test-Time Backpropagation:**
  * Real-time inference is strictly forward-pass without weight gradients. No inference latency overhead or CUDA stalls.
* **Consolidation Cycle ("Sleep Phase"):**
  * **Trigger:** Periodic offline schedule (circadian/sleep phase). Real-time inference remains unblocked.
  * **Dual-Source Ingestion:** Replays episodic traces from short-term context buffers and integrates verified facts into the neocortical parametric weights.
  * **Drift Mitigation:** Dynamically scaled learning rates and Fisher-regularized updates prevent catastrophic forgetting or model autophagy during sleep consolidation.
* **Self-Supervised "Learning While Reading" Protocol (CLS Theory):**
  * **Biological Grounding:** Inspired by Complementary Learning Systems (CLS) theory. Humans do not memorize sentences word-for-word; reading fast-buffers episodic representations, which are slowly distilled into neocortical semantic gists.
  * **Multi-Occurrence Representation Hooking:** As unannotated text streams through the frozen Wernicke encoder, occurrences of a target concept token $C$ (e.g., `"apple"`) are intercepted across varied contexts to capture distinct situational embeddings $\{h_C^{(1)}, h_C^{(2)}, \dots, h_C^{(N)}\}$ (e.g. eating, falling, slicing).
  * **Semantic Field Reconstruction (Not Next-Token):** Instead of standard autoregressive next-token prediction (which overfits to syntax filler words), the neocortex memory readout is trained to reconstruct the **bag of salient semantic co-occurrences** across occurrences:
    $$\mathcal{L}_{\text{field}} = - \sum_{w \in \{\text{sweet}, \text{red}, \text{fall}, \text{tree}, \text{slice}\}} \log P(w \mid \text{Neocortex}(C))$$
  * **Multi-Occurrence Contrastive Consolidation:** Gradients pull diverse occurrences of the same concept toward a shared $W_{\text{key}}$ attractor basin while pushing away distinct concepts, yielding smooth physical affordance manifolds without manual triple curation.

---

## 2. Attention as Reasoning: Anaphora Resolution & The Winograd Schema

### Why Attention is an Executive Frontal Lobe Function
In standard Transformers, "self-attention" is treated as an undifferentiated, bottom-up matrix multiplication repeated uniformly across all layers. In biological cognition, **top-down attention is an active, deductive reasoning process managed by the Frontal Lobe (dlPFC & Frontal Eye Fields)**.

### The Winograd Schema Case Study
Consider the classical Winograd sentences:
* **Sentence A:** *"The trophy didn't fit into the brown suitcase because **it** was too **large**."*
* **Sentence B:** *"The trophy didn't fit into the brown suitcase because **it** was too **small**."*

The grammatical syntax and vocabulary in both sentences are identical. Yet, human cognition effortlessly resolves `"it"` to the **trophy** in Sentence A, and to the **suitcase** in Sentence B.

```text
Information Processing Flow during Winograd Resolution:

1. TEMPORAL / PARIETAL LOBE (Sensory & Lexical Substrate)
   ├── Feeds vocabulary tokens: [trophy], [suitcase], [fit], [large], [small]
   └── Provides physical commonsense: "Fitting requires size(object) < size(container)"

2. FRONTAL LOBE (dlPFC Working Memory & Executive Reasoning)
   ├── Step 1 (Variable Detection): Reads "it" as an empty variable slot: "it" = [?]
   ├── Step 2 (Working Memory Instantiation): Holds candidate antecedents:
   │           Candidate A: {trophy (object)}, Candidate B: {suitcase (container)}
   ├── Step 3 (Causal Physics Simulation):
   │           • Sentence A ("too large"): 
   │             Can a container being too large prevent fitting? No.
   │             Can an object being too large prevent fitting? Yes → Winner: trophy.
   │           • Sentence B ("too small"): 
   │             Can an object being too small prevent fitting? No.
   │             Can a container being too small prevent fitting? Yes → Winner: suitcase.
   └── Step 4 (Executive Attention Beam):
               The dlPFC projects an endogenous attention query (Q_Frontal) backward into 
               Working Memory, binding "it" ──► trophy (Sentence A) or suitcase (Sentence B).
```

**Key Takeaway:** The lower sensory and associative lobes merely supply the tokens and physical definitions; the **Frontal Lobe’s executive attention beam performs the physical simulation that binds variables in working memory**.

---

## 3. Core Architectural Mechanisms: Evaluation & Recommendations

### A. Working Memory vs. Long-Term Memory Separation

| Memory Tier | Anatomical Locus | Implementation | Core Function |
| :--- | :--- | :--- | :--- |
| **Working Memory** | Frontal Lobe (dlPFC) | **Rolling Active Context Buffer** | High-speed, lossless active reasoning scratchpad. Holds immediate conversation, entities, and active hypotheses. |
| **Neocortical Knowledge** | Neocortex / Associative Cortex | **Parametric KV FFNs + Associative Index** | Scalable declarative/world memory. Instant forward-only associative recall with zero test-time backprop; offline sleep consolidation. |

---

### B. Top-Down Attentional Feedback: Adaptive Dynamic Gating

| Approach | Benefits | Flaws | Verdict |
| :--- | :--- | :--- | :--- |
| **Recurrent Step ($t \to t+1$)** | • Zero extra compute cost per step<br/>• Natural for streaming video/audio | • Cannot re-examine or "double-take" on a static image/document within time step $t$ | Base streaming mode |
| **Two-Pass Loop (Gist $\to$ Focal)** | • Mirrors human visual saccades and cognitive scrutiny<br/>• Suppresses irrelevant clutter | • Doubles inference latency ($2\times$) if executed on every input | Deep scrutiny mode |
| **Adaptive Dynamic Gating (Hybrid)** | • Single fast pass for routine inputs ($t-1 \to t$ recurrence)<br/>• Conditionally activates Pass 2 only when Frontal Core signals ambiguity/surprise | • Requires gating threshold calibration | **RECOMMENDED & ADOPTED** |

> **Operational Rule:**
> 1. **Default Flow:** Single fast pass where recurrent executive priors ($h_{\text{frontal}}^{(t-1)}$) bias the Frontal queries $Q_{\text{Frontal}}$ over the Parietal bank $K, V$.
> 2. **Conditional Focal Scrutiny (Pass 2):** When encountering anaphora ambiguity (Winograd schemas), high prediction error, or complex multi-step reasoning, the Frontal Core executes an active second pass, projecting targeted queries to bind variables and resolve conflicts.

---

### C. Top-Down Sensory Feedback: Cascading Sensory Focus ($C \to B \to A$)

Attention operates across two distinct biological axes:
1. **Executive Attention (Frontal $\to$ Parietal Hub):** The dlPFC selects which multimodal tokens to read and bind via endogenous queries $Q_{\text{Frontal}}$.
2. **Sensory Attention (Frontal $\to$ Sensory Encoders):** Top-down instructions directing lower sensory encoders to modulate their feature extraction.

Rather than treating sensory feedback as an undifferentiated channel, the system adopts a **Cascading Sensory Focus ($C \to B \to A$)** architecture. This cascade forms the operational backbone of Pass 2 scrutiny under **Adaptive Dynamic Gating**:

| Stage | Mechanism | Scope | Signal Emitted by Frontal Core | Compute Cost |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1 (Option C)** | **Adapter Routing** | Coarse / Domain level | Adapter mixture weights $r \in \Delta^{N-1}$ selecting active LoRA pools | Negligible |
| **Stage 2 (Option B)** | **Feature Channel Gating** | Semantic / Feature level | Channel modulation vector $g \in [0, 1]^C$ biasing internal activation maps | Very Low |
| **Stage 3 (Option A)** | **Spatial / Temporal Masking** | Coordinate / Region level | Continuous mask $M \in [0, 1]^{H \times W}$ (vision) or $M \in [0, 1]^T$ (audio) | Low |

```text
Pass 1 (Default Fast Path):
    Frontal Core emits adapter routing weights r (Stage 1: Option C)
       │
       ▼ Encoders process with selected LoRAs ──► Latents to Parietal Hub
       │
       ▼ Frontal attends over Parietal K/V ──► Ambiguity detected?
                                                 │
                                                 ├── NO  ──► Emit Action / Prediction [DONE]
                                                 └── YES ──► TRIGGER PASS 2

Pass 2 (Conditional Focal Scrutiny via Cascade):
    Frontal Core emits feature gating vector g (Stage 2: Option B)
    Frontal Core emits spatial/temporal mask M (Stage 3: Option A)
       │
       ▼ Encoders reprocess with (g, M) ──► High-resolution, disambiguated latents
       │
       ▼ Refined latents update Parietal K/V bank
       │
       ▼ Frontal executes focused cross-attention ──► Disambiguation resolved
```

> [!NOTE]
> **Why Pass 2 Needs the Cascade:** Without sensory feedback, Pass 2 would merely re-attend over the exact same coarse Parietal tokens. With the $C \to B \to A$ cascade, the Frontal Core instructs the lower encoders to produce *qualitatively superior, targeted tokens* specifically optimized to resolve the detected ambiguity.

---

### D. Parietal Hub Scalability: Adaptive Pooling

Concatenating sensory latents creates a joint key-value substrate:
$$K_{\text{Parietal}} = [H_{\text{lang}}; H_{\text{vis}}; H_{\text{aud}}], \quad V_{\text{Parietal}} = [H_{\text{lang}}; H_{\text{vis}}; H_{\text{aud}}]$$

With attention cost $O(K_{\text{Frontal}} \times (L_{\text{lang}} + L_{\text{vis}} + L_{\text{aud}}))$, unconstrained multimodal inputs can saturate compute. To resolve this, the architecture adopts **Adaptive Pooling** as the primary token budgeting strategy:

```text
Sensory Stream (Raw Tokens)
       │
       ▼
Adaptive Pooling Compression Layer
       │
       ├─► Complexity Estimator (Token entropy / Activation variance)
       │
       ├── Low Complexity (Blank backgrounds, silence, repetitive syntax)
       │     └─► Heavy Pooling Compression (e.g. 16x - 20x token reduction)
       │
       ├── Medium Complexity (Conversational dialog, standard objects)
       │     └─► Moderate Compression (e.g. 4x - 8x token reduction)
       │
       └── High Complexity (Dense code, cluttered visual scenes, polyphonic audio)
             └─► Light Compression (Preserves full token fidelity)
       │
       ▼
Parietal Key-Value Bank (Budget-Bounded)
```

* **Local Cache for Scrutiny:** When adaptive pooling summarizes sensory streams, the sensory encoders preserve the unpooled, full-resolution token cache in a local ring buffer. If Pass 2 focal scrutiny is triggered, the Frontal spatial mask $M$ unmasks only the targeted patch at full resolution.
* **Secondary Defenses (Reserved for Massive Scale):**
  * *Saliency-Guided Sparse Attention:* Uses the Parietal Hub's bottom-up exogenous saliency scores ($s_i$) to top-$k$ filter tokens prior to Frontal attention.
  * *Modality-Specific Attention Heads:* Partitions Frontal attention heads into dedicated sensory banks ($H_{\text{lang}}, H_{\text{vis}}, H_{\text{aud}}$) plus a small set of cross-modal binding heads.

---

### E. Modular Curriculum & Catastrophic Forgetting Defense Stack

Catastrophic forgetting is addressed through a strict architectural hierarchy rather than a single regularization term:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        CATASTROPHIC FORGETTING DEFENSE STACK            │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 5: Offline Sleep-Cycle Consolidation                             │
│   • Asynchronous episodic replay into neocortical weights              │
│   • Fisher regularization & importance gating prevents parameter drift │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 4: Decoupled Neocortical Knowledge Base                          │
│   • Forward-only Key-Value FFN lookup & semi-parametric factual memory │
│   • Zero test-time backpropagation or inference latency overhead       │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Joint Executive-Perceptual Training Loop & Gradient Wall      │
│   • Parietal Hub (theta_parietal) & Frontal Core (theta_frontal)       │
│     are trained JOINTLY on downstream causal reasoning tasks           │
│   • Gradients flow from reasoning loss through Frontal into Parietal   │
│   • STRICT GRADIENT WALL: Gradients STOP at Neocortex & Sensory gates  │
│   • Zero backprop into Wernicke, Occipital, Temporal, or Neocortex     │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Orthogonal LoRA Adapter Pools                                 │
│   • Domain-specific tasks routed to isolated parameter subspaces       │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Frozen Foundational Sensory Backbones                         │
│   • Universal perceptual primitives (V1 edges, phonemes, syntax)       │
│     are permanently locked against parameter degradation              │
└────────────────────────────────────────────────────────────────────────┘
```

#### The Rationale for the Strict Gradient Wall
Allowing backpropagation gradients to flow from the Frontal Core through the Parietal Hub and into the lower sensory encoders or Neocortex would destroy modularity. Each shift in executive reasoning policy would corrupt foundational perception and long-term factual memory.

By enforcing the **Strict Gradient Wall**, gradients from the reasoning loss optimize the **Frontal Decision Core and Parietal Hub jointly**:
$$\mathcal{L} = \mathcal{L}_{\text{reasoning}}\Big( \text{Frontal}\big(\text{Parietal}(\text{Sensory}_{\text{frozen}}, \text{Neocortex}_{\text{frozen}})\big) \Big)$$
The Parietal Hub directly learns what queries to pose to the Neocortex to minimize executive reasoning error, while foundational perception and long-term factual memories remain permanently locked against catastrophic forgetting.

---

### F. Training Datasets & Staged Modular Pipeline

To train each decoupled component without cross-subspace interference, dedicated training datasets are mapped to each stage:

| Stage & Component | Internet Dataset Source | Local Verified Sample | Target Supervision / Objective |
| :--- | :--- | :--- | :--- |
| **Stage 0: Wernicke Core (`theta_lang`)** | [Universal Dependencies English-EWT](https://github.com/UniversalDependencies/UD_English-EWT) | `data/sample/stage1_wernicke_syntax.jsonl` | POS tagging, dependency tree structure, masked syntax prediction. **Permanently frozen.** |
| **Stage 0: Occipital Core (`theta_vis`)** | ImageNet / SA-1B / WebVid | Feature extractor backbone | Spatial invariant primitives, contours, depth (MAE / DINOv2). **Permanently frozen.** |
| **Stage 0: Temporal Core (`theta_aud`)** | LibriSpeech / AudioSet | Feature extractor backbone | Acoustic invariant primitives, phonetics (wav2vec 2.0 / HuBERT). **Permanently frozen.** |
| **Stage 1: Neocortical Memory (`theta_know`)** | Natural Prose & [MS-COCO Captions](http://cocodataset.org/) | `data/sample/stage2_neocortex_knowledge.jsonl` & `stage3_parietal_multimodal.jsonl` | Universal multimodal associative memory ($W_{\text{key}}, W_{\text{value}}$): binds words $\leftrightarrow$ visual patches $\leftrightarrow$ affordances self-supervised. **Locked during reasoning.** |
| **Stage 2: Joint Executive-Perceptual Loop (`theta_parietal` + `theta_frontal`)** | [OpenAI GSM8K](https://github.com/openai/grade-school-math) & Causal Reasoning Corpora | `data/sample/stage4_frontal_reasoning.jsonl` | **Supervised Learning (SL) on Reasoning:** Frontal generates multi-step reasoning sequences (`prompt` $\to$ `target`) via re-entrant next-token prediction over Parietal canvas. Parietal queries Neocortex for context. **Gradients update only Frontal + Parietal; Sensory & Neocortex are frozen.** |
| **Stage 3: Offline Sleep Consolidation** | Daytime Episodic Traces | Internal Replay Buffer | Asynchronous offline consolidation into Neocortical weights via Elastic Weight Consolidation (EWC). |

All sample datasets are verified via `scripts/verify_datasets.py` to ensure schema conformance and absence of discrete graph tables in the neural weights.



