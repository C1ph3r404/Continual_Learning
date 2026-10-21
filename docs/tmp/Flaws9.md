# Architectural Flaws and Ninth-Order Vulnerabilities in Brain_v8.md (Flaws 9)

This document provides a comprehensive, research-backed critique of the architecture in [`docs/Brain_v8.md`](file:///home/nate/Projects/Continual_Learning/docs/Brain_v8.md). While `Brain_v8.md` addressed eighth-order vulnerabilities (discrete-gradient latency stalls, non-commutative Lévy area omission, continuous slot blind deconvolution, mean-pooling spatial-syntactic erasure, peripheral clutter smearing, deep Fisher curvature incomputability, and Diaconis-Freedman projection collapse), this analysis exposes **ninth-order foundational, algebraic, topological, and game-theoretic failure modes** that emerge at the intersection of non-symmetric numerical dynamics, optical phase vortex singularities, multi-instance attention binding, multi-clause syntactic scope bleeding, recurrent slot homogenization, and minimax distillation oscillations.

---

## 1. Flaw 9.1: The Asymmetric Matrix Cholesky Solvability Fallacy and Non-Conservative Energy Drift in Linearly-Implicit PHRI

### 1.1 The v8 Formulation
In `Brain_v8.md` (Section 1.3, lines 127–129; Section 3.B, line 257) and `solution8.md` (Section 1.1, lines 21–27), the Linearly-Implicit Discrete-Gradient Port-Hamiltonian Rough Integrator (LIDG-PHRI) updates working memory state via:
$$\left[ \mathbf{I} - \frac{\Delta t}{2} \left( \mathbf{J}_H(S_n) - \mathbf{R}_D(S_n) \right) \mathbf{M}(S_n) \right] \Delta S_n = \Delta t \left[ \mathbf{J}_H(S_n) - \mathbf{R}_D(S_n) \right] \nabla \mathcal{H}(S_n) + \Delta t \, \mathbf{F}_{\text{ext}}$$
The architecture specifies that this system:
$$\mathbf{A}(S_n) \Delta S_n = \mathbf{b}_n$$
is solved via a *"single Cholesky linear solve in $<0.25 \text{ ms}$ on CUDA tensor cores"* (lines 35, 61, 129, 205).

### 1.2 Theoretical and Empirical Failure Mode
1. **Mathematical Inapplicability of Cholesky Decomposition to Asymmetric Systems:**
   By fundamental linear algebra, Cholesky factorization $\mathbf{A} = \mathbf{L} \mathbf{L}^T$ is defined **strictly and exclusively for Symmetric Positive-Definite (SPD) matrices** ($\mathbf{A} = \mathbf{A}^T$ and $x^T \mathbf{A} x > 0, \forall x \neq 0$).
   However, the system matrix:
   $$\mathbf{A}(S_n) = \mathbf{I} - \frac{\Delta t}{2} \left( \mathbf{J}_H(S_n) - \mathbf{R}_D(S_n) \right) \mathbf{M}(S_n)$$
   is **structurally non-symmetric**:
   - $\mathbf{J}_H(S_n)$ is skew-symmetric ($\mathbf{J}_H^T = -\mathbf{J}_H$).
   - $\mathbf{R}_D(S_n)$ is symmetric positive semi-definite ($\mathbf{R}_D^T = \mathbf{R}_D \succeq 0$).
   - $\mathbf{M}(S_n) = \nabla^2 \mathcal{H}(S_n) + \lambda \mathbf{I}$ is symmetric positive definite.
   
   The product of a skew-symmetric matrix and a symmetric matrix is neither symmetric nor skew-symmetric:
   $$(\mathbf{J}_H \mathbf{M})^T = \mathbf{M}^T \mathbf{J}_H^T = -\mathbf{M} \mathbf{J}_H \neq \mathbf{J}_H \mathbf{M}$$
   Consequently, $\mathbf{A}(S_n)^T \neq \mathbf{A}(S_n)$. Calling a Cholesky solver (`torch.linalg.cholesky` or cuSOLVER `potrf`) on $\mathbf{A}(S_n)$ immediately fails with an algorithmic error or ignores the upper triangle, yielding mathematically invalid state updates.
   Solving an asymmetric system requires general LU factorization (`getrf`) or iterative solvers (GMRES, BiCGSTAB). General LU requires partial row pivoting, introducing dynamic branching, memory bank conflicts, and condition-number instabilities that break static CUDA graph compilation.

2. **Non-Conservative Truncation Heating and Secular Energy Drift:**
   The true Gonzalez discrete gradient $\bar{\nabla} \mathcal{H}$ satisfies the directional derivative property $\bar{\nabla} \mathcal{H}(S_n, S_{n+1})^T (S_{n+1} - S_n) \equiv \mathcal{H}(S_{n+1}) - \mathcal{H}(S_n)$ identically for any arbitrary smooth energy function.
   In contrast, the linearly-implicit discrete gradient:
   $$\bar{\nabla}_{\text{LI}} \mathcal{H}(S_n, S_{n+1}) = \nabla \mathcal{H}(S_n) + \frac{1}{2} \mathbf{M}(S_n)(S_{n+1} - S_n)$$
   is an exact discrete gradient **only if $\mathcal{H}(S)$ is strictly quadratic**. For general non-quadratic neural Hamiltonians (e.g. multi-well attractor landscapes or continuous Modern Hopfield energies), the third-order Taylor remainder produces an energy truncation error:
   $$\Delta \mathcal{H}_{\text{drift}} = \mathcal{H}(S_{n+1}) - \mathcal{H}(S_n) - \bar{\nabla}_{\text{LI}} \mathcal{H}^T \Delta S_n = \frac{1}{6} \nabla^3 \mathcal{H}(\xi)[\Delta S_n]^3 = O(\Delta t^3)$$
   Over long rollouts ($10^4$ real-time control steps), these unconstrained errors accumulate monotonically (**secular numerical heating/cooling**), destabilizing the attractor basins and triggering spontaneous, non-physical working memory state transitions (Hairer, Lubich, & Wanner, 2006; Celledoni et al., 2017).

3. **Combinatorial Explosion of Step-2 Lie Brackets:**
   In line 134 of `Brain_v8.md`, the driving term evaluates:
   $$\mathbf{F}_{\text{ext}} = \mathbf{G}_{\text{sat}}(S_n) \frac{\Delta X_n}{\Delta t} + \frac{1}{2 \Delta t} \sum_{1 \le i < j \le m} \mathbb{A}_{n, n+1}^{i, j} [\mathbf{G}_i, \mathbf{G}_j](S_n)$$
   When sensory streams from the Parietal Hub have $m = 64$ to $512$ latent channels, the number of distinct Lie brackets is:
   $$\binom{m}{2} = \frac{m(m-1)}{2} \in [2016, \,\, 130816]$$
   Evaluating each Lie bracket $[\mathbf{G}_i, \mathbf{G}_j] = \frac{\partial \mathbf{G}_j}{\partial S} \mathbf{G}_i - \frac{\partial \mathbf{G}_i}{\partial S} \mathbf{G}_j$ requires two Jacobian-vector products (JVPs) via autograd. Computing $>4,000$ to $260,000$ JVPs at every 1 ms step is completely computationally intractable, exceeding the 15 ms latency budget by several orders of magnitude (Lyons, Caruana, & Lévy, 2007; Friz & Victoir, 2010).

---

## 2. Flaw 9.2: Destructive Interference Optical Vortex Singularities and Multi-Instance Attention Ghosting in CDH-HB

### 2.1 The v8 Formulation
In `Brain_v8.md` (Section 1.3, lines 118–125) and `solution8.md` (Section 2.2, lines 61–75), bound tuples are aggregated via Polar Torus-Projected Superposition (PT-PS):
$$M = \mathcal{P}_{\mathbb{T}^d}\left( \sum_{k=1}^K w_k B_k \right), \quad M_j = \exp\left( i \cdot \text{atan2}(Y_j, X_j) \right)$$
where $X_j = \sum_k w_k \cos(\psi_{k, j})$ and $Y_j = \sum_k w_k \sin(\psi_{k, j})$.
Unbinding is followed by the Complex Slot-Attention Denoising Filter (CSAD):
$$S_k^{\text{clean}} = \mathcal{P}_{\mathbb{T}^d}\left( \text{LayerNorm}_{\mathbb{C}}\left( \hat{S}_k + \text{Attention}_{\mathbb{C}}(Q = \hat{S}_k \mathbf{W}_Q^{\mathbb{C}}, \, K = Z_{\text{parietal}} \mathbf{W}_K^{\mathbb{C}}, \, V = Z_{\text{parietal}} \mathbf{W}_V^{\mathbb{C}}) \right) \right)$$

### 2.2 Theoretical and Empirical Failure Mode
1. **Optical Phase Vortex Singularities and Gradient Detonation:**
   In complex phasor superposition on $\mathbb{T}^d$, when two or more components have opposing phases on coordinate $j$ (destructive interference), the resultant complex magnitude $R_j = \sqrt{X_j^2 + Y_j^2}$ approaches zero ($R_j \to 0$).
   At $R_j = 0$, the phase angle $\theta_j = \text{atan2}(Y_j, X_j)$ has an essential topological singularity known as an **optical phase vortex** or **wavefront dislocation** (Nye & Berry, 1974; Berry, 2000).
   The Jacobian of the polar projection operator with respect to the pre-projection real and imaginary parts is:
   $$\frac{\partial M_j}{\partial X_j} = \frac{-Y_j}{X_j^2 + Y_j^2} \exp(i \theta_j), \quad \frac{\partial M_j}{\partial Y_j} = \frac{X_j}{X_j^2 + Y_j^2} \exp(i \theta_j)$$
   As destructive interference occurs ($R_j \to 0$), the gradient norm diverges:
   $$\left\| \nabla_{(X_j, Y_j)} M_j \right\|_2 = \frac{1}{\sqrt{X_j^2 + Y_j^2}} = \frac{1}{R_j} \xrightarrow{R_j \to 0} \infty$$
   In backward-mode automatic differentiation during end-to-end task learning, backward passes through interference nulls produce catastrophic gradient explosions ($\pm \infty$ / `NaN`), destabilizing network weights.

2. **Multi-Instance Attention Ghosting under Identical Perceptual Queries:**
   CSAD conditions the slot cleanup strictly on $Z_{\text{parietal}}$ using the ungrounded complex query $\hat{S}_k = S_k + \eta_{\text{crosstalk}}$.
   Consider a visual scene with multiple identical or visually indistinguishable objects assigned to different semantic or task roles (e.g., two identical red mugs, where Mug 1 is the `TARGET_CONTAINER` and Mug 2 is an `OBSTACLE`).
   Because both objects share near-identical visual embeddings in $Z_{\text{parietal}}$, the unbinding query $\hat{S}_{\text{CONTAINER}}$ contaminated with crosstalk noise $\eta_{\text{crosstalk}}$ has near-identical dot-product affinities with both Mug 1 and Mug 2 tokens. The attention softmax splits probability evenly ($p_1 \approx 0.5, p_2 \approx 0.5$), resulting in an unphysical, blurred spatial and semantic blend ("attention ghosting") that merges the coordinates and states of both objects into a single degraded slot (Greff et al., 2019; Locatello et al., 2020).

---

## 3. Flaw 9.3: Syntactic Scope Blindness and Negation Bleeding in Root-Directed Attention Pooling (RSAP)

### 3.1 The v8 Formulation
In `Brain_v8.md` (Section 1.2, lines 89–93) and `solution8.md` (Section 3.2, lines 88–96), Root-Directed Syntactic Attention Pooling computes a single sentence-level polarity gating factor:
$$\gamma_{\text{neg}} = \tanh\left( \mathbf{w}_{\text{neg}}^T \hat{H}_{\text{lang}} \right) \in [-1, 1]$$
$$\text{RSAP}(\hat{H}_{\text{lang}}) = \left( \sum_{t=1}^{L_{\text{lang}}} \text{Softmax}\left( \frac{h_{\text{root}} \mathbf{W}_{Q,\text{syn}} (\hat{H}_{\text{lang}, t} \mathbf{W}_{K,\text{syn}})^T}{\sqrt{d}} \right) \hat{H}_{\text{lang}, t} \mathbf{W}_{V,\text{syn}} \right) \odot (1 + \gamma_{\text{neg}} \mathbf{e}_{\text{polarity}})$$

### 3.2 Theoretical and Empirical Failure Mode
1. **Global Negation Bleeding Across Multi-Clause Instructions:**
   In real-world robotic control and reasoning, linguistic instructions frequently contain mixed affirmative and negative clauses:
   - *"Pick up the blue cylinder, but do not touch the red cylinder."*
   - *"The robot verified the latch was locked and did not open the security door."*
   In `Brain_v8.md`, the presence of a single negation cue ("not", "didn't") drives $\gamma_{\text{neg}} \to -1$. Because the polarity vector $(1 + \gamma_{\text{neg}} \mathbf{e}_{\text{polarity}})$ is multiplied element-wise across the **entire pooled sentence representation**, the negation bleeds indiscriminately into every clause! The directive *"Pick up the blue cylinder"* is inverted into *"Do not pick up the blue cylinder"*, while the affirmative latch verification is negated.
2. **Violation of Syntactic C-Command Domain:**
   Formal linguistics and natural language semantics establish that negation is not a global sentence operator; rather, it is a hierarchical operator whose semantic influence is strictly confined to its **c-command domain** in the syntactic dependency tree (Chomsky, 1981; Heim & Kratzer, 1998; Steedman, 2000). Collapsing all tokens into a single pooled vector with a scalar polarity gate destroys clause boundaries and syntactic hierarchy, causing catastrophic scope collapse (Kassner & Schütze, 2020; Ribeiro et al., 2020).

---

## 4. Flaw 9.4: Recurrent Slot Homogenization and Ill-Conditioned Background Covariance in Bidirectional Paging

### 4.1 The v8 Formulation
In `Brain_v8.md` (Section 1.3, lines 110–115; Section 3.B, line 261) and `solution8.md` (Section 4.2, lines 118–120), bidirectional cross-tier paging executes the Reverse Top-Down Intentionality Broadcast (RHB):
$$S_j^{\text{periph}} \leftarrow S_j^{\text{periph}} + \text{GELU}\left( u_{\text{global}} \mathbf{W}_{\text{down}} \right) \mathbf{W}_{\text{gate}, j}$$
Urgency scoring for peripheral slots uses a Mahalanobis distance:
$$\sigma_j = \text{Sigmoid}\left( \mathbf{w}_{\text{urgency}}^T S_j^{\text{periph}} + \beta_{\text{novelty}} \mathcal{D}_{\text{Mahalanobis}}(S_j^{\text{periph}}, \mathcal{C}_{\text{background}}) \right)$$

### 4.2 Theoretical and Empirical Failure Mode
1. **Recurrent Slot Over-Smoothing and Feature Collapse:**
   Every peripheral slot $j \in \{1, \dots, 48\}$ receives an additive projection of the identical global vector $u_{\text{global}}$ at every recurrent step. In non-linear recurrent dynamical systems, continuous injection of an identical global state acts as a contractive operator on the pairwise Dirichlet energy:
   $$\mathcal{E}_{\text{Dir}}(S^{\text{periph}}) = \frac{1}{2} \sum_{j \neq k} \|S_j^{\text{periph}} - S_k^{\text{periph}}\|_2^2 \xrightarrow{t \to \infty} 0$$
   This causes **recurrent slot over-smoothing / feature homogenization** (Oono & Suzuki, 2020; Rusch et al., 2023). Within 5 to 10 recurrent execution steps, peripheral slots lose their individual perceptual identities and converge toward the shared global vector, disabling the model's ability to track distinct peripheral scene entities.
2. **Singularity and Inversion Explosion of Online Background Covariance:**
   Computing $\mathcal{D}_{\text{Mahalanobis}}(S, \mathcal{C}_{\text{bg}}) = (S - \mu_{\text{bg}})^T \mathbf{\Sigma}_{\text{bg}}^{-1} (S - \mu_{\text{bg}})$ in high dimension ($d = 512$) requires maintaining an invertible covariance matrix $\mathbf{\Sigma}_{\text{bg}} \in \mathbb{R}^{512 \times 512}$.
   In continual, non-stationary streaming environments (e.g. transitions between rooms, weather changes, lighting shifts), sample distributions evolve rapidly. An online empirical moving average covariance $\mathbf{\Sigma}_{\text{bg}}$ is persistently rank-deficient along unobserved feature dimensions. As eigenvalues $\lambda_i \to 0$, the inverse covariance explodes ($\|\mathbf{\Sigma}_{\text{bg}}^{-1}\|_2 \to \infty$), causing the novelty score to register false-positive maximum emergencies on arbitrary noise (Ledoit & Wolf, 2004).

---

## 5. Flaw 9.5: The Gauge Invariance Breakdown and Incomplete Subspace Tracking on the Low-Rank Fixed-Rank Manifold

### 5.1 The v8 Formulation
In `Brain_v8.md` (Section 1.4, lines 145–150) and `solution8.md` (Section 5.1, lines 126–149), parameter drift in low-rank LoRA adapters $\mathbf{W}(t) = \mathbf{W}_0 + \mathbf{A}(t) \mathbf{B}(t)$ is tracked by restricting $\mathbf{A}(t) \in \mathbb{R}^{d \times r}$ to the Grassmann manifold $\mathcal{Gr}(r, d)$ and computing exact closed-form Levi-Civita parallel transport $\mathcal{T}_{0 \to t}^{\mathcal{Gr}}(\mathbf{\Xi})$.

### 5.2 Theoretical and Empirical Failure Mode
1. **The Gauge Invariance Breakdown in LoRA Factorization:**
   The low-rank matrix decomposition $\Delta \mathbf{W}(t) = \mathbf{A}(t) \mathbf{B}(t)$ possesses an internal **gauge symmetry**: for any invertible matrix $\mathbf{R} \in \text{GL}(r)$,
   $$(\mathbf{A}(t) \mathbf{R}) (\mathbf{R}^{-1} \mathbf{B}(t)) \equiv \mathbf{A}(t) \mathbf{B}(t)$$
   Standard first-order optimizers (SGD, AdamW) optimize $\mathbf{A}$ and $\mathbf{B}$ concurrently. The optimization trajectory frequently undergoes continuous internal gauge rotations and scalings in $\text{GL}(r)$ that leave the true weight matrix $\Delta \mathbf{W}(t)$ and network behavior completely unchanged (Absil et al., 2008).
   Because GGPT tracks the column space of $\mathbf{A}(t)$ alone, gauge transformations within $\text{GL}(r)$ register as artificial, non-zero geometric velocity $\mathbf{\Delta} = \dot{\mathbf{A}}$ on $\mathcal{Gr}(r, d)$. GGPT parallel-transports memory conditioning vectors along fictitious trajectories caused by optimizer gauge noise rather than actual semantic drift.
2. **Row-Space and Singular-Value Oblivion on the Fixed-Rank Manifold:**
   Tracking only $\text{col}(\mathbf{A}(t)) \in \mathcal{Gr}(r, d)$ completely ignores the row space $\text{row}(\mathbf{B}(t)) \in \mathcal{Gr}(r, k)$ and the singular values $\mathbf{\Sigma}(t) \in \mathbb{R}_+^r$. In neural networks, the row space determines which input features the adapter responds to, while the singular values govern adaptation magnitude. Ignoring $\mathbf{B}(t)$ blinds GGPT to input-selective reweighting.
   Mathematically, low-rank parameter drift lives on the **fixed-rank matrix manifold** $\mathcal{M}_r = \{ \mathbf{W} \in \mathbb{R}^{d \times k} \mid \text{rank}(\mathbf{W}) = r \}$, which is a quotient Riemannian manifold $\mathcal{M}_r \cong (\text{St}(r, d) \times \mathbb{R}_+^r \times \text{St}(r, k)) / \mathcal{O}(r)$ (Vandereycken, 2013). Parallel transport must be defined on $\mathcal{M}_r$, not on an isolated Grassmannian factor.

---

## 6. Flaw 9.6: Adversarial Subspace Flickering and Sorting Discontinuity in Max-Margin Stiefel Distillation

### 6.1 The v8 Formulation
In `Brain_v8.md` (Section 1.4, lines 153–157) and `solution8.md` (Section 6.1, lines 158–179), sleep consolidation optimizes an orthogonal $k$-frame $\mathbf{\Theta}^* \in \text{St}(k, d)$ ($k = 8$) on the Stiefel manifold to maximize Sliced-Wasserstein discrepancy:
$$\mathbf{\Theta}^* = \arg\max_{\mathbf{\Theta} \in \text{St}(k, d)} \sum_{j=1}^k \mathcal{W}_2^2\left( \theta_j^T P_{\text{anc}}, \, \theta_j^T P_{\text{rep}}(\theta_{\text{slow}}) \right)$$
Student weights $\theta_{\text{slow}}$ are updated to minimize this worst-case discrepancy.

### 6.2 Theoretical and Empirical Failure Mode
1. **Adversarial Subspace Flickering and Rotational Limit Cycles:**
   The objective defines an adversarial minimax game:
   $$\min_{\theta_{\text{slow}}} \max_{\mathbf{\Theta} \in \text{St}(k, d)} \sum_{j=1}^k \mathcal{W}_2^2\left( \theta_j^T P_{\text{anc}}, \, \theta_j^T P_{\text{rep}}(\theta_{\text{slow}}) \right)$$
   In non-convex-concave minimax games on manifolds, simultaneous or alternating gradient descent-ascent dynamics are proven to enter **Hamiltonian limit cycles, Poincaré recurrences, and rotational divergence** (Daskalakis et al., 2018; Lin, Jin, & Jordan, 2020). As the student weights take a small step, the maximal discrepancy directions $\mathbf{\Theta}^*$ abruptly rotate into an orthogonal subspace (**subspace flickering**). The student network receives wildly oscillating gradient vectors that overwrite previously aligned dimensions, preventing convergence to a stable distributional equilibrium.
2. **Sorting Permutation Discontinuities in 1D Sliced-Wasserstein:**
   The 1D Wasserstein distance between projected 1D empirical measures is computed via sorted order statistics:
   $$\mathcal{W}_2^2(\theta_j^T P_{\text{anc}}, \theta_j^T P_{\text{rep}}) = \frac{1}{N} \sum_{i=1}^N \left| (\theta_j^T P_{\text{anc}})_{\pi(i)} - (\theta_j^T P_{\text{rep}})_{\sigma(i)} \right|^2$$
   The sorting permutations $\pi$ and $\sigma$ are piecewise constant with respect to $\theta_j$. Whenever two projected points cross along the projection axis ($(\theta_j^T x_a) = (\theta_j^T x_b)$), the sorting permutations swap indices instantaneously. This injects non-differentiable jump discontinuities into the Euclidean gradient $G = \nabla_{\mathbf{\Theta}} \mathcal{W}_2^2$, causing the Cayley retraction in Riemannian Adam to suffer from chattering, gradient noise spikes, and step-size collapse (Cuturi, 2013; Bonneel et al., 2015).

---

## Summary of Ninth-Order Failure Modes

| Flaw ID | System Boundary | Exact Mathematical Failure Mode | Consequence |
| :--- | :--- | :--- | :--- |
| **9.1** | Working Memory Dynamics | Asymmetric system matrix $\mathbf{A}(S_n) \neq \mathbf{A}(S_n)^T$ makes Cholesky solve invalid; non-quadratic third-order truncation error $\nabla^3 \mathcal{H}$; combinatorial explosion of $\binom{m}{2}$ Lie brackets | Cholesky crash/invalidity; secular numerical heating/cooling over time; incomputable real-time Lie bracket autograd ($>10^5$ JVPs) |
| **9.2** | Holographic Binding | Optical phase vortex singularity ($\nabla M_j \to \infty$ as $R_j \to 0$); multi-instance attention ambiguity in CSAD | Infinite NaN gradients during backward pass; attention ghosting / blurring of identical objects across roles |
| **9.3** | Spatial-Syntactic Query Modulation | Single global polarity gating factor $\gamma_{\text{neg}}$ applied uniformly to pooled language tokens; ignores syntactic c-command domains | Global negation bleeding across multi-clause instructions; affirmative directives incorrectly inverted |
| **9.4** | Hierarchical Slot Paging | Additive uniform reverse broadcast contracts Dirichlet energy $\mathcal{E}_{\text{Dir}} \to 0$; empirical background covariance rank-deficiency $\lambda_{\min} \to 0$ | Recurrent slot homogenization / over-smoothing within 10 steps; novelty score explosion on static background noise |
| **9.5** | Neocortical Adaptation | Tracking $\mathbf{A}(t)$ alone is susceptible to $\text{GL}(r)$ gauge rotations; ignores row-space $\mathbf{B}(t)$ and singular values $\mathbf{\Sigma}(t)$ on $\mathcal{M}_r$ | Spurious parallel transport driven by optimizer gauge noise; complete blindness to row-space input reweighting |
| **9.6** | Sleep Consolidation Distillation | Non-convex-concave minimax on Stiefel manifold causes limit cycles / subspace flickering; 1D sorting order swaps inject discontinuous gradient jumps | Distillation gradient oscillation and non-convergence; Riemannian Adam step-size collapse and chattering |
