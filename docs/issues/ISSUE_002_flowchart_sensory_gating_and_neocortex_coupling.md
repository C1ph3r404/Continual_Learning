# ISSUE-002: Inaccurate Flowchart Routing and Neocortex Association in Brain.md

- **Date:** 2026-10-21
- **Component:** Documentation / Architecture Flowchart (`docs/Brain.md`)
- **Status:** NOT FIXED

### Symptoms / Traceback
The architecture flowchart in `docs/Brain.md` contained two misleading structural inaccuracies:
1. **Misplaced Input and Downward Arrows to Neocortex:** `[Audio / Speech]`, `[Text / Code]`, and `[Images / Video]` were positioned below the encoders with downward arrows pointing directly into the Neocortical Knowledge System during standard flow.
2. **Ambiguous Neocortex Association Responsibility:** Sensory encoders (Temporal, Occipital, Wernicke) appeared to be directly querying the Neocortex during runtime, rather than this being the sole responsibility of the Parietal Hub.
3. **Missing Explicit Post-Encoder Gating Nodes:** Gating was described in text but not visually articulated as post-encoder decision gates (Temporal Gate and Occipital Gate) routing between Wernicke (language) and Parietal Hub (direct physical senses).

### Root Cause Analysis
- During real-time inference, sensory encoders must be strictly decoupled from the Neocortex. Their only job is feature extraction.
- Only the **Parietal Hub** performs forward associative lookup over the Neocortex.
- The Neocortex only ingests representations from Wernicke, Occipital, and Temporal during the **offline learning / consolidation phase** (distilling semantic field co-occurrences and cross-modal affordances).

### Fix Details
1. Re-architect the flowchart in `docs/Brain.md` to cleanly show:
   - Raw sensory inputs entering their dedicated encoders (Occipital, Temporal, Wernicke).
   - Explicit post-encoder decision gates (`Occipital Gate` and `Temporal Gate`) routing language to Wernicke and physical latents ($H_{\text{vis}}, H_{\text{aud}}$) to Parietal Hub.
   - Parietal Hub receiving all latents ($H_{\text{lang}}, H_{\text{vis}}, H_{\text{aud}}$) and exclusively managing the runtime associative query to the Neocortex.
   - Neocortex receiving offline consolidation traces rather than direct runtime input feeds.
2. Clarify text in Sections 1, 2, and 4 to reinforce this strict operational separation.

### Verification
- Visually verify ASCII alignment, arrow flow, and conceptual accuracy against modular brain principles.

### Commit Reference
- Commit: Pending
