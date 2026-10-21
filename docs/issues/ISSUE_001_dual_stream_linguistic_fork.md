# ISSUE-001: Missing Dual-Stream Linguistic Forking from Sensory Encoders into Wernicke

- **Date:** 2026-10-21
- **Component:** Sensory Ingestion / Wernicke Engine / Parietal Hub Architecture
- **Status:** NOT FIXED

### Symptoms / Traceback
In the initial architecture specification (`docs/Brain.md`), sensory pipelines were drawn as three isolated vertical streams:
- `[Text / Code]` $\to$ Wernicke Engine $\to$ `Lang Latents`
- `[Images / Video]` $\to$ Occipital Engine $\to$ `Visual Latents`
- `[Audio / Speech]` $\to$ Temporal Engine $\to$ `Audio Latents`

This created a severe architectural flaw:
1. **Linguistic Blindness in Sensory Modalities:** Spoken speech and images of text (e.g. road signs, book pages, document OCR) bypassed Wernicke entirely and were sent to the Parietal Hub as raw acoustic/visual tokens, depriving the model of Wernicke's deep syntactic and grammatical parsing.
2. **Hallucination Risk if Naively Re-routed:** Conversely, forcing *all* audio and imagery through Wernicke would force the language model to process non-linguistic scenes (e.g. mountain sunsets, wind noise, sirens), leading to language hallucination or model failure.

### Root Cause Analysis
The architecture lacked an explicit **Dual-Stream Sensory-Linguistic Gating Mechanism**:
- Sensory encoders must preserve non-linguistic perception (spatial geometry, pitch, timbre) by always streaming direct latents ($H_{\text{vis}}, H_{\text{aud}}$) to the Parietal Hub.
- In parallel, sensory encoders must feature a conditional linguistic gateway (Visual Word Form Gateway for vision, Acoustic Phoneme Gateway for audio) that detects symbolic/phonetic structure and forks it through Wernicke to produce structured syntactic latents ($H_{\text{lang}}$).

### Fix Details
1. Update `docs/Brain.md` ASCII diagram and Component Breakdown to incorporate the Dual-Stream Sensory-Linguistic Gating Architecture.
2. Add the formal gating mechanism:
   - Fork 1 (Direct Sensory Stream, Always Active): Sends $H_{\text{vis}}$ and $H_{\text{aud}}$ directly to Parietal Hub.
   - Fork 2 (Linguistic Gated Fork, Conditionally Active): Routes detected text glyphs and vocal phonemes through Wernicke.
3. Establish the Zero-Hallucination guarantee for non-linguistic inputs.

### Verification
- Validate schema and architectural consistency in `docs/Brain.md`.
- Ensure all downstream components (Parietal Hub, Frontal dlPFC, Neocortex) properly consume both direct sensory latents and Wernicke-parsed linguistic latents.

### Commit Reference
- Commit: Pending
