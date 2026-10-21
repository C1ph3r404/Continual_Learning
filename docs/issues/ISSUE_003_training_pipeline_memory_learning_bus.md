# ISSUE-003: Training Pipeline Alignment with Sensory Gating and Neocortex Memory Learning Bus

- **Date:** 2026-10-21
- **Component:** Training Pipeline / Sensory Gating / Neocortex Learning Architecture
- **Status:** NOT FIXED

### Symptoms / Traceback
The initial training pipeline specification did not formally account for:
1. **Sensory Gating Supervision:** The Occipital Gate (text vs. scene) and Temporal Gate (speech vs. ambient audio) require explicit calibration/training so they can accurately route linguistic tokens to Wernicke while routing physical tokens directly to the Parietal Hub.
2. **Memory Learning Bus Pathway:** The Neocortex training pipeline needed explicit specification that Neocortical weights ($W_{\text{key}}, W_{\text{value}}$) are trained by feeding the latents produced by the frozen sensory encoders (Wernicke, Occipital, Temporal) through the dedicated offline Memory Learning Bus, completely decoupled from runtime inference.

### Root Cause Analysis
- Without explicit gating training in Stage 0, uncalibrated gates would misroute non-linguistic scenes to Wernicke or speech to the physical audio stream.
- The Neocortex memory learning phase must be explicitly formalized as receiving feature latents from the frozen sensory encoders via the offline Memory Learning Bus.

### Fix Details
1. Clean up and align the flowchart in `docs/Brain.md` to properly render the `Memory Learning Bus (Offline Training Only)` tapping the outputs of Wernicke, Occipital, and Temporal into the Neocortex.
2. Update Section F (Training Pipeline) in `docs/Brain.md`:
   - Stage 0: Include calibration of Occipital Gate and Temporal Gate alongside foundational sensory encoder freezing.
   - Stage 1: Explicitly describe Neocortex memory learning over the Memory Learning Bus using frozen encoder representations.
   - Stage 2: Clarify that Sensory Encoders, Gates, and Neocortex are strictly frozen; only Parietal Hub and Frontal Core are updated.

### Verification
- Validate consistency between the flowchart and the staged training curriculum in `docs/Brain.md`.
- Verify datasets in `scripts/verify_datasets.py` remain aligned with each stage.

### Commit Reference
- Commit: Pending
