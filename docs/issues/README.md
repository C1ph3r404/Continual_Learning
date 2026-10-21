# Issue & Bug Database (`docs/issues/`)

This directory serves as the persistent issue and bug database for the Continual Learning architecture codebase. All encountered bugs, unexpected numerical artifacts, and failure modes must be tracked here to ensure they are never repeated.

---

## Issue Log Index

| Issue ID | Date | Component / Module | Description | Status |
| :--- | :--- | :--- | :--- | :--- |
| [ISSUE-001](ISSUE_001_dual_stream_linguistic_fork.md) | 2026-10-21 | Sensory / Wernicke / Parietal | Dual-stream sensory-linguistic split to prevent bypassing Wernicke or hallucinating language | FIXED (e32ea82) |
| [ISSUE-002](ISSUE_002_flowchart_sensory_gating_and_neocortex_coupling.md) | 2026-10-21 | Architecture Flowchart / Parietal / Neocortex | Correct sensory post-encoder gating, eliminate false runtime Neocortex queries from sensory engines | FIXED (9d2c93d) |
| [ISSUE-003](ISSUE_003_training_pipeline_memory_learning_bus.md) | 2026-10-21 | Training Pipeline / Sensory Gating / Neocortex | Update training curriculum for sensory gating calibration and offline memory learning bus | NOT FIXED |

---

## Issue Entry Template

When a bug or issue is identified, create a file `docs/issues/ISSUE_<id>.md` (or append to the log below) with the following structure:

```markdown
# ISSUE-<id>: <Brief Description>

- **Date:** YYYY-MM-DD
- **Component:** (e.g., Parietal Hub / Frontal dlPFC / Neocortex / Wernicke / Training Loop)
- **Status:** NOT FIXED | FIXED (<commit_hash>)

### Symptoms / Traceback
Describe the symptom, assertion failure, or include the exact traceback:
\`\`\`text
<Traceback or error message>
\`\`\`

### Root Cause Analysis
Explain why the bug occurred (e.g., tensor shape mismatch, gradient leak across the gradient wall, device mismatch, etc.).

### Fix Details
Describe what code changes resolved the issue.

### Verification
Explain how the fix was validated (e.g., test command, assertion check).

### Commit Reference
- Commit: `<hash>`
```
