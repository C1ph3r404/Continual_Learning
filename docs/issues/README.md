# Issue & Bug Database (`docs/issues/`)

This directory serves as the persistent issue and bug database for the Continual Learning architecture codebase. All encountered bugs, unexpected numerical artifacts, and failure modes must be tracked here to ensure they are never repeated.

---

## Issue Log Index

| Issue ID | Date | Component / Module | Description | Status |
| :--- | :--- | :--- | :--- | :--- |
| [ISSUE-001](ISSUE_001_dual_stream_linguistic_fork.md) | 2026-10-21 | Sensory / Wernicke / Parietal | Dual-stream sensory-linguistic split to prevent bypassing Wernicke or hallucinating language | NOT FIXED |

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
