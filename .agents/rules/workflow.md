# Development Workflow & Operational Rules

This document defines the mandatory project workflow and engineering rules for all development sessions.

---

## Core Rules

### Rule 1: Bug & Issue Tracking in `docs/issues/`
* **Persistent Issue Database:** Every bug, runtime exception, numerical instability, or architectural discrepancy encountered during development must be documented in `docs/issues/`.
* **Mandatory File Format:** Create an issue entry (or individual issue file `docs/issues/ISSUE_<id>.md` and register it in `docs/issues/README.md`).
* **Required Issue Metadata:**
  * **ID & Title:** Unique identifier (e.g., `ISSUE-001: <Title>`)
  * **Date & Context:** When and where the issue occurred (module, script, or test).
  * **Symptoms & Traceback:** Exact error message, stack trace, or aberrant behavior.
  * **Root Cause Analysis:** Why the failure occurred.
  * **Resolution & Fix Details:** Specific code and mathematical adjustments made.
  * **Status:** Must strictly be one of:
    * `NOT FIXED`
    * `FIXED (<full_or_short_commit_hash>)`
* **Zero Duplication Guarantee:** Before debugging or modifying modules, always consult `docs/issues/` to verify past solutions and prevent repeating previous mistakes.

### Rule 2: Mandatory Detailed Git Commits
* **Semantic & Descriptive:** All git commits must be detailed. Never use one-liner, vague commit messages such as "fix bug", "updates", or "wip".
* **Standard Commit Structure:**
  ```text
  <type>(<scope>): <concise high-level summary>

  - Motivation / Problem: Explain why this change is necessary.
  - Detailed Changes: Bulleted breakdown of exact modifications per file/module.
  - Verification: How the changes were tested (commands run, assertions passed).
  - Issues Referenced: Link to docs/issues/ISSUE_<id>.md and its resolution status.
  ```
* **Status Sync:** Once a bug fix commit is generated, update the corresponding issue entry in `docs/issues/` immediately with the resulting commit hash: `Status: FIXED (<commit_hash>)`.

---

## Architectural Development Workflow

1. **Step-by-Step Modular Construction:**
   * Build each brain-inspired module independently ([Occipital](file:///home/nate/Projects/Continual_Learning/docs/Brain.md#L51-L54), [Temporal](file:///home/nate/Projects/Continual_Learning/docs/Brain.md#L55-L58), [Wernicke](file:///home/nate/Projects/Continual_Learning/docs/Brain.md#L59-L63), [Parietal](file:///home/nate/Projects/Continual_Learning/docs/Brain.md#L64-L76), [Frontal dlPFC](file:///home/nate/Projects/Continual_Learning/docs/Brain.md#L77-L99), [Neocortex](file:///home/nate/Projects/Continual_Learning/docs/Brain.md#L100-L127)).
   * Adhere strictly to the **Strict Gradient Wall**: Sensory and Neocortical weights remain frozen during downstream reasoning loop execution.
2. **Pre-Execution Verification:**
   * Test every module with unit tests and schema verification scripts.
   * Log any unexpected failure immediately into `docs/issues/`.
3. **Commit Cadence:**
   * Commit logically atomic units of work with detailed commit messages and update `docs/issues/` status tags.
