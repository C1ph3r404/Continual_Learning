# Workspace Guidelines: Modular Brain-Inspired Continual Learning

All agent sessions must strictly adhere to the following workflow and rules:

## 1. Issue & Bug Database (`docs/issues/`)
- Any bug, error, exception, or failure encountered during development MUST be documented in `docs/issues/`.
- Every entry must specify its current status:
  - `Status: NOT FIXED`
  - `Status: FIXED (<commit_hash>)`
- This directory acts as the single source of truth for resolved failure modes so we never repeat the same bug by mistake.
- Check `docs/issues/` before troubleshooting any component.

## 2. Mandatory Detailed Git Commits
- All git commits must be detailed and descriptive.
- No vague, single-line messages (e.g., "fix", "updated code").
- Structure:
  - Header: `<type>(<scope>): <summary>`
  - Motivation / Problem Statement
  - Detailed Changes (bulleted by file/subsystem)
  - Verification & Test results
  - Issue references (linking to `docs/issues/` entries and their commit hashes)
- When a bug is resolved, immediately update its status in `docs/issues/` with the commit hash.
