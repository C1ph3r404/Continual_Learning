# ISSUE-006: GitHub Push Protection Violation Due to Exposed Hugging Face Secret

- **Date:** 2026-10-22
- **Component:** Security / Remote Training Pipeline (`notebooks/kaggle_dual_gpu_training.ipynb`, `.gitignore`)
- **Status:** FIXED (9a9c299)

### Symptoms / Traceback
When pushing local branch `main` to the remote repository `git@github-main:C1ph3r404/Continual_Learning.git`, GitHub pre-receive hook rejected the push with GH013 push protection violation:

```text
remote: error: GH013: Repository rule violations found for refs/heads/main.
remote: 
remote: - GITHUB PUSH PROTECTION
remote:   —————————————————————————————————————————
remote:     Resolve the following violations before pushing again
remote: 
remote:     - Push cannot contain secrets
remote: 
remote:       —— Hugging Face User Access Token ————————————————————
remote:        locations:
remote:          - commit: fe9bc9db294ed394c9be89d83ed6bd110edf62a4
remote:            path: notebooks/kaggle_dual_gpu_training.ipynb:45
remote:          - commit: 5c05eaefd75b679189fcd1b61f3b11d975b83ae8
remote:            path: notebooks/kaggle_dual_gpu_training.ipynb:45
remote: 
To github-main:C1ph3r404/Continual_Learning.git
 ! [remote rejected] main -> main (push declined due to repository rule violations)
error: failed to push some refs to 'github-main:C1ph3r404/Continual_Learning.git'
```

### Root Cause Analysis
A raw Hugging Face user access token (`hf_knRRlpzs...`) was hardcoded directly in cell 1 of `notebooks/kaggle_dual_gpu_training.ipynb` for automated checkpoint uploading/downloading, and committed into git history in commits `5c05eae` and `fe9bc9d`. Furthermore, `.gitignore` lacked comprehensive ignore rules for environment files (`.env`), token files, keys, and Kaggle/HuggingFace credential files.

### Fix Details
1. Scrub hardcoded `HF_TOKEN` from `notebooks/kaggle_dual_gpu_training.ipynb` across both active code and commit history, loading it dynamically via `kaggle_secrets.UserSecretsClient` or `os.environ.get("HF_TOKEN")`.
2. Expand `.gitignore` to strictly exclude `.env*`, `*.token`, `*.key`, `*.pem`, `kaggle.json`, `*secret*.json`, `.huggingface/`, and Hugging Face credential files.
3. Rewrite the offending git commits (`5c05eae` and `fe9bc9d`) to remove the secret token from repository history so that GitHub push protection passes.

### Verification
1. Run `git grep -i "hf_" $(git rev-list --all)` to verify 0 raw tokens exist in any commit object in the repository.
2. Verify updated `.gitignore` rules prevent staging credential or secret files.
3. Push to remote `origin main` to confirm GitHub push protection successfully admits the branch.

### Commit Reference
- Commit: `9a9c299`
