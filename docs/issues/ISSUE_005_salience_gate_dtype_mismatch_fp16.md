# ISSUE-005: Salience Gate Dtype Mismatch (Half vs Float) in Remote GPU Training

- **Date:** 2026-10-22
- **Component:** Salience Gate / Training Pipeline (`colab/train_salience_gate.py`)
- **Status:** FIXED (238006e)

### Symptoms / Traceback
When executing `colab/train_salience_gate.py` on remote Colab with a Tesla T4 GPU, the first training step fails during `gate.get_logits(h_lang)`:

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float
  File "/tmp/ipykernel_.../train_salience_gate.py", line 180, in main
    logits = gate.get_logits(h_lang)
  File "/tmp/ipykernel_.../train_salience_gate.py", line 60, in get_logits
    return self.net(h).squeeze(-1)
  File "torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
```

### Root Cause Analysis
The frozen Wernicke backbone (`microsoft/deberta-v3-small`) on CUDA produces activations in `torch.float16` (`Half`), whereas the `SalienceGate` MLP is initialized with default parameters in `torch.float32` (`Float`). When `F.linear` executes matrix multiplication between `h_lang` (Half) and `linear.weight` (Float), PyTorch raises a dtype mismatch RuntimeError.

### Fix Details
1. Ensure input tensor `h` in `SalienceGate.forward` and `SalienceGate.get_logits` is dynamically cast to `self.net[0].weight.dtype` (or `.float()`) before the linear layers.
2. Ensure consistent handling in `colab/train_salience_gate.py` and `src/continual_brain/neocortex/salience_gate.py`.

### Verification
Execute training on remote Colab session with GPU T4 and verify all 8 epochs complete successfully without dtype errors.

### Commit Reference
- Commit: `238006e`
