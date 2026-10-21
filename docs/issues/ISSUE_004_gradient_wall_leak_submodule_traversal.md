# ISSUE-004: Gradient Wall Leak via Submodule Parameter Traversal in Parietal Hub

- **Date:** 2026-10-21
- **Component:** Parietal Hub / Model Hierarchy / Strict Gradient Wall
- **Status:** NOT FIXED

### Symptoms / Traceback
During `test_strict_gradient_wall`, the test failed with:
```text
AssertionError: Gradient Wall breach: Neocortex param 'W_key' received gradient!
src/continual_brain/pipeline/stage2_reasoning.py:36: AssertionError
```
Even though `self.neocortex.freeze()` was called, `W_key` and `W_value` had `requires_grad = True` and accumulated gradients during Stage 2 backpropagation.

### Root Cause Analysis
In `ParietalHub.__init__`:
```python
self.neocortex = neocortex
```
In PyTorch, assigning an `nn.Module` instance as an attribute registers it as a child submodule (`self._modules['neocortex'] = neocortex`).
When `ContinualBrain.set_training_stage(2)` called `self._set_requires_grad(self.parietal, True)`, PyTorch's recursive `module.parameters()` traversal descended into `self.neocortex`, inadvertently re-enabling `requires_grad = True` on all Neocortex weights! This breached the Strict Gradient Wall.

### Fix Details
1. In `ContinualBrain`, manage `self.neocortex` at the root model level and pass its forward pass or decouple its parameter registration in `ParietalHub` (e.g., using `object.__setattr__(self, 'neocortex', neocortex)` or passing `neocortex` explicitly during `parietal.forward()`).
2. Alternatively, ensure `_set_requires_grad(self.parietal, True, recurse=False)` or explicit parameter iteration that excludes `self.neocortex`, and ensure `self.neocortex.freeze()` is applied with zero gradient accumulation.
3. In `ParietalHub.forward()`, pass the Neocortex memory reference explicitly or wrap the associative recall in `torch.no_grad()` during runtime Stage 2 to guarantee zero gradient tracking through Neocortex parameters.

### Verification
- Run `pytest -v tests/test_text_brain.py`.
- Verify `param.grad is None` for all parameters of `wernicke` and `neocortex`.

### Commit Reference
- Commit: Pending
