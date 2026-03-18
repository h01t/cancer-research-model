# Lessons Learned

Patterns discovered during the v0.2.0 refactor that should never be repeated.

## Data Pipeline

- **Never apply transforms inside both the dataset AND the trainer.** Pick one layer of responsibility. In FixMatch, the dataset returns raw PIL images; SSL wrappers apply transforms.
- **Validation data must use deterministic (test) transforms.** Never split from an augmented dataset — the val split inherits the training augmentation.
- **Always use `.copy()` when filtering a pandas DataFrame before modifying it.** Otherwise you get `SettingWithCopyWarning` and potential silent bugs.
- **Use local RNG instances (`random.Random(seed)`) instead of mutating global `random.seed()`.** Global seed mutation causes non-obvious side effects.

## Training

- **Override method names must match exactly.** If a subclass calls `self.train_epoch()` but the method is named `train_epoch_fixmatch()`, the base class method gets called with wrong arguments and crashes.
- **Guard `confusion_matrix().ravel()` for single-class predictions.** Early in training, the model may predict all one class. Pass `labels=[0, 1]` to force a 2x2 matrix.
- **Check `roc_auc_score` return value for NaN.** Newer sklearn versions return NaN instead of raising ValueError when only one class is present.

## Device Compatibility

- **Always check for MPS in addition to CUDA.** Apple Silicon uses `torch.backends.mps.is_available()`.
- **`pin_memory=True` is incompatible with MPS.** Only enable it for CUDA.
- **AMP on MPS uses `bfloat16`, not `float16`.** CUDA uses `float16` with `GradScaler`; MPS uses `bfloat16` without `GradScaler`.

## Dependencies

- **Use torchvision's built-in models over third-party packages.** `efficientnet-pytorch` is unmaintained; `torchvision.models.efficientnet_b0` is official.
- **Don't list unused dependencies.** `albumentations` and `opencv-python-headless` were in requirements.txt but never imported.
