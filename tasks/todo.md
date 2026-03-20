# Project TODO

## Completed

- [x] Project setup: environment, dependencies, license, project structure
- [x] Data exploration & preprocessing: analyze dataset, create data loaders
- [x] Implement supervised baseline with EfficientNet-B0
- [x] Implement FixMatch algorithm components
- [x] Create training pipeline for SSL experiments
- [x] Write documentation: notebooks, README, USER_GUIDE
- [x] Create experiment scripts and helper scripts

### v0.2.0 Refactor (completed)

- [x] Fix augmentation architecture (dataset returns raw PIL, transforms in wrappers)
- [x] Fix FixMatch trainer (method name, batch ratio, augmentation pipeline)
- [x] Fix supervised validation transforms (val uses test transforms now)
- [x] Fix device detection (MPS + CUDA + CPU auto-detection)
- [x] Fix metrics edge cases (single-class AUC, confusion matrix guard)
- [x] Fix Pandas SettingWithCopyWarning (.copy() on filtered DataFrames)
- [x] Add mixed precision training (AMP) for CUDA and MPS
- [x] Add learning rate warmup (linear, config-driven)
- [x] Add gradient clipping (configurable max_grad_norm)
- [x] Add EMA model for FixMatch pseudo-label generation
- [x] Add class-weighted loss (inverse frequency)
- [x] Switch to torchvision EfficientNet (remove efficientnet-pytorch dependency)
- [x] Make augmentation config-driven (read from YAML)
- [x] Add pseudo-label quality logging (mask ratio, sup/unsup loss)
- [x] Add W&B integration (optional)
- [x] Per-experiment checkpoints (inside output_dir)
- [x] Add pyproject.toml for proper packaging
- [x] Clean dependencies (remove albumentations, opencv, efficientnet-pytorch)
- [x] Clean dead config keys
- [x] Write 42 pytest tests with synthetic data (no real dataset required)
- [x] Fix shell scripts (Python version check, --max_epochs arg)
- [x] Cross-device checkpoint loading (weights_only, map_location)
- [x] Create SSL dataset wrappers (FixMatchLabeledDataset, FixMatchUnlabeledDataset)
- [x] Update README.md and documentation

### v0.3.0 Correctness Fixes (completed)

- [x] Patient-level data leakage fix (split by patient ID, not image index)
- [x] EMA tracks frozen-then-unfrozen backbone params (ema.refresh())
- [x] Complete FixMatch checkpoint save/load (EMA, scaler, ramp state, epoch, history)
- [x] Fix distribution alignment formula (post-softmax per ReMixMatch)
- [x] Add exception safety to EMA apply/restore (try/finally)
- [x] Fix print() warning in dataset.py → logger.warning()
- [x] Fix hardcoded blank image size (512 → 256 fallback)
- [x] 10 new tests (patient split, EMA refresh, checkpoint roundtrip) — 52 total

## Next Steps

- [x] Shared experiment runtime + dataset builder refactor
- [x] Thin training entrypoints for supervised, FixMatch, and Mean Teacher
- [x] Add Mean Teacher training path
- [x] Move ad hoc debug utilities out of the main `scripts/` surface
- [ ] Expand dataset.py coverage beyond split-only behavior
- [ ] Add tests for grad accumulation and distribution alignment edge cases
- [ ] Tighten config/docs consistency after rescue results land
- [ ] Re-run ablation study with patient-aware splits on CUDA workstation
- [ ] Update notebooks to work with refactored code
- [ ] Update USER_GUIDE.md with actual rescue and Mean Teacher results

## Future Enhancements

- [ ] Add more SSL algorithms (MixMatch, VAT, pseudo-label calibration variants)
- [ ] Support multi-class classification (calcification vs mass)
- [ ] Medical-specific augmentations (CLAHE, contrast normalization)
- [ ] Model export pipeline (ONNX, TorchScript)
- [ ] Deploy model as web service (FastAPI)
