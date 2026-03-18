# Project TODO

## Completed (✓)

- [x] Project setup: environment, dependencies, license, project structure
- [x] Data exploration & preprocessing: analyze dataset, create data loaders
- [x] Implement supervised baseline with EfficientNet-B0
- [x] Implement FixMatch algorithm components
- [x] Create training pipeline for SSL experiments (supervised + FixMatch trainers)
- [x] Write documentation: Jupyter notebook (01_data_exploration), README, comprehensive USER_GUIDE.md
- [x] Write unit tests for critical components (test_supervised.py)
- [x] Create experiment scripts (train_supervised.py, train_fixmatch.py)
- [x] Create remaining Jupyter notebooks (02_training_curves, 03_embeddings, 04_model_interpretation)
- [x] Create helper scripts (quick_start.sh, run_ablation.sh)
- [x] Update README.md to reference USER_GUIDE.md and new notebooks

## In Progress / Optional

- [ ] Run ablation study with labeled subsets (100,250,500) – requires GPU time

## Future Enhancements

- [ ] Add more SSL algorithms (e.g., MixMatch, ReMixMatch)
- [ ] Support multi-class classification (calcification vs mass)
- [ ] Integrate advanced augmentations (e.g., medical-specific transforms)
- [ ] Deploy model as web service (FastAPI)