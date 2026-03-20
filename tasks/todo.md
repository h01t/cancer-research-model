# Project TODO

## Completed

- [x] Project setup, packaging, and test infrastructure
- [x] Data exploration notebooks and CBIS-DDSM loading pipeline
- [x] Supervised EfficientNet baseline
- [x] FixMatch training path
- [x] Mean Teacher training path
- [x] Shared experiment runtime and dataset-builder refactor
- [x] Patient-aware splitting and correctness fixes
- [x] Mixed precision, warmup, clipping, class weighting, checkpointing
- [x] Optional W&B and TensorBoard integrations
- [x] Notebook refresh for the refactored codebase
- [x] Rescue, follow-up, and closure analysis scripts
- [x] Correct the supervised freeze/unfreeze behavior
- [x] Close the vanilla SSL comparison with the no-freeze supervised baseline
- [x] Refresh docs around the supervised pivot

## Active Next Steps

- [ ] Tune supervised augmentation and regularization
- [ ] Explore stronger initialization or pretraining strategies
- [ ] Evaluate higher-resolution and lesion-aware cropping strategies
- [ ] Improve calibration and threshold-selection analysis
- [ ] Expand dataset coverage tests beyond split-focused behavior
- [ ] Add a compact portfolio-ready project summary in the portfolio repo

## Historical / Archived Work

- [x] Preserve FixMatch and Mean Teacher codepaths for reproducibility
- [x] Preserve historical sweep scripts and summary utilities
- [x] Replace tracked raw result trees with compact versioned summaries
