# Current Progress Brief

## Project

This project is a mammography classification research effort built on the CBIS-DDSM dataset. The broader goal is not just a high AUC model, but a reproducible foundation for clinically oriented breast-imaging decision support.

## Where It Started

The original research question was whether semi-supervised learning could help in a low-label medical-imaging setting. The early project compared:

- supervised learning
- FixMatch
- Mean Teacher

across multiple labeled-data budgets.

## How We Got Here

The first major finding was methodological rather than algorithmic: the original supervised baseline had been artificially weakened by freezing behavior. Once that baseline was corrected, supervised fine-tuning clearly beat the vanilla SSL methods explored in the repo.

That shifted the project into a supervised-first phase. From there, the work focused on:

- no-freeze supervised baselines
- controlled sweeps over resolution, backbone, regularization, and optimizer
- clinical-style evaluation including PR AUC, calibration, fixed-sensitivity specificity, and grouped exam metrics

The final optimizer head-to-head showed that the best overall candidate is now the `label_smoothing + adamw` supervised setup.

## Current Result

Promoted baseline: `default_nofreeze_ls_adamw`

- backbone: EfficientNet-B0
- input size: 512x512
- optimizer: AdamW
- label smoothing: 0.1

Combined 6-seed clinical-style summary:

| Metric | Value |
|--------|-------|
| Mean validation ROC AUC | 0.8633 |
| Mean test ROC AUC | 0.7589 |
| Mean test PR AUC | 0.6748 |
| Mean test sensitivity | 0.7110 |
| Mean test specificity | 0.6651 |
| Mean specificity at 0.90 target sensitivity | 0.5833 |
| Mean exam-level ROC AUC | 0.7668 |

## What Comes Next

The next development phase is no longer broad architecture search. It is focused on making the promoted baseline more clinically credible:

- calibration refinement, starting with temperature scaling
- deeper false-positive and false-negative review
- subgroup and grouped-exam analysis
- exam-level and multi-view modeling
- external validation on additional mammography datasets

## Possible Clinical Path

This project is still research, not a deployable medical device. A realistic clinical path would frame the model as radiologist decision support rather than autonomous diagnosis.

Before any real clinical implementation, the model would need:

- stronger calibration and threshold selection
- external, source-shifted validation
- subgroup robustness analysis
- reader-study style evaluation
- traceable model documentation and change control

The current state is best described as an experimentally grounded, clinically aware research baseline that is now strong enough to justify that next stage of work.
