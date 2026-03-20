# Final Results

This project started as a comparison between supervised learning and semi-supervised learning on CBIS-DDSM mammography classification.

## Grouped Mean Validation AUC

| Method | 100 labels | 250 labels | 500 labels |
|--------|------------|------------|------------|
| Supervised (frozen historical baseline) | 0.6033 | 0.7097 | 0.8084 |
| Supervised no-freeze (official baseline) | 0.7442 | 0.8370 | 0.8575 |
| FixMatch | 0.6559 | 0.6788 | - |
| Mean Teacher | 0.6325 | 0.6595 | 0.6861 |

## Conclusion

The corrected no-freeze supervised baseline is the strongest setup tested in this repository. The earlier SSL promise was mostly a baseline-quality issue: once supervised fine-tuning was fixed, both vanilla FixMatch and the current Mean Teacher implementation underperformed.

## What This Means

- `supervised_nofreeze` is the official baseline going forward
- FixMatch and Mean Teacher remain in the repo as historical comparison paths
- the next research phase should focus on stronger supervised training, pretraining, and analysis rather than more vanilla SSL tuning
