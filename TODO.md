Improvement Plan
Tier 1: Fix FixMatch Collapse (Critical)
#	Change	Rationale
1.1	Lower confidence threshold from 0.95 to 0.80	0.95 is too high for early training — the model never generates pseudo-labels during the critical learning phase (mask=0.0 for 3 epochs). Lowering to 0.80 lets pseudo-labels participate earlier when the model is less biased. The original FixMatch paper used 0.95 on CIFAR with 4000 labeled samples — with 100 samples on medical images, a lower threshold is needed.
1.2	Reduce learning rate to 0.0003	0.001 causes the model to overfit to 100 labeled samples within 5-6 epochs. By the time pseudo-labels start, the model is already biased. Slower learning gives pseudo-labels time to participate before overfitting.
1.3	Increase warmup from 5 to 10 epochs	Combined with the lower LR, this keeps the model in an "exploration" phase longer, generating more diverse predictions before confidence hardens.
1.4	Reduce lambda_u from 1.0 to 0.5 initially	The unsupervised loss, once it kicks in, overwhelms the supervised signal (0.26 unsup vs 0.18 sup by epoch 18). Halving it prevents pseudo-label bias from dominating.
1.5	Increase early_stopping_patience from 10 to 25	With 100 labeled samples, FixMatch needs more epochs for pseudo-labels to stabilize. Early stopping at epoch 11/24 is premature.
Tier 2: Reduce VRAM Usage
#	Change	Impact
2.1	Reduce unlabeled_batch_ratio from 7 to 3	Cuts unlabeled batch from 112 to 48 images. VRAM drops from ~7GB to ~3GB. The FixMatch paper used mu=7 with batch_size=64 on CIFAR-10 (tiny images). For 224x224 medical images with batch_size=16, mu=3 is more appropriate. More batches per epoch compensates for smaller batches.
2.2	Detach weak forward from computation graph	The logits_weak pseudo-label generation already uses torch.no_grad() — this is correct. But verify EMA apply/restore doesn't leak gradients.
2.3	Add torch.backends.cudnn.benchmark = True	Minor but free optimization for fixed-size inputs (all images are 224x224).
Tier 3: Structural Improvements for Better SSL
#	Change	Rationale
3.1	Freeze backbone for first N epochs, then unfreeze	With only 100 labeled samples, EfficientNet's 4M parameters overfit instantly. Freezing the backbone (training only the classifier head) for the first 10-15 epochs prevents early overfitting, keeps features general, and produces better pseudo-labels when they start. Then unfreeze for fine-tuning.
3.2	Distribution alignment for pseudo-labels	Track the class distribution of pseudo-labels per epoch. If it drifts (e.g., 90% benign), apply a correction factor. This is a known fix for confirmation bias in FixMatch.
3.3	Create a FixMatch-specific config	The current default.yaml was designed for supervised training. FixMatch needs different hyperparameters (lower LR, lower threshold, fewer warmup epochs, etc.). Create configs/fixmatch.yaml with SSL-optimized defaults.
3.4	Fix the numpy serialization in test_metrics.yaml	The supervised_250 results show raw numpy objects in YAML. The fixmatch_100 and supervised_100 scripts already cast to float() — but ensure all paths do this consistently.
3.5	Increase image resolution back to 512 when VRAM allows	224px loses significant detail in mammography. With unlabeled_batch_ratio: 3 and batch_size: 16, 512px should fit in ~7-8GB (same as current FixMatch at 224px with ratio 7).
Recommended New FixMatch Config
# configs/fixmatch.yaml — optimized for SSL with limited labels
training:
  batch_size: 16
  learning_rate: 0.0003      # lower than supervised (was 0.001)
  warmup_epochs: 10           # longer warmup (was 5)
  early_stopping_patience: 25 # more patience (was 10)
  num_epochs: 150             # more epochs (was 100)
ssl:
  confidence_threshold: 0.80  # lower threshold (was 0.95)
  lambda_u: 0.5               # softer unsup weight (was 1.0)
  unlabeled_batch_ratio: 3    # lower ratio (was 7) — saves VRAM
  use_ema: true
  ema_decay: 0.999
model:
  freeze_backbone: true       # freeze for first epochs
Priority Order
1. Create configs/fixmatch.yaml with the SSL-optimized params (Tier 1 + 2)
2. Implement backbone freeze/unfreeze schedule in the trainer (Tier 3.1)
3. Implement pseudo-label distribution alignment (Tier 3.2)
4. Add cudnn.benchmark (Tier 2.3)
---
