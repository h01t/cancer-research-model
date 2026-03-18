"""
FixMatch semi-supervised learning trainer.

Data flow:
- Labeled loader yields (img_tensor, label)  — weak augmentation applied by dataset
- Unlabeled loader yields (img_weak, img_strong) — both views created from same PIL image
- Trainer computes supervised loss on labeled data + consistency loss on unlabeled data

Key differences from base trainer:
- Two data loaders (labeled + unlabeled)
- Pseudo-label generation with confidence thresholding
- Optional EMA model for stable pseudo-labels
- Tracks pseudo-label quality metrics
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ema import EMAModel
from .metrics import compute_metrics
from .trainer import BaseTrainer

logger = logging.getLogger(__name__)


class FixMatchTrainer(BaseTrainer):
    """Trainer for FixMatch semi-supervised learning."""

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        device: torch.device | None = None,
        output_dir: str | Path | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(model, config, device, output_dir, class_weights)

        # SSL parameters
        ssl_config = config["ssl"]
        self.confidence_threshold = ssl_config["confidence_threshold"]
        self.lambda_u = ssl_config["lambda_u"]

        # EMA model for pseudo-label generation
        self.use_ema = ssl_config.get("use_ema", True)
        self.ema_decay = ssl_config.get("ema_decay", 0.999)
        self.ema: EMAModel | None = None
        if self.use_ema:
            self.ema = EMAModel(self.model, decay=self.ema_decay)
            logger.info(f"EMA enabled with decay={self.ema_decay}")

        # Extended history for SSL-specific metrics
        self.history["sup_loss"] = []
        self.history["unsup_loss"] = []
        self.history["mask_ratio"] = []

    def train_epoch_ssl(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
    ) -> tuple[float, dict[str, float]]:
        """Train for one epoch with FixMatch.

        Args:
            labeled_loader: yields (img_tensor, label)
            unlabeled_loader: yields (img_weak_tensor, img_strong_tensor)

        Returns:
            (epoch_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        total_sup_loss = 0.0
        total_unsup_loss = 0.0
        total_mask_ratio = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        # Number of batches per epoch (driven by labeled loader)
        num_batches = len(labeled_loader)

        pbar = tqdm(range(num_batches), desc="Training (FixMatch)", leave=False)
        for batch_idx in pbar:
            # Get labeled batch
            try:
                labeled_data, labeled_targets = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_data, labeled_targets = next(labeled_iter)

            # Get unlabeled batch (weak + strong views)
            try:
                unlabeled_weak, unlabeled_strong = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_weak, unlabeled_strong = next(unlabeled_iter)

            # Move to device
            labeled_data = labeled_data.to(self.device)
            labeled_targets = labeled_targets.to(self.device)
            unlabeled_weak = unlabeled_weak.to(self.device)
            unlabeled_strong = unlabeled_strong.to(self.device)

            self.optimizer.zero_grad()

            # --- Forward pass with optional AMP ---
            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    loss, sup_loss, unsup_loss, mask_ratio, logits_labeled = (
                        self._compute_fixmatch_loss(
                            labeled_data,
                            labeled_targets,
                            unlabeled_weak,
                            unlabeled_strong,
                        )
                    )
            else:
                loss, sup_loss, unsup_loss, mask_ratio, logits_labeled = (
                    self._compute_fixmatch_loss(
                        labeled_data,
                        labeled_targets,
                        unlabeled_weak,
                        unlabeled_strong,
                    )
                )

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.use_grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Update EMA
            if self.ema is not None:
                self.ema.update(self.model)

            # Track losses
            total_loss += loss.item()
            total_sup_loss += sup_loss.item()
            total_unsup_loss += unsup_loss.item()
            total_mask_ratio += mask_ratio

            # Collect predictions for labeled data metrics
            with torch.no_grad():
                probs_labeled = torch.softmax(logits_labeled.float(), dim=1)
                preds_labeled = torch.argmax(probs_labeled, dim=1)

            all_preds.extend(preds_labeled.cpu().numpy())
            all_labels.extend(labeled_targets.cpu().numpy())
            all_probs.extend(probs_labeled[:, 1].cpu().numpy())

            if batch_idx % self.log_freq == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "sup": f"{sup_loss.item():.4f}",
                        "unsup": f"{unsup_loss.item():.4f}",
                        "mask": f"{mask_ratio:.2f}",
                    }
                )

        # Epoch-level metrics
        epoch_loss = total_loss / max(num_batches, 1)
        epoch_sup_loss = total_sup_loss / max(num_batches, 1)
        epoch_unsup_loss = total_unsup_loss / max(num_batches, 1)
        epoch_mask_ratio = total_mask_ratio / max(num_batches, 1)

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        metrics.update(
            {
                "loss": epoch_loss,
                "sup_loss": epoch_sup_loss,
                "unsup_loss": epoch_unsup_loss,
                "mask_ratio": epoch_mask_ratio,
            }
        )

        return epoch_loss, metrics

    def _compute_fixmatch_loss(
        self,
        labeled_data: torch.Tensor,
        labeled_targets: torch.Tensor,
        unlabeled_weak: torch.Tensor,
        unlabeled_strong: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """Compute FixMatch combined loss.

        Returns:
            (total_loss, sup_loss, unsup_loss, mask_ratio, logits_labeled)
        """
        # Supervised loss on labeled data
        logits_labeled = self.model(labeled_data)
        sup_loss = self.criterion(logits_labeled, labeled_targets)

        # Generate pseudo-labels from weak augmentation of unlabeled data
        if self.ema is not None:
            # Use EMA model for more stable pseudo-labels
            self.ema.apply(self.model)
            with torch.no_grad():
                logits_weak = self.model(unlabeled_weak)
            self.ema.restore(self.model)
        else:
            with torch.no_grad():
                logits_weak = self.model(unlabeled_weak)

        probs_weak = torch.softmax(logits_weak, dim=1)
        max_probs, pseudo_labels = torch.max(probs_weak, dim=1)
        confidence_mask = max_probs.ge(self.confidence_threshold).float()
        mask_ratio = confidence_mask.mean().item()

        # Consistency loss: strong augmentation should match pseudo-labels
        logits_strong = self.model(unlabeled_strong)
        unsup_loss = (
            F.cross_entropy(logits_strong, pseudo_labels, reduction="none") * confidence_mask
        ).mean()

        # Combined loss
        total_loss = sup_loss + self.lambda_u * unsup_loss

        return total_loss, sup_loss, unsup_loss, mask_ratio, logits_labeled

    def train(  # type: ignore[override]
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Main FixMatch training loop."""
        logger.info(
            f"Starting FixMatch training: {self.num_epochs} epochs on {self.device} "
            f"(AMP={'on' if self.use_amp else 'off'}, "
            f"EMA={'on' if self.ema else 'off'}, "
            f"tau={self.confidence_threshold}, lambda_u={self.lambda_u})"
        )

        for epoch in range(self.num_epochs):
            self._apply_warmup(epoch)

            current_lr = self._get_current_lr()
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} (lr={current_lr:.6f})")

            # Train with FixMatch
            train_loss, train_metrics = self.train_epoch_ssl(labeled_loader, unlabeled_loader)

            # Validate (uses EMA model if available)
            if self.ema is not None:
                self.ema.apply(self.model)
            val_loss, val_metrics = self.validate(val_loader)
            if self.ema is not None:
                self.ema.restore(self.model)

            # Update scheduler (after warmup)
            if epoch >= self.warmup_epochs and self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("auc", 0.0))
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["train_auc"].append(train_metrics.get("auc", 0.0))
            self.history["val_auc"].append(val_metrics.get("auc", 0.0))
            self.history["learning_rate"].append(current_lr)
            self.history["sup_loss"].append(train_metrics["sup_loss"])
            self.history["unsup_loss"].append(train_metrics["unsup_loss"])
            self.history["mask_ratio"].append(train_metrics["mask_ratio"])

            # Print metrics
            print(
                f"  Train — Loss: {train_loss:.4f} "
                f"(sup: {train_metrics['sup_loss']:.4f}, unsup: {train_metrics['unsup_loss']:.4f})"
            )
            print(
                f"  Train — Acc: {train_metrics['accuracy']:.4f}, "
                f"AUC: {train_metrics.get('auc', 0.0):.4f}, "
                f"Mask: {train_metrics['mask_ratio']:.2f}"
            )
            print(
                f"  Val   — Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                f"AUC: {val_metrics.get('auc', 0.0):.4f}"
            )

            # W&B logging
            self._log_wandb(
                {
                    "train/loss": train_loss,
                    "train/sup_loss": train_metrics["sup_loss"],
                    "train/unsup_loss": train_metrics["unsup_loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/auc": train_metrics.get("auc", 0.0),
                    "train/mask_ratio": train_metrics["mask_ratio"],
                    "val/loss": val_loss,
                    "val/accuracy": val_metrics["accuracy"],
                    "val/auc": val_metrics.get("auc", 0.0),
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            # Save periodic checkpoint
            if (epoch + 1) % self.save_freq == 0:
                self.save_checkpoint(epoch + 1, val_metrics.get("auc", 0.0))

            # Early stopping
            current_auc = val_metrics.get("auc", 0.0)
            if current_auc > self.best_val_auc:
                self.best_val_auc = current_auc
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch + 1, current_auc, best=True)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"  Early stopping triggered after {epoch + 1} epochs")
                    break

        # Finish W&B run
        if self.use_wandb and self.wandb_run is not None:
            import wandb

            wandb.finish()
