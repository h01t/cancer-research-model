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

        # Confidence threshold (adaptive or static)
        self.confidence_threshold = ssl_config["confidence_threshold"]  # base/fallback
        self.confidence_threshold_start = ssl_config.get(
            "confidence_threshold_start", self.confidence_threshold
        )
        self.confidence_threshold_end = ssl_config.get(
            "confidence_threshold_end", self.confidence_threshold
        )
        self.confidence_threshold_ramp_epochs = ssl_config.get(
            "confidence_threshold_ramp_epochs", 0
        )
        self.current_confidence_threshold = self.confidence_threshold

        if self.confidence_threshold_ramp_epochs > 0:
            logger.info(
                f"Confidence threshold scheduling: {self.confidence_threshold_start} → {self.confidence_threshold_end} "
                f"over {self.confidence_threshold_ramp_epochs} epochs"
            )
        else:
            logger.info(f"Confidence threshold static: {self.confidence_threshold}")

        # Lambda_u scheduling (dynamic or static)
        self.lambda_u = ssl_config["lambda_u"]  # base/default value
        self.lambda_u_start = ssl_config.get("lambda_u_start", self.lambda_u)
        self.lambda_u_end = ssl_config.get("lambda_u_end", self.lambda_u)
        self.lambda_u_ramp_epochs = ssl_config.get("lambda_u_ramp_epochs", 0)

        if self.lambda_u_ramp_epochs > 0:
            logger.info(
                f"Lambda_u scheduling: {self.lambda_u_start} → {self.lambda_u_end} "
                f"over {self.lambda_u_ramp_epochs} epochs"
            )
        else:
            logger.info(f"Lambda_u static: {self.lambda_u}")

        # Current lambda_u (updated each epoch)
        self.current_lambda_u = self.lambda_u

        # Gradient accumulation (memory efficiency)
        training_config = config["training"]
        self.gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
        if self.gradient_accumulation_steps > 1:
            effective_batch = training_config["batch_size"] * self.gradient_accumulation_steps
            logger.info(
                f"Gradient accumulation: {self.gradient_accumulation_steps} steps "
                f"(effective batch size: {effective_batch})"
            )

        # Distribution alignment for pseudo-labels
        self.distribution_alignment = ssl_config.get("distribution_alignment", True)
        self.temperature_smoothing = ssl_config.get("temperature_smoothing", 0.01)
        self.labeled_class_distribution = None  # π_labeled (stored on CPU)
        num_classes = config["model"]["num_classes"]
        # Initialize π_pseudo as uniform distribution on CPU (memory efficient)
        self.pseudo_label_distribution = (
            torch.ones(num_classes, dtype=torch.float32, device="cpu") / num_classes
        )
        self.distribution_ratio = None  # π_labeled / π_pseudo, stored on device when computed

        if self.distribution_alignment:
            logger.info(
                f"Distribution alignment enabled for pseudo-labels (num_classes={num_classes})"
            )

        # EMA model for pseudo-label generation
        self.use_ema = ssl_config.get("use_ema", True)
        self.ema_decay = ssl_config.get("ema_decay", 0.999)
        self.ema: EMAModel | None = None
        if self.use_ema:
            self.ema = EMAModel(self.model, decay=self.ema_decay)
            logger.info(f"EMA enabled with decay={self.ema_decay}")

        # Freeze/unfreeze schedule
        model_config = config["model"]
        self.freeze_backbone_epochs = model_config.get("freeze_backbone_epochs", 0)
        self.backbone_unfrozen = False
        if model_config.get("freeze_backbone", False):
            logger.info(
                f"Backbone frozen — will unfreeze after epoch {self.freeze_backbone_epochs}"
            )
        else:
            logger.info("Backbone not frozen (full fine-tuning from start)")

        # Extended history for SSL-specific metrics
        self.history["sup_loss"] = []
        self.history["unsup_loss"] = []
        self.history["mask_ratio"] = []
        self.history["lambda_u"] = []

    def _get_current_lambda_u(self, epoch: int) -> float:
        """Compute current lambda_u value based on scheduling."""
        if self.lambda_u_ramp_epochs <= 0 or epoch >= self.lambda_u_ramp_epochs:
            return self.lambda_u_end
        # Linear ramp from start to end
        progress = epoch / self.lambda_u_ramp_epochs
        return self.lambda_u_start + progress * (self.lambda_u_end - self.lambda_u_start)

    def _get_current_confidence_threshold(self, epoch: int) -> float:
        """Compute current confidence threshold based on scheduling."""
        if (
            self.confidence_threshold_ramp_epochs <= 0
            or epoch >= self.confidence_threshold_ramp_epochs
        ):
            return self.confidence_threshold_end
        # Linear ramp from start to end
        progress = epoch / self.confidence_threshold_ramp_epochs
        return self.confidence_threshold_start + progress * (
            self.confidence_threshold_end - self.confidence_threshold_start
        )

    def _compute_labeled_class_distribution(self, labeled_loader: DataLoader) -> None:
        """Compute π_labeled from the labeled dataset (stored on CPU)."""
        class_counts = torch.zeros(
            self.config["model"]["num_classes"], dtype=torch.float32, device=self.device
        )
        total = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in labeled_loader:
                labels = labels.to(self.device)
                for cls_idx in range(self.config["model"]["num_classes"]):
                    class_counts[cls_idx] += (labels == cls_idx).sum()
                total += labels.shape[0]
        self.labeled_class_distribution = (class_counts / total).cpu()
        logger.info(
            f"Labeled class distribution π_labeled: {self.labeled_class_distribution.numpy()}"
        )

    def _update_pseudo_label_distribution(self, pseudo_labels: torch.Tensor) -> None:
        """Update π_pseudo with exponential moving average across epochs (CPU)."""
        if pseudo_labels.dim() == 0 or pseudo_labels.numel() == 0:
            return
        # Move to CPU for memory efficiency (small integer tensor)
        pseudo_labels_cpu = pseudo_labels.cpu()
        num_classes = self.config["model"]["num_classes"]
        # Compute class distribution using bincount (fast)
        counts = torch.bincount(pseudo_labels_cpu, minlength=num_classes).float()
        current_dist = counts / pseudo_labels_cpu.numel()
        # EMA update (alpha = 0.9)
        alpha = 0.9
        self.pseudo_label_distribution = (
            alpha * self.pseudo_label_distribution + (1 - alpha) * current_dist
        )
        self._compute_distribution_ratio()

    def _compute_distribution_ratio(self) -> None:
        """Compute π_labeled / π_pseudo for distribution alignment (with smoothing).

        The ratio is used to re-weight pseudo-label probabilities so that
        the pseudo-label class distribution matches the labeled distribution.
        Ref: ReMixMatch (Berthelot et al., 2020).
        """
        if self.labeled_class_distribution is None or self.pseudo_label_distribution is None:
            return
        eps = self.temperature_smoothing
        self.distribution_ratio = (
            (self.labeled_class_distribution + eps) / (self.pseudo_label_distribution + eps)
        ).to(self.device)
        logger.debug(f"Distribution ratio (π_l/π_p): {self.distribution_ratio.cpu().numpy()}")

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
        all_pseudo_labels = []  # for distribution alignment

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        # Number of batches per epoch (driven by labeled loader)
        num_batches = len(labeled_loader)

        # Gradient accumulation state
        accumulation_step = 0

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

            # Gradient accumulation: zero gradients only at start of accumulation cycle
            if accumulation_step == 0:
                self.optimizer.zero_grad()

            # --- Forward pass with optional AMP ---
            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    (
                        loss_raw,
                        sup_loss,
                        unsup_loss,
                        mask_ratio,
                        logits_labeled,
                        batch_pseudo_labels,
                        batch_confidence_mask,
                    ) = self._compute_fixmatch_loss(
                        labeled_data,
                        labeled_targets,
                        unlabeled_weak,
                        unlabeled_strong,
                    )
            else:
                (
                    loss_raw,
                    sup_loss,
                    unsup_loss,
                    mask_ratio,
                    logits_labeled,
                    batch_pseudo_labels,
                    batch_confidence_mask,
                ) = self._compute_fixmatch_loss(
                    labeled_data,
                    labeled_targets,
                    unlabeled_weak,
                    unlabeled_strong,
                )

            # Scale loss for gradient accumulation
            loss_for_backward = loss_raw / self.gradient_accumulation_steps

            # Collect pseudo-labels for distribution alignment (only confident ones)
            if self.distribution_alignment:
                masked_pseudo = batch_pseudo_labels[batch_confidence_mask.bool()]
                if masked_pseudo.numel() > 0:
                    all_pseudo_labels.append(masked_pseudo)

            # Backward pass with scaled loss
            if self.scaler is not None:
                self.scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            accumulation_step += 1

            # Perform optimizer step if accumulation cycle complete
            if accumulation_step == self.gradient_accumulation_steps:
                if self.scaler is not None:
                    if self.use_grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.use_grad_clip:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                accumulation_step = 0

            # Track losses (use raw loss, not scaled)
            total_loss += loss_raw.item()

            # Update EMA
            if self.ema is not None:
                self.ema.update(self.model)

            # Track losses
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
                        "loss": f"{loss_raw.item():.4f}",
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

        # Update pseudo-label distribution for distribution alignment
        if self.distribution_alignment and all_pseudo_labels:
            concatenated_pseudo = torch.cat(all_pseudo_labels)
            self._update_pseudo_label_distribution(concatenated_pseudo)
            logger.debug(f"Updated π_pseudo: {self.pseudo_label_distribution.cpu().numpy()}")

        return epoch_loss, metrics

    def _compute_fixmatch_loss(
        self,
        labeled_data: torch.Tensor,
        labeled_targets: torch.Tensor,
        unlabeled_weak: torch.Tensor,
        unlabeled_strong: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Compute FixMatch combined loss.

        Returns:
            (total_loss, sup_loss, unsup_loss, mask_ratio, logits_labeled, pseudo_labels, confidence_mask)
        """
        # Supervised loss on labeled data
        logits_labeled = self.model(labeled_data)
        sup_loss = self.criterion(logits_labeled, labeled_targets)

        # Generate pseudo-labels from weak augmentation of unlabeled data
        if self.ema is not None:
            # Use EMA model for more stable pseudo-labels
            self.ema.apply(self.model)
            try:
                with torch.no_grad():
                    logits_weak = self.model(unlabeled_weak)
            finally:
                self.ema.restore(self.model)
        else:
            with torch.no_grad():
                logits_weak = self.model(unlabeled_weak)

        probs_weak = torch.softmax(logits_weak, dim=1)

        # Distribution alignment: re-weight probabilities so pseudo-label
        # class distribution matches the labeled distribution (ReMixMatch)
        if self.distribution_alignment and self.distribution_ratio is not None:
            probs_weak = probs_weak * self.distribution_ratio
            probs_weak = probs_weak / probs_weak.sum(dim=1, keepdim=True)

        max_probs, pseudo_labels = torch.max(probs_weak, dim=1)
        confidence_mask = max_probs.ge(self.current_confidence_threshold).float()
        mask_ratio = confidence_mask.mean().item()

        # Consistency loss: strong augmentation should match pseudo-labels
        logits_strong = self.model(unlabeled_strong)
        unsup_loss = (
            F.cross_entropy(logits_strong, pseudo_labels, reduction="none") * confidence_mask
        ).mean()

        # Combined loss
        total_loss = sup_loss + self.current_lambda_u * unsup_loss

        return (
            total_loss,
            sup_loss,
            unsup_loss,
            mask_ratio,
            logits_labeled,
            pseudo_labels,
            confidence_mask,
        )

    def save_checkpoint(self, epoch: int, auc: float, best: bool = False) -> None:
        """Save checkpoint with full FixMatch state for proper resume."""
        # Let BaseTrainer save the core state
        super().save_checkpoint(epoch, auc, best)

        # Determine the path that was just written
        if best:
            path = self.output_dir / "best_model.pth"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"

        # Re-load, augment with SSL state, re-save
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["ssl_state"] = {
            "current_lambda_u": self.current_lambda_u,
            "current_confidence_threshold": self.current_confidence_threshold,
            "backbone_unfrozen": self.backbone_unfrozen,
            "pseudo_label_distribution": self.pseudo_label_distribution,
            "labeled_class_distribution": self.labeled_class_distribution,
            "distribution_ratio": self.distribution_ratio,
        }
        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Load checkpoint and restore full FixMatch state."""
        epoch = super().load_checkpoint(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Restore SSL-specific state (with safe defaults for older checkpoints)
        ssl_state = checkpoint.get("ssl_state", {})
        if ssl_state:
            self.current_lambda_u = ssl_state.get("current_lambda_u", self.current_lambda_u)
            self.current_confidence_threshold = ssl_state.get(
                "current_confidence_threshold", self.current_confidence_threshold
            )
            self.backbone_unfrozen = ssl_state.get("backbone_unfrozen", self.backbone_unfrozen)
            self.pseudo_label_distribution = ssl_state.get(
                "pseudo_label_distribution", self.pseudo_label_distribution
            )
            self.labeled_class_distribution = ssl_state.get(
                "labeled_class_distribution", self.labeled_class_distribution
            )
            self.distribution_ratio = ssl_state.get("distribution_ratio", self.distribution_ratio)

        # Restore EMA shadow weights
        if self.ema is not None and "ema_shadow" in checkpoint:
            self.ema.shadow = checkpoint["ema_shadow"]

        return epoch

    def train(  # type: ignore[override]
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: str | Path | None = None,
    ) -> None:
        """Main FixMatch training loop.

        Args:
            labeled_loader: DataLoader for labeled training data
            unlabeled_loader: DataLoader for unlabeled training data
            val_loader: DataLoader for validation data
            resume_from: Path to checkpoint to resume from (optional)
        """
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
            logger.info(f"Resumed from epoch {start_epoch}")

        # Format lambda_u info
        if self.lambda_u_ramp_epochs > 0:
            lambda_u_info = f"lambda_u={self.lambda_u_start}→{self.lambda_u_end} over {self.lambda_u_ramp_epochs} epochs"
        else:
            lambda_u_info = f"lambda_u={self.lambda_u}"

        # Format confidence threshold info
        if self.confidence_threshold_ramp_epochs > 0:
            tau_info = f"tau={self.confidence_threshold_start}→{self.confidence_threshold_end} over {self.confidence_threshold_ramp_epochs} epochs"
        else:
            tau_info = f"tau={self.confidence_threshold}"

        logger.info(
            f"Starting FixMatch training: epochs {start_epoch + 1}-{self.num_epochs} on {self.device} "
            f"(AMP={'on' if self.use_amp else 'off'}, "
            f"EMA={'on' if self.ema else 'off'}, "
            f"{tau_info}, {lambda_u_info})"
        )

        # Initialize distribution alignment if enabled
        # (skip if already loaded from checkpoint)
        if self.distribution_alignment and self.labeled_class_distribution is None:
            self._compute_labeled_class_distribution(labeled_loader)
            self._compute_distribution_ratio()
            if self.distribution_ratio is not None:
                logger.info(
                    f"Initial distribution ratio (π_l/π_p): {self.distribution_ratio.cpu().numpy()}"
                )

        for epoch in range(start_epoch, self.num_epochs):
            self._apply_warmup(epoch)

            # Unfreeze backbone if schedule reached
            if (
                not self.backbone_unfrozen
                and self.freeze_backbone_epochs > 0
                and epoch >= self.freeze_backbone_epochs
            ):
                if hasattr(self.model, "unfreeze_backbone"):
                    self.model.unfreeze_backbone()
                    self.backbone_unfrozen = True
                    logger.info(f"Unfrozen backbone at epoch {epoch + 1}")
                    # Add newly-unfrozen params to EMA shadow so they get averaged
                    if self.ema is not None:
                        self.ema.refresh(self.model)
                else:
                    logger.warning("Model does not have unfreeze_backbone method")

            current_lr = self._get_current_lr()
            self.current_lambda_u = self._get_current_lambda_u(epoch)
            self.current_confidence_threshold = self._get_current_confidence_threshold(epoch)
            print(
                f"\nEpoch {epoch + 1}/{self.num_epochs} (lr={current_lr:.6f}, λ={self.current_lambda_u:.3f}, τ={self.current_confidence_threshold:.3f})"
            )

            # Train with FixMatch
            train_loss, train_metrics = self.train_epoch_ssl(labeled_loader, unlabeled_loader)

            # Validate (uses EMA model if available)
            if self.ema is not None:
                self.ema.apply(self.model)
            try:
                val_loss, val_metrics = self.validate(val_loader)
            finally:
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
            self.history["lambda_u"].append(self.current_lambda_u)

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
