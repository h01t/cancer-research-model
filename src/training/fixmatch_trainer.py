import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .trainer import BaseTrainer


class FixMatchTrainer(BaseTrainer):
    """Trainer for FixMatch semi-supervised learning."""

    def __init__(self, model, config, device=None):
        super().__init__(model, config, device)

        # SSL parameters
        ssl_config = config["ssl"]
        self.confidence_threshold = ssl_config["confidence_threshold"]
        self.lambda_u = ssl_config["lambda_u"]
        self.unlabeled_batch_ratio = ssl_config["unlabeled_batch_ratio"]

        # Augmentations
        self.weak_aug = self._create_weak_augmentation()
        self.strong_aug = self._create_strong_augmentation()

    def _create_weak_augmentation(self):
        """Create weak augmentation transform."""
        from src.data.transforms import get_transforms

        image_size = self.config["dataset"]["image_size"]
        return get_transforms("weak", image_size=image_size)

    def _create_strong_augmentation(self):
        """Create strong augmentation transform."""
        from src.data.transforms import get_transforms

        image_size = self.config["dataset"]["image_size"]
        ssl_config = self.config["ssl"]
        return get_transforms(
            "strong",
            image_size=image_size,
            n=ssl_config.get("randaugment_n", 2),
            m=ssl_config.get("randaugment_m", 10),
        )

    def train_epoch_fixmatch(self, labeled_loader, unlabeled_loader):
        """Train for one epoch with FixMatch."""
        self.model.train()
        total_loss = 0
        total_sup_loss = 0
        total_unsup_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        # Create iterators for both loaders
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        # Determine number of batches per epoch (based on labeled loader)
        num_batches = len(labeled_loader)

        pbar = tqdm(range(num_batches), desc="Training", leave=False)
        for batch_idx in pbar:
            # Get labeled batch
            try:
                labeled_data, labeled_targets = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_data, labeled_targets = next(labeled_iter)

            # Get unlabeled batch (multiple unlabeled samples per labeled sample)
            unlabeled_data = []
            for _ in range(self.unlabeled_batch_ratio):
                try:
                    unlabeled_batch, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_batch, _ = next(unlabeled_iter)
                unlabeled_data.append(unlabeled_batch)
            unlabeled_data = torch.cat(unlabeled_data, dim=0)

            # Move to device
            labeled_data = labeled_data.to(self.device)
            labeled_targets = labeled_targets.to(self.device)
            unlabeled_data = unlabeled_data.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # --- Supervised loss ---
            weak_labeled = self.weak_aug(labeled_data)
            logits_labeled = self.model(weak_labeled)
            sup_loss = self.criterion(logits_labeled, labeled_targets)

            # --- Unsupervised loss (FixMatch) ---
            # Weak augmentation for unlabeled data
            weak_unlabeled = self.weak_aug(unlabeled_data)
            with torch.no_grad():
                logits_weak = self.model(weak_unlabeled)
                probs_weak = torch.softmax(logits_weak, dim=1)
                # Pseudo-labels and confidence
                max_probs, pseudo_labels = torch.max(probs_weak, dim=1)
                confidence_mask = max_probs.ge(self.confidence_threshold).float()

            # Strong augmentation for unlabeled data
            strong_unlabeled = self.strong_aug(unlabeled_data)
            logits_strong = self.model(strong_unlabeled)

            # Compute unsupervised loss (only on confident pseudo-labels)
            unsup_loss = (
                F.cross_entropy(logits_strong, pseudo_labels, reduction="none")
                * confidence_mask
            ).mean()

            # Combine losses
            loss = sup_loss + self.lambda_u * unsup_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_sup_loss += sup_loss.item()
            total_unsup_loss += unsup_loss.item() if unsup_loss.item() > 0 else 0

            # Predictions for labeled data (for metrics)
            probs_labeled = torch.softmax(logits_labeled, dim=1)
            preds_labeled = torch.argmax(logits_labeled, dim=1)

            all_preds.extend(preds_labeled.cpu().detach().numpy())
            all_labels.extend(labeled_targets.cpu().numpy())
            all_probs.extend(probs_labeled[:, 1].cpu().detach().numpy())

            # Update progress bar
            if batch_idx % self.log_freq == 0:
                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "sup": sup_loss.item(),
                        "unsup": unsup_loss.item(),
                        "mask": confidence_mask.mean().item(),
                    }
                )

        # Compute epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_sup_loss = total_sup_loss / num_batches
        epoch_unsup_loss = total_unsup_loss / num_batches

        metrics = self._compute_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )
        metrics.update(
            {
                "loss": epoch_loss,
                "sup_loss": epoch_sup_loss,
                "unsup_loss": epoch_unsup_loss,
            }
        )

        return epoch_loss, metrics

    def _compute_metrics(self, y_true, y_pred, y_prob):
        """Compute classification metrics."""
        from .metrics import compute_metrics

        return compute_metrics(y_true, y_pred, y_prob)

    def train(self, labeled_loader, unlabeled_loader, val_loader):
        """Main training loop for FixMatch."""
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train with FixMatch
            train_loss, train_metrics = self.train_epoch(
                labeled_loader, unlabeled_loader
            )

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["auc"])
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["train_auc"].append(train_metrics.get("auc", 0.0))
            self.history["val_auc"].append(val_metrics.get("auc", 0.0))

            # Print metrics
            print(
                f"Train Loss: {train_loss:.4f} (sup: {train_metrics['sup_loss']:.4f}, unsup: {train_metrics['unsup_loss']:.4f})"
            )
            print(
                f"Train Acc: {train_metrics['accuracy']:.4f}, Train AUC: {train_metrics.get('auc', 0.0):.4f}"
            )
            print(
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics.get('auc', 0.0):.4f}"
            )

            # Save checkpoint
            if (epoch + 1) % self.save_freq == 0:
                self.save_checkpoint(epoch + 1, val_metrics["auc"])

            # Early stopping
            current_auc = val_metrics.get("auc", 0.0)
            if current_auc > self.best_val_auc:
                self.best_val_auc = current_auc
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch + 1, current_auc, best=True)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
