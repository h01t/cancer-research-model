"""
Mean Teacher semi-supervised learning trainer.
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ema import EMAModel
from .trainer import BaseTrainer

logger = logging.getLogger(__name__)


class MeanTeacherTrainer(BaseTrainer):
    """Trainer for Mean Teacher with EMA teacher consistency."""

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        device: torch.device | None = None,
        output_dir: str | Path | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(model, config, device, output_dir, class_weights)

        ssl_config = config["ssl"]
        self.consistency_weight = ssl_config.get("consistency_weight_end", 1.0)
        self.consistency_weight_start = ssl_config.get(
            "consistency_weight_start", self.consistency_weight
        )
        self.consistency_weight_end = ssl_config.get(
            "consistency_weight_end", self.consistency_weight
        )
        self.consistency_weight_ramp_epochs = ssl_config.get("consistency_weight_ramp_epochs", 0)
        self.current_consistency_weight = self.consistency_weight_start
        self.consistency_loss_name = ssl_config.get("consistency_loss", "mse").lower()

        if self.consistency_loss_name != "mse":
            raise ValueError(f"Unsupported consistency loss: {self.consistency_loss_name}")

        self.ema_decay = ssl_config.get("ema_decay", 0.999)
        self.ema = EMAModel(self.model, decay=self.ema_decay)

        model_config = config["model"]
        self.freeze_backbone_epochs = model_config.get("freeze_backbone_epochs", 0)
        self.backbone_unfrozen = False

        self.history["sup_loss"] = []
        self.history["consistency_loss"] = []
        self.history["consistency_weight"] = []

    def _get_current_consistency_weight(self, epoch: int) -> float:
        if self.consistency_weight_ramp_epochs <= 0 or epoch >= self.consistency_weight_ramp_epochs:
            return self.consistency_weight_end
        progress = epoch / self.consistency_weight_ramp_epochs
        return self.consistency_weight_start + progress * (
            self.consistency_weight_end - self.consistency_weight_start
        )

    def train_epoch_ssl(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
    ) -> tuple[float, dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        total_sup_loss = 0.0
        total_consistency_loss = 0.0

        all_preds = []
        all_labels = []
        all_probs = []

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        num_batches = len(labeled_loader)

        pbar = tqdm(range(num_batches), desc="Training (Mean Teacher)", leave=False)
        for batch_idx in pbar:
            try:
                labeled_data, labeled_targets = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_data, labeled_targets = next(labeled_iter)

            try:
                teacher_inputs, student_inputs = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                teacher_inputs, student_inputs = next(unlabeled_iter)

            labeled_data = labeled_data.to(self.device)
            labeled_targets = labeled_targets.to(self.device)
            teacher_inputs = teacher_inputs.to(self.device)
            student_inputs = student_inputs.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    total_batch_loss, sup_loss, consistency_loss, logits_labeled = (
                        self._compute_mean_teacher_loss(
                            labeled_data, labeled_targets, teacher_inputs, student_inputs
                        )
                    )
            else:
                total_batch_loss, sup_loss, consistency_loss, logits_labeled = (
                    self._compute_mean_teacher_loss(
                        labeled_data, labeled_targets, teacher_inputs, student_inputs
                    )
                )

            if self.scaler is not None:
                self.scaler.scale(total_batch_loss).backward()
                if self.use_grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_batch_loss.backward()
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            self.ema.update(self.model)

            total_loss += total_batch_loss.item()
            total_sup_loss += sup_loss.item()
            total_consistency_loss += consistency_loss.item()

            with torch.no_grad():
                probs_labeled = torch.softmax(logits_labeled.float(), dim=1)
                preds_labeled = torch.argmax(probs_labeled, dim=1)

            all_preds.extend(preds_labeled.cpu().numpy())
            all_labels.extend(labeled_targets.cpu().numpy())
            all_probs.extend(probs_labeled[:, 1].cpu().numpy())

            if batch_idx % self.log_freq == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{total_batch_loss.item():.4f}",
                        "sup": f"{sup_loss.item():.4f}",
                        "cons": f"{consistency_loss.item():.4f}",
                    }
                )

        epoch_loss = total_loss / max(num_batches, 1)
        epoch_sup_loss = total_sup_loss / max(num_batches, 1)
        epoch_consistency_loss = total_consistency_loss / max(num_batches, 1)

        from .metrics import compute_metrics
        import numpy as np

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        metrics.update(
            {
                "loss": epoch_loss,
                "sup_loss": epoch_sup_loss,
                "consistency_loss": epoch_consistency_loss,
            }
        )
        return epoch_loss, metrics

    def _compute_mean_teacher_loss(
        self,
        labeled_data: torch.Tensor,
        labeled_targets: torch.Tensor,
        teacher_inputs: torch.Tensor,
        student_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_labeled = self.model(labeled_data)
        sup_loss = self.criterion(logits_labeled, labeled_targets)

        self.ema.apply(self.model)
        try:
            with torch.no_grad():
                teacher_logits = self.model(teacher_inputs)
        finally:
            self.ema.restore(self.model)

        student_logits = self.model(student_inputs)
        teacher_probs = torch.softmax(teacher_logits, dim=1)
        student_probs = torch.softmax(student_logits, dim=1)
        consistency_loss = F.mse_loss(student_probs, teacher_probs)

        total_loss = sup_loss + self.current_consistency_weight * consistency_loss
        return total_loss, sup_loss, consistency_loss, logits_labeled

    def save_checkpoint(self, epoch: int, auc: float, best: bool = False) -> None:
        super().save_checkpoint(epoch, auc, best)

        if best:
            path = self.output_dir / "best_model.pth"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["ssl_state"] = {
            "current_consistency_weight": self.current_consistency_weight,
            "backbone_unfrozen": self.backbone_unfrozen,
        }
        checkpoint["ema_shadow"] = self.ema.shadow
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        epoch = super().load_checkpoint(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        ssl_state = checkpoint.get("ssl_state", {})
        self.current_consistency_weight = ssl_state.get(
            "current_consistency_weight", self.current_consistency_weight
        )
        self.backbone_unfrozen = ssl_state.get("backbone_unfrozen", self.backbone_unfrozen)

        if "ema_shadow" in checkpoint:
            self.ema.shadow = checkpoint["ema_shadow"]
        return epoch

    def train(  # type: ignore[override]
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: str | Path | None = None,
    ) -> None:
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)

        logger.info(
            f"Starting Mean Teacher training: epochs {start_epoch + 1}-{self.num_epochs} on "
            f"{self.device} (AMP={'on' if self.use_amp else 'off'}, EMA=on)"
        )

        for epoch in range(start_epoch, self.num_epochs):
            self._apply_warmup(epoch)

            if (
                not self.backbone_unfrozen
                and self.freeze_backbone_epochs > 0
                and epoch >= self.freeze_backbone_epochs
            ):
                if hasattr(self.model, "unfreeze_backbone"):
                    self.model.unfreeze_backbone()
                    self.backbone_unfrozen = True
                    self.ema.refresh(self.model)

            current_lr = self._get_current_lr()
            self.current_consistency_weight = self._get_current_consistency_weight(epoch)
            print(
                f"\nEpoch {epoch + 1}/{self.num_epochs} (lr={current_lr:.6f}, "
                f"consistency={self.current_consistency_weight:.3f})"
            )

            train_loss, train_metrics = self.train_epoch_ssl(labeled_loader, unlabeled_loader)

            self.ema.apply(self.model)
            try:
                val_loss, val_metrics = self.validate(val_loader)
            finally:
                self.ema.restore(self.model)

            if epoch >= self.warmup_epochs and self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("auc", 0.0))
                else:
                    self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["train_auc"].append(train_metrics.get("auc", 0.0))
            self.history["val_auc"].append(val_metrics.get("auc", 0.0))
            self.history["learning_rate"].append(current_lr)
            self.history["sup_loss"].append(train_metrics["sup_loss"])
            self.history["consistency_loss"].append(train_metrics["consistency_loss"])
            self.history["consistency_weight"].append(self.current_consistency_weight)

            print(
                f"  Train — Loss: {train_loss:.4f} "
                f"(sup: {train_metrics['sup_loss']:.4f}, "
                f"consistency: {train_metrics['consistency_loss']:.4f})"
            )
            print(
                f"  Train — Acc: {train_metrics['accuracy']:.4f}, "
                f"AUC: {train_metrics.get('auc', 0.0):.4f}"
            )
            print(
                f"  Val   — Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                f"AUC: {val_metrics.get('auc', 0.0):.4f}"
            )

            self._log_wandb(
                {
                    "train/loss": train_loss,
                    "train/sup_loss": train_metrics["sup_loss"],
                    "train/consistency_loss": train_metrics["consistency_loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/auc": train_metrics.get("auc", 0.0),
                    "val/loss": val_loss,
                    "val/accuracy": val_metrics["accuracy"],
                    "val/auc": val_metrics.get("auc", 0.0),
                    "learning_rate": current_lr,
                    "train/consistency_weight": self.current_consistency_weight,
                },
                step=epoch,
            )

            if (epoch + 1) % self.save_freq == 0:
                self.save_checkpoint(epoch + 1, val_metrics.get("auc", 0.0))

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

        if self.use_wandb and self.wandb_run is not None:
            import wandb

            wandb.finish()
