"""
Base trainer with training loop, validation, checkpointing, early stopping.

Supports: MPS (Apple Silicon), CUDA, CPU
Features: AMP, LR warmup, gradient clipping, class-weighted loss,
          W&B logging, per-experiment checkpoints
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics
from .metrics import find_best_threshold

logger = logging.getLogger(__name__)


def get_device(requested: str | None = None) -> torch.device:
    """Get the best available device.

    Priority: requested > CUDA > MPS > CPU

    Args:
        requested: explicit device string ('cuda', 'mps', 'cpu') or None for auto
    """
    if requested:
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BaseTrainer:
    """Base trainer class with common training utilities."""

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        device: torch.device | None = None,
        output_dir: str | Path | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        self.model = model
        self.config = config
        self.device = device or get_device()
        self.model.to(self.device)

        # Output directory (per-experiment)
        self.output_dir = Path(output_dir) if output_dir else Path("results/default")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training parameters
        train_cfg = config["training"]
        self.batch_size = train_cfg["batch_size"]
        self.num_epochs = train_cfg["num_epochs"]
        self.learning_rate = train_cfg["learning_rate"]
        self.weight_decay = train_cfg["weight_decay"]

        # Gradient clipping
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.use_grad_clip = train_cfg.get("gradient_clipping", True)

        # Mixed precision (AMP)
        self.use_amp = train_cfg.get("use_amp", True) and self.device.type in (
            "cuda",
            "mps",
        )
        if self.use_amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None  # MPS uses AMP without GradScaler
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        # Optimizer
        optimizer_name = train_cfg.get("optimizer", "adam").lower()
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=train_cfg.get("momentum", 0.9),
                weight_decay=self.weight_decay,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Warmup + Scheduler
        self.warmup_epochs = train_cfg.get("warmup_epochs", 0)
        self._base_lr = self.learning_rate

        scheduler_name = train_cfg.get("scheduler", "cosine")
        if scheduler_name is None:
            self.scheduler = None
        else:
            scheduler_name = scheduler_name.lower()
            if scheduler_name == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=max(1, self.num_epochs - self.warmup_epochs)
                )
            elif scheduler_name == "step":
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
            elif scheduler_name == "plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode="max", patience=5, factor=0.5
                )
            else:
                self.scheduler = None

        # Loss function (with optional class weights + label smoothing)
        label_smoothing = train_cfg.get("label_smoothing", 0.0)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=label_smoothing
            )
            logger.info(
                f"Using class-weighted loss: {class_weights.tolist()}, label_smoothing={label_smoothing}"
            )
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            if label_smoothing > 0:
                logger.info(f"Using label smoothing: {label_smoothing}")

        # Logging
        log_cfg = config.get("logging", {})
        self.save_freq = log_cfg.get("save_freq", 10)
        self.log_freq = log_cfg.get("log_freq", 10)

        # History
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_auc": [],
            "val_auc": [],
            "learning_rate": [],
        }

        # Early stopping
        self.early_stopping_patience = train_cfg.get("early_stopping_patience", 10)
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0

        # W&B integration
        self.use_wandb = config.get("wandb", {}).get("enabled", False)
        self.wandb_run = None
        if self.use_wandb:
            self._init_wandb(config)

    def predict(self, data_loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        """Collect labels and positive-class probabilities for a data loader."""
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data, targets in tqdm(data_loader, desc="Validation", leave=False):
                data = data.to(self.device)
                targets = targets.to(self.device)

                if self.use_amp:
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        outputs = self.model(data)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                probs = torch.softmax(outputs.float(), dim=1)

                all_labels.extend(targets.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        epoch_loss = total_loss / max(len(data_loader), 1)
        return epoch_loss, np.array(all_labels), np.array(all_probs)

    def _init_wandb(self, config: dict) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            wandb_cfg = config.get("wandb", {})
            self.wandb_run = wandb.init(
                project=wandb_cfg.get("project", "ssl-mammography"),
                name=wandb_cfg.get("run_name"),
                config=config,
                dir=str(self.output_dir),
                reinit=True,
            )
            logger.info("W&B initialized successfully")
        except ImportError:
            logger.warning("wandb not installed. Disabling W&B logging.")
            self.use_wandb = False
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}. Disabling.")
            self.use_wandb = False

    def _log_wandb(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics to W&B if enabled."""
        if self.use_wandb and self.wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def _apply_warmup(self, epoch: int) -> None:
        """Apply linear learning rate warmup."""
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self._base_lr * warmup_factor

    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        return self.optimizer.param_groups[0]["lr"]

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """Train for one epoch.

        Args:
            train_loader: DataLoader yielding (data, targets) batches

        Returns:
            (epoch_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

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

            total_loss += loss.item()

            # Collect predictions (outside AMP context)
            with torch.no_grad():
                probs = torch.softmax(outputs.float(), dim=1)
                preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

            if batch_idx % self.log_freq == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = total_loss / max(len(train_loader), 1)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))

        return epoch_loss, metrics

    def validate(self, val_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """Validate model on a data loader."""
        epoch_loss, y_true, y_prob = self.predict(val_loader)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)

        return epoch_loss, metrics

    def tune_decision_threshold(self, data_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """Tune the binary decision threshold on a validation loader."""
        epoch_loss, y_true, y_prob = self.predict(data_loader)
        threshold, metrics = find_best_threshold(y_true, y_prob)
        metrics["loss"] = epoch_loss
        metrics["decision_threshold"] = threshold
        return threshold, metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Main training loop with warmup, scheduling, early stopping."""
        logger.info(
            f"Starting training: {self.num_epochs} epochs on {self.device} "
            f"(AMP={'on' if self.use_amp else 'off'})"
        )

        for epoch in range(self.num_epochs):
            # Apply warmup
            self._apply_warmup(epoch)

            current_lr = self._get_current_lr()
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} (lr={current_lr:.6f})")

            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            # Update scheduler (only after warmup completes)
            if epoch >= self.warmup_epochs and self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
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

            # Print metrics
            print(
                f"  Train — Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                f"AUC: {train_metrics.get('auc', 0.0):.4f}"
            )
            print(
                f"  Val   — Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                f"AUC: {val_metrics.get('auc', 0.0):.4f}"
            )

            # W&B logging
            self._log_wandb(
                {
                    "train/loss": train_loss,
                    "train/accuracy": train_metrics["accuracy"],
                    "train/auc": train_metrics.get("auc", 0.0),
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

            # Early stopping on AUC
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

    def save_checkpoint(self, epoch: int, auc: float, best: bool = False) -> None:
        """Save model checkpoint to output_dir."""
        if best:
            path = self.output_dir / "best_model.pth"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (self.scheduler.state_dict() if self.scheduler else None),
            "scaler_state_dict": (self.scaler.state_dict() if self.scaler else None),
            "auc": auc,
            "best_val_auc": self.best_val_auc,
            "epochs_without_improvement": self.epochs_without_improvement,
            "history": self.history,
            "config": self.config,
        }

        torch.save(checkpoint, path)

        if best:
            print(f"  Best model saved (AUC: {auc:.4f})")

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Load model checkpoint (cross-device compatible).

        Returns:
            The epoch number the checkpoint was saved at (1-indexed).
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_val_auc = checkpoint.get("best_val_auc", 0.0)
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        self.history = checkpoint.get("history", self.history)
        return checkpoint["epoch"]

    def evaluate(
        self, test_loader: DataLoader, decision_threshold: float = 0.5
    ) -> dict[str, float]:
        """Evaluate model on test set."""
        epoch_loss, y_true, y_prob = self.predict(test_loader)
        y_pred = (y_prob >= decision_threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["loss"] = epoch_loss
        metrics["decision_threshold"] = decision_threshold
        return metrics
