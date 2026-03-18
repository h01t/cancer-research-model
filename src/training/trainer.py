import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import yaml
from pathlib import Path

from .metrics import compute_metrics


class BaseTrainer:
    """Base trainer class with common training utilities."""

    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Training parameters
        self.batch_size = config["training"]["batch_size"]
        self.num_epochs = config["training"]["num_epochs"]
        self.learning_rate = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]

        # Optimizer
        optimizer_name = config["training"].get("optimizer", "adam").lower()
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=config["training"].get("momentum", 0.9),
                weight_decay=self.weight_decay,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Scheduler
        scheduler_name = config["training"].get("scheduler", "cosine")
        if scheduler_name is None:
            self.scheduler = None
        else:
            scheduler_name = scheduler_name.lower()
            if scheduler_name == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.num_epochs
                )
            elif scheduler_name == "step":
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=30, gamma=0.1
                )
            elif scheduler_name == "plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode="max", patience=5, factor=0.5
                )
            else:
                self.scheduler = None

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Logging
        self.log_dir = Path(config["logging"].get("log_dir", "logs"))
        self.save_dir = Path(config["logging"].get("save_dir", "checkpoints"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_freq = config["logging"].get("save_freq", 10)
        self.log_freq = config["logging"].get("log_freq", 10)

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_auc": [],
            "val_auc": [],
        }

        # Early stopping
        self.early_stopping_patience = config["training"].get(
            "early_stopping_patience", 10
        )
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().detach().numpy())

            # Update progress bar
            if batch_idx % self.log_freq == 0:
                pbar.set_postfix({"loss": loss.item()})

        # Compute epoch metrics
        epoch_loss = total_loss / len(train_loader)
        metrics = compute_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )

        return epoch_loss, metrics

    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc="Validation", leave=False):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().detach().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().detach().numpy())

        epoch_loss = total_loss / len(val_loader)
        metrics = compute_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )

        return epoch_loss, metrics

    def train(self, train_loader, val_loader):
        """Main training loop."""
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
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
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, Train AUC: {train_metrics.get('auc', 0.0):.4f}"
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

    def save_checkpoint(self, epoch, auc, best=False):
        """Save model checkpoint."""
        if best:
            checkpoint_path = self.save_dir / "best_model.pth"
        else:
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler
                else None,
                "auc": auc,
                "history": self.history,
                "config": self.config,
            },
            checkpoint_path,
        )

        if best:
            print(f"Best model saved with AUC: {auc:.4f}")
        else:
            print(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"], checkpoint["auc"]

    def evaluate(self, test_loader):
        """Evaluate model on test set."""
        _, metrics = self.validate(test_loader)
        return metrics
