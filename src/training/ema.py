"""
Exponential Moving Average (EMA) model for FixMatch.

The EMA model maintains a smoothed copy of the student model's weights.
It produces more stable predictions for pseudo-label generation.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EMAModel:
    """Exponential Moving Average of model parameters.

    Usage:
        ema = EMAModel(model, decay=0.999)
        # After each training step:
        ema.update(model)
        # For pseudo-label generation:
        ema.apply(model)    # copy EMA weights to model
        # ... generate pseudo-labels ...
        ema.restore(model)  # restore original weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

        # Initialize shadow parameters as a copy of current model
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters with current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def refresh(self, model: nn.Module) -> None:
        """Add any new requires_grad parameters not yet tracked by EMA.

        Call this after unfreezing backbone layers so that the newly
        unfrozen parameters are included in EMA averaging going forward.
        Their shadow values are initialized to their current weights.
        """
        added = 0
        for name, param in model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.data.clone()
                added += 1
        if added:
            logger.info(f"EMA: added {added} newly-unfrozen parameters to shadow")

    def apply(self, model: nn.Module) -> None:
        """Apply EMA weights to model (backup current weights first)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
