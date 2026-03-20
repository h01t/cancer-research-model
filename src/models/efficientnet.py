"""
Classifier backbones for supervised mammography experiments.

Currently supports EfficientNet-B0/B2/B3 via torchvision.
"""

import logging

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    efficientnet_b0,
    efficientnet_b2,
    efficientnet_b3,
)

logger = logging.getLogger(__name__)


_EFFICIENTNET_BUILDERS = {
    "efficientnet-b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet-b2": (efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1),
    "efficientnet-b3": (efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
}


class EfficientNetClassifier(nn.Module):
    """EfficientNet classifier for binary mammography classification."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
        backbone_name: str = "efficientnet-b0",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name.lower()

        if self.backbone_name not in _EFFICIENTNET_BUILDERS:
            raise ValueError(f"Unsupported EfficientNet backbone: {backbone_name}")

        builder, pretrained_weights = _EFFICIENTNET_BUILDERS[self.backbone_name]
        weights = pretrained_weights if pretrained else None
        self.backbone = builder(weights=weights)
        self.feature_dim = self.backbone.classifier[1].in_features

        if pretrained:
            logger.info(
                "Loaded %s with ImageNet pretrained weights", self.backbone_name
            )
        else:
            logger.info("Loaded %s without pretrained weights", self.backbone_name)

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(self.feature_dim, num_classes),
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze all layers except the classifier head."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only classifier head will be trained")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen — all layers will be trained")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector before classification head."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
