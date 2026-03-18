"""
EfficientNet-B0 classifier using torchvision (official PyTorch implementation).

Replaces the third-party `efficientnet-pytorch` package.
"""

import logging

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

logger = logging.getLogger(__name__)


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 for binary classification.

    Uses torchvision's official EfficientNet implementation with
    optional ImageNet pretrained weights, configurable dropout,
    and optional backbone freezing.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load model with or without pretrained weights
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b0(weights=weights)
            logger.info("Loaded EfficientNet-B0 with ImageNet pretrained weights")
        else:
            self.backbone = efficientnet_b0(weights=None)
            logger.info("Loaded EfficientNet-B0 without pretrained weights")

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        # Optional backbone freezing (fine-tune only the classifier head)
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
        """Extract feature vector before classification head.

        Useful for t-SNE/UMAP embedding visualization.
        """
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
