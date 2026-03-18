import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 for binary classification."""

    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.2):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained EfficientNet-B0
        self.efficientnet = (
            EfficientNet.from_pretrained("efficientnet-b0")
            if pretrained
            else EfficientNet.from_name("efficientnet-b0")
        )

        # Get feature dimension
        in_features = self.efficientnet._fc.in_features

        # Replace final fully connected layer
        self.efficientnet._fc = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

    def get_features(self, x):
        """Extract feature vector before classification head."""
        # Global pooling features
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x
