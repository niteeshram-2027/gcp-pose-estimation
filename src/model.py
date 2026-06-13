import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


class GCPModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        in_features = (
            self.backbone.classifier[1]
            .in_features
        )

        self.backbone.classifier = nn.Identity()

        self.coord_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 2),

            nn.Sigmoid()
        )

        self.shape_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 3)
        )

    def forward(self, x):

        features = self.backbone(x)

        coords = self.coord_head(
            features
        )

        shape_logits = self.shape_head(
            features
        )

        return coords, shape_logits