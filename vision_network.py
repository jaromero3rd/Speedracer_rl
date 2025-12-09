"""
CNN encoder for predicting CartPole state from stacked frames.
Uses ResNet18 backbone trained from scratch.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34


class ResNetStateEncoder(nn.Module):
    """ResNet-based encoder that predicts state buffer from 16 stacked RGB frames."""

    def __init__(self, input_channels=48, output_dim=64, backbone='resnet18'):
        super().__init__()

        self.input_channels = input_channels
        self.output_dim = output_dim

        if backbone == 'resnet18':
            self.resnet = resnet18(weights=None)
            self.feature_dim = 512
        elif backbone == 'resnet34':
            self.resnet = resnet34(weights=None)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # modify first conv for 48 input channels (16 frames * 3 RGB)
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # output 64 dims (16 timesteps * 4 state vars)
        self.resnet.fc = nn.Linear(self.feature_dim, output_dim)

    def forward(self, x):
        return self.resnet(x)

    def get_features(self, x):
        """Get 512-dim features before final FC (for embeddings)."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def create_encoder(backbone='resnet18'):
    return ResNetStateEncoder(
        input_channels=48,
        output_dim=64,
        backbone=backbone
    )


if __name__ == "__main__":
    model = create_encoder('resnet18')

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet18 encoder: {total_params:,} params")

    # test forward pass
    x = torch.randn(2, 48, 224, 224)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
