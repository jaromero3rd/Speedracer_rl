"""
Supervised CNN encoder for CartPole state prediction.
Uses ResNet18 architecture trained from scratch.

Input: (B, 48, 224, 224) - 16 stacked RGB frames, channels concatenated
Output: (B, 64) - flattened observation buffer (16 timesteps x 4 state dims)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34


class ResNetStateEncoder(nn.Module):
    """ResNet encoder for predicting CartPole state buffer from stacked frames."""

    def __init__(self, input_channels=48, output_dim=64, backbone='resnet18'):
        """
        Args:
            input_channels: Number of input channels (48 = 16 frames x 3 RGB)
            output_dim: Output dimension (64 = 16 timesteps x 4 state dims)
            backbone: 'resnet18' or 'resnet34'
        """
        super().__init__()

        self.input_channels = input_channels
        self.output_dim = output_dim

        # Load backbone
        if backbone == 'resnet18':
            self.resnet = resnet18(weights=None)
            self.feature_dim = 512
        elif backbone == 'resnet34':
            self.resnet = resnet34(weights=None)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Modify first conv layer for 48 input channels
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace final FC layer to output 64 dims
        self.resnet.fc = nn.Linear(self.feature_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 48, 224, 224) - stacked RGB frames
        Returns:
            state: (B, 64) - predicted state buffer
        """
        return self.resnet(x)

    def get_features(self, x):
        """
        Get intermediate features (512-dim) before final FC layer.
        Useful for extracting embeddings.

        Args:
            x: (B, 48, 224, 224)
        Returns:
            features: (B, 512)
        """
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
    """Factory function to create encoder with default settings."""
    return ResNetStateEncoder(
        input_channels=48,  # 16 frames x 3 RGB
        output_dim=64,      # 16 timesteps x 4 state dims
        backbone=backbone
    )


if __name__ == "__main__":
    # Test the network
    model = create_encoder('resnet18')

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"ResNet18 State Encoder")
    print(f"  Input: (B, 48, 224, 224)")
    print(f"  Output: (B, 64)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    x = torch.randn(2, 48, 224, 224)
    y = model(x)
    print(f"\n  Test input shape: {x.shape}")
    print(f"  Test output shape: {y.shape}")

    # Test feature extraction
    features = model.get_features(x)
    print(f"  Feature shape: {features.shape}")
