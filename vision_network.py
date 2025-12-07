import torch
import torch.nn as nn
from torchvision.models import ResNet, resnet34


def resnet_setup() -> ResNet:
    # might need to make smaller for faster training,
    model: ResNet = resnet34(weights=None)
    # 16 x 3 x 224 x 224 input
    model.conv1 = nn.Conv2d(48, 64, kernel_size=7, stride=2, padding=3)
    # output raw features
    model.fc.out_features = model.fc.in_features
    model.fc.weight.data = torch.eye(model.fc.in_features).requires_grad_(False)
    model.fc.bias.data.zero_().requires_grad_(False)

    return model


class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim=128) -> None:
        super().__init__()
        # encoder
        self.resnet = resnet_setup()
        # embedding in latent dim and learning mean and std of latent space
        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        if x.shape == (-1, 16, 3, 224, 224):
            x = x.reshape(-1, 48, 224, 224)
        elif x.shape != (48, 224, 224):
            raise ValueError(f"{self._get_name()} receieved wrong shape {x.shape}")

        features = self.resnet(x)
        return self.mu(features), self.logvar(features)


class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)

        # Mirror ResNet downsampling with ConvTranspose
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 7→14
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 14→28
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 28→56
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 56→112
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, 48, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 112→224
            nn.Sigmoid(),  # pixel values in [0,1]
        )

    def forward(self, z):
        batch = z.shape[0]
        x = self.fc(z)
        x = x.view(batch, 512, 7, 7)  # match start of decoder
        x = self.decoder(x)
        # Output: (B, 48, 224, 224)
        return x


class ResNetVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = ResNetEncoder(latent_dim)
        self.decoder = ResNetDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
