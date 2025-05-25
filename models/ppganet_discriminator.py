import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class CNNDiscriminator1D(nn.Module):
    """
    Inputs: (batch_size, 1, 512)
    Outputs: probability: values in [0,1]
    """
    def __init__(self, in_channels=1):
        super(CNNDiscriminator1D, self).__init__()
        # Conv layers with spectral normalization for stability
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(16)
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(32)
        )
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(64)
        )
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128)
        )
        # Classifier
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 32, 1),  # 32 time steps after 4 downsamples
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)  # -> [B,16,256]
        x = self.conv2(x)  # -> [B,32,128]
        x = self.conv3(x)  # -> [B,64,64]
        x = self.conv4(x)  # -> [B,128,32]
        x = self.flatten(x)  # -> [B, 128*32]
        prob = self.classifier(x)  # -> [B,1]
        return prob

# Example usage:
# from ppganet_discriminator import CNNDiscriminator1D
# D = CNNDiscriminator1D()
# input_tensor = torch.randn((16, 1, 512))
# output = D(input_tensor)  # shape: (16,1)
