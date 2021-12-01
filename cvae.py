import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
from model import cvae

"""
A Convolutional Variational Autoencoder
"""
class ConvBlock(nn.Module):
    def __init__(self, n_in, n_middle, n_out, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(n_in, n_middle, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(n_middle),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_middle, n_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(n_out),
        )
        if n_in != n_out:
            self.shortcut = nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.main(x)
        return x + identity


class DownSample(nn.Module):
    def __init__(self, n_in, n_out):
        super(DownSample, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_out),
            # nn.ReLU(inplace=True)
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class VAE(nn.Module):
    def __init__(self, num_channels=3, num_features=32):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        nf = num_features
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, nf, kernel_size=3, stride=1, padding=1),    # [32, 32]
            # nn.ReLU(inplace=True),
            nn.SiLU(inplace=True),
            DownSample(nf, nf * 2),   # [16, 16]
            ConvBlock(nf * 2, nf * 2, nf * 2),
            DownSample(nf * 2, nf * 4),    # [8, 8]
            # ConvBlock(nf * 4, nf * 4, nf * 4),
            # DownSample(nf * 4, nf * 8),    # [4, 4]
        )
        # self.mu = ConvBlock(nf * 8, nf * 8, nf * 8)
        # self.logvar = ConvBlock(nf * 8, nf * 8, nf * 8)

        self.mu = ConvBlock(nf * 4, nf * 4, nf * 4)
        self.logvar = ConvBlock(nf * 4, nf * 4, nf * 4)

        self.decoder = nn.Sequential(
            # ConvBlock(nf * 8, nf * 8, nf * 4),
            # nn.Upsample(scale_factor=2),    # [8, 8]
            ConvBlock(nf * 4, nf * 4, nf * 2),
            nn.Upsample(scale_factor=2),  # [16, 16]
            ConvBlock(nf * 2, nf * 2, nf),
            nn.Upsample(scale_factor=2),  # [32, 32]
            nn.Conv2d(nf, num_channels, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid(),
            nn.Tanh(),
        )


    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar * 0.1)
        eps = torch.randn_like(std)
        return mu + std * eps


    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        encoded_x = self.encoder(x)
        mu, logVar = self.mu(encoded_x), self.logvar(encoded_x)

        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar