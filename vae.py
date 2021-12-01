import random
import torch
import torchvision
import os
import torch.nn as nn
from torch.nn import functional as F
class VAE(nn.Module):
    def __init__(self, image_size, hidden_size_1, hidden_size_2, latent_size, batch_size):
        super(VAE, self).__init__()
        self.batch_size = batch_size
        self.fc1 = nn.Linear(image_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc31 = nn.Linear(hidden_size_2, latent_size)
        self.fc32 = nn.Linear(hidden_size_2, latent_size)

        self.fc4 = nn.Linear(latent_size, hidden_size_2)
        self.fc5 = nn.Linear(hidden_size_2, hidden_size_1)
        self.fc6 = nn.Linear(hidden_size_1, image_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        f1, f2 = self.fc31(h2), self.fc32(h2)
        return f1, f2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decode(self, z):
        h3 = F.relu(self.fc4(z))
        h4 = F.relu(self.fc5(h3))
        output = torch.sigmoid(self.fc6(h4))
        #output = torch.tanh(self.fc6(h4))
        # probability
        return output

    def forward(self, x):
        batch_size, n_channels, ny, nx = x.size()
        # print(x.size()) # 256, 3, 32, 32
        x = x.view(batch_size, n_channels * ny * nx)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        # print('mu:', mu.size())
        # print('logvar:', logvar.size())
        # print(output.size())
        return output.view(batch_size, n_channels, ny, nx), mu, logvar


