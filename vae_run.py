"""
The following is an import of PyTorch libraries.
"""
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
from torchvision.utils import save_image
import PIL
"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Initialize Hyperparameters
"""
batch_size = 64
learning_rate = 0.001
num_epochs = 20


"""
Create dataloaders to feed data into the neural network
Default MNIST dataset is used and standard train/test split is performed
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=True, download=True,
                    transform=transform),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=False, transform=transform),
    batch_size=1)

"""
Initialize the network and the Adam optimizer
"""
net = cvae.VAE().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""
for epoch in range(num_epochs):
    for idx, data in enumerate(train_loader, 0):
        imgs, _ = data
        imgs = imgs.to(device)

        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = net(imgs)
        # 256, 3, 32, 32
        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = -0.5 * (1 + logVar - mu ** 2 - logVar.exp()).mean()#.sum(dim=-1).sum(dim=-1).sum(dim=-1).mean(dim=0)

        # loss = F.binary_cross_entropy(out, imgs)# + kl_divergence
        reconstruction_loss = F.mse_loss(out, imgs)
        loss = reconstruction_loss + kl_divergence

        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {}: Loss : {:.4f} | recon_loss : {:.4f} | kl_divergence : {:.4f}'.format(epoch, loss, reconstruction_loss, kl_divergence))

    def denorm(img):
        return (img + 1.) * 0.5

    save_image(torch.cat([denorm(out).cpu(), denorm(imgs).cpu()], dim=0), 'test_{}_(1.0).png'.format(epoch))
