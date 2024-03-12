import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torchvision.datasets import MNIST
from VAE import VAE


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(
        root=config.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = MNIST(
        root=config.data_dir, train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=config.batch_size, shuffle=False
    )

    example = train_loader.__iter__().__next__()

    # plt.imshow(example[0][0][0], cmap="gray")
    # plt.show()

    model = VAE()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)


if __name__ == "__main__":
    main()
