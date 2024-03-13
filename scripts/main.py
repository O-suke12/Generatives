import hydra
import omegaconf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torchvision.datasets import MNIST
from tqdm import tqdm
from VAE import VAE

import wandb


def loss_function(x_hat, x, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD


def train(model, optimizer, epochs, train_loader):
    model.train()
    for epoch in range(epochs):
        over_all_loss = 0
        for batch_idx, (x, _) in tqdm(enumerate(train_loader)):
            x = x.view(-1, 784)
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_function(x_hat, x, mean, log_var)
            loss.backward()
            over_all_loss += loss.item()
            optimizer.step()
        wandb.log({"loss": over_all_loss})

    return over_all_loss


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    transform = transforms.Compose([transforms.ToTensor()])
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    wandb.init(project="VAE")
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

    train(model, optimizer, config.epochs, train_loader)

    torch.save(model.state_dict(), config.model_path)
    wandb.finish()


if __name__ == "__main__":
    main()
