import hydra
import matplotlib.pyplot as plt
import torch
import VAE
from omegaconf import DictConfig


def generate(mean, var, model):
    with torch.no_grad():
        z = torch.tensor([mean, var], dtype=torch.float32)
        sample = model.decode(z)
        sample = sample.view(28, 28)
        plt.imshow(sample, cmap="gray")
        plt.show()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    model = VAE.VAE()
    model.load_state_dict(torch.load(config.model_path))

    generate(1, 1, model)


if __name__ == "__main__":
    main()
