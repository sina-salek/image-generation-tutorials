import numpy as np
import torch
import matplotlib.pyplot as plt

from constants import BATCH_SIZE, DEVICE


def plot_latent_space(model, data, targets, num_points=1000):
    model.eval()
    with torch.no_grad():
        mu_list = []
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE].to(DEVICE)
            mu, _ = model.encode(batch)
            mu_list.append(mu.cpu().numpy())
        mus = np.concatenate(mu_list)

        plt.figure(figsize=(8, 6))
        plt.scatter(mus[:, 0], mus[:, 1], c=targets[:len(mus)], cmap='viridis', s=2)
        plt.colorbar()
        plt.title('Latent Space')
        plt.xlabel('Latent dim 1')
        plt.ylabel('Latent dim 2')
        plt.show()

def generate_images(decoder, grid_size=10, dim_range=(-3, 3)):
    with torch.no_grad():
        figure = np.zeros((grid_size * 64, grid_size * 64))
        grid_x = np.linspace(dim_range[0], dim_range[1], grid_size)
        grid_y = np.linspace(dim_range[0], dim_range[1], grid_size)
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]]).float().to(DEVICE)
                x_decoded = decoder(z_sample).cpu().numpy().reshape(64, 64)
                figure[i * 64: (i + 1) * 64, j * 64: (j + 1) * 64] = x_decoded
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='gray')
        plt.title('Generated Faces from Latent Space')
        plt.axis('off')
        plt.show()