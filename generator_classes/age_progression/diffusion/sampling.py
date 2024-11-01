import numpy as np
import torch
from age_progression.diffusion.constants import DEVICE, HEIGHT, TIMESTEPS
from age_progression.diffusion.diffusion_utils import denoise_add_noise


# sample using standard algorithm
@torch.no_grad()
def sample_ddpm(n_sample, nn_model, context, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, HEIGHT, HEIGHT).to(DEVICE)

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(TIMESTEPS, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / TIMESTEPS])[:, None, None, None].to(DEVICE)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t, c = context)  # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate == 0 or i == TIMESTEPS or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate
