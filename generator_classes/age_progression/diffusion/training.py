import os

import torch
import torch.nn.functional as F
from age_progression.diffusion.constants import DEVICE, TIMESTEPS
from age_progression.diffusion.diffusion_utils import perturb_input
from tqdm import tqdm


# training with context code
def train_model(nn_model, n_epochs, lrate, dataloader, optim, with_context, save_dir):
    for ep in range(n_epochs):
        print(f'epoch {ep}')

        # linearly decay learning rate
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epochs)

        pbar = tqdm(dataloader, mininterval=2)
        for data in pbar:
            if data is None:
                continue
            x = data['image'] # x: images  c: context
            c = data['race']
            optim.zero_grad()
            x = x.to(DEVICE)
            if with_context:
                c = c.to(x)

                # randomly mask out c
                context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(DEVICE)
                c = c * context_mask.unsqueeze(-1)
            else:
                c = None

            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, TIMESTEPS + 1, (x.shape[0],)).to(DEVICE)
            x_pert = perturb_input(x, t, noise)

            # use network to recover noise
            pred_noise = nn_model(x_pert, t / TIMESTEPS, c=c)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            optim.step()

        # save model periodically
        if ep % 4 == 0 or ep == int(n_epochs - 1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(nn_model.state_dict(), save_dir + f"context_model_{ep}.pth")
            print('saved model at ' + save_dir + f"context_model_{ep}.pth")
