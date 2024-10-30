import os

import numpy as np
import torch
import torchvision.transforms as transforms
from age_progression.diffusion.diffusion_utils import CustomDataset, transform
from data_loader import UTKFaceDataset
from samplers import ContextUnet
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# faces transform
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # Resize images to a consistent size
#     transforms.ToTensor(),          # Convert PIL images to tensors
#     transforms.Normalize(           # Normalize pixel values
#         mean=[0.485, 0.456, 0.406], # These are standard mean values for RGB images
#         std=[0.229, 0.224, 0.225]   # These are standard std values for RGB images
#     )
# ])

# Initialize the dataset
root_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(root_dir, '..', 'data', 'UTKFace')
data_path = os.path.join(root_dir, '..', 'data', 'sprites')
if __name__ == '__main__':
    # hyperparameters

    # diffusion hyperparameters
    timesteps = 500
    beta1 = 1e-4
    beta2 = 0.02

    # network hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    n_feat = 64  # 64 hidden dimension feature
    n_cfeat = 5  # context vector is of size 5
    height = 128  # 16x16 image
    save_dir = './weights/'

    # training hyperparameters
    batch_size = 100
    n_epoch = 32
    lrate = 1e-3

    # construct DDPM noise schedule
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    # construct model
    nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

    # load dataset and construct optimizer
    # dataset = UTKFaceDataset(root_dir=data_path, transform=transform)

    sprites_path= os.path.join(data_path, 'sprites_1788_16x16.npy')
    labels_path = os.path.join(data_path, 'sprite_labels_nc_1788_16x16.npy')
    dataset = CustomDataset(sprites_path, labels_path, transform,
                            null_context=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)


    # helper function: perturbs an image to a specified noise level
    def perturb_input(x, t, noise):
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise


    # set into train mode
    nn_model.train()

    for ep in range(n_epoch):
        print(f'epoch {ep}')

        # linearly decay learning rate
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader, mininterval=2)
        for data_dict in pbar:  # x: images
            optim.zero_grad()
            # x = data_dict['image']
            # age = data_dict['age']
            # gender = data_dict['gender']
            # race = data_dict['race']
            x, _ = data_dict

            x = x.to(device)

            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
            x_pert = perturb_input(x, t, noise)

            # use network to recover noise
            pred_noise = nn_model(x_pert, t / timesteps)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            optim.step()

        # save model periodically
        if ep % 4 == 0 or ep == int(n_epoch - 1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(nn_model.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")


    # helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
    def denoise_add_noise(x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
        return mean + noise


    # sample using standard algorithm
    @torch.no_grad()
    def sample_ddpm(n_sample, save_rate=20):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, height, height).to(device)

        # array to keep track of generated steps for plotting
        intermediate = []
        for i in range(timesteps, 0, -1):
            print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            eps = nn_model(samples, t)  # predict noise e_(x_t,t)
            samples = denoise_add_noise(samples, i, eps, z)
            if i % save_rate == 0 or i == timesteps or i < 8:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate


    # load in model weights and set to eval mode
    nn_model.load_state_dict(torch.load(f"{save_dir}/model_0.pth", map_location=device))
    nn_model.eval()
    print("Loaded in Model")
