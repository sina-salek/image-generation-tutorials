import os

import torch
import torch.nn.functional as F
from age_progression.diffusion.constants import (
    BATCH_SIZE,
    DEVICE,
    HEIGHT,
    LRATE,
    N_CFEAT,
    N_EPOCHS,
    N_FEAT,
    TIMESTEPS,
)
from age_progression.diffusion.data_loader import UTKFaceDataset
from age_progression.diffusion.training_utils import train_model
from age_progression.diffusion.unet import ContextUnet
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
    # Set up data loader
    local_data_path = os.environ['SM_CHANNEL_TRAIN']
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = UTKFaceDataset(root_dir=local_data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # Construct model
    nn_model = ContextUnet(in_channels=3, n_feat=N_FEAT, n_cfeat=N_CFEAT, height=HEIGHT).to(DEVICE)
    optim = torch.optim.Adam(nn_model.parameters(), lr=LRATE)

    # Train model
    nn_model.train()
    nn_model = train_model(nn_model, N_EPOCHS, LRATE, dataloader, optim, with_context=True, save_dir=os.environ['SM_MODEL_DIR'])

if __name__ == '__main__':
    main()
