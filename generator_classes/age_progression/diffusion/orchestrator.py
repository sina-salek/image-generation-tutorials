import os

import matplotlib.pyplot as plt
import torch
from age_progression.diffusion.constants import (
    BATCH_SIZE,
    DEVICE,
    HEIGHT,
    LRATE,
    N_CFEAT,
    N_EPOCHS,
    N_FEAT,
    TRAINING,
)
from age_progression.diffusion.data_loader import UTKFaceDataset

# from age_progression.diffusion.diffusion_utils import CustomDataset, transform
from age_progression.diffusion.plotting import plot_sample, show_images
from age_progression.diffusion.sampling import sample_ddpm
from age_progression.diffusion.training import train_model
from age_progression.diffusion.unet import ContextUnet
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':

    model_save_dir = './weights/'
    # model_name = 'model_31.pth'
    # model_name = 'context_model_trained.pth'
    model_name = 'context_model_31.pth'
    model_path = os.path.join(model_save_dir, model_name)
    plot_save_dir = './plots/'

    # load dataset
    root_dir_path = os.path.dirname(os.path.abspath(__file__))

    # sprites dataset
    # data_dir_path = os.path.join(root_dir_path, '..', 'data', 'sprites')
    # images_path = os.path.join(data_dir_path, 'sprites_1788_16x16.npy')
    # labels_path = os.path.join(data_dir_path, 'sprite_labels_nc_1788_16x16.npy')

    # faces dataset
    data_path = os.path.join(root_dir_path, '..', 'data', 'UTKFace')

    # faces transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to a consistent size (128x128)
        transforms.ToTensor(),  # Convert PIL images to tensors
        transforms.Normalize(  # Normalize pixel values
            mean=[0.485, 0.456, 0.406],  # These are standard mean values for RGB images
            std=[0.229, 0.224, 0.225]  # These are standard std values for RGB images
        )
    ])
    dataset = UTKFaceDataset(root_dir=data_path, transform=transform)
    # dataset = CustomDataset(images_path, labels_path, transform,
    #                         null_context=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # construct model
    nn_model = ContextUnet(in_channels=3, n_feat=N_FEAT, n_cfeat=N_CFEAT, height=HEIGHT).to(DEVICE)
    optim = torch.optim.Adam(nn_model.parameters(), lr=LRATE)

    if TRAINING:
        # set into train mode
        nn_model.train()
        # train model
        nn_model = train_model(nn_model, N_EPOCHS, LRATE, dataloader, optim, with_context=True, save_dir=model_save_dir)
    elif not TRAINING:
        # load in model weights and set to eval mode
        nn_model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        print("Loaded in Model")
    else:
        print("Please set the training flag to True or False to train or load in a model")

    ctx = torch.tensor([
        # sprites: hero, non-hero, food, spell, side-facing
        # faces: white, black, asian, indian, others
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]).float().to(DEVICE)

    nn_model.eval()
    samples, intermediate_ddpm = sample_ddpm(ctx.shape[0], nn_model, ctx)

    # Generate the animation (assuming plot_sample returns a matplotlib FuncAnimation object)
    # ani = plot_sample(intermediate_ddpm, 32, 4, plot_save_dir, "ani_run", None, save=True)

    show_images(samples)

    print("Done")
