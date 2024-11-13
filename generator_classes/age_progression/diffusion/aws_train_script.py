import argparse
import os

import torch
from age_progression.diffusion.data_loader import UTKFaceDataset
from age_progression.diffusion.training_utils import train_model
from age_progression.diffusion.unet import ContextUnet
from torch.utils.data import DataLoader
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()

    # Data and model parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--n_feat', type=int, default=128)
    parser.add_argument('--n_cfeat', type=int, default=5)
    parser.add_argument('--height', type=int, default=16)

    # SageMaker parameters
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up data loading
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = UTKFaceDataset(root_dir=args.train, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize model
    model = ContextUnet(
        in_channels=3,
        n_feat=args.n_feat,
        n_cfeat=args.n_cfeat,
        height=args.height
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train model
    model = train_model(
        model,
        args.n_epochs,
        args.learning_rate,
        dataloader,
        optimizer,
        with_context=True,
        save_dir=args.model_dir
    )
