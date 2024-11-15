import logging
import os

import boto3
import matplotlib.pyplot as plt
import sagemaker
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
from age_progression.diffusion.training_utils import train_model
from age_progression.diffusion.unet import ContextUnet
from sagemaker.pytorch import PyTorch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

if __name__ == '__main__':

    model_save_dir = './weights/'
    model_name = 'context_model_31.pth'
    model_path = os.path.join(model_save_dir, model_name)
    plot_save_dir = './plots/'

    # Initialize the S3 client
    s3 = boto3.client('s3')

    root_dir_path = os.path.dirname(os.path.abspath(__file__))
    local_data_path = os.path.join(root_dir_path, '..', '..', 'data', 'UTKFace')

    bucket_name = 'face-diffuser'
    s3_data_path = 'UTKFace'
    s3_output_path = 'output'

    UPLOAD = False
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if UPLOAD:
        for root, dirs, files in os.walk(local_data_path):
            for file in tqdm(files, desc="Uploading files"):
                local_file_path = os.path.join(root, file)
                s3_file_path = os.path.join(s3_data_path, file)
                try:
                    s3.upload_file(local_file_path, bucket_name, s3_file_path)
                    logger.info(f'Successfully uploaded {local_file_path} to s3://{bucket_name}/{s3_file_path}')
                except Exception as e:
                    logger.error(f'Failed to upload {local_file_path} to s3://{bucket_name}/{s3_file_path}: {e}')
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    pytorch_estimator = PyTorch(
        entry_point='aws_train_script.py',
        role=role,
        framework_version='2.2.2',
        py_version='py311',
        instance_count=1,
        instance_type='ml.p3.2xlarge',
        hyperparameters={
            'batch_size': BATCH_SIZE,
            'n_epochs': N_EPOCHS,
            'learning_rate': LRATE,
            'n_feat': N_FEAT,
            'n_cfeat': N_CFEAT,
            'height': HEIGHT,
        },
        output_path=s3_output_path
    )


    if TRAINING:
        pytorch_estimator.fit({'training': s3_data_path})
    elif not TRAINING:
        nn_model = ContextUnet(in_channels=3, n_feat=N_FEAT, n_cfeat=N_CFEAT, height=HEIGHT).to(DEVICE)
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
