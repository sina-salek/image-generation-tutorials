import torch

# hyperparameters

# diffusion hyperparameters
TIMESTEPS = 500
BETA1 = 1e-4
BETA2 = 0.02

# network hyperparameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
N_FEAT = 64  # 64 hidden dimension feature
N_CFEAT = 5  # context vector is of size 5
HEIGHT = 128  # 16x16 image

# training hyperparameters
BATCH_SIZE = 100
N_EPOCHS = 32
LRATE = 1e-3

# training flag
TRAINING = True
