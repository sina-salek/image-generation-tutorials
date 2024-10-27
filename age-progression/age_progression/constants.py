import torch

# Hyperparameters
LATENT_DIM = 2
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# Model, optimizer
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')