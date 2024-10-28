import torch
from constants import BATCH_SIZE, DEVICE, EPOCHS, LATENT_DIM
from torch import optim
from vae import VAE, loss_function


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


# Define the objective function
def objective(trial, x_train, x_val):
    # Suggest a learning rate
    # learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    # Initialize the model and optimizer with the suggested learning rate
    model = VAE(LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        train_loss = 0
        for i in range(0, len(x_train), BATCH_SIZE):
            batch = x_train[i : i + BATCH_SIZE].to(DEVICE)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)

            # Apply sigmoid if needed
            recon_batch = torch.sigmoid(recon_batch)

            loss = loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        # Calculate validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(x_val), BATCH_SIZE):
                batch = x_val[i : i + BATCH_SIZE].to(DEVICE)
                recon_batch, mu, logvar = model(batch)
                recon_batch = torch.sigmoid(
                    recon_batch
                )  # Ensure output is between 0 and 1
                loss = loss_function(recon_batch, batch, mu, logvar)
                val_loss += loss.item()
        val_loss /= len(x_val)

        # Print metrics for the current epoch
        print(
            f"Trial {trial.number}, Epoch {epoch + 1}, Train Loss: {train_loss / len(x_train):.4f}, Val Loss: {val_loss:.4f}"
        )

        # Check early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Return the validation loss for Optuna to minimize
    return val_loss
