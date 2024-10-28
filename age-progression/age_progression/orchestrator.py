import numpy as np
import optuna
import torch
import torch.optim as optim
from constants import BATCH_SIZE, DEVICE, EPOCHS, LATENT_DIM, LEARNING_RATE
from model_utils import EarlyStopping, objective
from plotting import generate_images, plot_latent_space
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from vae import VAE, loss_function

# Load Olivetti Faces Dataset
data = fetch_olivetti_faces()
faces = data.images

# Normalize the data and convert to PyTorch tensors
faces = faces.astype("float32") / 255.0
faces = np.expand_dims(faces, 1)  # Add channel dimension
faces = torch.tensor(faces)

# Split the dataset and targets
x_train, x_test, y_train, y_test = train_test_split(
    faces, data.target, test_size=0.2, random_state=42
)
# Further split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
x_train, x_test, x_val = (
    x_train.clone().detach().to(DEVICE),
    x_test.clone().detach().to(DEVICE),
    x_val.clone().detach().to(DEVICE),
)

model = VAE(LATENT_DIM).to(DEVICE)

# Create a study and optimize the objective function
study = optuna.create_study(direction="minimize")
study.optimize(lambda trial: objective(trial, x_train, x_val), n_trials=10)

# Print the best hyperparameters
print(f"Best learning rate: {study.best_params['learning_rate']}")

optimizer = optim.Adam(model.parameters(), lr=study.best_params["learning_rate"])

# Initialize early stopping
early_stopping = EarlyStopping(patience=5, min_delta=0.01)

# Training loop with early stopping
model.train()
for epoch in range(EPOCHS):
    train_loss = 0
    for i in range(0, len(x_train), BATCH_SIZE):
        batch = x_train[i : i + BATCH_SIZE].to(DEVICE)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # Calculate validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(x_val), BATCH_SIZE):
            batch = x_val[i : i + BATCH_SIZE].to(DEVICE)
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)
            val_loss += loss.item()
    val_loss /= len(x_val)

    print(
        f"Epoch {epoch + 1}, Train Loss: {train_loss / len(x_train):.4f}, Val Loss: {val_loss:.4f}"
    )

    # Check early stopping
    if early_stopping(val_loss):
        print("Early stopping triggered")
        break

# Visualize latent space and generate images
plot_latent_space(model, x_test, y_test)
generate_images(model.decode)

# Save the model
torch.save(model.state_dict(), "saved_models/vae_model.pth")
