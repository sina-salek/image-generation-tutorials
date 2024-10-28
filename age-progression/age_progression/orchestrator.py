import numpy as np
import torch
import torch.optim as optim
from constants import BATCH_SIZE, DEVICE, EPOCHS, LATENT_DIM, LEARNING_RATE
from plotting import generate_images, plot_latent_space
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from vae import VAE, loss_function

# Load Olivetti Faces Dataset
data = fetch_olivetti_faces()
faces = data.images

# Normalize the data and convert to PyTorch tensors
faces = faces.astype('float32') / 255.0
faces = np.expand_dims(faces, 1)  # Add channel dimension
faces = torch.tensor(faces)

# Split the dataset and targets
x_train, x_test, y_train, y_test = train_test_split(faces, data.target, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
x_train, x_test = torch.tensor(x_train), torch.tensor(x_test)


model = VAE(LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
model.train()
for epoch in range(EPOCHS):
    train_loss = 0
    for i in range(0, len(x_train), BATCH_SIZE):
        batch = x_train[i:i+BATCH_SIZE].to(DEVICE)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(x_train):.4f}')

# Visualize latent space and generate images
plot_latent_space(model, x_test, y_test)
generate_images(model.decode)

# Save the model
torch.save(model.state_dict(), 'saved_models/vae_model.pth')
