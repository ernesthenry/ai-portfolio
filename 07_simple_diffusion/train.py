import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusion import DiffusionUtils
from unet import UNet

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Scale to [-1, 1]
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Setup Model & Utils
    model = UNet(device=device).to(device)
    diff = DiffusionUtils(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    print("Starting Training...")

    for epoch in range(5):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # A. Sample random timesteps
            t = diff.sample_timesteps(images.shape[0])

            # B. Add noise to images
            x_t, noise = diff.noise_images(images, t)

            # C. Predict the noise
            predicted_noise = model(x_t, t)

            # D. Loss: Compare Actual Noise vs Predicted Noise
            loss = loss_fn(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item()}")

    # Save
    torch.save(model.state_dict(), "ddpm_mnist.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()
