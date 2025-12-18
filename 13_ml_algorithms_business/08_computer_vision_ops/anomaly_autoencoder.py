import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------------------------------------------------------
# BUSINESS PROBLEM: PREDICTIVE MAINTENANCE (Anomaly Detection)
# ---------------------------------------------------------
# Scenario: Determine if a machine is about to break based on vibration data.
# Challenge: We have lots of "Normal" data, but almost zero "Broken" data.
# Solution: Autoencoder (Train on normal, if reconstruction error is high -> Anomaly).
# ---------------------------------------------------------

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 10),
            nn.Identity() # Try to reconstruct input
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def run_anomaly_detector():
    # 1. Train on NORMAL data only
    normal_data = torch.randn(100, 10) # 10 sensors
    
    model = Autoencoder()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("--- Autoencoder (Training on Normal State) ---")
    for epoch in range(50):
        optimizer.zero_grad()
        recon = model(normal_data)
        loss = criterion(recon, normal_data)
        loss.backward()
        optimizer.step()
    
    print(f"Final Training MSE: {loss.item():.4f}")
    
    # 2. Test Phase
    test_normal = torch.randn(1, 10)
    test_anomaly = torch.randn(1, 10) + 5 # Major deviation
    
    with torch.no_grad():
        loss_normal = criterion(model(test_normal), test_normal)
        loss_anomaly = criterion(model(test_anomaly), test_anomaly)
        
    print("\n--- Detection Results ---")
    print(f"Normal Sample Reconstruction Error: {loss_normal.item():.4f}")
    print(f"Anomaly Sample Reconstruction Error: {loss_anomaly.item():.4f}")
    
    threshold = 2.0
    if loss_anomaly > threshold:
        print(">> ALARM: Anomaly Detected! Machine requires maintenance.")

if __name__ == "__main__":
    run_anomaly_detector()
