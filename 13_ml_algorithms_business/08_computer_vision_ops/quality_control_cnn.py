import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------
# BUSINESS PROBLEM: MANUFACTURING QUALITY CONTROL
# ---------------------------------------------------------
# Scenario: A factory leverages a camera to check for defects on a conveyor belt.
# Input: 1D Sensor signal or simplified Image representation.
# Output: [Pass, Defect]
# ---------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Simulating 1D Signal Processing (e.g., Vibration sensor or flattened line scan)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16 * 4, 2) # Adjusted for input size 10 -> conv(8) -> pool(4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def run_quality_control():
    # Mock Data: 100 samples, 1 channel, 10 time-steps/pixels
    # Normal patterns are smooth, Defects have spikes
    X = torch.randn(100, 1, 10) 
    y = torch.randint(0, 2, (100,)) # Binary labels
    
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("--- CNN Quality Control (Training) ---")
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
    print("\nInsight: The CNN filters (convolutions) learn to detect local patterns (spikes/scratches) invariant of position.")

if __name__ == "__main__":
    run_quality_control()
