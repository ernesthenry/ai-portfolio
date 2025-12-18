import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------------------------------------------------------
# BUSINESS PROBLEM: INFRASTRUCTURE SCALING (Time Series)
# ---------------------------------------------------------
# Scenario: Cloud Server Load Prediction.
# Goal: Predict CPU usage for the next hour to autoscaling before traffic spikes.
# Data: Sequential time-series (Previous 5 hours -> Next 1 hour).
# ---------------------------------------------------------

class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=10, batch_first=True)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, hn = self.rnn(x)
        # We use the output of the last time step
        return self.fc(out[:, -1, :])

def run_server_forecast():
    # Mock Data: Sine wave (daily traffic pattern)
    t = np.linspace(0, 100, 200)
    data = np.sin(t) + np.random.normal(0, 0.1, 200)
    
    # Prepare sequences
    seq_len = 5
    X = []
    y = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
        
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) # (N, L, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    
    model = SimpleRNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("--- RNN Server Load Forecasting ---")
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, MSE Loss: {loss.item():.4f}")
            
    print("\nInsight: RNN retains 'memory' of the sequence (traffic trend) to predict the next step.")

if __name__ == "__main__":
    run_server_forecast()
