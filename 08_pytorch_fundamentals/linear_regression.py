import torch
import torch.nn as nn
import torch.optim as optim

# 1. DATA GENERATION (Synthetic)
# X = House Size, Y = Price
# We reshape to [N, 1] because PyTorch expects 2D tensors for features
X = torch.tensor([[1000], [1200], [1500], [1800], [2000], [2200], [2500]], dtype=torch.float32)
y = torch.tensor([[300], [350], [400], [450], [500], [550], [600]], dtype=torch.float32) # In thousands

# 2. MODEL DEFINITION
# A single linear layer: Output = Weight * Input + Bias
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1) # 1 input feature, 1 output feature

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 3. OPTIMIZATION SETUP
criterion = nn.MSELoss() # Mean Squared Error (Standard for Regression)
optimizer = optim.SGD(model.parameters(), lr=0.0000001) # Low key learning rate effectively non-normalized data

# 4. TRAINING LOOP
print("--- Training Linear Regression ---")
for epoch in range(2000):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward pass (The Learning)
    optimizer.zero_grad() # Clear old gradients
    loss.backward()       # Calculate new gradients
    optimizer.step()      # Update weights

    if (epoch+1) % 400 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 5. INFERENCE
new_house = torch.tensor([[1600]], dtype=torch.float32)
with torch.no_grad():
    predicted_price = model(new_house).item()
print(f"\nPredicted price for 1600 sqft: ${predicted_price:.2f}k")
