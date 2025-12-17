import torch
import torch.nn as nn
import torch.optim as optim

# 1. DATA (2 features: Word Count, Link Count)
X_spam = torch.tensor([[10, 0], [50, 1], [200, 5], [5, 0], [300, 10]], dtype=torch.float32)
y_spam = torch.tensor([[0], [0], [1], [0], [1]], dtype=torch.float32) # 0=Safe, 1=Spam

# 2. MODEL
class SpamClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1) 
        self.sigmoid = nn.Sigmoid() # Squashes output between 0 and 1

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model_cls = SpamClassifier()
criterion_cls = nn.BCELoss() # Binary Cross Entropy (Standard for Yes/No)
optimizer_cls = optim.Adam(model_cls.parameters(), lr=0.01)

# 3. TRAIN
print("\n--- Training Spam Classifier ---")
for epoch in range(1000):
    y_pred = model_cls(X_spam)
    loss = criterion_cls(y_pred, y_spam)
    optimizer_cls.zero_grad()
    loss.backward()
    optimizer_cls.step()

# 4. TEST
# Lots of words (250), lots of links (6) -> Should be Spam
email_features = torch.tensor([[250, 6]], dtype=torch.float32) 
with torch.no_grad():
    probability = model_cls(email_features).item()
print(f"Spam Probability: {probability:.4f} ({'Spam' if probability > 0.5 else 'Safe'})")
