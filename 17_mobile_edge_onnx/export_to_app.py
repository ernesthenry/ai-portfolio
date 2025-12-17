import torch
import torch.nn as nn
import torch.onnx

# 1. DEFINE A SIMPLE MODEL (e.g. The Spam Classifier)
class SmallModel(nn.Module):
    def __init__(self):
        super(SmallModel, self).__init__()
        self.linear = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# 2. INSTANTIATE & LOAD WEIGHTS
model = SmallModel()
model.eval() # Important: set to eval mode for export

# 3. CREATE DUMMY INPUT
# Shape must match the model's input
dummy_input = torch.randn(1, 5)

# 4. EXPORT TO ONNX
# Open Neural Network Exchange (ONNX) is the standard for 
# running models on iOS (CoreML), Android, or Browser (WASM).
output_path = "spam_classifier.onnx"

torch.onnx.export(
    model, 
    dummy_input, 
    output_path, 
    verbose=True,
    input_names=['input_features'], 
    output_names=['spam_prob'],
    dynamic_axes={'input_features': {0: 'batch_size'}} # Allow variable batch sizes
)

print(f"✅ Model exported to {output_path}")
print("This file can now be loaded in C++, C#, Java, or JavaScript apps.")
