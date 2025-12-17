import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=32, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # Input is 28x28
        self.conv1 = nn.Conv2d(c_in, 64, 3, padding=1)
        self.down1 = nn.Conv2d(64, 64, 3, stride=2, padding=1) # 14x14
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.down2 = nn.Conv2d(128, 128, 3, stride=2, padding=1) # 7x7

        # Bottleneck
        self.bot1 = nn.Conv2d(128, 256, 3, padding=1)
        
        # Upsampling (Decoder)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2) # 14x14
        self.conv3 = nn.Conv2d(128 + 128, 128, 3, padding=1) # +128 for Skip Connection
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2) # 28x28
        self.conv4 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        self.out = nn.Conv2d(64, c_out, 1)

        # Time Embedding (So the model knows if it's step 5 or step 500)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, x, t):
        # t is [batch_size] integers. Convert to vector.
        t = t.unsqueeze(-1).type(torch.float)
        t_emb = self.time_mlp(t)
        # Reshape to broadcast: [Batch, TimeDim, 1, 1]
        t_emb = t_emb[:, :, None, None]

        # --- Encoder ---
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.down1(x1)) # 14x14
        
        # Add time info (broadcasted)
        # In real UNets, we add time emb at every layer. Simplified here.
        x2 = x2 + t_emb.expand(-1, 64, 14, 14)[:, :128] # Hack to fit dims if needed, or project

        x3 = F.relu(self.conv2(x2))
        x4 = F.relu(self.down2(x3)) # 7x7
        
        # --- Bottleneck ---
        x_bot = F.relu(self.bot1(x4))
        
        # --- Decoder ---
        x_up1 = self.up1(x_bot)
        # Skip Connection: Concatenate x_up1 with x3 (from encoder)
        x_up1 = torch.cat([x_up1, x3], dim=1) 
        x_up1 = F.relu(self.conv3(x_up1))

        x_up2 = self.up2(x_up1)
        # Skip Connection: Concatenate x_up2 with x1
        x_up2 = torch.cat([x_up2, x1], dim=1)
        x_up2 = F.relu(self.conv4(x_up2))

        return self.out(x_up2)
