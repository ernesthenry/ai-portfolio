import torch

class DiffusionUtils:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.noise_steps = noise_steps
        self.device = device
        
        # Define the linear schedule for beta
        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        
        # Calculate Alpha (1 - beta)
        self.alpha = 1. - self.beta
        
        # Calculate Cumulative Alpha (alpha_hat)
        # This allows us to jump to any step 't' in one go without looping
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        """
        Takes an image x, and a timestep t.
        Returns: The image with 't' amount of noise added.
        Formula: sqrt(alpha_hat) * x + sqrt(1 - alpha_hat) * epsilon
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        
        epsilon = torch.randn_like(x) # Random Gaussian Noise
        
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        """Get random time steps for training"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)
