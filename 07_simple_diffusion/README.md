# Project: Simple Diffusion (DDPM)

**Goal:** Build a Denoising Diffusion Probabilistic Model (DDPM) from scratch to generate MNIST digits.

**The Logic:**

1. **Forward Process (Noise):** Slowly add Gaussian noise to an image over 1000 steps until it is pure static.
2. **Reverse Process (Denoise):** Train a U-Net to predict the noise added at specific step `t`.
3. **Generation:** Start with random noise and subtract the predicted noise 1000 times to reveal a number.

**Files:**

- `diffusion.py`: The mathematical scheduler (Beta, Alpha).
- `unet.py`: The Neural Network architecture (with Time Embeddings).
- `train.py`: The training loop.
