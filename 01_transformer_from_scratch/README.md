# Building a Transformer from Scratch

This project implements the "Attention Is All You Need" architecture from scratch using PyTorch. It is designed to demystify Large Language Models (LLMs) by building them bottom-up.

## Architecture Steps

1. **Input Embeddings**: Converting tokens to vectors.
2. **Positional Encoding**: Injecting order information (Sine/Cosine waves).
3. **Multi-Head Attention**: The core mechanism enabling the model to focus on different parts of the sequence.
4. **Feed-Forward Network**: Processing information per token.
5. **Encoder & Decoder Stacks**: The full architecture.

## Usage

Run the training script to see a dummy forward pass:

```bash
python train.py
```

## Key Concepts

- **Self-Attention**: $ \text{Softmax}(\frac{QK^T}{\sqrt{d_k}})V $
- **Layer Normalization**: Stabilizing training.
- **Residual Connections**: Preventing vanishing gradients.
