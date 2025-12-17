# Fine-Tuning LLMs with QLoRA

This project demonstrates how to fine-tune a massive 20B+ parameter model (like Gemma-27B or Llama-3-70B) on consumer hardware using Quantized Low-Rank Adaptation (QLoRA).

## Why QLoRA?

A 27B model requires ~54GB of VRAM just to load. QLoRA reduces this by:

1. **4-Bit Quantization**: Loading weights in 4-bit precision (reducing memory by 4x).
2. **LoRA**: Freezing the main model and only training tiny adapter matrices.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure `config.yaml` with your target model.
3. Train:
   ```bash
   python train.py
   ```

## Files

- `train.py`: The training loop using HuggingFace `SFTTrainer`.
- `inference.py`: Merging adapters with the base model for generation.
