# Project: Mixture of Experts (MoE) Transformer

**Goal:** Modify the standard Transformer FFN to use a sparse Mixture of Experts.

**The Logic:**
Instead of one massive FFN for every token, we have `N` experts (e.g., 4).
A "Router" (Gating Network) decides which experts process which tokens.
We only use top-k (e.g., 2) experts per token, saving compute while increasing capacity.

**Files:**

- `moe.py`: The `MoELayer` and `Expert` classes.
- `transformer_moe.py`: Shows how to swap the standard FeedForward with `MoELayer` inside an Encoder.
