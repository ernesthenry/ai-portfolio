import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    An 'Expert' is just a standard Feed Forward Network.
    It specializes in certain types of tokens (e.g., verbs, technical terms).
    """
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 1. Create a list of Expert Networks
        self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout) for _ in range(num_experts)])

        # 2. The Gating Network (The Router)
        # It takes input size (d_model) and outputs a score for each expert (num_experts)
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # Flatten inputs to [batch*seq, d_model] for processing
        x_flat = x.view(-1, d_model)

        # 1. Calculate Router Logits
        # shape: [batch*seq, num_experts]
        gate_logits = self.gate(x_flat)
        
        # 2. Select Top-K Experts
        # weights: The probabilities (softmax)
        # indices: Which experts were chosen (0, 3, etc.)
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # Normalize weights so they sum to 1 for the selected experts
        weights = F.softmax(weights, dim=-1)

        # 3. Route tokens to experts
        # We prepare a zero-tensor to hold the results
        final_output = torch.zeros_like(x_flat)

        # Naive Loop Implementation (For educational clarity)
        # In production (CUDA), this is done via scatter/gather operations
        for i, expert in enumerate(self.experts):
            # Create a mask: "Is this expert (i) in the top_k indices for this token?"
            # shape: [batch*seq, top_k] -> boolean
            batch_mask = (indices == i)
            
            # If this expert is used anywhere in the batch
            if batch_mask.any():
                # We flatten the mask to find relevant tokens
                # We sum across the top_k dimension to get a 1D mask for the batch
                token_mask = batch_mask.any(dim=-1)
                
                # Extract tokens assigned to this expert
                selected_tokens = x_flat[token_mask]
                
                # Pass through expert
                expert_out = expert(selected_tokens)
                
                # We need to multiply by the router weight (Importance)
                # Find which "rank" (1st choice, 2nd choice) this expert was
                # This logic gets complex in pure PyTorch, simplified here:
                weight_val = weights[batch_mask] # Extract specific weights
                
                # Reshape weight to match expert output [num_selected, 1]
                weight_val = weight_val.view(-1, 1)
                
                # Add to final output
                # We use index_add_ to put the values back into the correct rows
                indices_to_add = torch.nonzero(token_mask).squeeze()
                final_output.index_add_(0, indices_to_add, expert_out * weight_val)

        # 4. Load Balancing Loss (Auxiliary Loss)
        # We want to penalize the model if it only uses Expert #0.
        # We calculate the standard deviation of expert usage.
        # (Simplified definition)
        usage = gate_logits.softmax(dim=-1).mean(dim=0) # Avg probability per expert
        aux_loss = torch.sum(usage ** 2) * self.num_experts # Penalize spikiness

        return final_output.view(batch_size, seq_len, d_model), aux_loss
