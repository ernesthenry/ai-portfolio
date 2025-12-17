from moe import MoELayer
import torch.nn as nn

class EncoderLayerMoE(nn.Module):
    def __init__(self, d_model, n_head, d_ff, num_experts, top_k, dropout):
        super().__init__()
        # ... (Self Attention initialization stays the same) ...
        # NOTE: For a real impl, you'd need the SelfAttention class from the other file
        # or import it. Here we just show the MoE integration.
        # self.self_attn = MultiHeadAttention(d_model, n_head) 
        
        # REPLACE Standard FeedForward WITH MoELayer
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Sublayer 1: Attention
        # ... (Same as before) ...
        # attn_output = self.self_attn(x, x, x, mask)
        # x = self.norm1(x + self.dropout(attn_output))
        
        # Sublayer 2: MoE
        # Note: MoE returns (output, aux_loss)
        moe_output, aux_loss = self.moe(x)
        x = self.norm2(x + self.dropout(moe_output))
        
        return x, aux_loss
