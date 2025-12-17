import torch
import torch.nn as nn
import torch.optim as optim
from model import Transformer

def run_training_example():
    """
    Demonstrates a dummy training loop to verify the architecture works.
    In a real scenario, you would load a dataset (e.g., Multi30k or WMT).
    """
    
    # --- HYPERPARAMETERS ---
    # These match the 'Base' Transformer from the paper
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    n_head = 8
    d_ff = 2048
    n_layer = 6
    max_seq_len = 100
    dropout = 0.1
    batch_size = 3
    lr = 0.0001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- INITIALIZE MODEL ---
    model = Transformer(
        src_vocab_size, 
        tgt_vocab_size, 
        d_model, 
        n_head, 
        d_ff, 
        n_layer, 
        max_seq_len, 
        dropout
    ).to(device) # Move model to GPU if available

    # Initialize weights (Xavier init is standard for Transformers)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # --- DUMMY DATA ---
    # Random integers representing word IDs
    src_data = torch.randint(1, src_vocab_size, (batch_size, 10)).to(device) # Sentences of length 10
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, 10)).to(device)

    # --- OPTIMIZER & LOSS ---
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # Ignore the padding index (0) so the model isn't punished for predicting padding correctly
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()

    print("\n--- STARTING DUMMY TRAINING LOOP ---")
    for step in range(10):
        # 1. Forward Pass
        # NB: In seq2seq training, the Decoder input is the target shifted by one.
        # Here we just pass the full target for dimension verification purposes.
        # ideally: decoder_input = tgt[:, :-1], label = tgt[:, 1:]
        
        # For this demo, let's pretend tgt_data is the input to decoder
        output = model(src_data, tgt_data) # [batch, seq_len, tgt_vocab]

        # 2. Calculate Loss
        # Flatten output to [batch * seq_len, vocab_size] for CrossEntropy
        loss = criterion(output.view(-1, tgt_vocab_size), tgt_data.view(-1))
        
        # 3. Backbone
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1} | Loss: {loss.item():.4f}")

    print("Training loop complete. Model is functional.")

if __name__ == "__main__":
    run_training_example()
