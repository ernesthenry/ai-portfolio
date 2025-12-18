import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    STEP 1: CONVERTING WORDS TO VECTORS

    Computers don't understand text strings like "cat" or "dog". They only understand numbers.
    An 'Embedding' layer is essentially a lookup table that maps each word (represented by an ID)
    to a vector of continuous numbers.

    Why vectors? 
    Vectors allow us to capture semantic meaning. In a good embedding space,
    vector("King") - vector("Man") + vector("Woman") ≈ vector("Queen").
    """
    def __init__(self, d_model: int, vocab_size: int):
        """
        Args:
            d_model: The dimension of the vector space (e.g., 512). larger = more overlapping concepts captured.
            vocab_size: The size of our vocabulary (e.g., 10,000 words).
        """
        super().__init__()
        self.d_model = d_model
        # nn.Embedding is a learnable matrix of shape [vocab_size, d_model]
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x is a tensor of word IDs (integers)
        # We multiply by sqrt(d_model) to scale the embeddings.
        # Why? The Positional Encodings (added next) have a fixed scale.
        # If the learned embeddings are too small (close to 0), the position info will dominate.
        # Scaling them up balances the two signals.
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    STEP 2: INJECTING ORDER (The "Rhythm" of the sequence)

    Unlike RNNs (which process words one by one), Transformers process all words at once in parallel.
    This generates a problem: The model sees "The cat ate the mouse" exactly the same as "The mouse ate the cat".
    It has no concept of order.

    We fix this by adding a fixed vector pattern to each position.
    """
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)

        # Position indices [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term (frequencies)
        # The formula uses sine and cosine waves of geometrically increasing wavelengths.
        # This allows the model to learn to attend to relative positions (e.g., "3 words back").
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply Sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply Cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension [1, seq_len, d_model] so it broadcasts over any batch size
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # We perform simple addition: Embedding + Positional Signal
        # Slice pe to the current sequence length of x
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    STEP 3: MULTI-HEAD ATTENTION (The "Brain")

    This mechanism allows the model to look at other words in the sentence to gather context.
    "Multi-Head" means we do this multiple times in parallel, allowing the model to focus on 
    different aspects (e.g., Head 1 tracks grammar, Head 2 tracks synonyms).
    """
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_head = d_model // n_head # Dimension per head
        self.n_head = n_head
        self.d_model = d_model

        # The Projections
        # Q (Query): What am I looking for?
        # K (Key): What do I contain?
        # V (Value): What information do I pass along if I am selected?
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final output projection
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 1. Project and Split Heads
        # Transform [batch, seq, d_model] -> [batch, seq, n_head, d_head]
        # Then transpose to [batch, n_head, seq, d_head] for matrix multiplication
        Q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        # "How similar is Query i to Key j?"
        # Matrix multiplication: [batch, n_head, seq, d_head] x [batch, n_head, d_head, seq]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 3. Masking (Optional)
        # Masks are used to:
        # - Hide padding tokens (so we don't zero-in on empty space)
        # - Prevent peeking at future tokens (in the Decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # -infinity gets softmaxed to 0

        # 4. Softmax (Probabilities)
        # Convert scores to percentages (summing to 1)
        attention_weights = scores.softmax(dim=-1)

        # 5. Aggregate Values
        # Weighted sum of V based on attention scores
        context = torch.matmul(attention_weights, V)

        # 6. Concat Heads
        # [batch, n_head, seq, d_head] -> [batch, seq, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 7. Final Linear Projection
        return self.w_o(context)


class PositionwiseFeedForward(nn.Module):
    """
    STEP 4: FEED FORWARD NETWORK (The "Processor")

    After the tokens have "talked" to each other via Attention, each token processes 
    what it learned independently. This layer adds non-linearity and capacity to the model.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # Expansion (usually 4x d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # Compression back to d_model

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    STEP 5: ENCODER LAYER

    One block of the Encoder stack. Contains:
    1. Multi-Head Attention
    2. Feed Forward
    3. Residual Connections (Add) & Layer Normalization (Norm)
    """
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Sublayer 1: Self Attention
        attn_output = self.self_attn(x, x, x, mask)
        # Add & Norm
        x = self.norm1(x + self.dropout(attn_output))

        # Sublayer 2: Feed Forward
        ffn_output = self.ffn(x)
        # Add & Norm
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class DecoderLayer(nn.Module):
    """
    STEP 6: DECODER LAYER

    One block of the Decoder stack.
    Differs from Encoder because it has TWO attention layers:
    1. Self-Attention (Masked to see only past)
    2. Cross-Attention (Looks at Encoder output)
    """
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.cross_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 1. Masked Self Attention (Decoder only looks at past tokens)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Cross Attention (Query=Decoder, Key/Value=Encoder)
        # This is where the translation happens: "Given what I've said (Q), look at source (K/V)"
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 3. Feed Forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x


class Transformer(nn.Module):
    """
    STEP 7: THE FULL TRANSFORMER

    Assembles the Encoder and Decoder stacks.
    INPUT -> Encoder Stack -> Memory -> Decoder Stack -> OUTPUT
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff, n_layer, max_seq_len, dropout):
        super().__init__()

        # Encoder
        self.encoder_embedding = InputEmbeddings(d_model, src_vocab_size)
        self.encoder_pos = PositionalEncoding(d_model, max_seq_len, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])

        # Decoder
        self.decoder_embedding = InputEmbeddings(d_model, tgt_vocab_size)
        self.decoder_pos = PositionalEncoding(d_model, max_seq_len, dropout)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])

        # Final Projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src):
        # Mask padding (assuming 0 is padding index)
        # [batch, 1, 1, seq_len]
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        # 1. Padding mask
        padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)

        # 2. No-Peak mask (Causal mask) - Lower triangular matrix
        seq_len = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=tgt.device)).bool()

        return padding_mask & nopeak_mask

    def forward(self, src, tgt):
        # Create masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Encoder Pass
        enc_out = self.encoder_embedding(src)
        enc_out = self.encoder_pos(enc_out)
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        # Decoder Pass
        dec_out = self.decoder_embedding(tgt)
        dec_out = self.decoder_pos(dec_out)
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        # Output prediction
        return self.fc_out(dec_out)
