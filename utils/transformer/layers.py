import torch
import torch.nn as nn
import math
from typing import Optional


class FFNN(nn.Module):
    """Feed-forward network used in each Transformer layer."""
    def __init__(self, d_model: int, d_ff: int) -> None:
        super(FFNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two-layer feed-forward transformation to input."""
        return self.network(x)
    

class DecoderLayer(nn.Module):
    """A single layer of the Transformer decoder (no cross-attention)."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = FFNN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply self-attention and feed-forward layers to input sequence."""
        # x: (batch_size, seq_length, d_model)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    

class SinusoidalPositionalEncoding(nn.Module):
    """Adds sinusoidal positional encoding to token embeddings."""
    def __init__(self, d_model: int, max_seq_length: int) -> None:
        super(SinusoidalPositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to input tensor."""
        return x + self.pe[:, :x.size(1)]



def get_relative_positions(seq_len: int) -> torch.Tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


def get_alibi_slope(num_heads):
    x = (2 ** (8 / num_heads))
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )