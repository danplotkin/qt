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
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FFNN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply self-attention and feed-forward layers to input sequence."""
        attn_output = self.self_attn(x, x, x, tgt_mask)
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



def get_relative_positions(seq_len: int) -> torch.tensor:
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



class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism used in Transformer models."""
    def __init__(self, d_model: int, num_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.register_buffer("m", get_alibi_slope(self.num_heads))

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute attention output using scaled dot-product with optional masking."""
        seq_len = Q.shape[1] # get sequence length fo ALiBi
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ALiBi
        bias = (self.m * get_relative_positions(seq_len)).unsqueeze(0)
        attn_scores += bias
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split input tensor into multiple attention heads."""
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine attention heads back into a single tensor."""
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute multi-head attention output from input queries, keys, and values."""
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        output = self.W_o(self.combine_heads(attn_output))
        return output