from utils.transformer.layers import *
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, tgt_vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, max_seq_length: int, dropout: float) -> None:
        super(Transformer, self).__init__()
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(tgt.device)
        tgt_mask = tgt_mask & nopeak_mask
        return tgt_mask

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt_mask = self.generate_mask(tgt)
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_mask)

        output = self.fc(dec_output)
        return output