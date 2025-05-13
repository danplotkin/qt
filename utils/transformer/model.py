from utils.transformer.layers import *
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from typing import Union, Literal

class Transformer(nn.Module):
    def __init__(
        self, 
        tgt_vocab_size: int, 
        d_model: int, num_heads: int, 
        num_layers: int, d_ff: int, 
        max_seq_length: int, 
        dropout: float,
        tokenizer: Union[GPT2Tokenizer, GPT2TokenizerFast],
        device: torch.device
    ) -> None:
        super(Transformer, self).__init__()
        self.to(device) # Put model on device on init
        self.device = device
        self.tokenizer = tokenizer
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Generate casual and padding mask"""
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(tgt.device)
        tgt_mask = tgt_mask & nopeak_mask
        return tgt_mask

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        """Decoder-only transformer forward pass"""
        tgt_mask = self.generate_mask(tgt)
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_mask)

        output = self.fc(dec_output)
        return output
    
    def change_device(self, device: torch.device) -> None:
        """Method to change device of model"""
        self.to(device)
        self.device = device

    @torch.no_grad()
    def decode(
        self,
        text: str,
        method: Literal["greedy", "beam", "sample"] = "greedy",
        max_tokens: int = 100,
        return_tokens: bool = False,
        beam_size: int = 3,
        tau: float = 1.0
    ) -> Union[str, list[int]]:
        """
        Decode text using specified decoding strategy.
        Args:
            text: input prompt
            method: decoding strategy - 'greedy', 'beam', or 'sample'
            max_tokens: max tokens to generate
            return_tokens: if True, return token ids instead of text
            beam_size: beam width for beam search
            tau: temperature for sampling
        Returns:
            Generated text or token list
        """
        if method == "greedy":
            return self._decode_greedy(text, max_tokens=max_tokens, return_tokens=return_tokens)
        elif method == "beam":
            return self._decode_beam(text, max_tokens=max_tokens, return_tokens=return_tokens, beam_size=beam_size)
        elif method == "sample":
            return self._decode_sample(text, max_tokens=max_tokens, return_tokens=return_tokens, tau=tau)
        else:
            raise ValueError(f"Unsupported decode method: {method}")

    @torch.no_grad()
    def _decode_greedy(self, text: str, max_tokens: int = 100, return_tokens: bool = False) -> Union[str, list[int]]:
        trg = self.tokenizer.encode(text, return_tensors='pt').to(self.device) # Tokenize text
        inference_tokens = []
        
        for _ in range(max_tokens):
            logits = self.forward(trg=trg)
            next_token = torch.argmax(logits[:, -1], dim=-1)  # [batch]

            # Check for end of seq
            if next_token.item() == self.tokenizer.eos_token_id:  # EOS token for GPT2 tokenier
                break

            # Append to inference token
            inference_tokens.append(next_token.item())

            # Append next token to trg
            trg = torch.cat([trg, next_token.unsqueeze(0)], dim=1)
            trg = trg[:, -self.seqlen:]  # context cuttoff

        if return_tokens:
            return inference_tokens
        return self.tokenizer.decode(inference_tokens)
    
    @torch.no_grad()
    def _decode_beam(self, text: str, max_tokens: int = 100, return_tokens: bool = False, beam_size: int = 3) -> Union[str, list[int]]:
        raise NotImplementedError
    
    @torch.no_grad()
    def _decode_sample(self, text: str, max_tokens: int = 100, return_tokens: bool = False, tau: float = 1.0) -> Union[str, list[int]]:
        raise NotImplementedError