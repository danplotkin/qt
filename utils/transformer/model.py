import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from typing import Union, Literal, Iterable
import numpy as np
import collections
from tqdm import tqdm

from utils.transformer.layers import *


class QT(nn.Module):
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
        super(QT, self).__init__()
        self.max_seq_length = max_seq_length
        self.to(device) # Put model on device on init
        self.device = device
        self.tokenizer = tokenizer
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # NOTE no positional encodings
        # self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.fc.weight = self.decoder_embedding.weight # NOTE tie weights
        self.dropout = nn.Dropout(dropout)

    def initialize_output_bias(self, token_sequences: Iterable[int]):
        """
        Initializes the output layer's bias based on token frequency distribution.

        Args:
            token_sequences: An iterable of token IDs.
        Notes:
            All special tokens, as defined by self.tokenizer.all_special_ids, are excluded from frequency counts.
        """
        counts = collections.Counter(token_sequences)

        vocab_size = self.decoder_embedding.num_embeddings
        counts_arr = np.zeros(shape=(vocab_size,))
        for idx, count in counts.items():
            if idx < vocab_size:
                counts_arr[idx] = count

        # Zero out counts for all special tokens defined by the tokenizer
        for special_id in self.tokenizer.all_special_ids:
            if special_id < vocab_size:
                counts_arr[special_id] = 0

        total = counts_arr.sum()
        p = counts_arr / total
        p[counts_arr == 0] = 1.0
        log_p = np.log(p)

        entropy = -(log_p * p).sum()
        print(f"\nUniform entropy: {np.log(vocab_size):0.2f}")
        print(f"Marginal entropy: {entropy:0.2f}")

        log_p[counts_arr == 0] = -1e9
        self.fc.bias.data = torch.tensor(log_p, dtype=torch.float32, device=self.device)

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
        # tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        # NOTE no positional encodings, using ALiBi 
        tgt_embedded = self.dropout(self.decoder_embedding(tgt))

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
        method: Literal["greedy", "beam", "sample", "topk"] = "greedy",
        max_tokens: int = 100,
        return_tokens: bool = False,
        beam_size: int = 3,
        tau: float = 1.0
    ) -> Union[str, list[int]]:
        """
        Decode text using specified decoding strategy.
        Args:
            text: input prompt
            method: decoding strategy - 'greedy', 'beam', 'sample', or 'topk'
            max_tokens: max tokens to generate
            return_tokens: if True, return token ids instead of text
            beam_size: beam width for beam search or top-k
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
        elif method == "topk":
            return self._decode_topk(
                text, max_tokens=max_tokens, return_tokens=return_tokens, k=beam_size
            )
        else:
            raise ValueError(f"Unsupported decode method: {method}")

    @torch.no_grad()
    def _decode_greedy(self, text: str, max_tokens: int = 100, return_tokens: bool = False) -> Union[str, list[int]]:
        tgt = self.tokenizer.encode(text, return_tensors='pt').to(self.device) # Tokenize text
        inference_tokens = []
        
        for _ in range(max_tokens):
            logits = self.forward(tgt=tgt)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)  # [batch]          

            # Check for end of seq
            if next_token.item() == self.tokenizer.eos_token_id:  # EOS token for GPT2 tokenier
                break

            # Append to inference token
            inference_tokens.append(next_token.item())

            # Append next token to tgt
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
            tgt = tgt[:, -self.max_seq_length:]  # context cuttoff

        if return_tokens:
            return inference_tokens
        return self.tokenizer.decode(inference_tokens)
    
    @torch.no_grad()
    def _decode_beam(self, text: str, max_tokens: int = 100, return_tokens: bool = False, beam_size: int = 3) -> Union[str, list[int]]:
        tgt = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        sequences = [(tgt, 0)]  # (tokens, log_prob)

        for _ in range(max_tokens):
            all_candidates = []
            for seq, score in sequences:
                if seq[0, -1].item() == self.tokenizer.eos_token_id:
                    all_candidates.append((seq, score))
                    continue

                logits = self.forward(tgt=seq)[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    candidate_seq = torch.cat([seq, topk_indices[:, i].unsqueeze(0)], dim=-1)
                    candidate_score = score + topk_log_probs[0, i].item()
                    all_candidates.append((candidate_seq, candidate_score))

            sequences = sorted(all_candidates, key=lambda tup: tup[1] / len(tup[0][0]), reverse=True)[:beam_size]

            # Early stop if all sequences end with EOS
            if all(seq[0, -1].item() == self.tokenizer.eos_token_id for seq, _ in sequences):
                break

        best_seq = sequences[0][0][0, tgt.size(1):]  # Remove prompt
        if return_tokens:
            return best_seq.tolist()
        return self.tokenizer.decode(best_seq)

    @torch.no_grad()
    def _decode_sample(self, text: str, max_tokens: int = 100, return_tokens: bool = False, tau: float = 1.0) -> Union[str, list[int]]:
        tgt = self.tokenizer.encode(text, return_tensors='pt').to(self.device) # Tokenize text
        inference_tokens = []

        for _ in range(max_tokens):
            logits = self.forward(tgt=tgt)

            # Apply tempurature sampling 
            dist = Categorical(logits=logits[:, -1, :]/tau)
            next_token = dist.sample() # (batch)

            # Check for end of seq
            if next_token.item() == self.tokenizer.eos_token_id:  # EOS token for GPT2 tokenier
                break

            # Append to inference token
            inference_tokens.append(next_token.item())

            # Append next token to tgt
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
            tgt = tgt[:, -self.max_seq_length:]  # context cuttoff

        if return_tokens:
            return inference_tokens
        return self.tokenizer.decode(inference_tokens)

    @torch.no_grad()
    def _decode_topk(
        self,
        text: str,
        max_tokens: int = 100,
        return_tokens: bool = False,
        k: int = 10
    ) -> Union[str, list[int]]:
        """
        Top-k sampling decoding: at each step, sample from the top k logits.
        """
        tgt = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        inference_tokens = []

        for _ in range(max_tokens):
            logits = self.forward(tgt=tgt)[:, -1, :]  # [batch, vocab]
            # Keep only top k logits
            topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, topk_idx, topk_vals)
            dist = Categorical(logits=mask)
            next_token = dist.sample()  # (batch)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            inference_tokens.append(next_token.item())
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
            tgt = tgt[:, -self.max_seq_length:]
        
        if return_tokens:
            return inference_tokens
        return self.tokenizer.decode(inference_tokens)
