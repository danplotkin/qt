import numpy as np
import collections
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from typing import Union, Literal, Iterable
from tqdm import tqdm

from utils.transformer.layers import *
from utils.configs import TransformerConfigs

logger = logging.getLogger(__name__)


class QT(nn.Module):
    def __init__(
        self, 
        config: TransformerConfigs,
        tokenizer: Union[GPT2Tokenizer, GPT2TokenizerFast],
        device: torch.device
    ) -> None:
        super(QT, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.decoder_embedding = nn.Embedding(config.tgt_vocab_size, config.d_model)
        self.config = config
        # NOTE no positional encodings
        # self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config.d_model, config.num_heads, config.d_ff, config.dropout) for _ in range(config.num_layers)])
        self.fc = nn.Linear(config.d_model, config.tgt_vocab_size)
        self.fc.weight = self.decoder_embedding.weight # NOTE tie weights
        self.dropout = nn.Dropout(config.dropout)
        self.to(device) # Put model on device on init

    def generate_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Generate casual and padding mask"""
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            padding = torch.ones_like(tgt, dtype=torch.bool, device=tgt.device)
        else:
            padding = tgt != pad_id
        tgt_mask = padding.unsqueeze(1).unsqueeze(3)
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

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """Resize decoder embedding and tied output layer to match new vocab size"""
        old_embedding_weight = self.decoder_embedding.weight.data
        old_vocab_size, embedding_dim = old_embedding_weight.shape

        new_embedding = nn.Embedding(new_vocab_size, embedding_dim)
        new_embedding.weight.data[:old_vocab_size] = old_embedding_weight
        self.decoder_embedding = new_embedding
        self.fc = nn.Linear(embedding_dim, new_vocab_size)
        self.fc.weight = self.decoder_embedding.weight  # Re-tie weights
        self.to(self.device)  # Ensure new layers are moved to the correct device

    @torch.no_grad()
    def decode(
        self,
        text: str,
        method: Literal["greedy", "beam", "sample", "topk", "topp"] = "greedy",
        max_tokens: int = 100,
        return_stream: bool = False,
        return_tokens: bool = False,
        beam_size: int = 3,
        tau: float = 1.0,
        top_p: float = 0.9
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
        self.eval()
        if method == "greedy":
            return self._decode_greedy(text, max_tokens=max_tokens, return_tokens=return_tokens, return_stream=return_stream)
        elif method == "beam":
            return self._decode_beam(text, max_tokens=max_tokens, return_tokens=return_tokens, return_stream=return_stream, beam_size=beam_size)
        elif method == "sample":
            return self._decode_sample(text, max_tokens=max_tokens, return_tokens=return_tokens, return_stream=return_stream, tau=tau)
        elif method == "topk":
            return self._decode_topk(
                text, max_tokens=max_tokens, return_tokens=return_tokens, return_stream=return_stream, k=beam_size
            )
        elif method == "topp":
            return self._decode_topp(
                text,
                max_tokens=max_tokens,
                return_tokens=return_tokens,
                return_stream=return_stream,
                p=top_p,
                tau=tau
            )
        else:
            raise ValueError(f"Unsupported decode method: {method}")

    @torch.no_grad()
    def _decode_greedy(self, text: str, max_tokens: int = 100, return_tokens: bool = False, return_stream: bool = False) -> Union[str, list[int]]:
        device = next(self.parameters()).device
        tgt = self.tokenizer.encode(text, return_tensors='pt').to(device) # Tokenize text
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
            tgt = tgt[:, -self.config.max_seq_length:]  # context cuttoff

        if return_tokens:
            return inference_tokens
        if return_stream:
            print(self.tokenizer.decode(next_token.item()), end='')
        return self.tokenizer.decode(inference_tokens)
    
    @torch.no_grad()
    def _decode_beam(self, text: str, max_tokens: int = 100, return_tokens: bool = False, return_stream: bool = False, beam_size: int = 3) -> Union[str, list[int]]:
        device = next(self.parameters()).device
        tgt = self.tokenizer.encode(text, return_tensors='pt').to(device)
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
    def _decode_sample(self, text: str, max_tokens: int = 100, return_tokens: bool = False, return_stream: bool = False, tau: float = 1.0) -> Union[str, list[int]]:
        device = next(self.parameters()).device
        tgt = self.tokenizer.encode(text, return_tensors='pt').to(device) # Tokenize text
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
            tgt = tgt[:, -self.config.max_seq_length:]  # context cuttoff

        if return_tokens:
            return inference_tokens
        return self.tokenizer.decode(inference_tokens)

    @torch.no_grad()
    def _decode_topk(
        self,
        text: str,
        max_tokens: int = 100,
        return_tokens: bool = False,
        return_stream: bool = False,
        k: int = 10
    ) -> Union[str, list[int]]:
        """
        Top-k sampling decoding: at each step, sample from the top k logits.
        """
        device = next(self.parameters()).device
        tgt = self.tokenizer.encode(text, return_tensors='pt').to(device)
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
            tgt = tgt[:, -self.config.max_seq_length:]
        
        if return_tokens:
            return inference_tokens
        return self.tokenizer.decode(inference_tokens)

    @torch.no_grad()
    def _decode_topp(
        self,
        text: str,
        max_tokens: int = 100,
        return_tokens: bool = False,
        return_stream: bool = False,
        p: float = 0.9,
        tau: float = 1.0
    ) -> Union[str, list[int]]:
        """
        Nucleus (top-p) sampling decoding: at each step, sample from the smallest set of tokens
        whose cumulative probability mass exceeds p.
        """
        device = next(self.parameters()).device
        tgt = self.tokenizer.encode(text, return_tensors='pt').to(device)
        inference_tokens = []

        for _ in range(max_tokens):
            logits = self.forward(tgt=tgt)[:, -1, :] / tau
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = probs.cumsum(dim=-1)

            # Identify tokens to remove
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Map back to original indices and mask
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

            dist = Categorical(logits=logits)
            next_token = dist.sample()

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            inference_tokens.append(next_token.item())
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
            tgt = tgt[:, -self.config.max_seq_length:]

        if return_tokens:
            return inference_tokens
        return self.tokenizer.decode(inference_tokens)
