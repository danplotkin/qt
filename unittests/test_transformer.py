import sys
import os
import unittest
sys.path.append(os.getcwd())
from utils.transformer.model import Transformer
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2TokenizerFast


class TestTransformerModel(unittest.TestCase):
    @torch.no_grad()
    def test_forward_output_shape(self):
        model = Transformer(
            tgt_vocab_size=100,
            d_model=100,
            num_heads=2,
            max_seq_length=100,
            dropout=0.1,
            num_layers=1,
            d_ff=10,
            tokenizer=GPT2TokenizerFast.from_pretrained('gpt2'),
            device='cpu'
        )
        dummy_input = torch.randint(0, 10, size=(1, 50))
        output = model.forward(dummy_input)
        self.assertEqual(output.size(), torch.Size([1, 50, 100]))


if __name__ == '__main__':
    unittest.main()