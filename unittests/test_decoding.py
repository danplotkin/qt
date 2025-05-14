import sys
import os
import unittest
sys.path.append(os.getcwd())
from utils.transformer.model import Transformer
import torch
from transformers import GPT2Tokenizer, GPT2TokenizerFast

MODEL = Transformer(
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

class TestDecodingMethods(unittest.TestCase):
    def test_greedy(self):
        dummy_input = "."
        output = MODEL.decode(dummy_input, method='greedy')
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)

    def test_sampling(self):
        dummy_input = "."
        output = MODEL.decode(dummy_input, method='sample', tau=1.0)
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)

    def test_beam(self):
        dummy_input = "."
        output = MODEL.decode(dummy_input, method='beam')
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)

if __name__ == '__main__':
    unittest.main()