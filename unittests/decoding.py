import sys
import os
import unittest
sys.path.append(os.getcwd())
from utils.transformer.model import QT
import torch
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from utils.configs import load_configs

config = load_configs(path='unittests/test_config.yaml')
MODEL = QT(
    config=config['transformer'],
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

    def test_topk(self):
        dummy_input = "."
        output = MODEL.decode(dummy_input, method='topk', max_tokens=10)
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)

    def test_topp(self):
        dummy_input = "."
        output = MODEL.decode(dummy_input, method='topp', max_tokens=10)
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)

if __name__ == '__main__':
    unittest.main()