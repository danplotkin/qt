import sys
import os
import unittest
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2TokenizerFast

sys.path.append(os.getcwd())

from utils.transformer.model import QT
from utils.tokenizer import get_tokenizer
from utils.configs import load_configs

config = load_configs(path='unittests/test_config.yaml')

class TestTransformerModel(unittest.TestCase):
    @torch.no_grad()
    def test_forward_output_shape(self):
        model = QT(
            config=config['transformer'],
            tokenizer=get_tokenizer(),
            device='cpu'
        )
        dummy_input = torch.randint(0, 10, size=(1, 50))
        output = model.forward(dummy_input)
        self.assertEqual(output.size(), torch.Size([1, 50, 100]))


if __name__ == '__main__':
    unittest.main()