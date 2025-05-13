import sys
import os
sys.path.append(os.getcwd())
from utils.transformer.model import Transformer
import torch
import torch.nn as nn


@torch.no_grad()
def main():
    model = Transformer(100, 10, 2, 1, 100, 50, 0.1)
    dummy_input = torch.randint(0, 10, size=(1, 50))
    print(f'Input Shape: {dummy_input.size()}')
    dummy_out = model.forward(dummy_input)
    print(f'Output Shape: {dummy_out.size()}')


if __name__ == '__main__':
    main()