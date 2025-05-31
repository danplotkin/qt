import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from tqdm import tqdm
from typing import Literal
import os
from utils.tokenizer import get_tokenizer


class MiniPileDataset(Dataset):
    def __init__(self, split: Literal['train', 'validation', 'test'], block_size: int, stride: int = None, offset: int = 0, dir: str = 'data/flattened_corpa/minipile'):
        self.tokens = torch.load(os.path.join(dir, split + '.pt'))
        self.offset = offset
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        self.indices = list(range(0 + offset, len(self.tokens) - block_size, self.stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        chunk = self.tokens[start : start + self.block_size + 1]
        input_tensor = chunk[:-1].clone().long()
        label_tensor = chunk[1:].clone().long()
        return input_tensor, label_tensor
    

class ExampleCorpusDataset(Dataset):
    def __init__(self, split: Literal['train', 'val'], block_size: int, stride: int = None):
        with open('unittests/corpus.txt') as f:
            self.corpus = f.read()
        self.tokenizer = get_tokenizer()
        tokens = self.tokenizer.encode(self.corpus)
        total_tokens = len(tokens)
        train_end = int(0.8 * total_tokens)

        if split == 'train':
            tokens = tokens[:train_end]
        elif split == 'val':
            tokens = tokens[train_end:]

        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.indices = list(range(0, len(self.tokens) - block_size, self.stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        chunk = self.tokens[start : start + self.block_size + 1]
        input_tensor = chunk[:-1].clone().long()
        label_tensor = chunk[1:].clone().long()
        return input_tensor, label_tensor