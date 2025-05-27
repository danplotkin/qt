import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from tqdm import tqdm


class MiniPileDataset(Dataset):
    def __init__(self, path: str, block_size: int, stride: int = None, offset: int = 0):
        self.tokens = torch.load(path)
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