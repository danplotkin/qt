import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from tqdm import tqdm


class MiniPileDataset(Dataset):
    def __init__(self, path: str, block_size: int, stride: int = None):
        self.tokens = torch.load(path)
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        self.indices = list(range(0, len(self.tokens) - block_size - 1, self.stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        chunk = self.tokens[start : start + self.block_size + 1]
        input_tensor = torch.tensor(chunk[:-1].clone(), dtype=torch.long)
        label_tensor = torch.tensor(chunk[1:].clone(), dtype=torch.long)
        return {"input_ids": input_tensor, "labels": label_tensor}