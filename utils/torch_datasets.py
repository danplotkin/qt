import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from tqdm import tqdm


class MiniPileDataset(Dataset):
    def __init__(self, tokenized_ds: HFDataset, block_size: int = 512):
        self.block_size = block_size
        self.ds = tokenized_ds
        self.indices = []

        # Precompute (example_idx, start_pos) pairs
        for i, example in enumerate(tqdm(tokenized_ds, desc="Initalizing Dataset...")):
            input_ids = example["input_ids"]
            for j in range(0, len(input_ids) - block_size, block_size):
                self.indices.append((i, j))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        example_idx, start_pos = self.indices[idx]
        input_ids = self.ds[example_idx]["input_ids"][start_pos : start_pos + self.block_size + 1]
        input_tensor = torch.tensor(input_ids[:-1], dtype=torch.long)
        label_tensor = torch.tensor(input_ids[1:], dtype=torch.long)
        return {"input_ids": input_tensor, "labels": label_tensor}