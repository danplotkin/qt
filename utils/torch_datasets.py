import torch
from torch.utils.data import Dataset
from datasets import Dataset
from tqdm import tqdm


class MiniPileDataset(Dataset):
    def __init__(self, tokenized_ds: Dataset, block_size: int = 512):
        self.block_size = block_size

        # Concatenate all input_ids from the dataset into a single long token stream
        all_ids = []
        for example in tqdm(tokenized_ds):
            all_ids.extend(example["input_ids"])

        # Create examples of block_size + 1 for shifting
        total_len = (len(all_ids) // (block_size + 1)) * (block_size + 1)
        self.tokens = all_ids[:total_len]

        self.examples = [
            self.tokens[i:i+block_size+1]
            for i in tqdm(range(0, total_len, block_size + 1))
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}