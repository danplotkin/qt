from torch.utils.data import ConcatDataset
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from tqdm import tqdm
from typing import Literal
import os
from utils.tokenizer import get_tokenizer
#

# Define subreddit file lists for train/val/test splits (all 14 subreddits)
REDDIT_TRAIN_FILES = [
    'bestof.pt', 'bodyweightfitness.pt', 'buildapc.pt',
    'Documentaries.pt', 'explainlikeimfive.pt', 'history.pt',
    'philosophy.pt', 'podcasts.pt'
]

REDDIT_VAL_FILES = [
    'programming.pt', 'socialskills.pt', 'tifu.pt'
]

REDDIT_TEST_FILES = [
    'WritingPrompts.pt', 'YouShouldKnow.pt'
]



class MiniPileDataset(Dataset):
    def __init__(self, split: Literal['train', 'validation', 'test'], block_size: int, stride: int = None, dir: str = 'data/flattened_corpa/minipile'):
        self.tokens = torch.load(os.path.join(dir, split + '.pt'))
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        self.indices = list(range(0, len(self.tokens) - block_size, self.stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        chunk = self.tokens[start : start + self.block_size + 1]
        input_tensor = chunk[:-1].clone().long()
        label_tensor = chunk[1:].clone().long()
        return input_tensor, label_tensor
    

class RedditCommentsDataset(Dataset):
    def __init__(self, split: Literal['train', 'val', 'test'], block_size: int, stride: int = None, dir: str = 'data/flattened_corpa/reddit_comments'):
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size

        if split == 'train':
            self.files = REDDIT_TRAIN_FILES
        elif split == 'val':
            self.files = REDDIT_VAL_FILES
        elif split == 'test':
            self.files = REDDIT_TEST_FILES
        else:
            raise ValueError(f"Invalid split: {split}")

        self.file_lengths = []
        self.file_offsets = []
        self.total_chunks = 0
        self._token_cache = {}

        # Precompute index ranges per file and load all tokens into cache
        for i, file in enumerate(tqdm(self.files, desc=f"Reddit data: indexing {split} set...", leave=False)):
            path = os.path.join(dir, file)
            tokens = torch.load(path)
            self._token_cache[i] = tokens
            length = len(tokens)
            num_chunks = max(0, (length - block_size) // self.stride)
            self.file_lengths.append(num_chunks)
            self.file_offsets.append(self.total_chunks)
            self.total_chunks += num_chunks

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        # Determine which file the idx belongs to
        for i, offset in enumerate(self.file_offsets):
            if idx < offset + self.file_lengths[i]:
                file_index = i
                local_idx = idx - offset
                break
        else:
            raise IndexError(f"Index {idx} out of range")

        tokens = self._token_cache[file_index]
        start = local_idx * self.stride
        chunk = tokens[start : start + self.block_size + 1]
        input_tensor = chunk[:-1].clone().long()
        label_tensor = chunk[1:].clone().long()
        return input_tensor, label_tensor
    

class ExampleCorpusDataset(Dataset):
    """This is just for unittesting"""
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
    
    
class PretrainedCorpaDataset(Dataset):
    def __init__(self, split: Literal['train', 'val', 'test'], block_size: int, stride: int = None):
        # Reddit uses 'val', MiniPile uses 'validation'
        reddit_dataset = RedditCommentsDataset(split=split, block_size=block_size, stride=stride)
        minipile_split = 'validation' if split == 'val' else split
        minipile_dataset = MiniPileDataset(split=minipile_split, block_size=block_size, stride=stride)
        self.dataset = ConcatDataset([reddit_dataset, minipile_dataset])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]