from torch.utils.data import ConcatDataset
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from tqdm import tqdm
from typing import Literal
import os
import json
from utils.tokenizer import get_tokenizer
#

# Define subreddit file lists for train/val/test splits (all 14 subreddits)
REDDIT_TRAIN_FILES = [
    'bodyweightfitness.pt', 'socialskills.pt'
]

REDDIT_VAL_FILES = [
    'YouShouldKnow.pt'
]

REDDIT_TEST_FILES = [
    'podcasts.pt'
]


class NoRobotsDataset(Dataset):
    def __init__(self, split: Literal['train', 'validation', 'test'], block_size: int, stride: int = None, dir: str = 'data/no_robots'):
        from datasets import load_from_disk
        self.tokenizer = get_tokenizer()
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size // 2
        self.dataset = load_from_disk(dir)[split]

        # Precompute strided index mappings: (example_idx, chunk_start)
        self.index_map = []
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            messages = item["messages"]

            # Extract system prompt if it exists
            system_prompt = "You are a helpful assistant."
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                    break

            prompt_parts = [f"<|system|> {system_prompt}"]
            for msg in messages[:-1]:
                role = msg["role"]
                if role == "user":
                    prompt_parts.append(f"<|user|> {msg['content']}")
                elif role == "assistant":
                    prompt_parts.append(f"<|assistant|> {msg['content']}")
            prompt_text = " ".join(prompt_parts)
            final_response = messages[-1]["content"]
            full_text = f"<s>{prompt_text} <|assistant|> {final_response} </s>"

            tokens = self.tokenizer(full_text)["input_ids"]
            for start in range(0, max(1, len(tokens) - self.block_size + 1), self.stride):
                self.index_map.append((i, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        item_idx, start = self.index_map[idx]
        item = self.dataset[item_idx]
        messages = item["messages"]

        # Extract system prompt if it exists
        system_prompt = "You are a helpful assistant."
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                break

        prompt_parts = [f"<|system|> {system_prompt}"]
        for msg in messages[:-1]:
            role = msg["role"]
            if role == "user":
                prompt_parts.append(f"<|user|> {msg['content']}")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|> {msg['content']}")
        prompt_text = " ".join(prompt_parts)
        final_response = messages[-1]["content"]
        full_text = f"<s>{prompt_text} <|assistant|> {final_response} </s>"

        tokens = self.tokenizer(full_text)["input_ids"]
        chunk = tokens[start : start + self.block_size + 1]
        chunk = chunk + [self.tokenizer.pad_token_id] * (self.block_size + 1 - len(chunk))

        chunk = torch.tensor(chunk, dtype=torch.long)
        input_tensor = chunk[:-1].clone().long()
        label_tensor = chunk[1:].clone().long()
        return input_tensor, label_tensor
    

class TwitterCustomerCareDataset(Dataset):
    def __init__(self, split: Literal['train', 'validation', 'test'], block_size: int, stride: int = None, dir: str = 'data/twitter'):
        from datasets import load_from_disk
        self.tokenizer = get_tokenizer()
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size // 2
        self.dataset = load_from_disk(dir)[split]

        self.index_map = []
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            system_prompt = "You are a helpful assistant helping with users with customer service requests."

            prompt_parts = [f"<|system|> {system_prompt}"]
            for turn in item["dialogContent"]:
                role_tag = "<|user|>" if turn["agent"] is None else "<|assistant|>"
                message = turn["message"].strip()
                if message:
                    prompt_parts.append(f"{role_tag} {message}")
            agent_response = item["agentURL"]["url_utterance"].strip()
            full_text = f"<s>{' '.join(prompt_parts)} <|assistant|> {agent_response} </s>"
            tokens = self.tokenizer(full_text)["input_ids"]
            for start in range(0, max(1, len(tokens) - self.block_size + 1), self.stride):
                self.index_map.append((i, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        item_idx, start = self.index_map[idx]
        item = self.dataset[item_idx]
        system_prompt = "You are a helpful assistant helping with users with customer service requests."

        prompt_parts = [f"<|system|> {system_prompt}"]
        for turn in item["dialogContent"]:
            role_tag = "<|user|>" if turn["agent"] is None else "<|assistant|>"
            message = turn["message"].strip()
            if message:
                prompt_parts.append(f"{role_tag} {message}")
        agent_response = item["agentURL"]["url_utterance"].strip()
        full_text = f"<s>{' '.join(prompt_parts)} <|assistant|> {agent_response} </s>"
        tokens = self.tokenizer(full_text)["input_ids"]
        chunk = tokens[start : start + self.block_size + 1]
        chunk = chunk + [self.tokenizer.pad_token_id] * (self.block_size + 1 - len(chunk))
        chunk = torch.tensor(chunk, dtype=torch.long)
        input_tensor = chunk[:-1].clone().long()
        label_tensor = chunk[1:].clone().long()
        return input_tensor, label_tensor
    

class MiniPileDataset(Dataset):
    def __init__(self, split: Literal['train', 'validation', 'test'], block_size: int, stride: int = None, dir: str = 'data/flattened_corpa/minipile'):
        self.tokens = torch.load(os.path.join(dir, split + '.pt'), mmap=True)
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

        # Precompute index ranges per file and store file paths only
        for i, file in tqdm(enumerate(self.files), total=len(self.files), desc=f"Reddit data: indexing {split} set...", leave=False):
            path = os.path.join(dir, file)
            if not os.path.exists(path):
                print(f"Warning: File does not exist and will be skipped: {path}")
                continue
            meta_path = path.replace(".pt", ".json")
            if not os.path.exists(meta_path):
                print(f"Warning: Metadata file not found for {path}. Skipping.")
                continue
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                length = meta['length']
            except Exception as e:
                print(f"Warning: Failed to load metadata for '{path}': {e}. Skipping this file.")
                continue
            num_chunks = max(0, (length - block_size) // self.stride)
            self.file_lengths.append(num_chunks)
            self.file_offsets.append(self.total_chunks)
            self.total_chunks += num_chunks
            self._token_cache[i] = path  # store the path instead of the tokens

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

        path = self._token_cache.get(file_index)
        if path is None:
            raise KeyError(f"[RedditCommentsDataset] file_index {file_index} not found in _token_cache")
        tokens = torch.load(path, mmap=True)
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


class FineTuneCorpusDataset(Dataset):
    def __init__(self, split: Literal['train', 'validation', 'test'], block_size: int, stride: int = None):
        no_robots_dataset = NoRobotsDataset(split=split, block_size=block_size, stride=stride)
        twitter_dataset = TwitterCustomerCareDataset(split=split, block_size=block_size, stride=stride)
        self.dataset = ConcatDataset([no_robots_dataset, twitter_dataset])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]