"""
This script loads, tokenizes, and saves tokenized data to disk for the minipile dataset. Wrote by Dan.
"""

import os
import sys
from typing import Union
sys.path.append(os.getcwd())

from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

from utils.tokenizer import get_tokenizer

OUTPUT_DIR = "data/tokenized/minipile"


def get_minipile() -> DatasetDict:
    if not os.path.exists('data/minipile'):
        ds = load_dataset("JeanKaddour/minipile")
        ds.save_to_disk('data/minipile')
    return load_from_disk('data/minipile')


def tokenize_dataset(tokenizer: any, ds: DatasetDict) -> DatasetDict:
    def _tokenize(examples):
        return tokenizer(
            examples['text'],
            add_special_tokens=False,
            return_attention_mask=False
        )
    tok_dataset = {}
    for set_type in ('train', 'validation', 'test'):
        tok_dataset[set_type] = ds[set_type].map(_tokenize, batched=True, remove_columns=['text'])
    tokenized = DatasetDict(tok_dataset) 
    return tokenized


def main():
    raw_ds = get_minipile()
    tokenizer = get_tokenizer()
    tokenized_ds = tokenize_dataset(tokenizer, raw_ds)
    tokenized_ds.save_to_disk(OUTPUT_DIR)


if __name__ == "__main__":
    main()
