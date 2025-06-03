"""
This script loads, tokenizes, and saves tokenized data to disk for the minipile dataset. Wrote by Dan.
"""

import os
import sys
from time import time
from typing import Union
import torch
sys.path.append(os.getcwd())

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

from utils.tokenizer import get_tokenizer
from tqdm import tqdm

OUTPUT_DIR = "data/tokenized/wikipedia"
FLATTENED_CORPA_DIR = "data/flattened_corpa/wikipedia"
os.makedirs(FLATTENED_CORPA_DIR, exist_ok=True)

def get_wikipedia():
    from datasets import load_dataset
    # get 2022 english wikipedia
    dataset = load_dataset("wikipedia", "20220301.en")

    save_dir = "data/wikipedia"
    dataset.save_to_disk(save_dir) 

def tokenize_dataset(tokenizer: any, ds: DatasetDict) -> DatasetDict:
    def _tokenize(examples):
        return tokenizer(
            examples['text'],
            add_special_tokens=False,
            return_attention_mask=False
        )
    tok_dataset = {}
    # for set_type in ('train', 'validation', 'test'):
    for set_type in ['train']:
        logger.info(f"Tokenizing {set_type} split...")
        tok_dataset[set_type] = ds[set_type].map(_tokenize, batched=True, remove_columns=['text'])
    tokenized = DatasetDict(tok_dataset) 
    logger.info("Tokenization complete.")
    return tokenized


def main():
    logger.info("Starting tokenization pipeline...")

    get_wikipedia()

    raw_ds = load_dataset('data/wikipedia')


    tokenizer = get_tokenizer()
    if os.path.exists(OUTPUT_DIR):
        logger.info("Tokenized dataset already exists. Loading from disk...")
        tokenized_ds = load_from_disk(OUTPUT_DIR)
    else:
        tokenized_ds = tokenize_dataset(tokenizer, raw_ds)
        tokenized_ds.save_to_disk(OUTPUT_DIR)
        logger.info(f"Tokenized dataset saved to {OUTPUT_DIR}")    

    for split in tokenized_ds:
        out_path = os.path.join(FLATTENED_CORPA_DIR, f"{split}.pt")
        if os.path.exists(out_path):
            logger.info(f"Flattened tensor for {split} already exists. Skipping...")
            continue
        logger.info(f"Flattening and saving {split} split as .pt tensor...")
        all_ids = []
        for batch in tqdm(tokenized_ds[split], desc=f"Processing {split}"):
            all_ids.extend(batch["input_ids"])
        tensor = torch.tensor(all_ids, dtype=torch.long)
        torch.save(tensor, out_path)
        logger.info(f"{split} tensor saved to {out_path} with shape {tensor.shape}")


if __name__ == "__main__":
    start = time()
    main()
    print(f'Tokenized MiniPile in {(time()-start):.4f} s')
