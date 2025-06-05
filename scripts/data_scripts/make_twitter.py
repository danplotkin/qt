import os
import sys
import logging
import json
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
sys.path.append(os.getcwd())

DATA_PATH = 'twitter-customer-care-document-prediction'
DS_OUTPUT_DIR = 'data/twitter'

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_twitter() -> None:
    # if os.path.exists(DS_OUTPUT_DIR):
    #     logging.info(f"Dataset already exists at {DS_OUTPUT_DIR}. Skipping download.")
    #     return

    logging.info("Loading dataset from Disk...")

    with open(os.path.join(DATA_PATH, f'train.json'), 'r') as f:
        train_json = json.load(f)
    with open(os.path.join(DATA_PATH, f'dev.json'), 'r') as f:
        val_json = json.load(f)
    with open(os.path.join(DATA_PATH, f'test.json'), 'r') as f:
        test_json = json.load(f)
    
    new_ds = DatasetDict({
        "train": Dataset.from_list(train_json),
        "validation": Dataset.from_list(val_json),
        "test": Dataset.from_list(test_json),
    })

    logging.info(f"Saving dataset to {DS_OUTPUT_DIR}...")
    new_ds.save_to_disk(DS_OUTPUT_DIR)
    logging.info("Dataset saved successfully.")

if __name__ == "__main__":
    get_twitter()