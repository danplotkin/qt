import os
import sys
import logging
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
sys.path.append(os.getcwd())

HUGGING_FACE_PATH = 'HuggingFaceH4/no_robots'
DS_OUTPUT_DIR = 'data/no_robots'

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_no_robots() -> None:
    if os.path.exists(DS_OUTPUT_DIR):
        logging.info(f"Dataset already exists at {DS_OUTPUT_DIR}. Skipping download.")
        return

    logging.info("Loading dataset from Hugging Face...")
    ds = load_dataset(HUGGING_FACE_PATH)

    logging.info("Splitting test set into validation and test...")
    test_val_split = ds["test"].train_test_split(test_size=0.5, seed=42)
    new_ds = DatasetDict({
        "train": ds["train"],
        "validation": test_val_split["train"],
        "test": test_val_split["test"]
    })

    logging.info(f"Saving dataset to {DS_OUTPUT_DIR}...")
    new_ds.save_to_disk(DS_OUTPUT_DIR)
    logging.info("Dataset saved successfully.")

if __name__ == "__main__":
    get_no_robots()