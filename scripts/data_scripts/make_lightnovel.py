import os
import logging
from datasets import load_dataset, DatasetDict

LIGHTNOVEL_PATH = "Chat-Error/lightnovel-2048"
DS_OUTPUT_DIR = "data/lightnovel"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_lightnovel() -> None:
    if os.path.exists(DS_OUTPUT_DIR):
        logging.info(f"Dataset already exists at {DS_OUTPUT_DIR}. Skipping download.")
        return

    logging.info("Loading LightNovel dataset from Hugging Face...")
    ds = load_dataset(LIGHTNOVEL_PATH)

    logging.info(f"Saving dataset to {DS_OUTPUT_DIR}...")
    ds.save_to_disk(DS_OUTPUT_DIR)
    logging.info("Dataset saved successfully.")

if __name__ == "__main__":
    get_lightnovel()