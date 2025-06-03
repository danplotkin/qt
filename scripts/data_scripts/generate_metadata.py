import os
import torch
import json
from tqdm import tqdm

def generate_metadata_for_dir(data_dir):
    pt_files = [f for f in os.listdir(data_dir) if f.endswith(".pt")]

    for file in tqdm(pt_files, desc=f"Generating metadata in {data_dir}"):
        pt_path = os.path.join(data_dir, file)
        json_path = pt_path.replace(".pt", ".json")

        if os.path.exists(json_path):
            continue  # skip if metadata already exists

        try:
            tokens = torch.load(pt_path, mmap=True)
            length = len(tokens)
            with open(json_path, "w") as f:
                json.dump({"length": length}, f)
        except Exception as e:
            print(f"Failed to process {file}: {e}")

if __name__ == "__main__":
    # Path to your Reddit tokenized dataset
    reddit_dir = "data/flattened_corpa/reddit_comments"
    generate_metadata_for_dir(reddit_dir)