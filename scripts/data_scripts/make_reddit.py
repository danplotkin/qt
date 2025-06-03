import os
import sys
sys.path.append(os.getcwd())
from datasets import load_dataset
import torch
from tqdm import tqdm
import logging
from utils.tokenizer import get_tokenizer

FLATTENED_CORPA_DIR = "data/flattened_corpa/reddit_comments"
os.makedirs(FLATTENED_CORPA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def tokenize_and_flatten_stream(tokenizer, subreddit_name):
    label = subreddit_name
    output_path = os.path.join(FLATTENED_CORPA_DIR, f"{label}.pt")

    logger.info(f"Streaming and tokenizing: {label}")
    dataset = load_dataset("HuggingFaceGECLM/REDDIT_comments", split=subreddit_name, streaming=True)

    all_ids = []
    for example in tqdm(dataset, desc=f"Tokenizing {label}"):
        tokenized = tokenizer(example["body"], add_special_tokens=False, return_attention_mask=False)
        all_ids.extend(tokenized["input_ids"])

    tensor = torch.tensor(all_ids, dtype=torch.long)
    torch.save(tensor, output_path)
    logger.info(f"Saved tensor: {output_path} | shape={tensor.shape}")
 

def main():
    tokenizer = get_tokenizer()
    # Dan's reddits
    # subreddits = ["bestof", "bodyweightfitness", "buildapc", "tifu", "explainlikeimfive", "WritingPrompts"]
    # Jack's reddits
    subreddits = ["podcasts"]

    for subreddit in subreddits:
        output_path = os.path.join(FLATTENED_CORPA_DIR, f"{subreddit}.pt")
        if os.path.exists(output_path):
            logger.info(f"Already tokenized: {output_path}. Skipping.")
            continue
        tokenize_and_flatten_stream(tokenizer, subreddit)


if __name__ == "__main__":
    main()