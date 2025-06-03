import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())
from utils.configs import load_configs, TrainingConfigs, TransformerConfigs
from utils.training import Trainer
from utils.transformer.model import QT
from utils.losses import SequenceLoss
from utils.metrics import MaskedAccuracy
from utils.tokenizer import get_tokenizer
from utils.torch_datasets import RedditCommentsDataset


@torch.no_grad()
def init_output_bias_from_dataloader(model: QT, loader: DataLoader) -> None:
    """
    Compute a log-frequency bias by streaming through loader batches.
    Only keeps a [vocab_size] counter tensor in memory.
    """
    vocab_size = model.config.tgt_vocab_size
    counts = torch.zeros(vocab_size, dtype=torch.long)
    total = 0

    for batch in tqdm(loader, desc="Initializing output bias"):
        tokens = batch[0].view(-1)
        tokens = tokens[tokens < vocab_size]  # Remove OOV tokens
        counts += torch.bincount(tokens, minlength=vocab_size)
        total += tokens.numel()

    # Zero out special tokens from frequency counts
    for special_id in model.tokenizer.all_special_ids:
        if 0 <= special_id < vocab_size:
            counts[special_id] = 0

    freqs = counts.float().div(total).clamp(min=1e-8)
    bias = torch.log(freqs)

    # assume your model’s final bias lives at model.fc.bias
    model.fc.bias.data.copy_(bias)


def configure_trainer(init_bias: bool = False) -> tuple[Trainer, DataLoader]:
    tokenizer = get_tokenizer()
    configs = load_configs()
    training_configs: TrainingConfigs = configs['training']
    transformer_configs: TransformerConfigs = configs['transformer']
    model = QT(config=transformer_configs, tokenizer=tokenizer, device='cpu')
    train_ds = RedditCommentsDataset(split='train', block_size=transformer_configs.max_seq_length)
    val_ds = RedditCommentsDataset(split='val', block_size=transformer_configs.max_seq_length)
    test_loader = RedditCommentsDataset(split='test', block_size=transformer_configs.max_seq_length)
    train_loader = DataLoader(train_ds, batch_size=training_configs.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=training_configs.batch_size, shuffle=False)

    if init_bias:
        init_output_bias_from_dataloader(model, train_loader)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_configs,
        criterion=SequenceLoss(ignore_index=tokenizer.pad_token_id),
        metric=MaskedAccuracy(padding_token_id=tokenizer.pad_token_id)
    )
    return trainer, test_loader


def main():
    parser = argparse.ArgumentParser(description="Pretrain QT model.")
    parser.add_argument(
        "--init-bias",
        action="store_true",
        help="If set, initializes the output bias using frequency distribution from the training set"
    )
    args = parser.parse_args()

    trainer, test_loader = configure_trainer(init_bias=args.init_bias)
    trainer.train()
    trainer.test(test_loader=test_loader)


if __name__ == '__main__':
    main()